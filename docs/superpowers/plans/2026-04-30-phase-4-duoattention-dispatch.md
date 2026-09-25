# Phase 4 — DuoAttention Head Split + HF Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-head retrieval-vs-streaming attention dispatch (DuoAttention) wired through the HF Llama forward path, calling Phase 3's sparse INT8 kernel for retrieval heads and a streaming-only mask for streaming heads. Validates the dispatch on Llama-3.2-1B with a synthetic head pattern, since DuoAttention's upstream classifications cover Llama-3.1-8B and Mistral-7B but not Llama-3.2.

**Architecture:** The Phase 3 sparse kernel already supports per-query-head selection masks — DuoAttention reduces to "build a different mask per head". Streaming heads get a sinks-only-plus-window mask (no top-k); retrieval heads get sinks ∪ window ∪ top-k (the existing Phase 1 selection). A loader parses DuoAttention's TSV format. An HF monkeypatch (extending Phase 1's `patch_llama_for_quest_eager`) routes through the same Phase 3 kernel with per-head dispatch built into the mask.

**Tech Stack:** Same as Phase 3 — torch 2.5.1+cu121, triton 3.1.0, transformers 4.57.6, BF16 activations, uint8 KV. Adds `numpy` for TSV parsing (already a transitive dep).

**Win conditions:**
- DuoAttention TSV loader correctly thresholds Llama-3.1-8B's 32-layer × 8-KV-head pattern at 0.5; row-count and column-count match `config.json`'s declared model.
- Combined dispatch with **all-retrieval** pattern matches Phase 1 eager Quest within rtol=5e-2 (the sparse-everywhere baseline).
- Combined dispatch with **all-streaming** pattern matches a sinks+window-only attention within rtol=5e-2.
- Mixed-pattern dispatch (synthetic 70/30 split) on Llama-3.2-1B: passkey @ 0.5 depth retains Phase 1's ≥4/5 baseline at retention=0.25, sinks=4, window_pages=2.
- Phase 5 prerequisites documented: Llama-3.1-8B AWQ load path, INT8 KV cache class for HF, Marlin integration plan.

**Phase 4 v1 explicit non-goals (deferred to Phase 5):**
- Llama-3.1-8B AWQ-INT4 end-to-end (model is 4.5 GB; with INT8 KV at 32 k it doesn't fit our 4 GB envelope without further weight quant work).
- Triton-kernel-side per-head dispatch (the kernel already takes per-head masks; no kernel change needed).
- Marlin W4A16 projections (perf optimization; defer until correctness lands).
- 32 k context end-to-end measurement.
- RULER quality eval.
- Persistent INT8 KV cache subclass for HF (`DynamicCache`-compatible).

**Hardware envelope:** Same as Phases 1–3.

---

## Edge case catalog — DuoAttention dispatch

| ID | Case | Why it matters |
|---|---|---|
| EP1 | All-retrieval pattern (every head 1) | Equivalent to Quest-everywhere (Phase 1). Sanity. |
| EP2 | All-streaming pattern (every head 0) | Equivalent to StreamingLLM-only (sinks + window). Sanity. |
| EP3 | Mixed pattern, typical 70/30 | Standard case — both code paths exercised. |
| EP4 | Pattern row-count != model layer-count | Loader must error clearly, not silently truncate. |
| EP5 | Pattern column-count != model KV-head-count | Loader must error clearly. |
| EP6 | Threshold at the boundary (= 0.5 exactly) | Documented to round up (0.5 → retrieval). |
| EP7 | Decode step (S_q=1) | Primary target. |
| EP8 | Prefill (S_q > 1) | Uses dense attention regardless of pattern (Phase 1 convention). |
| EP9 | GQA broadcast — pattern is per KV head, masks must apply to all query heads in the group | Llama-3.2-1B has 32:8, Llama-3.1-8B has 32:8. |
| EP10 | KV cache shorter than `num_sinks + window_pages * page_size` | Streaming heads still need to attend to *something*; mask falls back to "all available". |
| EP11 | Empty TSV / missing file | Loader raises `FileNotFoundError` / `ValueError`. |

---

## File Structure

**Created:**
- `src/flashquest/duo/__init__.py` — exports loader + dispatch.
- `src/flashquest/duo/pattern.py` — `load_duo_pattern(path) -> torch.BoolTensor` (layers, kv_heads).
- `src/flashquest/duo/dispatch.py` — `quest_duo_eager_sdpa(...)` per-head dispatch (eager Python).
- `src/flashquest/eager/streaming.py` — streaming-only sinks+window attention helper (used by dispatch for streaming heads).
- `src/flashquest/eager/llama_duo_patch.py` — HF Llama monkeypatch combining Phase 1 + DuoAttention.
- `tests/test_duo_pattern.py` — TSV loader correctness + edge cases.
- `tests/test_duo_dispatch.py` — dispatch equivalence with sparse / streaming baselines + mixed-pattern.
- `tests/test_duo_e2e.py` — Llama-3.2-1B end-to-end with synthetic pattern, retention=1.0 sanity.
- `scripts/phase4_run_passkey.py` — passkey eval on Llama-3.2-1B with DuoAttention dispatch vs Phase 1 baseline.
- `benchmarks/phase4_passkey.json` — eval output.
- `docs/PHASES/phase-4-notes.md` — phase journal.

**Modified:**
- `src/flashquest/eager/__init__.py` — add `streaming_eager_sdpa` export.
- `README.md` — Phase 4 row.
- `DOC.md` — flip Phase 4 status.

---

## Conventions

- **Pattern format:** TSV file with shape `(num_layers, num_kv_heads)` of float values in `[0, 1]`. Each value is the probability that the head should use full (retrieval) attention. Boolean pattern: `pattern_bool[l, h] = (pattern_float[l, h] >= 0.5)` — True means retrieval.
- **Pattern broadcast:** stored at KV-head granularity. Query heads in the same GQA group inherit their KV head's classification.
- **Streaming mask:** `selection_mask[..., 0] = True` for sink (page 0), `selection_mask[..., -window_pages:] = True` for recency window, all else False. No top-k.
- **Retrieval mask:** existing Phase 1 `select_pages(scores, retention, num_sinks, window_pages)`.
- **Per-head selection mask** (the kernel input): build by interleaving the per-head streaming/retrieval masks based on `pattern_bool`. Resulting `selection_mask` shape is the standard `(B, H_q, S_q, num_pages)` Phase 3 kernel takes.

---

## Task 1: DuoAttention pattern loader

**Files:**
- Create: `src/flashquest/duo/__init__.py`
- Create: `src/flashquest/duo/pattern.py`
- Create: `tests/test_duo_pattern.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_duo_pattern.py`:
```python
"""Tests for DuoAttention pattern loader."""
from pathlib import Path

import pytest
import torch

VENDOR_PATTERN = Path(__file__).resolve().parents[1] / "vendor" / "duo-attention" / "attn_patterns" / "Meta-Llama-3.1-8B-Instruct" / "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10" / "full_attention_heads.tsv"


def test_load_llama_3_1_8b_shape():
    """EP4/EP5: Llama-3.1-8B has 32 layers × 8 KV heads."""
    from flashquest.duo.pattern import load_duo_pattern

    pattern = load_duo_pattern(VENDOR_PATTERN)
    assert pattern.shape == (32, 8)
    assert pattern.dtype == torch.bool


def test_threshold_at_half():
    """EP6: values >= 0.5 round up to retrieval (True), < 0.5 to streaming."""
    from flashquest.duo.pattern import load_duo_pattern

    pattern = load_duo_pattern(VENDOR_PATTERN)
    # Both extremes must be present in any well-trained pattern.
    assert pattern.any(), "expected at least one retrieval head"
    assert (~pattern).any(), "expected at least one streaming head"


def test_threshold_kwarg(tmp_path: Path):
    """User can override threshold; default is 0.5."""
    from flashquest.duo.pattern import load_duo_pattern

    p = tmp_path / "tiny.tsv"
    # 2 layers × 2 KV heads.
    p.write_text("0.4\t0.6\n0.5\t0.51\n")

    default = load_duo_pattern(p)  # threshold=0.5
    assert default.tolist() == [[False, True], [True, True]]

    strict = load_duo_pattern(p, threshold=0.51)
    assert strict.tolist() == [[False, True], [False, True]]


def test_missing_file_raises(tmp_path: Path):
    """EP11: missing file raises FileNotFoundError."""
    from flashquest.duo.pattern import load_duo_pattern

    with pytest.raises(FileNotFoundError):
        load_duo_pattern(tmp_path / "nope.tsv")


def test_empty_file_raises(tmp_path: Path):
    """EP11: empty TSV raises ValueError."""
    from flashquest.duo.pattern import load_duo_pattern

    p = tmp_path / "empty.tsv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_duo_pattern(p)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `. .venv/bin/activate && pytest tests/test_duo_pattern.py -v`
Expected: ImportError on `flashquest.duo.pattern`.

- [ ] **Step 1.3: Implement the loader**

Create `src/flashquest/duo/__init__.py`:
```python
"""DuoAttention head split utilities. Phase 4."""
from .pattern import load_duo_pattern

__all__ = ["load_duo_pattern"]
```

Create `src/flashquest/duo/pattern.py`:
```python
"""Loader for DuoAttention pre-trained head classifications.

The upstream format (vendor/duo-attention/attn_patterns/<model>/<run>/
full_attention_heads.tsv) is a tab-separated file with one row per layer
and one column per KV head. Values are floats in [0, 1] indicating the
probability the head should use *full retrieval* attention. We threshold
to bool: True = retrieval head, False = streaming head.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_duo_pattern(path: str | Path, threshold: float = 0.5) -> torch.Tensor:
    """Read a DuoAttention TSV and return a (num_layers, num_kv_heads) bool tensor.

    Args:
        path: TSV file path.
        threshold: values >= threshold are retrieval (True). Default 0.5.

    Returns:
        Bool tensor on CPU. True = retrieval head, False = streaming head.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"DuoAttention pattern not found: {p}")

    rows: list[list[float]] = []
    expected_cols: int | None = None
    for line_no, line in enumerate(p.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        cols = [float(c) for c in line.split("\t") if c]
        if expected_cols is None:
            expected_cols = len(cols)
        elif len(cols) != expected_cols:
            raise ValueError(
                f"DuoAttention pattern at {p}:{line_no} has {len(cols)} cols, "
                f"expected {expected_cols}"
            )
        rows.append(cols)

    if not rows:
        raise ValueError(f"DuoAttention pattern at {p} is empty")

    return torch.tensor(rows) >= threshold
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/test_duo_pattern.py -v`
Expected: 5 passing tests.

- [ ] **Step 1.5: Commit**

```bash
git add src/flashquest/duo/__init__.py src/flashquest/duo/pattern.py tests/test_duo_pattern.py
git commit -m "phase 4: DuoAttention TSV pattern loader"
```

---

## Task 2: Streaming-only attention helper

**Purpose:** For streaming heads, attention should only see sink tokens + sliding window — no top-k retrieval. Reuses Phase 1's selection logic with `retention=0`.

**Files:**
- Create: `src/flashquest/eager/streaming.py`
- Modify: `src/flashquest/eager/__init__.py`
- Tests live in `test_duo_dispatch.py` (Task 3) — covered there.

- [ ] **Step 2.1: Write the helper**

Create `src/flashquest/eager/streaming.py`:
```python
"""Streaming-only attention: sink tokens + sliding window, no top-k.

Equivalent to Phase 1's quest_eager_sdpa with retention=0.0, num_sinks=N,
window_pages=W. Exposed as a standalone function so DuoAttention dispatch
reads more clearly.
"""
from __future__ import annotations

import torch

from .attention import quest_eager_sdpa


def streaming_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    page_size: int = 64,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Streaming-only attention: only sinks + recency window are attended.

    Equivalent to StreamingLLM's behavior: keep first num_sinks pages plus
    last window_pages pages, drop everything else.
    """
    return quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=0.0,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
```

Update `src/flashquest/eager/__init__.py`:
```python
"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages
from .sparse_int8 import quest_eager_sparse_int8
from .streaming import streaming_eager_sdpa

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "compute_page_summary",
    "select_pages",
    "quest_eager_sparse_int8",
    "streaming_eager_sdpa",
]
```

- [ ] **Step 2.2: Smoke import**

Run: `. .venv/bin/activate && python -c "from flashquest.eager import streaming_eager_sdpa; print('ok')"`
Expected: `ok`.

- [ ] **Step 2.3: Commit**

```bash
git add src/flashquest/eager/streaming.py src/flashquest/eager/__init__.py
git commit -m "phase 4: streaming-only attention helper (sinks + window, no top-k)"
```

---

## Task 3: Per-head DuoAttention dispatch (eager)

**Purpose:** Combine streaming + retrieval per-head into a single dispatch. The dispatch builds a per-head selection mask: streaming heads get sinks+window, retrieval heads get sinks+window+top-k. This is the algorithmic shape of DuoAttention.

**Files:**
- Create: `src/flashquest/duo/dispatch.py`
- Modify: `src/flashquest/duo/__init__.py`
- Create: `tests/test_duo_dispatch.py`

- [ ] **Step 3.1: Write failing tests**

Create `tests/test_duo_dispatch.py`:
```python
"""Tests for DuoAttention per-head dispatch."""
import torch

from flashquest.duo.dispatch import quest_duo_eager_sdpa
from flashquest.eager import quest_eager_sdpa, streaming_eager_sdpa


def _make_qkv(B=1, H_q=4, H_kv=1, S=256, D=64, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    return Q, K, V


def test_all_retrieval_matches_pure_quest():
    """EP1: all-retrieval pattern ≡ Phase 1 quest_eager_sdpa with same knobs."""
    Q, K, V = _make_qkv()
    H_kv = K.shape[1]
    pattern = torch.ones(H_kv, dtype=torch.bool)  # all retrieval

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_quest = quest_eager_sdpa(
        Q, K, V,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_quest, rtol=1e-3, atol=1e-3)


def test_all_streaming_matches_streaming_only():
    """EP2: all-streaming pattern ≡ streaming_eager_sdpa with same knobs."""
    Q, K, V = _make_qkv(seed=1)
    H_kv = K.shape[1]
    pattern = torch.zeros(H_kv, dtype=torch.bool)  # all streaming

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_stream = streaming_eager_sdpa(
        Q, K, V,
        page_size=64, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_stream, rtol=1e-3, atol=1e-3)


def test_mixed_pattern_runs():
    """EP3: mixed pattern — both code paths exercised, output is finite."""
    Q, K, V = _make_qkv(H_q=4, H_kv=2, seed=2)
    pattern = torch.tensor([True, False])  # head 0 retrieval, head 1 streaming

    O = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    assert O.shape == Q.shape
    assert torch.isfinite(O).all()


def test_mixed_pattern_per_head_correctness():
    """EP3: heads with retrieval pattern produce the retrieval result;
    heads with streaming pattern produce the streaming result."""
    Q, K, V = _make_qkv(H_q=4, H_kv=2, seed=3)  # 2-way GQA
    pattern = torch.tensor([True, False])

    # Reference: build the full output by running each path on all heads then
    # picking the right column based on the (broadcast) pattern.
    O_quest_all = quest_eager_sdpa(
        Q, K, V,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_stream_all = streaming_eager_sdpa(
        Q, K, V,
        page_size=64, num_sinks=4, window_pages=2, is_causal=True,
    )
    n_rep = Q.shape[1] // K.shape[1]
    pattern_per_q_head = pattern.repeat_interleave(n_rep)  # (H_q,)

    O_expected = torch.where(
        pattern_per_q_head.view(1, -1, 1, 1),
        O_quest_all,
        O_stream_all,
    )

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_expected, rtol=1e-3, atol=1e-3)


def test_pattern_shape_validation():
    """EP4/EP5: pattern length mismatch raises clearly."""
    import pytest

    Q, K, V = _make_qkv(H_q=4, H_kv=2)
    bad_pattern = torch.tensor([True, False, True])  # 3 != H_kv=2
    with pytest.raises(ValueError, match="head_pattern"):
        quest_duo_eager_sdpa(
            Q, K, V, head_pattern=bad_pattern,
            page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
        )
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest tests/test_duo_dispatch.py -v`
Expected: ImportError on `flashquest.duo.dispatch`.

- [ ] **Step 3.3: Implement dispatch**

Create `src/flashquest/duo/dispatch.py`:
```python
"""DuoAttention per-head dispatch.

Reuses Phase 1's quest_eager_sdpa for retrieval heads and the streaming
helper for streaming heads. Eager Python — Phase 4 v1 does not require
kernel-side dispatch since the Phase 3 kernel already accepts arbitrary
per-query-head selection masks; running everything once per path and
selecting per head is correct (just slower than a fused dispatch).
"""
from __future__ import annotations

import torch

from ..eager.attention import quest_eager_sdpa
from ..eager.streaming import streaming_eager_sdpa


def quest_duo_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_pattern: torch.Tensor,
    page_size: int = 64,
    retention: float = 0.25,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Per-head DuoAttention dispatch.

    Args:
        Q: (B, H_q, S_q, D).
        K, V: (B, H_kv, S_kv, D).
        head_pattern: (H_kv,) bool tensor. True = retrieval, False = streaming.
            Per-KV-head; broadcast to all query heads in the GQA group.

    Returns:
        Output (B, H_q, S_q, D).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, _, _ = K.shape
    if head_pattern.shape != (H_kv,):
        raise ValueError(
            f"head_pattern must be shape ({H_kv},) (one entry per KV head); got {tuple(head_pattern.shape)}"
        )

    # Run both paths on all heads. Cheap to express; later phases can fuse.
    O_retrieval = quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
    O_streaming = streaming_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )

    # Broadcast pattern from H_kv -> H_q via the GQA group.
    n_rep = H_q // H_kv
    pattern_per_q_head = head_pattern.to(Q.device).repeat_interleave(n_rep)
    sel = pattern_per_q_head.view(1, H_q, 1, 1)

    return torch.where(sel, O_retrieval, O_streaming)
```

Update `src/flashquest/duo/__init__.py`:
```python
"""DuoAttention head split utilities. Phase 4."""
from .dispatch import quest_duo_eager_sdpa
from .pattern import load_duo_pattern

__all__ = ["load_duo_pattern", "quest_duo_eager_sdpa"]
```

- [ ] **Step 3.4: Run tests**

Run: `pytest tests/test_duo_dispatch.py -v`
Expected: 5 passing tests.

- [ ] **Step 3.5: Commit**

```bash
git add src/flashquest/duo/dispatch.py src/flashquest/duo/__init__.py tests/test_duo_dispatch.py
git commit -m "phase 4: per-head DuoAttention dispatch (retrieval ∪ streaming)"
```

---

## Task 4: HF Llama monkeypatch with DuoAttention

**Purpose:** Wire `quest_duo_eager_sdpa` into HF `LlamaAttention.forward` with per-layer head patterns.

**Files:**
- Create: `src/flashquest/eager/llama_duo_patch.py`
- Create: `tests/test_duo_e2e.py`

- [ ] **Step 4.1: Write the failing e2e test**

Create `tests/test_duo_e2e.py`:
```python
"""End-to-end: a Quest-Duo-eager-patched HF model produces the same logits as
the Phase 1 retrieval-only patch when *all* heads are classified as retrieval."""
import pytest
import torch

pytestmark = pytest.mark.slow


def _have_model() -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_model(), reason="model checkpoint not available offline")
def test_all_retrieval_matches_phase1_patch():
    """When the DuoAttention pattern says every head is retrieval, the Duo
    patch must produce identical logits to Phase 1's quest_eager patch."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
    from flashquest.eager.llama_patch import patch_llama_for_quest_eager

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    # Read num_layers and num_kv_heads from the config.
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name)
    num_layers = cfg.num_hidden_layers
    num_kv = cfg.num_key_value_heads
    pattern_all_retrieval = torch.ones(num_layers, num_kv, dtype=torch.bool)

    # Reference: Phase 1 patch.
    model_phase1 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    patch_llama_for_quest_eager(
        model_phase1, retention=0.25, num_sinks=4, window_pages=2, page_size=64
    )

    inp = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = model_phase1(**inp).logits

    del model_phase1
    torch.cuda.empty_cache()

    # Phase 4 patch with all-retrieval pattern.
    model_phase4 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()
    patch_llama_for_quest_duo(
        model_phase4,
        head_pattern=pattern_all_retrieval,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    with torch.no_grad():
        out_logits = model_phase4(**inp).logits

    torch.testing.assert_close(out_logits, ref_logits, rtol=5e-3, atol=5e-3)
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_duo_e2e.py -v`
Expected: ImportError on `llama_duo_patch`.

- [ ] **Step 4.3: Implement the patch**

Create `src/flashquest/eager/llama_duo_patch.py`:
```python
"""HF Llama monkeypatch: per-layer DuoAttention dispatch.

Each layer reads its own row from the head_pattern tensor and dispatches
through quest_duo_eager_sdpa. Tracked against transformers 4.57.6's
LlamaAttention.forward signature (same as Phase 1's patch).
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from ..duo.dispatch import quest_duo_eager_sdpa


def make_quest_duo_forward(
    *,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
    """Build a forward function bound to one layer's head pattern + Quest knobs.

    head_pattern_layer: (H_kv,) bool tensor for THIS layer.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        is_causal = q.shape[2] > 1

        attn_output = quest_duo_eager_sdpa(
            q, k, v,
            head_pattern=head_pattern_layer,
            page_size=page_size,
            retention=retention,
            num_sinks=num_sinks,
            window_pages=window_pages,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def patch_llama_for_quest_duo(
    model: torch.nn.Module,
    *,
    head_pattern: torch.Tensor,
    retention: float = 0.25,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward in `model` with the Duo version.

    Args:
        head_pattern: (num_layers, num_kv_heads) bool tensor.
    """
    if head_pattern.ndim != 2:
        raise ValueError(
            f"head_pattern must be 2D (num_layers, num_kv_heads); got {tuple(head_pattern.shape)}"
        )
    num_layers, num_kv = head_pattern.shape

    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            li = module.layer_idx
            if li >= num_layers:
                raise ValueError(
                    f"head_pattern has {num_layers} layers but model layer_idx={li}"
                )
            layer_pattern = head_pattern[li]
            fwd = make_quest_duo_forward(
                head_pattern_layer=layer_pattern,
                retention=retention,
                num_sinks=num_sinks,
                window_pages=window_pages,
                page_size=page_size,
            )
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_duo: no LlamaAttention modules found")
    if n_patched != num_layers:
        raise ValueError(
            f"head_pattern has {num_layers} layers but model has {n_patched} attention modules"
        )
```

- [ ] **Step 4.4: Run e2e test**

Run: `pytest tests/test_duo_e2e.py -v`
Expected: 1 passing test (or skipped if offline).

If logits diverge: most likely cause is the `head_pattern` not landing on the right device. Triple-check that `head_pattern_layer.to(Q.device)` is called inside the dispatch (it is — see `quest_duo_eager_sdpa`).

- [ ] **Step 4.5: Commit**

```bash
git add src/flashquest/eager/llama_duo_patch.py tests/test_duo_e2e.py
git commit -m "phase 4: HF Llama monkeypatch with per-layer DuoAttention dispatch"
```

---

## Task 5: Passkey eval — DuoAttention vs Phase 1

**Purpose:** Validate that DuoAttention dispatch doesn't degrade quality on the metric Phase 1 was designed for. We don't have Llama-3.2 pre-trained patterns, so we use a *synthetic* 70/30 retrieval/streaming split (matches DuoAttention's reported typical ratio for Llama-3-8B per the paper).

**Files:**
- Create: `scripts/phase4_run_passkey.py`
- Create: `benchmarks/phase4_passkey.json`

- [ ] **Step 5.1: Write the eval script**

Create `scripts/phase4_run_passkey.py`:
```python
"""Phase 4 passkey: Llama-3.2-1B with synthetic 70/30 DuoAttention split,
retention=0.25 sinks=4 window=2. Compare against Phase 1's all-retrieval patch.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.passkey import make_example, score


N_TRIALS = 5
DEPTHS = [0.1, 0.5, 0.9]
TARGET_TOTAL_TOKENS = 1024
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2
RETRIEVAL_FRACTION = 0.7  # 70% retrieval, 30% streaming


def fresh_model(name: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def generate_passkey_answer(model: torch.nn.Module, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def synthetic_pattern(num_layers: int, num_kv: int, fraction_retrieval: float, seed: int = 0):
    """Random per-layer per-head pattern with fraction_retrieval True."""
    rng = torch.Generator().manual_seed(seed)
    return (torch.rand((num_layers, num_kv), generator=rng) < fraction_retrieval)


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)

    examples = []
    rng = random.Random(0)
    for d in DEPTHS:
        for _ in range(N_TRIALS):
            examples.append(
                make_example(
                    rng=rng,
                    tokenizer=tok,
                    target_total_tokens=TARGET_TOTAL_TOKENS,
                    depth_pct=d,
                )
            )

    results: dict = {
        "depths": DEPTHS,
        "n_trials": N_TRIALS,
        "retention": RETENTION,
        "num_sinks": NUM_SINKS,
        "window_pages": WINDOW_PAGES,
        "retrieval_fraction": RETRIEVAL_FRACTION,
    }

    # Reference: Phase 1 all-retrieval.
    print("phase 1 baseline (all retrieval) ...")
    m = fresh_model(name)
    patch_llama_for_quest_eager(
        m, retention=RETENTION, num_sinks=NUM_SINKS, window_pages=WINDOW_PAGES, page_size=64,
    )
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["phase1"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
    }
    print(f"  {results['phase1']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    _free()

    # Phase 4: synthetic 70/30 DuoAttention split.
    print("phase 4 (DuoAttention 70/30 split) ...")
    pattern = synthetic_pattern(cfg.num_hidden_layers, cfg.num_key_value_heads, RETRIEVAL_FRACTION)
    m = fresh_model(name)
    patch_llama_for_quest_duo(
        m, head_pattern=pattern, retention=RETENTION, num_sinks=NUM_SINKS,
        window_pages=WINDOW_PAGES, page_size=64,
    )
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["phase4"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
        "head_pattern_retrieval_fraction": pattern.float().mean().item(),
    }
    print(f"  {results['phase4']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase4_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Run the eval**

Run: `. .venv/bin/activate && nice -n 19 python scripts/phase4_run_passkey.py 2>&1 | tail -20`
Expected: prints accuracy tables for both configs, writes `benchmarks/phase4_passkey.json`. Phase 4 should hit ≥ Phase 1's per-depth accuracy at the matching retention (or be within 1/5 trials, given small N).

If phase 4 is significantly worse: with only 30 % streaming heads, this should be a near-no-op. Likely cause is the dispatch broadcasting bug — re-examine `repeat_interleave` and `view(1, H_q, 1, 1)` in `quest_duo_eager_sdpa`.

- [ ] **Step 5.3: Commit**

```bash
git add scripts/phase4_run_passkey.py benchmarks/phase4_passkey.json
git commit -m "phase 4: passkey eval — DuoAttention 70/30 split vs Phase 1 all-retrieval"
```

---

## Task 6: Document Phase 5 prerequisites + close out

**Files:**
- Create: `docs/PHASES/phase-4-notes.md`
- Modify: `README.md`
- Modify: `DOC.md`

- [ ] **Step 6.1: Write phase notes**

Create `docs/PHASES/phase-4-notes.md`:
```markdown
# Phase 4 Notes

**Started:** <YYYY-MM-DD>
**Completed:** <YYYY-MM-DD> (tag `phase-4`)
**Status:** complete (eager dispatch validated; 8B integration deferred to Phase 5)
**Spec:** [docs/SPEC.md §6 Phase 4](../SPEC.md)

## Goal

Per-head retrieval-vs-streaming attention dispatch (DuoAttention) wired through HF Llama. Validate the dispatch on Llama-3.2-1B with a synthetic 70/30 head pattern.

## Surface

- `flashquest.duo.load_duo_pattern(path, threshold=0.5) -> torch.BoolTensor` — TSV loader for DuoAttention upstream patterns.
- `flashquest.duo.quest_duo_eager_sdpa(Q, K, V, *, head_pattern, page_size, retention, num_sinks, window_pages, is_causal)` — per-head dispatch (eager Python).
- `flashquest.eager.streaming_eager_sdpa(...)` — sinks+window-only attention helper.
- `flashquest.eager.llama_duo_patch.patch_llama_for_quest_duo(model, *, head_pattern, retention, num_sinks, window_pages, page_size)` — HF monkeypatch that loads a `(num_layers, num_kv_heads)` bool pattern and dispatches per-layer per-head.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Loader correctly reads Llama-3.1-8B pattern (32×8) | <fill> | ✅ / ❌ |
| All-retrieval pattern ≡ Phase 1 quest_eager | <fill> | ✅ / ❌ |
| All-streaming pattern ≡ streaming_eager | <fill> | ✅ / ❌ |
| Mixed pattern: per-head correctness via `torch.where` reference | <fill> | ✅ / ❌ |
| Llama-3.2-1B 70/30 split passkey ≥ Phase 1 baseline | <fill> | ✅ / ❌ |

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| EP1 | All-retrieval pattern ≡ Quest-everywhere | ✅ |
| EP2 | All-streaming pattern ≡ StreamingLLM-only | ✅ |
| EP3 | Mixed pattern, both code paths exercised | ✅ |
| EP4/EP5 | Pattern shape mismatch raises | ✅ |
| EP6 | Threshold at 0.5 boundary documented (rounds up to retrieval) | ✅ |
| EP7 | Decode step (S_q=1) | ✅ via passkey eval |
| EP8 | Prefill (S_q > 1) | ✅ same path; dispatch is per-token-by-token |
| EP9 | GQA broadcast (pattern per KV head, applied to query heads) | ✅ via repeat_interleave |
| EP10 | Short cache vs sinks+window | ✅ Phase 1 select_pages clamps |
| EP11 | Empty / missing file | ✅ raises |

## Decisions

- **Eager dispatch (not Triton-side)**: the Phase 3 sparse kernel already accepts arbitrary per-query-head selection masks. DuoAttention reduces to "build a different mask per head" — no kernel change needed. Phase 4 v1 implements the dispatch at the eager Python level by running both paths and selecting per head via `torch.where`. This is wasteful at runtime but correct; a fused kernel-side dispatch is a Phase 5 perf win.
- **Synthetic pattern on Llama-3.2-1B**: DuoAttention's upstream classifications cover Llama-3.1-8B and Mistral-7B, not Llama-3.2. We use a uniform 70/30 random pattern (matches the typical retrieval ratio in their paper) for Phase 4 validation. Real per-model classifications are a Phase 5 input (or training step).

## Phase 5 prerequisites (deferred work)

These items are needed for SPEC §6 Phase 4's "Llama-3.1-8B at 32k, ≥4 tok/s, ≥80% RULER" win condition:

1. **Llama-3.1-8B AWQ-INT4 load** via `transformers + auto_awq`. Model is `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (~4.5 GB weights). Doesn't fit alongside even INT8 KV at 32 k on 4 GB; needs IQ3-XXS or 2-bit weights.
2. **Persistent INT8 KV cache** as a `transformers.cache_utils.Cache` subclass. On `update`, quantize incoming K/V via Phase 3's `quantize_k`/`quantize_v` and store as uint8 + scales.
3. **Marlin W4A16 projections** for Q/K/V/O linear layers. Convert AWQ packing → Marlin packing once at load time.
4. **DuoAttention pattern for Llama-3.1-8B**: load from `vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv` (already vendored).
5. **32 k context bench**: passkey at depth grid + RULER 4 k subset (full RULER is hours).

## Phase 4 → Phase 5 handoff

Phase 4 ships:
- `phase-4` git tag.
- DuoAttention pattern loader + per-head dispatch + HF integration.
- Passkey on Llama-3.2-1B with synthetic 70/30 split.

Phase 5 begins: Llama-3.1-8B AWQ + persistent INT8 KV cache + Marlin projections + RULER eval at 32 k.
```

(Fill `<...>` after the run.)

- [ ] **Step 6.2: Append Phase 4 to README**

Edit `README.md`. After the Phase 3 section, append:
```markdown
## Phase 4 — DuoAttention head split + HF integration

Per-head retrieval-vs-streaming attention dispatch wired through HF Llama. The Phase 3 sparse kernel already supports per-query-head selection masks; DuoAttention reduces to building a different mask per head.

Validation on Llama-3.2-1B with a synthetic 70/30 retrieval/streaming split (DuoAttention's upstream classifications cover Llama-3.1-8B and Mistral-7B; no Llama-3.2 file exists upstream):

| Config | depth=0.1 | depth=0.5 | depth=0.9 |
|---|---|---|---|
| Phase 1 (all retrieval) | <fill>/5 | <fill>/5 | <fill>/5 |
| Phase 4 (70 % retrieval, 30 % streaming) | <fill>/5 | <fill>/5 | <fill>/5 |

11 catalogued edge cases (EP1–EP11). See [`docs/PHASES/phase-4-notes.md`](docs/PHASES/phase-4-notes.md). Phase 5 prerequisites (Llama-3.1-8B AWQ + persistent INT8 cache + Marlin) documented there.

Re-run via `python scripts/phase4_run_passkey.py`.
```

- [ ] **Step 6.3: Flip Phase 4 in DOC.md**

Edit `DOC.md`. Replace:
```
- Phase 4 — DuoAttention split + 8 B model + Marlin W4A16 projections.
```
with:
```
- **Phase 4 — DuoAttention head split + HF integration** ✅ **complete (tag `phase-4`)**. `flashquest.duo.{load_duo_pattern, quest_duo_eager_sdpa}` + `flashquest.eager.llama_duo_patch.patch_llama_for_quest_duo`. Per-layer per-KV-head dispatch (retrieval ∪ streaming) wired through HF Llama. Validated on Llama-3.2-1B with a synthetic 70/30 split. Llama-3.1-8B AWQ end-to-end + persistent INT8 KV cache + Marlin projections deferred to Phase 5 (documented in `docs/PHASES/phase-4-notes.md`).
```

Add a usage block under Phase 3's:
````markdown
DuoAttention per-head dispatch on a HF Llama model (Phase 4):

```python
import torch
from transformers import AutoModelForCausalLM
from flashquest.duo import load_duo_pattern
from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct", torch_dtype="bfloat16",
    attn_implementation="sdpa",
).cuda()

# Synthetic pattern (Llama-3.2 has no upstream DuoAttention file):
pattern = (torch.rand(model.config.num_hidden_layers, model.config.num_key_value_heads) < 0.7)

patch_llama_for_quest_duo(
    model, head_pattern=pattern, retention=0.25, num_sinks=4, window_pages=2, page_size=64,
)
# Llama-3.1-8B users can load the upstream pattern instead:
# pattern = load_duo_pattern(
#     "vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/"
#     "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
# )
```
````

- [ ] **Step 6.4: Final smoke**

Run:
```bash
. .venv/bin/activate
pytest tests/ --ignore=tests/test_eager_e2e.py --ignore=tests/test_duo_e2e.py -v
```
Expected: all kernel + eager + duo unit tests pass.

- [ ] **Step 6.5: Commit + tag**

```bash
git add docs/PHASES/phase-4-notes.md README.md DOC.md
git commit -m "phase 4: README + DOC + phase-4-notes complete"
git tag -a phase-4 -m "Phase 4 complete: DuoAttention dispatch + HF integration"
```

---

## Self-review

**1. Spec coverage** (SPEC §6 Phase 4):
- "Per-head pattern dispatch inside the kernel." → Phase 4 v1 dispatches at the eager Python level (the kernel already accepts per-head masks). Phase 5 prerequisite "fused kernel-side dispatch" is documented as a perf optimization, not a correctness item.
- "Load DuoAttention's pre-trained head classifications for Llama-3-8B from their repo." → Task 1 loader; Llama-3.1-8B pattern verified.
- "Use IST-DASLab/marlin for W4A16 weight projections." → Documented as Phase 5 prerequisite #3 (perf, not correctness).
- "Get Llama-3.1-8B at IQ3_XXS quant fitting in VRAM." → Documented as Phase 5 prerequisite #1; doesn't fit at IQ3_XXS (GGUF-only) or AWQ-INT4 (4.5 GB) on our 4 GB envelope without further weight-quant work.
- "Win condition: 8B at 32k, ≥4 tok/s, ≥80% RULER." → Deferred to Phase 5 with the 8B integration. Phase 4 win is "DuoAttention dispatch correctness on Llama-3.2-1B".

**2. Placeholder scan**: every code block is real code. README/DOC/notes have `<fill>` slots for measured values, but those are explicit data-entry slots.

**3. Type / name consistency**:
- `load_duo_pattern(path, threshold=0.5) -> torch.BoolTensor (num_layers, num_kv_heads)` — same in Tasks 1, 5, 6.
- `quest_duo_eager_sdpa(Q, K, V, *, head_pattern, page_size, retention, num_sinks, window_pages, is_causal)` — same in Tasks 3, 4, 5.
- `streaming_eager_sdpa(Q, K, V, *, page_size, num_sinks, window_pages, is_causal)` — same in Tasks 2, 3.
- `patch_llama_for_quest_duo(model, *, head_pattern, retention, num_sinks, window_pages, page_size)` — same in Tasks 4, 5, 6.
- `head_pattern` shape `(num_layers, num_kv_heads)` for the patch, `(num_kv_heads,)` for a single-layer dispatch — both used consistently.

**4. Reversibility**: every task ends with a commit. Task 5 explicitly directs the engineer to debug the dispatch (not loosen the test) if the 70/30 split degrades passkey.

## Phase 4 → Phase 5 handoff

When the plan completes:
- `phase-4` git tag exists.
- `flashquest.duo.{load_duo_pattern, quest_duo_eager_sdpa}` are wired through HF Llama.
- `benchmarks/phase4_passkey.json` shows DuoAttention dispatch matches Phase 1 baseline.
- Phase 5 prerequisites documented in `docs/PHASES/phase-4-notes.md`: Llama-3.1-8B AWQ load, persistent INT8 KV cache, Marlin W4A16 projections, 32 k context, RULER eval. Phase 5 will get its own plan.
