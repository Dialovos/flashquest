# Phase 6 Criticality Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the SPEC §6 task 1 gate — Llama-3.2-3B-AWQ decode at 32k context from 0.10 → ≥4 tok/s — by replacing the per-decode-step `dequantize_k → compute_page_summary → page_scores → select_pages` chain with an algebraic-direct page scorer (no dequant, no GQA broadcast on the cache) and a vectorised top-k that batches all heads in a single `torch.topk` call.

**Architecture:** Two new pure-PyTorch functions added alongside the existing reference implementations (which stay for the eager path and tests):
1. `page_scores_int8(Q, K_scale, K_mn) → scores` computes Quest's per-page upper-bound criticality directly from the cache's quantization parameters using the identity `page_min ≡ K_mn`, `page_max ≡ K_mn + 255 * K_scale` (exact by construction in `kv_quant.py`).
2. `select_pages_vectorized(scores, retention_per_h, num_sinks, window_pages) → mask` does one batched `topk(k_max)` over all heads, masks ranks beyond `k_per_h[h]` with a broadcasted comparison, scatters in a single op. No Python `for` loop, no `.item()` per head.

These are wired into `_quest_duo_fused_with_lse` in `llama_persistent_patch.py`. Phase 5 implementations stay untouched on disk; the patched LlamaAttention forward swaps which functions it calls.

**Tech Stack:** PyTorch (no Triton), pytest, transformers Llama. CUDA on RTX 3050 Ti Laptop (sm_86) under WSL2.

**Reference:** `docs/superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md`. **Profile evidence:** `benchmarks/phase6_profile.json` and `scripts/phase6_profile_decode.py`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/flashquest/eager/criticality.py` | `page_scores` (existing) and new `page_scores_int8` | Modify |
| `src/flashquest/eager/selection.py` | `select_pages` (existing) and new `select_pages_vectorized` | Modify |
| `src/flashquest/eager/__init__.py` | Re-export the two new functions | Modify |
| `src/flashquest/eager/llama_persistent_patch.py` | Swap the decode-path scoring + selection chain | Modify |
| `tests/test_page_scores_int8.py` | EQ18-EQ20 coverage | Create |
| `tests/test_select_pages_vectorized.py` | EQ21-EQ25 coverage | Create |
| `tests/test_persistent_e2e.py` | Already covers logit equivalence; stays as integration gate | No change |
| `scripts/phase6_bench_decode_32k.py` | Re-run decode bench, write `benchmarks/phase6_decode.json` | Create |
| `scripts/phase6_run_passkey_32k.py` | Re-run 32k passkey, write `benchmarks/phase6_passkey.json` | Create |
| `docs/PHASES/phase-6-notes.md` | Phase journal | Create at end |
| `DOC.md` | Append Phase 6 row + reproduce list | Modify at end |
| `README.md` | Update Phase 5 decode footnote, add Phase 6 row | Modify at end |
| `docs/SPEC.md` | Tick off Phase 6 task 1 | Modify at end |

---

## Task 1: Add `page_scores_int8` with EQ18 (equivalence to Phase 5 path)

**Files:**
- Modify: `src/flashquest/eager/criticality.py`
- Test: `tests/test_page_scores_int8.py`

- [ ] **Step 1: Write the failing test for INT8 algebraic ≡ dequant path**

Create `tests/test_page_scores_int8.py`:

```python
"""EQ18-EQ20: page_scores_int8 — algebraic Quest criticality direct from
quantization params. Validated against the Phase 5 path
(dequantize_k -> compute_page_summary -> page_scores).
"""
import torch
import pytest

from flashquest.eager.criticality import page_scores, page_scores_int8
from flashquest.eager.page_summary import compute_page_summary
from flashquest.kernel.kv_quant import dequantize_k, quantize_k


def _make_kv(B=1, H_kv=2, S=256, D=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(B, H_kv, S, D, generator=g, device="cuda", dtype=dtype)
    return K


def test_eq18_int8_scores_match_dequant_path():
    """EQ18: page_scores_int8 ≡ page_scores(compute_page_summary(dequantize_k(...)))."""
    page_size = 64
    B, H_kv, H_q, S, D = 1, 2, 4, 256, 64  # n_rep = 2
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)

    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    # Phase 5 path:
    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    n_rep = H_q // H_kv
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    ref = page_scores(Q.float(), page_min, page_max)

    # New path:
    out = page_scores_int8(Q, K_scale, K_mn)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails with ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_page_scores_int8.py::test_eq18_int8_scores_match_dequant_path -v
```
Expected: FAIL with `ImportError: cannot import name 'page_scores_int8' from 'flashquest.eager.criticality'`.

- [ ] **Step 3: Implement `page_scores_int8`**

Append to `src/flashquest/eager/criticality.py`:

```python
def page_scores_int8(
    Q: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
) -> torch.Tensor:
    """Quest criticality computed directly from INT8 quant params.

    Identity: under per-page channel-wise asymmetric uint8 quantization
    (kv_quant._scale_mn_per_page_channel), K_mn is the per-page per-channel
    minimum and K_mn + 255 * K_scale is the maximum (exact, modulo the
    eps-clamp on constant channels). So:

        score(b, h_q, q, p) = sum_d max(Q[d] * K_mn[p, d], Q[d] * (K_mn[p, d] + 255 * K_scale[p, d]))

    GQA broadcast happens on the small (P, D) summary, not on the (S, D) cache.

    Args:
        Q: (B, H_q, S_q, D) bf16/fp16/fp32.
        K_scale: (B, H_kv, P, D) bf16 — per-page per-channel quant scale.
        K_mn: (B, H_kv, P, D) bf16 — per-page per-channel quant min.

    Returns:
        (B, H_q, S_q, P) fp32 page-criticality scores.
    """
    B, H_q, S_q, D = Q.shape
    H_kv = K_scale.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
    n_rep = H_q // H_kv

    Kmn_f = K_mn.float()
    Kmx_f = Kmn_f + 255.0 * K_scale.float()
    if n_rep > 1:
        Kmn_f = Kmn_f.repeat_interleave(n_rep, dim=1)
        Kmx_f = Kmx_f.repeat_interleave(n_rep, dim=1)
    # (B, H_q, P, D) -> (B, H_q, 1, P, D); Q -> (B, H_q, S_q, 1, D)
    Q_e = Q.float().unsqueeze(3)
    Kmn_e = Kmn_f.unsqueeze(2)
    Kmx_e = Kmx_f.unsqueeze(2)
    cand_mx = Q_e * Kmx_e
    cand_mn = Q_e * Kmn_e
    return torch.maximum(cand_mx, cand_mn).sum(dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_page_scores_int8.py::test_eq18_int8_scores_match_dequant_path -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eager/criticality.py tests/test_page_scores_int8.py
git commit -m "phase 6: page_scores_int8 — algebraic Quest criticality from quant params (EQ18)"
```

---

## Task 2: EQ19 (GQA broadcast) and EQ20 (shape/dtype contract) for `page_scores_int8`

**Files:**
- Test: `tests/test_page_scores_int8.py`

- [ ] **Step 1: Append EQ19 (GQA broadcast)**

Append to `tests/test_page_scores_int8.py`:

```python
@pytest.mark.parametrize("H_kv,n_rep", [(2, 1), (2, 4), (8, 3)])
def test_eq19_gqa_broadcast(H_kv, n_rep):
    """EQ19: each query head sees its corresponding KV head's K_scale/K_mn."""
    page_size = 64
    B, S, D = 1, 256, 64
    H_q = H_kv * n_rep
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D, seed=H_kv * 7 + n_rep)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)

    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    ref = page_scores(Q.float(), page_min, page_max)

    out = page_scores_int8(Q, K_scale, K_mn)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_eq20_shape_and_dtype_contract():
    """EQ20: output shape (B, H_q, S_q, P) and fp32 dtype."""
    page_size = 64
    B, H_kv, H_q, S, D = 1, 2, 4, 192, 64  # 192/64 = 3 pages
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D)
    _, K_scale, K_mn = quantize_k(K, page_size=page_size)
    Q = torch.randn(B, H_q, 5, D, device="cuda", dtype=torch.bfloat16)
    out = page_scores_int8(Q, K_scale, K_mn)
    assert out.shape == (B, H_q, 5, 3)
    assert out.dtype == torch.float32


def test_eq20b_h_q_not_divisible_by_h_kv_raises():
    """EQ20: ValueError when GQA group is invalid."""
    page_size = 64
    B, H_kv, S, D = 1, 3, 64, 64
    _, K_scale, K_mn = quantize_k(_make_kv(B, H_kv, S, D), page_size=page_size)
    Q = torch.randn(B, 4, 1, D, device="cuda", dtype=torch.bfloat16)  # 4 not div by 3
    with pytest.raises(ValueError, match="divisible"):
        page_scores_int8(Q, K_scale, K_mn)
```

- [ ] **Step 2: Run all three new tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_page_scores_int8.py -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```
git add tests/test_page_scores_int8.py
git commit -m "phase 6: page_scores_int8 GQA broadcast + dtype tests (EQ19, EQ20)"
```

---

## Task 3: Add `select_pages_vectorized` with EQ21 (scalar retention equivalence)

**Files:**
- Modify: `src/flashquest/eager/selection.py`
- Test: `tests/test_select_pages_vectorized.py`

- [ ] **Step 1: Write the failing test for scalar-retention equivalence**

Create `tests/test_select_pages_vectorized.py`:

```python
"""EQ21-EQ25: select_pages_vectorized — single batched topk + scatter,
equivalent to the Phase 5 per-head loop in select_pages."""
import torch
import pytest

from flashquest.eager.selection import select_pages, select_pages_vectorized


def _scores(B=1, H=4, S_q=1, P=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, H, S_q, P, generator=g, device="cuda", dtype=torch.float32)


@pytest.mark.parametrize("retention", [0.0, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("P", [8, 16, 64, 128])
def test_eq21_scalar_retention_equiv(retention, P):
    """EQ21: vectorized output ≡ loop output for scalar retention across (P, k)."""
    s = _scores(B=2, H=6, S_q=1, P=P, seed=hash((retention, P)) & 0xFFFF)
    ref = select_pages(s, retention=retention, num_sinks=2, window_pages=1)
    out = select_pages_vectorized(s, retention=retention, num_sinks=2, window_pages=1)
    assert torch.equal(ref, out)
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_select_pages_vectorized.py::test_eq21_scalar_retention_equiv -v
```
Expected: FAIL with `ImportError: cannot import name 'select_pages_vectorized'`.

- [ ] **Step 3: Implement `select_pages_vectorized`**

Append to `src/flashquest/eager/selection.py`:

```python
def select_pages_vectorized(
    scores: torch.Tensor,
    retention: float | torch.Tensor,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Vectorized equivalent of select_pages — single batched topk + scatter,
    no Python per-head loop, no `.item()` per head.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: scalar in [0, 1] or 1-D tensor of shape (H,).
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include.

    Returns:
        Boolean mask shaped (B, H, S_q, P).
    """
    B, H, S_q, P = scores.shape

    if isinstance(retention, torch.Tensor):
        if retention.shape != (H,):
            raise ValueError(
                f"per-head retention must be shape ({H},); got {tuple(retention.shape)}"
            )
        retention_per_h = retention.to(scores.device).float()
    else:
        if not (0.0 <= retention <= 1.0):
            raise ValueError(f"scalar retention must be in [0, 1]; got {retention}")
        retention_per_h = torch.full((H,), float(retention), device=scores.device)

    # ceil(r * P), clamped to [0, P]. select_pages uses math.ceil per head;
    # tensor-level ceil() matches that exactly.
    k_per_h = (retention_per_h * P).ceil().long().clamp(min=0, max=P)  # (H,)

    mask = torch.zeros_like(scores, dtype=torch.bool)

    # One host sync per call (not per head). Acceptable.
    k_max = int(k_per_h.max().item())

    if k_max > 0:
        topk_idx = scores.topk(k_max, dim=-1).indices  # (B, H, S_q, k_max)
        ranks = torch.arange(k_max, device=scores.device).view(1, 1, 1, k_max)
        keep = ranks < k_per_h.view(1, H, 1, 1)  # (1, H, 1, k_max) bool
        # scatter_ writes src at topk_idx positions. Where keep is False, src
        # is False -> we write False to a False mask cell (no-op). Where keep
        # is True, src is True -> we write True at the top-k indices.
        src = keep.expand_as(topk_idx)
        mask.scatter_(-1, topk_idx, src)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_select_pages_vectorized.py::test_eq21_scalar_retention_equiv -v
```
Expected: all `test_eq21_*` PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eager/selection.py tests/test_select_pages_vectorized.py
git commit -m "phase 6: select_pages_vectorized — single-call topk+scatter (EQ21)"
```

---

## Task 4: EQ22-EQ25 for `select_pages_vectorized` (per-head, edges)

**Files:**
- Test: `tests/test_select_pages_vectorized.py`

- [ ] **Step 1: Append EQ22 (per-head retention tensor)**

Append to `tests/test_select_pages_vectorized.py`:

```python
def test_eq22_per_head_retention_tensor():
    """EQ22: per-head retention vector — same selections as the loop."""
    B, H, P = 1, 6, 32
    retention = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    s = _scores(B=B, H=H, S_q=1, P=P, seed=42)
    ref = select_pages(s, retention=retention, num_sinks=2, window_pages=1)
    out = select_pages_vectorized(s, retention=retention, num_sinks=2, window_pages=1)
    assert torch.equal(ref, out)


def test_eq23_retention_zero_only_sinks_window():
    """EQ23: retention=0 head — only sinks + window are set."""
    B, H, P = 1, 4, 16
    s = _scores(B=B, H=H, S_q=1, P=P, seed=7)
    out = select_pages_vectorized(s, retention=0.0, num_sinks=2, window_pages=1)
    expect = torch.zeros_like(s, dtype=torch.bool)
    expect[..., :2] = True
    expect[..., -1:] = True
    assert torch.equal(out, expect)


def test_eq24_retention_one_all_pages():
    """EQ24: retention=1 head — every page selected."""
    B, H, P = 1, 4, 16
    s = _scores(B=B, H=H, S_q=1, P=P, seed=11)
    out = select_pages_vectorized(s, retention=1.0, num_sinks=0, window_pages=0)
    assert out.all()


def test_eq25_sinks_window_clamp():
    """EQ25: num_sinks > P or window_pages > P clamps to P."""
    B, H, P = 1, 2, 4
    s = _scores(B=B, H=H, S_q=1, P=P, seed=99)
    out = select_pages_vectorized(s, retention=0.0, num_sinks=10, window_pages=10)
    assert out.all()  # everything covered
```

- [ ] **Step 2: Run all `select_pages_vectorized` tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_select_pages_vectorized.py -v
```
Expected: all PASS.

- [ ] **Step 3: Commit**

```
git add tests/test_select_pages_vectorized.py
git commit -m "phase 6: select_pages_vectorized per-head + edge tests (EQ22-EQ25)"
```

---

## Task 5: Re-export new functions from `flashquest.eager`

**Files:**
- Modify: `src/flashquest/eager/__init__.py`

- [ ] **Step 1: Update package exports**

Replace the file `src/flashquest/eager/__init__.py` with:

```python
"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores, page_scores_int8
from .page_summary import compute_page_summary
from .selection import select_pages, select_pages_vectorized
from .sparse_int8 import quest_eager_sparse_int8
from .streaming import streaming_eager_sdpa

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "page_scores_int8",
    "compute_page_summary",
    "select_pages",
    "select_pages_vectorized",
    "quest_eager_sparse_int8",
    "streaming_eager_sdpa",
]
```

- [ ] **Step 2: Verify import works**

Run:
```
source .venv/bin/activate && python -c "from flashquest.eager import page_scores_int8, select_pages_vectorized; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```
git add src/flashquest/eager/__init__.py
git commit -m "phase 6: export page_scores_int8 + select_pages_vectorized"
```

---

## Task 6: Wire new functions into `_quest_duo_fused_with_lse`

**Files:**
- Modify: `src/flashquest/eager/llama_persistent_patch.py`

- [ ] **Step 1: Replace the decode-path scoring chain**

In `src/flashquest/eager/llama_persistent_patch.py`, change two things.

First, update imports at the top. Replace these lines:

```python
from ..eager.criticality import page_scores
from ..eager.page_summary import compute_page_summary
from ..eager.selection import select_pages
from ..kernel import flash_attn_sparse_fwd
from ..kernel.kv_quant import dequantize_k, dequantize_v
```

With:

```python
from ..eager.criticality import page_scores_int8
from ..eager.selection import select_pages_vectorized
from ..kernel import flash_attn_sparse_fwd
from ..kernel.kv_quant import dequantize_k, dequantize_v
```

Second, in `_quest_duo_fused_with_lse`, replace the body that builds `K_dq`, `page_min`, `page_max`, `scores`, `sel`. Locate this block:

```python
    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    scores = page_scores(Q.float(), page_min, page_max)
    sel = select_pages(
        scores, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )
```

Replace with:

```python
    scores = page_scores_int8(Q, K_scale, K_mn)
    sel = select_pages_vectorized(
        scores, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )
```

Note: `dequantize_k` is still imported because the prefill branch (`S_q > 1`) still uses it for the dense path. Don't remove that import.

- [ ] **Step 2: Run the existing per-head retention test (sanity)**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_per_head_retention.py -v
```
Expected: PASS (this test exercises the patched forward indirectly via the persistent path).

- [ ] **Step 3: Run the persistent-cache logit equivalence test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_persistent_e2e.py -v
```
Expected: PASS at rtol=5e-2 (logit equivalence vs Phase 4 BF16-eager Duo). If this test was marked `slow` or skipped due to missing model, accept the skip — the integration is also covered by the bench script.

If it fails: do NOT loosen the tolerance. Re-check Step 1 — most likely the retention vector type changed shape or `n_rep` mismatch. Re-read `_quest_duo_fused_with_lse` to ensure `retention_per_q` is shape `(H_q,)` (it was constructed earlier in the same function from `head_pattern`).

- [ ] **Step 4: Run the full unit test suite (fast tests only)**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -v -m "not slow" 2>&1 | tail -30
```
Expected: PASS (no regressions in Phases 1–5 unit coverage).

- [ ] **Step 5: Commit**

```
git add src/flashquest/eager/llama_persistent_patch.py
git commit -m "phase 6: wire page_scores_int8 + select_pages_vectorized into decode path"
```

---

## Task 7: Decode bench at 32k — measure the improvement

**Files:**
- Create: `scripts/phase6_bench_decode_32k.py`

- [ ] **Step 1: Write the bench script**

Create `scripts/phase6_bench_decode_32k.py`:

```python
"""Phase 6 task 1 validation: re-run the Phase 5 32k decode bench with the
new algebraic + vectorized paths wired in. Target: ≥4 tok/s decode.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


N_PREFILL_TOKENS = 32768
N_DECODE_TOKENS = 32  # more than Phase 5 since each step is now fast
N_TRIALS = 1


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@torch.no_grad()
def measure_decode(model, tok, n_prefill: int, n_decode: int) -> dict:
    ids = synthetic_prompt_ids(tok, n_prefill)
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    out = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    for _ in range(n_decode):
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_dec

    return {
        "prefill_s": prefill_s,
        "prefill_tok_per_s": n_prefill / prefill_s,
        "decode_s": decode_s,
        "decode_tok_per_s": n_decode / decode_s,
    }


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    results = {
        "model": name, "n_prefill": N_PREFILL_TOKENS, "n_decode": N_DECODE_TOKENS,
        "n_trials": N_TRIALS, "trials": [],
    }

    print("=== Phase 6 (algebraic page_scores_int8 + vectorized select) ===")
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL_TOKENS + N_DECODE_TOKENS + 128,
        page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    for trial in range(N_TRIALS):
        cache._seen_tokens = [0] * cache.num_layers
        torch.cuda.reset_peak_memory_stats()
        m = measure_decode(model, tok, N_PREFILL_TOKENS, N_DECODE_TOKENS)
        m["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  trial {trial}: prefill={m['prefill_tok_per_s']:.1f} tok/s, "
              f"decode={m['decode_tok_per_s']:.2f} tok/s, "
              f"peak VRAM={m['peak_vram_mib']:.0f} MiB")
        results["trials"].append(m)

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_decode.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench (long — ~10 min prefill + ~1 min decode)**

Run:
```
source .venv/bin/activate && python -u scripts/phase6_bench_decode_32k.py 2>&1 | tee /tmp/phase6_bench.log
```
Expected: `decode=` value of **≥4 tok/s** (target). If less than 4 tok/s but more than ~1 tok/s, profile again at the new bottleneck — the vectorized select probably needs work, or partial-tail attention is now visible. If between 0.1 and 1 tok/s, something in the wiring is wrong; revisit Task 6.

- [ ] **Step 3: Verify the JSON was written**

Run:
```
ls -la benchmarks/phase6_decode.json && head -30 benchmarks/phase6_decode.json
```
Expected: file exists with `decode_tok_per_s` ≥ 4.0.

- [ ] **Step 4: Commit**

```
git add scripts/phase6_bench_decode_32k.py benchmarks/phase6_decode.json
git commit -m "phase 6: 32k decode bench — measured at $(jq -r '.trials[0].decode_tok_per_s' benchmarks/phase6_decode.json) tok/s"
```

(If the substitution feels uncomfortable, run `jq` first to read the value, then write the literal in the commit message.)

---

## Task 8: 32k passkey re-validation

**Files:**
- Create: `scripts/phase6_run_passkey_32k.py`

- [ ] **Step 1: Write the passkey re-run script**

Create `scripts/phase6_run_passkey_32k.py`:

```python
"""Phase 6 task 1 quality check: re-run Phase 5's 32k passkey eval with
the new algebraic + vectorized paths. Must still pass 6/6 across depths."""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_LENS = [8192, 32768]
DEPTHS = [0.1, 0.5, 0.9]
N_TRIALS = 2
MAX_NEW = 8


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_passkey_prompt(tok, ctx_len: int, depth: float, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    passkey = "".join(str(rng.randint(0, 9)) for _ in range(5))
    filler = "The grass is green. The sky is blue. The sun is yellow. " * (ctx_len // 8)
    target_pos = int(depth * len(filler))
    prefix = filler[:target_pos]
    suffix = filler[target_pos:]
    prompt = (
        f"There is an important info hidden in the text. Find it.\n\n"
        f"{prefix} The pass key is {passkey}. Remember it. {passkey} is the pass key. {suffix}\n\n"
        f"What is the pass key? The pass key is "
    )
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > ctx_len:
        ids = ids[:, :ctx_len]
    return tok.decode(ids[0]), passkey


@torch.no_grad()
def run_one(model, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=max(CONTEXT_LENS) + MAX_NEW + 128,
        page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    results = {"model": name, "tiers": []}
    for ctx in CONTEXT_LENS:
        tier = {"context": ctx, "depths": []}
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for depth in DEPTHS:
            hits = 0
            for trial in range(N_TRIALS):
                cache._seen_tokens = [0] * cache.num_layers
                _free()
                prompt, key = make_passkey_prompt(tok, ctx, depth, seed=ctx * 100 + trial)
                out = run_one(model, tok, prompt)
                if key in out:
                    hits += 1
                print(f"  ctx={ctx} depth={depth} trial={trial} key={key} hit={key in out} out={out!r}")
            tier["depths"].append({"depth": depth, "hits": hits, "trials": N_TRIALS})
        tier["wall_s"] = time.perf_counter() - t0
        tier["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"ctx={ctx}: wall={tier['wall_s']:.0f}s peak_vram={tier['peak_vram_mib']:.0f} MiB")
        results["tiers"].append(tier)

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the eval (long — Phase 5 took 50 min at 32k tier; should be much faster now)**

Run:
```
source .venv/bin/activate && python -u scripts/phase6_run_passkey_32k.py 2>&1 | tee /tmp/phase6_passkey.log
```
Expected: every `(ctx, depth)` tier shows `hits=2 trials=2`. If any depth misses, the algebraic identity may have a quality regression — DO NOT proceed; flag and re-examine.

- [ ] **Step 3: Verify JSON**

Run:
```
cat benchmarks/phase6_passkey.json | python -m json.tool
```
Expected: every `hits` field equals `trials`.

- [ ] **Step 4: Commit**

```
git add scripts/phase6_run_passkey_32k.py benchmarks/phase6_passkey.json
git commit -m "phase 6: 32k passkey — 6/6 across depths confirms quality"
```

---

## Task 9: Phase 6 notes

**Files:**
- Create: `docs/PHASES/phase-6-notes.md`

- [ ] **Step 1: Write phase notes**

Create `docs/PHASES/phase-6-notes.md`:

```markdown
# Phase 6 Task 1 Notes

**Started:** 2026-05-01
**Status:** task 1 (criticality + top-k) — in progress / complete (depending on bench result)
**Spec:** [../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md](../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md)

## Goal

Close the SPEC §6 task 1 gate: Llama-3.2-3B-AWQ decode at 32 k context from
Phase 5's 0.10 tok/s up to ≥4 tok/s.

## Surface

- `flashquest.eager.page_scores_int8(Q, K_scale, K_mn) -> (B, H_q, S_q, P)` —
  algebraic Quest criticality direct from quant params; no dequant.
- `flashquest.eager.select_pages_vectorized(scores, retention, num_sinks, window_pages) -> mask` —
  single batched topk + scatter; no per-head Python loop, no `.item()` per head.

## Win conditions

| Condition | Result | Pass? |
|---|---|---|
| EQ18-EQ20 unit tests for page_scores_int8 | (TODO: fill from pytest run) | ⬜ |
| EQ21-EQ25 unit tests for select_pages_vectorized | (TODO) | ⬜ |
| Phase 5 unit + e2e tests still green | (TODO) | ⬜ |
| 32 k passkey ≥80 % at depth=0.5 | (TODO from benchmarks/phase6_passkey.json) | ⬜ |
| Decode at 32 k ≥4 tok/s | (TODO from benchmarks/phase6_decode.json) | ⬜ |

## Decisions

- **Algebraic identity exact, not approximate.** `kv_quant._scale_mn_per_page_channel`
  computes `mn = K.min(dim=page).values` and `scale = (K.max - K.min)/255`, so
  `K_mn` is the page_min and `K_mn + 255*K_scale` is the page_max — bit-equal
  modulo the eps-clamp on constant channels (which the dequant path also
  rounds through). No quality cost.
- **Triton kernel deferred.** Original SPEC §6 task 1 called for porting
  Quest's CUDA criticality kernel to Triton with shared-mem bitonic top-k.
  Profile evidence (`benchmarks/phase6_profile.json`) showed the bottleneck
  was memory traffic (full BF16 cache materialization for criticality), not
  Python overhead per se. The fix is "don't materialize the cache," not "do
  the same compute in a kernel." Saves 1-2 weeks of work.
- **One `.item()` per layer** instead of per-head — for `k_max` only. Could
  be precomputed at patch time from the retention vector but the saving is
  ~1 host sync per layer per step (microseconds); not worth the API churn.

## Phase 6 next tasks (still SPEC §6 priority order)

2. RULER 4k subset eval.
3. `flashquest chat` CLI.
4. Head-to-head benchmark table.
5. INT4 KV.
6. EAGLE-2 wrapper.
7. Marlin W4A16 (if profile demands).
8. ExLlamaV2 backend integration.
```

- [ ] **Step 2: Fill in the result fields from your runs**

Manually edit the table in `docs/PHASES/phase-6-notes.md`, replacing each `(TODO ...)` with the actual measured value from `benchmarks/phase6_passkey.json` and `benchmarks/phase6_decode.json` (and from the `pytest` runs in Tasks 1–6). Use ✅ / ❌ for the pass column.

- [ ] **Step 3: Commit**

```
git add docs/PHASES/phase-6-notes.md
git commit -m "phase 6 task 1: notes + result table"
```

---

## Task 10: Update DOC.md, README.md, SPEC.md

**Files:**
- Modify: `DOC.md` (add Phase 6 row)
- Modify: `README.md` (add Phase 6 section after Phase 5)
- Modify: `docs/SPEC.md` (tick off task 1)

- [ ] **Step 1: Read current DOC.md**

Run:
```
cat DOC.md | head -100
```

Find the Phase 5 row in the Phases table. The Phase 6 row goes immediately after it.

- [ ] **Step 2: Add Phase 6 row to DOC.md**

Edit `DOC.md`. Insert a Phase 6 table row + section after Phase 5. Use this template, filling in the actual measured `decode_tok_per_s` from `benchmarks/phase6_decode.json`:

```markdown
| 6 (task 1) | algebraic page_scores_int8 + vectorized select_pages | tag `phase-6-task-1` | 32k decode ≥4 tok/s; passkey 6/6 |
```

And a Phase 6 section body:

```markdown
### Phase 6 task 1 — Criticality + top-k fix (no kernel rewrite)

Profile (see `benchmarks/phase6_profile.json`) showed Phase 5's 0.10 tok/s gap
at 32 k was 95 % `dequantize_k` + `compute_page_summary` + `repeat_interleave_K`
per layer (155 + 133 + 38 ms × 28). Algebraic identity from `kv_quant.py`:
`K_mn` is the per-page per-channel min and `K_mn + 255*K_scale` is the max,
exact by construction. Quest criticality computed directly from quant params,
no dequant. Vectorized top-k batches all heads in one `torch.topk` call.

Result: decode at 32 k from 0.10 → **{X}** tok/s, 32 k passkey 6/6 across
depths preserved.

Reproduce:
- `python scripts/phase6_bench_decode_32k.py`
- `python scripts/phase6_run_passkey_32k.py`
```

(Replace `{X}` with the actual measured value.)

- [ ] **Step 3: Add Phase 6 section to README.md**

Edit `README.md`. Locate the end of the "## Phase 5 — Persistent INT8 KV cache + AWQ-INT4 + fused DuoAttention" section (just before "## Non-goals"). Insert a Phase 6 section:

```markdown
## Phase 6 task 1 — Algebraic criticality + vectorized top-k

Profiling Phase 5's 0.10 tok/s decode gap at 32 k showed it was memory-bound
on `dequantize_k` (155 ms/layer) + `compute_page_summary` (133 ms/layer) +
`repeat_interleave_K` (38 ms/layer) — not Python overhead. The fix replaces
the dequant chain with a direct algebraic page-criticality computed from the
cache's quant params (`K_mn` ≡ page_min, `K_mn + 255*K_scale` ≡ page_max,
exact by construction). The per-head Python `for` loop in `select_pages` is
replaced by a single batched `torch.topk` + scatter.

| | Decode tok/s @ 32 k | Peak VRAM |
|---|---|---|
| Phase 5 (dequant + per-head loop) | 0.092 | 6 378 MiB |
| **Phase 6 task 1** | **{X}** | (TODO) |

32 k passkey re-validated 6/6 across depths — no quality cost.

Re-run via `python scripts/phase6_bench_decode_32k.py` and
`python scripts/phase6_run_passkey_32k.py`.
```

(Fill in `{X}` and the peak VRAM from `benchmarks/phase6_decode.json`.)

- [ ] **Step 4: Tick off task 1 in `docs/SPEC.md`**

Edit `docs/SPEC.md`. Locate the Phase 6 priority list in §6 (currently item 1: "Kernel-fused criticality + top-k. ..."). Replace item 1 with:

```markdown
1. ~~**Kernel-fused criticality + top-k.**~~ **DONE (algebraic + vectorized PyTorch
   instead of Triton kernel)** — `phase-6-task-1` tag. Profile showed the gap
   was memory traffic, not Python; algebraic identity from
   `kv_quant._scale_mn_per_page_channel` lets us skip the cache dequant
   entirely. Measured: 0.10 → **{X}** tok/s decode at 32 k. Triton kernel
   approach deferred to Phase 6 task 5 (when INT4 KV needs new dequant
   kernels anyway).
```

(Fill `{X}`.)

- [ ] **Step 5: Commit**

```
git add DOC.md README.md docs/SPEC.md
git commit -m "phase 6 task 1: update DOC + README + SPEC with measured result"
```

---

## Task 11: Tag and final verification

**Files:** None

- [ ] **Step 1: Run the full test suite one more time**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -v 2>&1 | tail -40
```
Expected: all PASS (slow tests may skip if no model checkpoint cached, that's fine).

- [ ] **Step 2: Tag the milestone**

Run:
```
git tag phase-6-task-1
git log --oneline phase-5..HEAD
```
Expected: 9–10 commits between `phase-5` and `phase-6-task-1`.

- [ ] **Step 3: Update task list state**

The on-disk task tracker doesn't gate this; just confirm by inspection that the SPEC §6 priority list now shows task 1 done.

---

## Self-review (post-write)

1. **Spec coverage** — design doc sections 1–5 mapped to tasks: §1 (`page_scores_int8`) → Tasks 1, 2; §2 (`select_pages_vectorized`) → Tasks 3, 4; §3 (wire) → Task 6; §4 (tests) → all of 1–4 + Task 6 step 3; §5 (validation gates) → Tasks 7, 8. Export plumbing (Task 5) was implicit but needed; added.
2. **Placeholder scan** — no "TBD/TODO/implement later" in instructions; the only `(TODO)` markers are explicit fields the engineer fills *with measured numbers* from the bench/eval JSONs (Tasks 9, 10). Documented.
3. **Type consistency** — `page_scores_int8` returns `(B, H_q, S_q, P)` fp32 in code, test, and design. `select_pages_vectorized` returns `(B, H, S_q, P)` bool in code, test, and design. Argument order matches in all references.
4. **Order** — Task 5 (re-export) comes before Task 6 (wire), so the `from ..eager.criticality import page_scores_int8` import in `llama_persistent_patch.py` works regardless of how the engineer caches their Python imports. (Both are valid paths since the function lives in `eager/criticality.py` directly; the re-export is for external users.)

No issues found.
