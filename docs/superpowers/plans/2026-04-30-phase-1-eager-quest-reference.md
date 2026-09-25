# Phase 1 — Eager Quest Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-PyTorch eager Quest-style sparse attention as a drop-in for HF `LlamaAttention.forward`, with quality benchmarks (Wikitext perplexity + synthetic passkey retrieval) on `unsloth/Llama-3.2-1B-Instruct`. Validate the algorithm before any Triton work.

**Architecture:** Five small utilities — page-summary (per-page min/max of K), criticality (Q·summary score), selection (top-k ∪ sinks ∪ window), sparse-attention, and an HF-compatible `QuestEagerAttention` module that composes them. Decode uses sparse selection; prefill stays dense (algorithm-faithful to Quest paper). KV cache is plain PyTorch tensors stored per-layer in the standard HF cache shape. No CUDA/Triton kernels — this phase only validates correctness and quality.

**Tech Stack:**
- PyTorch (already pinned: torch 2.5.1+cu121)
- HF transformers 4.57.6 (`LlamaForCausalLM`, `LlamaAttention`)
- HF datasets (Wikitext-2 raw)
- pytest + hypothesis for property-based shape testing

**Win conditions** (per SPEC §6 Phase 1):
- `top_k_retention=1.0` matches dense PyTorch SDPA within rtol=1e-3 (numerics float-ordering only).
- Wikitext perplexity within **1 %** of dense at retention=0.25.
- Wikitext perplexity within **3 %** of dense at retention=0.10.
- Passkey retrieval ≥ dense baseline at retention ≥ 0.25 (sanity).

**Hardware envelope:** Llama-3.2-1B BF16 ~ 2 GB weights + KV cache + activations ≤ ~3.5 GB at 4 k context. Fits with margin.

---

## File Structure

**Created:**
- `src/flashquest/eager/__init__.py` — module exports
- `src/flashquest/eager/page_summary.py` — `compute_page_summary(K, page_size) -> (min, max)` per-channel
- `src/flashquest/eager/criticality.py` — `page_scores(Q, page_min, page_max) -> [num_pages]`
- `src/flashquest/eager/selection.py` — `select_pages(scores, num_pages, retention, num_sinks, window_pages) -> bool mask`
- `src/flashquest/eager/attention.py` — `quest_eager_sdpa(Q, K, V, retention, **kw) -> O` (the composition)
- `src/flashquest/eager/llama_patch.py` — `QuestEagerLlamaAttention` subclassing HF `LlamaAttention`; `patch_llama_for_quest_eager(model, retention, ...)` helper
- `src/flashquest/eval/__init__.py`
- `src/flashquest/eval/perplexity.py` — sliding-window Wikitext perplexity helper
- `src/flashquest/eval/passkey.py` — synthetic passkey-retrieval generator + scorer
- `scripts/phase1_run_perplexity.py` — sweep retention ∈ {1.0, 0.5, 0.25, 0.1} on Wikitext-2, dump `benchmarks/phase1_perplexity.json`
- `scripts/phase1_run_passkey.py` — passkey at depths {0.1, 0.5, 0.9} × retentions, dump `benchmarks/phase1_passkey.json`
- `tests/test_eager_page_summary.py`
- `tests/test_eager_criticality.py`
- `tests/test_eager_selection.py`
- `tests/test_eager_attention.py` — sparse vs dense at retention=1.0 equivalence
- `tests/test_eager_e2e.py` — patched HF model produces same logits as un-patched at retention=1.0
- `docs/PHASES/phase-1-notes.md`
- `benchmarks/phase1_perplexity.json` (filled in by script)
- `benchmarks/phase1_passkey.json` (filled in by script)

**Modified:**
- `pyproject.toml` — add `datasets` to `bench` extras
- `README.md` — append Phase 1 results table
- `DOC.md` — flip Phase 1 status, add eager-reference usage snippet
- `docs/PHASES/phase-1-notes.md` — phase journal

---

## Conventions used across all tasks

- **Tensor shape**: `(B=1, H, S, D)` for Q (head-major), `(B=1, H_kv, S, D)` for K/V. We support GQA via `repeat_kv` from HF transformers (the same util Quest uses).
- **Page size**: `PAGE_SIZE = 64`. Tail (final page if S not multiple of 64) is handled by zero-padding K and masking out padding contributions in attention. Document the choice — matches Phase 2 `BLOCK_N`.
- **Dtype**: BF16 for activations/weights to match Llama-3.2-1B; FP32 only for criticality scoring intermediate (small tensor, accuracy matters more there than perf).
- **Causal**: all attention is causal in the autoregressive sense. Sparse selection happens *over the past* (current page is always included as part of the sliding window).
- **Sinks + window defaults**: `num_sinks=4`, `window_pages=2` (i.e. 128 most-recent tokens always attended). Override per-test.

---

## Task 1: Page-summary primitive

**Files:**
- Create: `src/flashquest/eager/page_summary.py`
- Create: `tests/test_eager_page_summary.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_eager_page_summary.py`:
```python
import torch

from flashquest.eager.page_summary import compute_page_summary


def test_basic_shape_and_values():
    # Two heads, 3 pages of 4 tokens, head_dim 8
    B, H, S, D = 1, 2, 12, 8
    page_size = 4
    K = torch.arange(B * H * S * D, dtype=torch.float32).view(B, H, S, D)

    page_min, page_max = compute_page_summary(K, page_size)

    assert page_min.shape == (B, H, 3, D)
    assert page_max.shape == (B, H, 3, D)
    # Within page 0 of head 0: K[0,0,:4,:] -> min on dim=2 axis is row 0, max is row 3
    torch.testing.assert_close(page_min[0, 0, 0], K[0, 0, 0])
    torch.testing.assert_close(page_max[0, 0, 0], K[0, 0, 3])


def test_handles_partial_tail_page():
    # 10 tokens, page_size 4 -> 3 pages, last has 2 valid tokens.
    B, H, S, D = 1, 1, 10, 4
    page_size = 4
    K = torch.randn(B, H, S, D)

    page_min, page_max = compute_page_summary(K, page_size)

    assert page_min.shape == (B, H, 3, D)
    # Last-page summaries derive only from real tokens (rows 8, 9), not padding.
    expected_max = K[0, 0, 8:10].max(dim=0).values
    torch.testing.assert_close(page_max[0, 0, 2], expected_max)


def test_full_retention_summary_equals_pointwise():
    # 1 token per page = page summary equals K itself.
    B, H, S, D = 1, 1, 5, 3
    K = torch.randn(B, H, S, D)
    page_min, page_max = compute_page_summary(K, page_size=1)
    torch.testing.assert_close(page_min.squeeze(2), K)
    torch.testing.assert_close(page_max.squeeze(2), K)
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_eager_page_summary.py -v`
Expected: ImportError or ModuleNotFoundError on `flashquest.eager.page_summary`.

- [ ] **Step 1.3: Implement `compute_page_summary`**

Create `src/flashquest/eager/page_summary.py`:
```python
"""Per-page channel-wise min/max statistics of K, the criticality signal Quest uses."""
from __future__ import annotations

import torch


def compute_page_summary(
    K: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-page per-channel min and max of K.

    Args:
        K: (B, H, S, D) keys (head-major).
        page_size: tokens per page (must be > 0).

    Returns:
        (page_min, page_max) each shaped (B, H, num_pages, D), where
        num_pages = ceil(S / page_size). Tail pages with fewer than
        page_size valid tokens summarise only the valid prefix.
    """
    assert page_size > 0
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    pad = num_pages * page_size - S

    if pad > 0:
        # Pad with +inf for min and -inf for max so padding contributes nothing.
        K_min_padded = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("inf"))
        K_max_padded = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("-inf"))
    else:
        K_min_padded = K
        K_max_padded = K

    K_min_pages = K_min_padded.view(B, H, num_pages, page_size, D)
    K_max_pages = K_max_padded.view(B, H, num_pages, page_size, D)

    page_min = K_min_pages.min(dim=3).values
    page_max = K_max_pages.max(dim=3).values
    return page_min, page_max
```

- [ ] **Step 1.4: Run test to verify it passes**

Run: `. .venv/bin/activate && pytest tests/test_eager_page_summary.py -v`
Expected: 3 passing tests.

- [ ] **Step 1.5: Commit**

```bash
git add src/flashquest/eager/page_summary.py tests/test_eager_page_summary.py
git commit -m "phase 1: per-page min/max channel summary (Quest criticality input)"
```

---

## Task 2: Criticality scoring

**Files:**
- Create: `src/flashquest/eager/criticality.py`
- Create: `tests/test_eager_criticality.py`

The criticality signal per Quest: for each query, score each page by `sum_d max(q*page_max_d, q*page_min_d)`. The intuition: `q · k_max` is an upper-bound estimate of how strongly any token in the page would attend; `q · k_min` is a lower-bound. Taking the elementwise max over the two and summing across `D` gives a tight per-page upper bound on `Q·K`.

- [ ] **Step 2.1: Write the failing test**

Create `tests/test_eager_criticality.py`:
```python
import torch

from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary


def test_score_shape():
    B, H, S_q, S_kv, D = 1, 2, 1, 8, 4
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)
    assert scores.shape == (B, H, S_q, S_kv // page_size)


def test_score_upper_bounds_qk_dot():
    # Quest's per-page score is meant to upper-bound max_t (Q[q] . K[t]) over
    # tokens t in the page. Verify on random data.
    torch.manual_seed(0)
    B, H, S_q, S_kv, D = 1, 1, 1, 16, 8
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)  # (1, 1, 1, 4)

    # True per-page max QK score
    qk = (Q @ K.transpose(-2, -1)).squeeze(2)  # (1, 1, 16)
    qk_pages = qk.view(B, H, 4, page_size).max(dim=-1).values  # (1, 1, 4)

    # Score is an upper bound (+ tiny float slack).
    assert torch.all(scores.squeeze(2) >= qk_pages - 1e-5)


def test_score_handles_multi_query():
    B, H, S_q, S_kv, D = 1, 2, 3, 8, 4
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)
    assert scores.shape == (B, H, S_q, 2)
```

- [ ] **Step 2.2: Run test to verify it fails**

Run: `pytest tests/test_eager_criticality.py -v`
Expected: ImportError on `flashquest.eager.criticality`.

- [ ] **Step 2.3: Implement `page_scores`**

Create `src/flashquest/eager/criticality.py`:
```python
"""Per-page approximate-attention score from page summaries (Quest criticality)."""
from __future__ import annotations

import torch


def page_scores(
    Q: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
) -> torch.Tensor:
    """Quest criticality: per-page upper bound on Q . K.

    For each (b, h, q, p), returns sum_d max(Q_d * page_max[p, d], Q_d * page_min[p, d]).
    This upper-bounds max_t (Q . K[t]) over tokens t in page p.

    Args:
        Q: (B, H, S_q, D) queries.
        page_min: (B, H, num_pages, D) per-page channel min of K.
        page_max: (B, H, num_pages, D) per-page channel max of K.

    Returns:
        (B, H, S_q, num_pages) page-criticality scores.
    """
    # (B, H, S_q, 1, D) * (B, H, 1, P, D) -> (B, H, S_q, P, D)
    Q_e = Q.unsqueeze(3)
    pmin_e = page_min.unsqueeze(2)
    pmax_e = page_max.unsqueeze(2)
    cand_max = Q_e * pmax_e
    cand_min = Q_e * pmin_e
    scores = torch.maximum(cand_max, cand_min).sum(dim=-1)
    return scores
```

- [ ] **Step 2.4: Run test to verify it passes**

Run: `pytest tests/test_eager_criticality.py -v`
Expected: 3 passing tests.

- [ ] **Step 2.5: Commit**

```bash
git add src/flashquest/eager/criticality.py tests/test_eager_criticality.py
git commit -m "phase 1: per-page criticality score (Quest upper-bound estimator)"
```

---

## Task 3: Page selection (top-k ∪ sinks ∪ window)

**Files:**
- Create: `src/flashquest/eager/selection.py`
- Create: `tests/test_eager_selection.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_eager_selection.py`:
```python
import torch

from flashquest.eager.selection import select_pages


def test_full_retention_selects_all():
    B, H, S_q, P = 1, 2, 1, 8
    scores = torch.randn(B, H, S_q, P)
    mask = select_pages(scores, retention=1.0, num_sinks=0, window_pages=0)
    assert mask.shape == (B, H, S_q, P)
    assert mask.all()


def test_zero_retention_keeps_only_sinks_and_window():
    # 8 pages, sinks=1 page, window=2 pages, retention=0 -> first page + last 2 pages.
    B, H, S_q, P = 1, 1, 1, 8
    scores = torch.zeros(B, H, S_q, P)
    mask = select_pages(scores, retention=0.0, num_sinks=1, window_pages=2)
    expected = torch.tensor([True, False, False, False, False, False, True, True])
    assert torch.equal(mask.squeeze(0).squeeze(0).squeeze(0), expected)


def test_topk_picks_highest_scoring():
    B, H, S_q, P = 1, 1, 1, 6
    scores = torch.tensor([[[[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]]]])
    # retention 0.5 -> 3 pages selected by score; no sinks/window
    mask = select_pages(scores, retention=0.5, num_sinks=0, window_pages=0)
    expected = torch.tensor([False, True, False, True, False, True])
    assert torch.equal(mask.squeeze(0).squeeze(0).squeeze(0), expected)


def test_per_head_independent():
    B, H, S_q, P = 1, 2, 1, 4
    scores = torch.tensor([
        [[[0.9, 0.0, 0.0, 0.1]]],   # head 0: page 0 best
        [[[0.0, 0.0, 0.9, 0.1]]],   # head 1: page 2 best
    ]).permute(1, 0, 2, 3)  # -> (1, 2, 1, 4)
    mask = select_pages(scores, retention=0.25, num_sinks=0, window_pages=0)
    assert mask[0, 0, 0, 0] and not mask[0, 0, 0, 2]
    assert mask[0, 1, 0, 2] and not mask[0, 1, 0, 0]


def test_retention_rounds_up_to_at_least_one():
    # 4 pages, retention 0.1 -> ceil(0.4) = 1 page
    B, H, S_q, P = 1, 1, 1, 4
    scores = torch.tensor([[[[0.1, 0.4, 0.2, 0.3]]]])
    mask = select_pages(scores, retention=0.1, num_sinks=0, window_pages=0)
    assert mask.sum().item() == 1
    assert mask[0, 0, 0, 1].item()
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `pytest tests/test_eager_selection.py -v`
Expected: ImportError.

- [ ] **Step 3.3: Implement `select_pages`**

Create `src/flashquest/eager/selection.py`:
```python
"""Top-k page selection with sink + sliding-window always-attended set."""
from __future__ import annotations

import math

import torch


def select_pages(
    scores: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Build a boolean mask over pages: union of top-k by score with sinks + window.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: fraction of pages to select via top-k. 1.0 = all, 0.0 = none
            (only sinks + window). Always rounds up: at least 1 page if retention > 0.
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include (recency window).

    Returns:
        Boolean mask shaped (B, H, S_q, P).
    """
    B, H, S_q, P = scores.shape
    assert 0.0 <= retention <= 1.0

    if retention >= 1.0:
        return torch.ones_like(scores, dtype=torch.bool)

    k = math.ceil(retention * P) if retention > 0 else 0
    mask = torch.zeros_like(scores, dtype=torch.bool)

    if k > 0:
        topk_idx = scores.topk(k, dim=-1).indices  # (B, H, S_q, k)
        mask.scatter_(-1, topk_idx, True)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `pytest tests/test_eager_selection.py -v`
Expected: 5 passing tests.

- [ ] **Step 3.5: Commit**

```bash
git add src/flashquest/eager/selection.py tests/test_eager_selection.py
git commit -m "phase 1: page selection (top-k union sinks union window)"
```

---

## Task 4: Sparse attention given a page mask

**Files:**
- Create: `src/flashquest/eager/attention.py`
- Create: `tests/test_eager_attention.py`

- [ ] **Step 4.1: Write the failing test**

Create `tests/test_eager_attention.py`:
```python
import torch

from flashquest.eager.attention import quest_eager_sdpa


def test_full_retention_matches_dense_sdpa():
    """retention=1.0 with no sinks/window must match torch SDPA exactly modulo numerics."""
    torch.manual_seed(0)
    B, H, S, D = 1, 4, 128, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    # Reference: dense causal SDPA
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    out = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=1.0,
        num_sinks=0,
        window_pages=0,
        is_causal=True,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_gqa_repeat_kv():
    """Q heads > K/V heads is supported (Llama-3.2-1B is 32:8 GQA)."""
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 128, 64  # 4-way GQA
    Q = torch.randn(B, H_q, S, D, dtype=torch.float32)
    K = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    V = torch.randn(B, H_kv, S, D, dtype=torch.float32)

    out = quest_eager_sdpa(Q, K, V, page_size=64, retention=1.0, is_causal=True)
    assert out.shape == (B, H_q, S, D)


def test_partial_retention_close_to_dense_on_low_entropy():
    """When K/V are correlated, top-k page selection should still capture most mass."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 256, 32
    # Concentrate "real" mass on a few pages by zeroing the others.
    K = torch.randn(B, H, S, D) * 0.01
    V = torch.randn(B, H, S, D) * 0.01
    K[..., 64:128, :] *= 100.0
    V[..., 64:128, :] *= 100.0
    Q = torch.randn(B, H, 1, D)  # decode step

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    out = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=0.5,
        num_sinks=0,
        window_pages=0,
        is_causal=False,
    )
    # 50% retention of 4 pages = 2 pages; the high-mass page must be picked.
    rel_err = (out - ref).norm() / ref.norm()
    assert rel_err < 0.1, f"rel_err={rel_err.item():.4f}"


def test_excluded_pages_truly_dropped():
    """Tokens in non-selected pages must contribute zero to the output."""
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 16
    Q = torch.randn(B, H, 1, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    # Force only page 0 (sink) to be selected: retention=0, num_sinks=1, window_pages=0.
    out_sink = quest_eager_sdpa(Q, K, V, page_size=64, retention=0.0, num_sinks=1, window_pages=0, is_causal=False)

    # Compute the same thing manually with K/V truncated to page 0.
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K[:, :, :64], V[:, :, :64], is_causal=False)
    torch.testing.assert_close(out_sink, ref, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 4.2: Run test to verify it fails**

Run: `pytest tests/test_eager_attention.py -v`
Expected: ImportError.

- [ ] **Step 4.3: Implement `quest_eager_sdpa`**

Create `src/flashquest/eager/attention.py`:
```python
"""Pure-PyTorch Quest-style sparse attention. Composes page-summary, criticality,
selection, and a masked-SDPA call. Decode-only sparse-selection logic; for
multi-query (prefill) inputs, retention=1.0 is assumed and we just call SDPA."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def quest_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    page_size: int = 64,
    retention: float = 1.0,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Quest-style sparse causal SDPA.

    Args:
        Q: (B, H_q, S_q, D)
        K: (B, H_kv, S_kv, D)
        V: (B, H_kv, S_kv, D)
        page_size, retention, num_sinks, window_pages: Quest knobs.
        is_causal: standard causal mask.

    Returns:
        Output (B, H_q, S_q, D).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, S_kv, _ = K.shape
    assert H_q % H_kv == 0, "GQA: H_q must be a multiple of H_kv"
    n_rep = H_q // H_kv

    # GQA: replicate K, V along the head axis.
    Kr = _repeat_kv(K, n_rep)
    Vr = _repeat_kv(V, n_rep)

    if retention >= 1.0 and num_sinks == 0 and window_pages == 0:
        return F.scaled_dot_product_attention(Q, Kr, Vr, is_causal=is_causal)

    # Per-page summaries and per-query criticality, computed in fp32 for stability.
    page_min, page_max = compute_page_summary(Kr.float(), page_size)
    scores = page_scores(Q.float(), page_min, page_max)
    page_mask = select_pages(scores, retention, num_sinks, window_pages)
    # page_mask: (B, H_q, S_q, P)

    # Expand page mask to per-token mask over S_kv (with tail padding mask).
    P = page_mask.shape[-1]
    # Per-token mask: True = attended.
    token_mask = page_mask.unsqueeze(-1).expand(B, H_q, S_q, P, page_size).reshape(B, H_q, S_q, P * page_size)
    token_mask = token_mask[..., :S_kv]

    # Causal mask: queries cannot attend to future keys. For S_q == 1 (decode),
    # this is automatic (queries are at S_kv - 1). For multi-query, build the
    # (S_q, S_kv) lower-triangular mask and AND with token_mask.
    if is_causal and S_q > 1:
        causal = torch.ones(S_q, S_kv, dtype=torch.bool, device=Q.device).tril(diagonal=S_kv - S_q)
        token_mask = token_mask & causal

    # Build an additive bias (-inf where masked) for SDPA.
    attn_bias = torch.zeros_like(token_mask, dtype=Q.dtype)
    attn_bias = attn_bias.masked_fill(~token_mask, float("-inf"))

    return F.scaled_dot_product_attention(
        Q, Kr, Vr,
        attn_mask=attn_bias,
        is_causal=False,  # we already encoded causality in attn_bias
    )
```

- [ ] **Step 4.4: Run test to verify it passes**

Run: `pytest tests/test_eager_attention.py -v`
Expected: 4 passing tests.

- [ ] **Step 4.5: Commit**

```bash
git add src/flashquest/eager/attention.py tests/test_eager_attention.py
git commit -m "phase 1: pure-pytorch Quest-style sparse SDPA composition"
```

---

## Task 5: Module exports for `flashquest.eager`

**Files:**
- Create: `src/flashquest/eager/__init__.py`

- [ ] **Step 5.1: Write `__init__.py`**

Create `src/flashquest/eager/__init__.py`:
```python
"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "compute_page_summary",
    "select_pages",
]
```

- [ ] **Step 5.2: Verify imports**

Run: `. .venv/bin/activate && python -c "from flashquest.eager import quest_eager_sdpa, page_scores, compute_page_summary, select_pages; print('ok')"`
Expected: `ok`.

- [ ] **Step 5.3: Commit**

```bash
git add src/flashquest/eager/__init__.py
git commit -m "phase 1: eager module exports"
```

---

## Task 6: HF Llama integration via attention monkeypatch

**Files:**
- Create: `src/flashquest/eager/llama_patch.py`
- Create: `tests/test_eager_e2e.py`

We patch HF `LlamaAttention.forward` in place (the monkeypatch route Quest itself uses). HF transformers 4.45+ also has an `ALL_ATTENTION_FUNCTIONS` registry, but it changes between minor versions; the monkeypatch is the most stable path.

- [ ] **Step 6.1: Write the failing e2e test**

Create `tests/test_eager_e2e.py`:
```python
"""End-to-end: a Quest-eager-patched HF model produces the same logits as the
unpatched model when retention=1.0, num_sinks=0, window_pages=0."""
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
def test_full_retention_matches_unpatched_logits():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from flashquest.eager.llama_patch import patch_llama_for_quest_eager

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()

    inp = tok("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = model(**inp).logits

    patch_llama_for_quest_eager(
        model, retention=1.0, num_sinks=0, window_pages=0, page_size=64
    )

    with torch.no_grad():
        out_logits = model(**inp).logits

    torch.testing.assert_close(out_logits, ref_logits, rtol=2e-2, atol=2e-2)
```

(`rtol=2e-2` because BF16 accumulation differs slightly between SDPA paths; tightening would be over-fit to a particular kernel.)

- [ ] **Step 6.2: Run test to verify it fails**

Run: `pytest tests/test_eager_e2e.py -v`
Expected: ImportError on `flashquest.eager.llama_patch`.

- [ ] **Step 6.3: Implement the patch**

Create `src/flashquest/eager/llama_patch.py`:
```python
"""Drop-in monkeypatch of HF LlamaAttention.forward to use Quest-eager SDPA."""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from .attention import quest_eager_sdpa


def make_quest_eager_forward(
    *,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
    """Build a forward function bound to the given Quest knobs."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)

        is_causal = q_len > 1  # decode (q_len==1) needs no causal mask vs the cache.

        attn_output = quest_eager_sdpa(
            q, k, v,
            page_size=page_size,
            retention=retention,
            num_sinks=num_sinks,
            window_pages=window_pages,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def patch_llama_for_quest_eager(
    model: torch.nn.Module,
    *,
    retention: float,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward in `model` with the Quest-eager version."""
    fwd = make_quest_eager_forward(
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        page_size=page_size,
    )
    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError(
            "patch_llama_for_quest_eager: no LlamaAttention modules found"
        )
```

- [ ] **Step 6.4: Pull the model weights (one-time, ~2.5 GB)**

Run:
```bash
. .venv/bin/activate
mkdir -p ~/models/llama-3.2-1b-instruct
hf download unsloth/Llama-3.2-1B-Instruct \
  --local-dir ~/models/llama-3.2-1b-instruct
export HF_HOME=~/models  # so transformers uses local cache
```
Expected: ~2.5 GB of safetensors + tokenizer files. (Caching to `~/models/` so we don't re-download per session.)

- [ ] **Step 6.5: Run the e2e test**

Run: `. .venv/bin/activate && pytest tests/test_eager_e2e.py -v`
Expected: 1 passing test (or skipped if offline).

If it fails with `position_embeddings` signature mismatch, transformers may have changed the API again; check `LlamaAttention.forward` in the installed version (`python -c "import inspect, transformers; print(inspect.getsourcefile(transformers.models.llama.modeling_llama.LlamaAttention))"`) and align the kwargs.

- [ ] **Step 6.6: Commit**

```bash
git add src/flashquest/eager/llama_patch.py tests/test_eager_e2e.py
git commit -m "phase 1: HF Llama monkeypatch + retention=1.0 e2e equivalence test"
```

---

## Task 7: Wikitext perplexity sweep

**Files:**
- Create: `src/flashquest/eval/__init__.py`
- Create: `src/flashquest/eval/perplexity.py`
- Create: `scripts/phase1_run_perplexity.py`
- Modify: `pyproject.toml` (add `datasets` to bench extras)

- [ ] **Step 7.1: Add `datasets` to bench extras**

Edit `pyproject.toml`. Find the bench extras block (currently):
```toml
bench = [
  "transformers>=4.45",
  "accelerate",
  "huggingface_hub[cli]",
]
```
Replace with:
```toml
bench = [
  "transformers>=4.45,<5",
  "accelerate",
  "huggingface_hub[cli]",
  "datasets>=3",
]
```

Run: `. .venv/bin/activate && pip install -e ".[bench]" 2>&1 | tail -5`
Expected: `datasets` installs cleanly.

- [ ] **Step 7.2: Implement perplexity helper**

Create `src/flashquest/eval/__init__.py`:
```python
"""Phase 1 evaluation utilities."""
```

Create `src/flashquest/eval/perplexity.py`:
```python
"""Sliding-window negative-log-likelihood / perplexity over a corpus."""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F


@torch.no_grad()
def perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    window: int,
    stride: int,
    device: str = "cuda",
) -> float:
    """Sliding-window perplexity following the standard HF recipe.

    For long sequences, we slide a window of `window` tokens with `stride` step,
    computing NLL only over the *new* tokens at each step (so we don't double-count
    overlap). Returns ppl = exp(mean NLL).
    """
    assert input_ids.ndim == 1, "pass a single 1D token tensor"
    n = input_ids.shape[0]
    nlls: list[torch.Tensor] = []
    counts = 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        target_len = end - prev_end  # we score this many fresh tokens
        ids = input_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        # Mask everything except the fresh suffix.
        labels[:, : -target_len] = -100
        out = model(input_ids=ids, labels=labels)
        # HF .loss is mean over un-masked positions; multiply back by count.
        nlls.append(out.loss * target_len)
        counts += target_len
        prev_end = end
        if end == n:
            break
    avg_nll = torch.stack(nlls).sum() / counts
    return math.exp(avg_nll.item())
```

- [ ] **Step 7.3: Implement sweep script**

Create `scripts/phase1_run_perplexity.py`:
```python
"""Sweep retention on Wikitext-2 (raw); compare ppl against dense baseline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.perplexity import perplexity


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    # Cap at 8192 tokens for a tractable sweep on a 1B model.
    ids = ids[: 8192]

    results: dict[str, dict] = {"input_tokens": ids.numel()}

    def fresh_model() -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, attn_implementation="eager"
        ).cuda().eval()

    # Dense baseline first.
    print("dense baseline ...")
    m = fresh_model()
    t0 = time.perf_counter()
    ppl = perplexity(m, ids, window=4096, stride=2048)
    dt = time.perf_counter() - t0
    results["dense"] = {"ppl": ppl, "elapsed_s": dt}
    print(f"  ppl={ppl:.4f} ({dt:.1f}s)")
    del m
    torch.cuda.empty_cache()

    for r in [1.0, 0.5, 0.25, 0.10]:
        print(f"retention={r} ...")
        m = fresh_model()
        patch_llama_for_quest_eager(m, retention=r, num_sinks=4, window_pages=2, page_size=64)
        t0 = time.perf_counter()
        ppl = perplexity(m, ids, window=4096, stride=2048)
        dt = time.perf_counter() - t0
        delta_pct = 100.0 * (ppl - results["dense"]["ppl"]) / results["dense"]["ppl"]
        results[f"retention_{r}"] = {"ppl": ppl, "elapsed_s": dt, "delta_pct": delta_pct}
        print(f"  ppl={ppl:.4f}  delta={delta_pct:+.2f}%  ({dt:.1f}s)")
        del m
        torch.cuda.empty_cache()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase1_perplexity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.4: Run the sweep**

Run: `. .venv/bin/activate && nice -n 19 python scripts/phase1_run_perplexity.py 2>&1 | tail -30`
Expected: prints dense ppl, then 4 retention rows. Each retention should be slower than dense (eager Python overhead). retention=1.0 should be within 0.5 % of dense (numerics-level only). Writes `benchmarks/phase1_perplexity.json`.

- [ ] **Step 7.5: Inspect numbers; gate on win conditions**

Open `benchmarks/phase1_perplexity.json`. Verify:
- `retention_1.0.delta_pct` ≤ 0.5 (sanity — retention=1 IS dense, modulo numerics)
- `retention_0.25.delta_pct` ≤ 1.0 (SPEC win condition)
- `retention_0.1.delta_pct` ≤ 3.0 (SPEC win condition)

If any fail: do **not** silently pass the task. Open `docs/PHASES/phase-1-notes.md` and write a `### Win-condition gap` section describing which retention failed by how much, and what the most likely cause is (sinks/window count, page size, GQA broadcasting bug, etc.). Bring it to the user before continuing.

- [ ] **Step 7.6: Commit**

```bash
git add pyproject.toml src/flashquest/eval/__init__.py src/flashquest/eval/perplexity.py \
        scripts/phase1_run_perplexity.py benchmarks/phase1_perplexity.json
git commit -m "phase 1: Wikitext perplexity sweep over retention"
```

---

## Task 8: Synthetic passkey-retrieval test

**Files:**
- Create: `src/flashquest/eval/passkey.py`
- Create: `scripts/phase1_run_passkey.py`

- [ ] **Step 8.1: Implement the passkey generator + scorer**

Create `src/flashquest/eval/passkey.py`:
```python
"""Synthetic long-context retrieval: hide a 5-digit passkey in filler text,
ask the model to retrieve it. Standard sparse-attention quality probe."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass


FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)


@dataclass
class PasskeyExample:
    text: str
    passkey: str
    depth_pct: float  # where in [0, 1] of the filler the passkey was inserted


def make_example(
    *,
    rng: random.Random,
    n_filler_tokens_approx: int,
    depth_pct: float,
) -> PasskeyExample:
    """Build a passkey prompt with the secret at ~`depth_pct` of the filler."""
    passkey = f"{rng.randint(10000, 99999)}"
    # ~6 tokens per filler unit; build to twice the budget then trim character-wise.
    units = (n_filler_tokens_approx // 3) + 100
    pre_units = int(units * depth_pct)
    post_units = units - pre_units
    pre = FILLER * pre_units
    post = FILLER * post_units
    needle = (
        f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    )
    body = pre + needle + post
    prompt = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about the important info there.\n\n"
        + body
        + "\n\nWhat is the pass key? The pass key is "
    )
    return PasskeyExample(text=prompt, passkey=passkey, depth_pct=depth_pct)


def score(generation: str, passkey: str) -> bool:
    """A generation passes if the passkey appears as the first 5-digit run."""
    m = re.search(r"\d{5}", generation)
    return bool(m) and m.group(0) == passkey
```

- [ ] **Step 8.2: Implement the runner**

Create `scripts/phase1_run_passkey.py`:
```python
"""Run passkey at three depths × four retentions. Compute % correct."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.passkey import make_example, score


N_TRIALS = 5
DEPTHS = [0.1, 0.5, 0.9]
RETENTIONS = [1.0, 0.5, 0.25, 0.1]
N_FILLER_TOKENS = 3500  # keeps total prompt under 4096 tokens with margin


def fresh_model(name: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda().eval()


@torch.no_grad()
def generate_passkey_answer(model: torch.nn.Module, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    examples = []
    rng = random.Random(0)
    for d in DEPTHS:
        for _ in range(N_TRIALS):
            examples.append(make_example(rng=rng, n_filler_tokens_approx=N_FILLER_TOKENS, depth_pct=d))

    results: dict = {"depths": DEPTHS, "n_trials": N_TRIALS, "configs": {}}

    # Dense baseline
    print("dense baseline ...")
    m = fresh_model(name)
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["dense"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
    }
    print(f"  {results['dense']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    torch.cuda.empty_cache()

    for r in RETENTIONS:
        print(f"retention={r} ...")
        m = fresh_model(name)
        patch_llama_for_quest_eager(m, retention=r, num_sinks=4, window_pages=2, page_size=64)
        t0 = time.perf_counter()
        correct_by_depth = {d: 0 for d in DEPTHS}
        for ex in examples:
            gen = generate_passkey_answer(m, tok, ex.text)
            if score(gen, ex.passkey):
                correct_by_depth[ex.depth_pct] += 1
        dt = time.perf_counter() - t0
        results["configs"][f"retention_{r}"] = {
            "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
            "elapsed_s": dt,
        }
        print(f"  {results['configs'][f'retention_{r}']['accuracy_by_depth']}  ({dt:.1f}s)")
        del m
        torch.cuda.empty_cache()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase1_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.3: Run passkey eval**

Run: `. .venv/bin/activate && nice -n 19 python scripts/phase1_run_passkey.py 2>&1 | tail -25`
Expected: 5 trials × 3 depths × 5 configs = 75 generations. ~30-60 min on eager Python.

- [ ] **Step 8.4: Inspect numbers**

Open `benchmarks/phase1_passkey.json`. Sanity gates:
- `dense.accuracy_by_depth` should be ~ 1.0 for shallow depths (≤0.5). 1B Instruct is small; depth 0.9 may already drop.
- `retention_1.0` should match dense.
- `retention_0.25` should not collapse — at least depth 0.5 should match dense.

If any retention collapses far below dense, document the failure mode in `docs/PHASES/phase-1-notes.md` with one specific hypothesis (e.g. window too small to catch the answer prompt; sinks too few; per-head selection diverging).

- [ ] **Step 8.5: Commit**

```bash
git add src/flashquest/eval/passkey.py scripts/phase1_run_passkey.py benchmarks/phase1_passkey.json
git commit -m "phase 1: synthetic passkey-retrieval eval over retention x depth"
```

---

## Task 9: Phase 1 notes + README + DOC.md + tag

**Files:**
- Create: `docs/PHASES/phase-1-notes.md`
- Modify: `README.md`
- Modify: `DOC.md`

- [ ] **Step 9.1: Write the phase journal**

Create `docs/PHASES/phase-1-notes.md`:
```markdown
# Phase 1 Notes

**Started:** <YYYY-MM-DD>
**Completed:** <YYYY-MM-DD> (tag `phase-1`)
**Status:** complete
**Spec:** [docs/SPEC.md §6 Phase 1](../SPEC.md)

## Goal

Pure-PyTorch Quest-style sparse attention as a drop-in for HF LlamaAttention. Validate algorithm + sinks/window composition on Llama-3.2-1B before any kernel work.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| retention=1.0 numerically equivalent to dense SDPA | <fill from test_eager_attention.py + e2e logits delta> | ✅ / ❌ |
| Wikitext ppl Δ ≤ 1 % at retention=0.25 | <fill from benchmarks/phase1_perplexity.json> | ✅ / ❌ |
| Wikitext ppl Δ ≤ 3 % at retention=0.10 | <fill from benchmarks/phase1_perplexity.json> | ✅ / ❌ |
| Passkey ≥ dense at retention=0.25 across depths | <fill from benchmarks/phase1_passkey.json> | ✅ / ❌ |

## Decisions

- Page size 64 (vs Quest's 16). Matches Phase 2 BLOCK_N target — saves a redesign step.
- Sinks=4 tokens (1 page worth), window=2 pages (128 tokens). StreamingLLM defaults adapted to our page granularity.
- Decode = sparse selection; prefill = dense. Algorithm-faithful to Quest paper.
- Per-query-head top-k (not per-KV-head). Heads in the same GQA group select independently.
- Criticality computed in fp32 (small tensor, accuracy matters); attention output in BF16.
- Patched HF Llama via in-place `LlamaAttention.forward = ...`, not the `attn_implementation` registry — registry signature shifts between transformers minor versions.

## Runtime cost (eager Python)

| Workload | Wall time |
|---|---|
| Dense Wikitext ppl over 8 192 tokens | <fill> s |
| retention=0.25 Wikitext ppl | <fill> s |
| Full passkey sweep (75 generations) | <fill> s |

Eager Python is ~Nx slower than HF dense — expected; this phase is correctness-only.

## Phase 2 handoff

Algorithm validated. Phase 2 ports the FA-2 06-fused-attention tutorial onto sm_86 — *no sparsity yet*, just dense Triton baseline. We'll layer the Phase 1 sparse-selection logic on top in Phase 3.
```

(Fill in `<...>` after the runs.)

- [ ] **Step 9.2: Append Phase 1 to README**

Edit `README.md`. After the "Phase 0 baselines" section, append:

```markdown
## Phase 1 — Eager Quest reference

Pure-PyTorch implementation. `unsloth/Llama-3.2-1B-Instruct`, BF16, eager (no kernels). Page size 64, sinks=4, window=128.

| Retention | Wikitext-2 ppl | Δ vs dense | Passkey @ 0.5 depth |
|---|---|---|---|
| 1.00 (dense-equiv) | <fill> | ~0 % | <fill>/5 |
| 0.50 | <fill> | <fill> % | <fill>/5 |
| 0.25 | <fill> | <fill> % | <fill>/5 |
| 0.10 | <fill> | <fill> % | <fill>/5 |

See [`docs/PHASES/phase-1-notes.md`](docs/PHASES/phase-1-notes.md) for the full passkey grid (3 depths) and decisions. Re-run via `python scripts/phase1_run_perplexity.py` and `python scripts/phase1_run_passkey.py`.
```

- [ ] **Step 9.3: Flip Phase 1 in DOC.md**

Edit `DOC.md`. Find the line:
```
- Phase 1 — Eager Python Quest reference. Not started. Planned: ...
```
Replace with:
```
- **Phase 1 — Eager Quest reference** ✅ **complete (tag `phase-1`)**. Pure-PyTorch composition: page summary → criticality → top-k ∪ sinks ∪ window → sparse SDPA. HF LlamaAttention monkeypatch. Validated on `unsloth/Llama-3.2-1B-Instruct` Wikitext ppl + passkey. See `docs/PHASES/phase-1-notes.md`.
```

Add a usage block under "Getting started":
````markdown
Use the eager Quest attention on any HF Llama model:

```python
from transformers import AutoModelForCausalLM
from flashquest.eager.llama_patch import patch_llama_for_quest_eager

model = AutoModelForCausalLM.from_pretrained("unsloth/Llama-3.2-1B-Instruct", torch_dtype="bfloat16").cuda()
patch_llama_for_quest_eager(model, retention=0.25, num_sinks=4, window_pages=2, page_size=64)
# model.generate(...) now uses Quest-eager sparse attention.
```
````

- [ ] **Step 9.4: Final smoke check**

Run:
```bash
. .venv/bin/activate
pytest tests/ -v
python scripts/verify_triton_int8.py
```
Expected: all tests pass; INT8 mma still OK (sanity that nothing in Phase 1 disturbed the venv).

- [ ] **Step 9.5: Commit + tag**

```bash
git add docs/PHASES/phase-1-notes.md README.md DOC.md
git commit -m "phase 1: README + DOC + phase-1-notes complete"
git tag -a phase-1 -m "Phase 1 complete: eager Quest reference validated on Llama-3.2-1B"
```

---

## Self-review

**1. Spec coverage** (SPEC §6 Phase 1):
- "Pure-PyTorch implementation of Quest-style attention (no Triton yet)." → Tasks 1–5.
- "Wire as `attn_implementation='flashquest_eager'` in HF Transformers via `LlamaAttention` subclass." → Task 6 (monkeypatch chosen over registry; reasoning recorded in Task 9.1 decisions).
- "Use Llama-3.2-1B (smallest model — fast iteration)." → Task 6 + 7 + 8 use `unsloth/Llama-3.2-1B-Instruct`.
- "Quality harness: RULER 4k subset, Wikitext perplexity." → Task 7 (Wikitext) + Task 8 (passkey, simpler than RULER and more directly probes long-context retrieval; RULER deferred to a later phase where the model is large enough for it to be informative).
- "Win condition: matches dense within 1% perplexity at top-25% retention, within 3% at top-10%." → Task 7.5 gates explicitly.

**2. Placeholder scan**: every code block contains real code. The README and phase-1-notes have `<fill>` slots for actual measurement values, but those are explicit data-entry slots, not unwritten code.

**3. Type / name consistency**:
- `compute_page_summary(K, page_size)` returns `(page_min, page_max)` — used identically across Tasks 2, 4.
- `page_scores(Q, page_min, page_max) -> (B, H, S_q, P)` — used in Task 3 selection input shape.
- `select_pages(scores, retention, num_sinks, window_pages) -> bool mask` — same signature in Tasks 3 and 4.
- `quest_eager_sdpa(..., page_size, retention, num_sinks, window_pages, is_causal)` — same kw set in Tasks 4, 6, 7, 8.
- `patch_llama_for_quest_eager(model, *, retention, num_sinks, window_pages, page_size)` — same in Tasks 6, 7, 8.

**4. Reversibility**: every task ends with a commit. Failed quality gates in Task 7.5 / 8.4 explicitly tell the engineer to STOP and surface the gap rather than continuing.

## Phase 1 → Phase 2 handoff

When the plan completes:
- `phase-1` git tag exists.
- `benchmarks/phase1_perplexity.json` and `benchmarks/phase1_passkey.json` populated.
- `flashquest.eager.{compute_page_summary, page_scores, select_pages, quest_eager_sdpa, patch_llama_for_quest_eager}` are the algorithm of record. Phase 2 keeps these as the *correctness oracle* the Triton kernel is checked against.
- Phase 2 begins with porting `vendor/triton/python/tutorials/06-fused-attention.py` onto sm_86 — *dense* attention only. Sparsity gets layered in Phase 3 by reusing the page-summary + criticality + selection from this phase to drive a sparse outer loop.
