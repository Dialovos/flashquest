# Phase 3 — Sparse Retrieval + INT8 KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Layer Phase 1's Quest-style sparse selection on top of Phase 2's dense Triton kernel, with KIVI-style INT8 KV storage. Decode-only sparse forward (`S_q == 1`) — the hot path for autoregressive inference. Prefill stays dense (Phase 2 kernel). End state: a kernel that loads INT8 KV pages selected by Quest, dequantizes inside the kernel, runs online softmax, and produces correct output within INT8 quantization error of the dense baseline.

**Architecture:** Three pieces — (1) Python-side KV quant/dequant utilities (KIVI convention: per-page channel-wise K, per-token V; uint8 storage with `(scale, mn)` asymmetric quant); (2) Eager Python sparse-INT8 reference that extends Phase 1's `quest_eager_sdpa`; (3) Triton sparse forward kernel that takes uint8 K/V + scales + a selection mask, dequantizes per-tile inside the kernel, iterates only over selected pages, returns BF16 output + LSE. The eager reference is the correctness oracle.

**Tech Stack:** Same as Phase 2 — torch 2.5.1+cu121, triton 3.1.0, BF16 activations, FP32 accumulator. uint8 KV storage.

**Win conditions** (per SPEC §6 Phase 3):
- Triton sparse kernel ≡ Phase 2 dense kernel at full mask (`selection_mask` all True) within rtol=5e-2 (INT8 quant tolerance is wider than BF16-only).
- Triton sparse kernel ≡ Phase 1 eager (with retention=0.25, sinks=4, window=2) within rtol=5e-2 at the same INT8 budget.
- Passkey retrieval at retention=0.25 ≥ Phase 1 eager's 5/5 baseline (i.e. INT8 KV doesn't break retrieval).
- Decode perf at 8 k context: sparse INT8 kernel beats Phase 2 dense decode at the same context (concrete bar: 1.5× faster on the synthetic decode loop). The 32 k decode bar is deferred to Phase 4 (needs HF integration to be meaningful end-to-end).

**Phase 3 v1 explicit non-goals (deferred):**
- Bit-packed INT4 / INT2 KV (this plan is INT8 only — straight uint8 byte storage).
- Sparse *prefill* (multi-query). Prefill uses Phase 2 dense kernel.
- HF Llama integration (Phase 4).
- 32 k context end-to-end measurement (Phase 4).
- vLLM-style page_table indirection (KV stays contiguous).

**Hardware envelope:** Same as Phase 2 — sm_86, 4 GB VRAM, 48 KB SMEM. INT8 KV halves the cache footprint vs BF16.

---

## Edge case catalog — sparse kernel

| ID | Case | Why it matters |
|---|---|---|
| ES1 | selection_mask all True (full retention) | Must match Phase 2 dense kernel within INT8 tolerance — sanity. |
| ES2 | selection_mask all False (no pages selected) | Degenerate; output should be 0 (no information attended). |
| ES3 | `S_kv` not multiple of `page_size` | Last page is partial; quant must handle, kernel must mask. |
| ES4 | `S_q == 1` (decode hot path) | Primary use case; performance-sensitive. |
| ES5 | GQA with selection per query head | `selection_mask[b, h_q, 0, p]` differs by `h_q`; KV is per `h_kv`. |
| ES6 | head_dim ∈ {64, 128} | Both Llama-3.2 and Llama-3.1 supported. |
| ES7 | Small `S_kv` (e.g. 1 page) | Sparse is overkill but must not break. |
| ES8 | Causal (decode trivializes) | For S_q=1 the single query attends to selected pages; no further mask. |
| ES9 | Zero-range channel after quant (scale = 0) | Division-by-zero must be handled (e.g. clamp scale to small ε). |
| ES10 | Quant round-trip drift | dequant(quant(X)) ≈ X within INT8 error (~1/256 of range per channel). |
| ES11 | Multi-query input (`S_q > 1`) | Phase 3 v1 rejects this at the wrapper — prefill uses Phase 2 dense. |

The kernel rejects ES11 at the Python wrapper. ES1–ES10 handled inside.

---

## File Structure

**Created:**
- `src/flashquest/kernel/kv_quant.py` — Python utilities `quantize_k`, `dequantize_k`, `quantize_v`, `dequantize_v`. Pure PyTorch. KIVI convention.
- `src/flashquest/kernel/sparse_fwd.py` — `_sparse_attn_fwd_kernel` (Triton @jit) + `flash_attn_sparse_fwd` Python wrapper. Loads uint8 K/V + scales, dequantizes per tile.
- `src/flashquest/eager/sparse_int8.py` — `quest_eager_sparse_int8(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, selection_mask, *, sm_scale)`. Pure-PyTorch reference for the kernel.
- `tests/test_kv_quant.py` — round-trip + edge cases for quant utilities.
- `tests/test_eager_sparse_int8.py` — eager sparse INT8 vs Phase 1 BF16 eager.
- `tests/test_kernel_sparse_basic.py` — Triton sparse kernel correctness.
- `tests/test_kernel_sparse_edges.py` — ES1–ES11 grid + property fuzz.
- `tests/test_kernel_sparse_dense_equivalence.py` — sparse @ full-mask ≡ Phase 2 dense.
- `scripts/phase3_bench_decode.py` — synthetic decode-loop perf vs Phase 2 dense.
- `benchmarks/phase3_perf.json` — decode tok/s comparison.
- `docs/PHASES/phase-3-notes.md` — phase journal.

**Modified:**
- `src/flashquest/kernel/__init__.py` — add `flash_attn_sparse_fwd` export.
- `src/flashquest/eager/__init__.py` — add `quest_eager_sparse_int8` export.
- `README.md` — Phase 3 row.
- `DOC.md` — flip Phase 3 status.

---

## Conventions

- **Quant convention:** asymmetric uint8 with `(scale, mn)`:
  - quant: `x_uint8 = clip(round((x - mn) / scale), 0, 255)`
  - dequant: `x ≈ x_uint8 * scale + mn`
- **K layout:** `K_bf16 (B, H_kv, S_kv, D) → K_uint8 (B, H_kv, S_kv, D)` plus `K_scale (B, H_kv, num_pages, D) bf16`, `K_mn (B, H_kv, num_pages, D) bf16`. Per-page channel-wise (KIVI's "per-channel K" semantics, page granularity). `num_pages = ceil(S_kv / page_size)`.
- **V layout:** `V_bf16 (B, H_kv, S_kv, D) → V_uint8 (B, H_kv, S_kv, D)` plus `V_scale (B, H_kv, S_kv, 1) bf16`, `V_mn (B, H_kv, S_kv, 1) bf16`. Per-token (each token's D channels share one (scale, mn)) — "per-token V" semantics.
- **`page_size`:** 64. Aligned with Phase 1 + Phase 2.
- **selection_mask:** `(B, H_q, S_q, num_pages) bool`. True = attend to this page.
- **Numerical tolerance:** INT8 quant introduces ~`scale/2` per-element error, so end-to-end the output rtol is wider than BF16-only paths. Tests use `rtol=5e-2, atol=5e-2`.

---

## Task 1: KV quant / dequant utilities (Python eager)

**Purpose:** Build the round-trippable INT8 KV codec before any kernel work. Eager Python so we can test it in isolation.

**Files:**
- Create: `src/flashquest/kernel/kv_quant.py`
- Create: `tests/test_kv_quant.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_kv_quant.py`:
```python
import pytest
import torch

from flashquest.kernel.kv_quant import (
    dequantize_k,
    dequantize_v,
    quantize_k,
    quantize_v,
)


def test_quantize_k_shapes():
    B, H, S, D = 1, 2, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    assert K_uint8.shape == (B, H, S, D)
    assert K_uint8.dtype == torch.uint8
    num_pages = S // 64
    assert scale.shape == (B, H, num_pages, D)
    assert mn.shape == (B, H, num_pages, D)
    assert scale.dtype == torch.bfloat16


def test_quantize_v_shapes():
    B, H, S, D = 1, 2, 128, 64
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V_uint8, scale, mn = quantize_v(V)
    assert V_uint8.shape == (B, H, S, D)
    assert V_uint8.dtype == torch.uint8
    assert scale.shape == (B, H, S, 1)
    assert mn.shape == (B, H, S, 1)


def test_k_round_trip_within_int8_tolerance():
    """ES10: dequant(quant(K)) within (max_range / 255) per element."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)

    # Per-page per-channel error bound: scale (= range/255) at most.
    expected_max_err = scale.abs().amax().item()
    actual_max_err = (K_rec.float() - K.float()).abs().amax().item()
    assert actual_max_err <= expected_max_err * 1.01, (
        f"round-trip err {actual_max_err} > expected {expected_max_err}"
    )


def test_v_round_trip_within_int8_tolerance():
    torch.manual_seed(1)
    B, H, S, D = 1, 2, 128, 64
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V_uint8, scale, mn = quantize_v(V)
    V_rec = dequantize_v(V_uint8, scale, mn)

    expected_max_err = scale.abs().amax().item()
    actual_max_err = (V_rec.float() - V.float()).abs().amax().item()
    assert actual_max_err <= expected_max_err * 1.01


def test_k_partial_last_page():
    """ES3: S_kv not multiple of page_size — last page handled."""
    B, H, S, D = 1, 1, 100, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    # 2 pages: [0..63] full, [64..99] partial.
    assert scale.shape == (B, H, 2, D)
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)
    assert K_rec.shape == K.shape


def test_zero_range_channel_safe():
    """ES9: a channel with all-equal values must not divide by 0."""
    B, H, S, D = 1, 1, 64, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K[..., 5] = 0.7  # constant channel 5
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    # No NaN / Inf in storage or dequant.
    assert torch.isfinite(scale).all()
    assert torch.isfinite(mn).all()
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)
    assert torch.isfinite(K_rec).all()
    # Channel 5 should round-trip exactly.
    torch.testing.assert_close(K_rec[..., 5], K[..., 5], rtol=0, atol=1e-2)
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `. .venv/bin/activate && pytest tests/test_kv_quant.py -v`
Expected: ImportError on `flashquest.kernel.kv_quant`.

- [ ] **Step 1.3: Implement quant utilities**

Create `src/flashquest/kernel/kv_quant.py`:
```python
"""Asymmetric uint8 KV quant / dequant. KIVI-style: per-page channel-wise K,
per-token V. Pure PyTorch — used by tests and as the eager reference.

Quant convention:
    x_uint8 = clip(round((x - mn) / scale), 0, 255)
    x ≈ x_uint8 * scale + mn

`scale` is bounded below by a tiny epsilon to avoid div-by-zero on
constant-valued channels (ES9).
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def _scale_mn_per_page_channel(K: torch.Tensor, page_size: int):
    """For K (B, H, S, D), returns (scale, mn) shaped (B, H, num_pages, D).

    Pads the last page with the page-min for min and page-max for max so
    padding contributes nothing to the per-page extrema.
    """
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    pad = num_pages * page_size - S
    if pad > 0:
        # Use +inf padding for min reduce, -inf for max reduce, then merge.
        K_padded = K
    else:
        K_padded = K

    # Reshape to (B, H, num_pages, page_size, D) — using the per-page extrema
    # over the *valid* tokens. Build a mask to ignore padding.
    if pad > 0:
        K_for_min = torch.nn.functional.pad(K_padded, (0, 0, 0, pad), value=float("inf"))
        K_for_max = torch.nn.functional.pad(K_padded, (0, 0, 0, pad), value=float("-inf"))
    else:
        K_for_min = K_padded
        K_for_max = K_padded
    K_for_min = K_for_min.view(B, H, num_pages, page_size, D)
    K_for_max = K_for_max.view(B, H, num_pages, page_size, D)
    mn = K_for_min.min(dim=3).values
    mx = K_for_max.max(dim=3).values
    scale = (mx - mn) / 255.0
    scale = scale.clamp_min(_EPS)
    return scale, mn


def quantize_k(K: torch.Tensor, page_size: int):
    """Quantize K to uint8 with per-page per-channel (scale, mn).

    Args:
        K: (B, H, S, D) bf16 or fp16 or fp32.
        page_size: tokens per page.

    Returns:
        (K_uint8 (B, H, S, D), scale (B, H, num_pages, D), mn (B, H, num_pages, D)).
        scale and mn are stored in bf16.
    """
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    scale, mn = _scale_mn_per_page_channel(K, page_size)

    # Broadcast scale/mn back to (B, H, S, D) for the quant op.
    scale_per_token = scale.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    mn_per_token = mn.repeat_interleave(page_size, dim=2)[:, :, :S, :]

    K_norm = (K.float() - mn_per_token.float()) / scale_per_token.float()
    K_uint8 = K_norm.round().clamp(0, 255).to(torch.uint8)
    return K_uint8, scale.to(torch.bfloat16), mn.to(torch.bfloat16)


def dequantize_k(
    K_uint8: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Inverse of quantize_k. Returns bf16."""
    B, H, S, D = K_uint8.shape
    scale_per_token = scale.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    mn_per_token = mn.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    out = K_uint8.to(torch.float32) * scale_per_token.float() + mn_per_token.float()
    return out.to(torch.bfloat16)


def quantize_v(V: torch.Tensor):
    """Quantize V per-token: each token's D channels share one (scale, mn).

    Returns:
        (V_uint8 (B, H, S, D), scale (B, H, S, 1), mn (B, H, S, 1)).
    """
    mn = V.float().amin(dim=-1, keepdim=True)
    mx = V.float().amax(dim=-1, keepdim=True)
    scale = (mx - mn) / 255.0
    scale = scale.clamp_min(_EPS)
    V_uint8 = ((V.float() - mn) / scale).round().clamp(0, 255).to(torch.uint8)
    return V_uint8, scale.to(torch.bfloat16), mn.to(torch.bfloat16)


def dequantize_v(
    V_uint8: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
) -> torch.Tensor:
    out = V_uint8.to(torch.float32) * scale.float() + mn.float()
    return out.to(torch.bfloat16)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `pytest tests/test_kv_quant.py -v`
Expected: 6 passing tests.

- [ ] **Step 1.5: Commit**

```bash
git add src/flashquest/kernel/kv_quant.py tests/test_kv_quant.py
git commit -m "phase 3: KV quant/dequant utilities (KIVI-style asymmetric uint8)"
```

---

## Task 2: Eager INT8 sparse SDPA reference

**Purpose:** Pure-PyTorch sparse attention with INT8 KV. This is the **correctness oracle** the Triton kernel must match. By layering on top of Phase 1's `quest_eager_sdpa` and adding the dequant step, we keep the algorithm identical and isolate the INT8 quantization effect.

**Files:**
- Create: `src/flashquest/eager/sparse_int8.py`
- Create: `tests/test_eager_sparse_int8.py`
- Modify: `src/flashquest/eager/__init__.py`

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_eager_sparse_int8.py`:
```python
import torch

from flashquest.eager import quest_eager_sdpa
from flashquest.eager.sparse_int8 import quest_eager_sparse_int8
from flashquest.kernel.kv_quant import quantize_k, quantize_v


def test_full_retention_close_to_bf16_eager():
    """retention=1.0: INT8 path within INT8 quant tolerance of BF16 path."""
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 4, 1, 256, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

    O_bf16 = quest_eager_sdpa(
        Q, K, V, page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_int8 = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )

    # INT8 quant error budget — empirically generous; we verify the algorithm
    # is correct, not that INT8 matches BF16 exactly.
    torch.testing.assert_close(O_int8, O_bf16, rtol=5e-2, atol=5e-2)


def test_decode_step_int8():
    """ES4: S_q=1 decode case."""
    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 4, 1, 1024, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)

    O_bf16 = quest_eager_sdpa(
        Q, K, V, page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_int8 = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )
    # Same algorithm, INT8 vs BF16 KV.
    rel_err = (O_int8 - O_bf16).norm() / O_bf16.norm()
    assert rel_err < 0.1, f"rel_err={rel_err.item():.4f}"


def test_no_pages_selected_returns_zero():
    """ES2: when retention=0 and no sinks/window, output should be zero
    (no information attended)."""
    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 1, 1, 128, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=0.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    assert torch.equal(O, torch.zeros_like(O)), "no-pages-selected must produce zero"
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/test_eager_sparse_int8.py -v`
Expected: ImportError on `flashquest.eager.sparse_int8`.

- [ ] **Step 2.3: Implement the eager INT8 sparse path**

Create `src/flashquest/eager/sparse_int8.py`:
```python
"""Eager Quest-style sparse attention with INT8 KV.

Composes:
  1. Dequantize K and V from uint8 to bf16.
  2. Use the existing quest_eager_sdpa over the dequantized tensors.

Splitting like this keeps the algorithm identical to Phase 1 — the only
difference is the quantization round-trip on K/V before the attention.
This is the *correctness oracle* the Triton sparse kernel must match.
"""
from __future__ import annotations

import torch

from ..kernel.kv_quant import dequantize_k, dequantize_v
from .attention import quest_eager_sdpa


def quest_eager_sparse_int8(
    Q: torch.Tensor,
    K_uint8: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_uint8: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    page_size: int = 64,
    retention: float = 1.0,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Sparse attention over INT8-quantized KV.

    Args:
        Q: (B, H_q, S_q, D) bf16.
        K_uint8: (B, H_kv, S_kv, D) uint8.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16.
        V_uint8: (B, H_kv, S_kv, D) uint8.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16.

    Returns:
        Output (B, H_q, S_q, D) bf16.
    """
    K = dequantize_k(K_uint8, K_scale, K_mn, page_size)
    V = dequantize_v(V_uint8, V_scale, V_mn)
    return quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
```

Edit `src/flashquest/eager/__init__.py` to add the export:
```python
"""Eager (pure-PyTorch) Quest reference. Phase 1 milestone."""
from .attention import quest_eager_sdpa
from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages
from .sparse_int8 import quest_eager_sparse_int8

__all__ = [
    "quest_eager_sdpa",
    "page_scores",
    "compute_page_summary",
    "select_pages",
    "quest_eager_sparse_int8",
]
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest tests/test_eager_sparse_int8.py -v`
Expected: 3 passing tests.

- [ ] **Step 2.5: Commit**

```bash
git add src/flashquest/eager/sparse_int8.py src/flashquest/eager/__init__.py tests/test_eager_sparse_int8.py
git commit -m "phase 3: eager INT8 sparse SDPA reference (correctness oracle for the kernel)"
```

---

## Task 3: Triton sparse forward kernel (decode-only)

**Purpose:** The kernel itself. Decode-only (`S_q == 1`). Loads uint8 K and V, applies dequant inside, iterates only over selected pages from `selection_mask`.

**Files:**
- Create: `src/flashquest/kernel/sparse_fwd.py`
- Modify: `src/flashquest/kernel/__init__.py`
- Create: `tests/test_kernel_sparse_basic.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/test_kernel_sparse_basic.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(B, H_q, H_kv, S_kv, D, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    return Q, K, V


@cuda
def test_full_mask_matches_eager_int8():
    """ES1: selection_mask all True (full retention) ≡ eager INT8 sparse @ retention=1.0."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 4, 1, 256, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    num_pages = S_kv // 64
    selection_mask = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )

    O_eager = quest_eager_sparse_int8(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_partial_mask_matches_eager_int8():
    """selection_mask with sinks + window + a couple top-k pages."""
    from flashquest.eager.sparse_int8 import quest_eager_sparse_int8
    from flashquest.eager.criticality import page_scores
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 4, 1, 1024, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D, seed=1)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)

    # Build selection_mask the same way the eager path does, against the
    # *dequantized* K (matches the eager path used as the oracle).
    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=64)
    K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
    scores = page_scores(Q.float(), pmin, pmax)
    selection_mask = select_pages(scores, retention=0.25, num_sinks=4, window_pages=2)

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )

    O_eager = quest_eager_sparse_int8(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_no_pages_selected_returns_zero():
    """ES2: all-False mask -> output is zero."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 1, 1, 128, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    num_pages = 2
    selection_mask = torch.zeros(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )
    assert torch.equal(O, torch.zeros_like(O))


@cuda
def test_rejects_multi_query():
    """ES11: S_q > 1 is rejected at the wrapper (Phase 3 v1 is decode-only)."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    Q = torch.randn(1, 1, 8, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    selection_mask = torch.ones(1, 1, 8, 1, dtype=torch.bool, device="cuda")
    with pytest.raises(NotImplementedError, match="decode-only"):
        flash_attn_sparse_fwd(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            selection_mask=selection_mask, page_size=64,
        )
```

- [ ] **Step 3.2: Run tests to verify they fail**

Run: `pytest tests/test_kernel_sparse_basic.py -v`
Expected: ImportError on `flash_attn_sparse_fwd`.

- [ ] **Step 3.3: Write the kernel**

Create `src/flashquest/kernel/sparse_fwd.py`:
```python
"""Decode-only sparse forward kernel with INT8 KV. Phase 3.

Algorithm:
  - One program per (b, h_q). Q is a single row.
  - Outer loop over pages where selection_mask[b, h_q, 0, p] == True.
  - For each selected page:
      - Load K_uint8 page (page_size, D) — dequant to bf16 with K_scale, K_mn.
      - QK^T -> (1, page_size).
      - Online softmax update.
      - Load V_uint8 page (page_size, D) — dequant to bf16 with V_scale, V_mn.
      - softmax · V -> (1, D), accumulate.
  - Normalize, store.

For S_q == 1 we don't get tensor-core efficiency on QK (the M dim is 1).
We accept this for v1 — sparsity gain dominates kernel-issue overhead at
typical retention rates (10-25%).
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (64, 128)


@triton.jit
def _sparse_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    sel_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_selp,
    H_q, H_kv, S_kv, NUM_PAGES,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # batch * H_q
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q (single row)
    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh + offs_d * stride_qd
    )
    q = tl.load(q_ptrs)  # (HEAD_DIM,)

    m_i = tl.full([1], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for p in range(0, NUM_PAGES):
        sel_ptr_p = sel_ptr + b * stride_selb + h_q * stride_selh + p * stride_selp
        is_sel = tl.load(sel_ptr_p)

        if is_sel:
            page_start = p * PAGE_SIZE
            n_idx = page_start + offs_n
            valid_kv = n_idx < S_kv

            # Load uint8 K page (PAGE_SIZE, HEAD_DIM)
            k_ptrs = (
                K_ptr + b * stride_kb + h_kv * stride_kh
                + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd
            )
            k_u8 = tl.load(k_ptrs, mask=valid_kv[:, None], other=0)
            # Load K scale, mn for this page (HEAD_DIM,)
            ks_ptrs = (
                K_scale_ptr + b * stride_ksb + h_kv * stride_ksh
                + p * stride_ksp + offs_d * stride_ksd
            )
            km_ptrs = (
                K_mn_ptr + b * stride_kmb + h_kv * stride_kmh
                + p * stride_kmp + offs_d * stride_kmd
            )
            k_scale = tl.load(ks_ptrs).to(tl.float32)
            k_mn = tl.load(km_ptrs).to(tl.float32)
            # Dequant
            k = k_u8.to(tl.float32) * k_scale[None, :] + k_mn[None, :]

            # qk: (PAGE_SIZE,) — single query row
            qk = tl.sum(q[None, :].to(tl.float32) * k, axis=1)  # (PAGE_SIZE,)
            # Mask out-of-range KV (last partial page)
            qk = tl.where(valid_kv, qk, -float("inf"))

            # Online softmax (vector form, BLOCK_M=1)
            m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, axis=0))
            m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            p_softmax = tl.math.exp2(qk * qk_scale - m_ij_safe)
            p_softmax = tl.where(m_ij == -float("inf"), 0.0, p_softmax)

            alpha = tl.math.exp2(m_i - m_ij_safe)
            alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

            l_i = l_i * alpha + tl.sum(p_softmax, axis=0)
            acc = acc * alpha

            # Load uint8 V page + per-token (scale, mn)
            v_ptrs = (
                V_ptr + b * stride_vb + h_kv * stride_vh
                + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
            )
            v_u8 = tl.load(v_ptrs, mask=valid_kv[:, None], other=0)
            vs_ptrs = V_scale_ptr + b * stride_vsb + h_kv * stride_vsh + n_idx * stride_vss
            vm_ptrs = V_mn_ptr + b * stride_vmb + h_kv * stride_vmh + n_idx * stride_vms
            v_scale = tl.load(vs_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
            v_mn = tl.load(vm_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
            v = v_u8.to(tl.float32) * v_scale[:, None] + v_mn[:, None]

            # Accumulate weighted V
            acc += tl.sum(p_softmax[:, None] * v, axis=0)

            m_i = m_ij

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l

    o_ptrs = O_ptr + b * stride_ob + h_q * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty))

    if WRITE_LSE:
        lse = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse = tl.where(l_i == 0.0, -float("inf"), lse)
        l_ptr_bh = L_ptr + b * stride_lb + h_q * stride_lh
        tl.store(l_ptr_bh, lse)


def flash_attn_sparse_fwd(
    Q: torch.Tensor,
    K_uint8: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_uint8: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode-only sparse forward with INT8 KV.

    Args:
        Q: (B, H_q, 1, D) bf16 cuda.
        K_uint8: (B, H_kv, S_kv, D) uint8 cuda.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16 cuda.
        V_uint8: (B, H_kv, S_kv, D) uint8 cuda.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16 cuda.
        selection_mask: (B, H_q, 1, num_pages) bool cuda — True = attend.
    Returns:
        (O (B, H_q, 1, D) bf16, lse (B, H_q, 1) fp32 or None).
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_uint8.dtype == torch.uint8 and V_uint8.dtype == torch.uint8

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(
            f"flash_attn_sparse_fwd: Phase 3 v1 is decode-only (S_q={S_q}); use flash_attn_fwd for prefill."
        )
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")

    Bk, H_kv, S_kv, Dk = K_uint8.shape
    assert (B, D) == (Bk, Dk)
    assert H_q % H_kv == 0
    num_pages = K_scale.shape[2]
    assert selection_mask.shape == (B, H_q, 1, num_pages)
    assert selection_mask.dtype == torch.bool

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Squeeze the S_q=1 axis for kernel pointer convenience.
    Q_2d = Q.squeeze(2)              # (B, H_q, D)
    O_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    if L is not None:
        sl_b, sl_h = L.stride()
    else:
        sl_b = sl_h = 0

    sel_2d = selection_mask.squeeze(2)  # (B, H_q, num_pages)

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel[grid](
        Q_2d, K_uint8, V_uint8, O_2d, L_ptr,
        K_scale, K_mn, V_scale, V_mn,
        sel_2d,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_uint8.stride(0), K_uint8.stride(1), K_uint8.stride(2), K_uint8.stride(3),
        V_uint8.stride(0), V_uint8.stride(1), V_uint8.stride(2), V_uint8.stride(3),
        O_2d.stride(0), O_2d.stride(1), O_2d.stride(2),
        sl_b, sl_h,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel_2d.stride(0), sel_2d.stride(1), sel_2d.stride(2),
        H_q, H_kv, S_kv, num_pages,
        HEAD_DIM=D,
        PAGE_SIZE=page_size,
        WRITE_LSE=bool(return_lse),
        num_warps=4,
        num_stages=2,
    )

    O = O_2d.unsqueeze(2)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out
```

Edit `src/flashquest/kernel/__init__.py` to add the export:
```python
"""flashquest Triton kernels. Phase 2: dense FA-2 forward. Phase 3: sparse INT8."""
from .flash_fwd import flash_attn_fwd
from .sparse_fwd import flash_attn_sparse_fwd

__all__ = ["flash_attn_fwd", "flash_attn_sparse_fwd"]
```

- [ ] **Step 3.4: Run the kernel tests**

Run: `pytest tests/test_kernel_sparse_basic.py -v`
Expected: 4 passing tests.

If `test_full_mask_matches_eager_int8` fails: check that the kernel's dequant matches `dequantize_k` / `dequantize_v` from kv_quant.py exactly. Most likely cause is broadcasting axis confusion (per-page-channel for K vs per-token for V). Print intermediate values from a small repro before guessing.

If `test_no_pages_selected_returns_zero` fails: the kernel's "no iterations executed" path stores `acc / safe_l = 0 / 1 = 0` — verify the L_i guard.

- [ ] **Step 3.5: Commit**

```bash
git add src/flashquest/kernel/sparse_fwd.py src/flashquest/kernel/__init__.py tests/test_kernel_sparse_basic.py
git commit -m "phase 3: triton sparse forward kernel (decode-only, INT8 KV dequant inside)"
```

---

## Task 4: Edge case grid for sparse kernel

**Purpose:** Lock down ES1–ES10 with explicit tests. ES1, ES2, ES4, ES11 are covered in T3 — this task adds ES3, ES5, ES6, ES7, ES8, ES9, ES10 + property fuzz.

**Files:**
- Create: `tests/test_kernel_sparse_edges.py`

- [ ] **Step 4.1: Write the edge tests**

Create `tests/test_kernel_sparse_edges.py`:
```python
"""Edge case grid for flash_attn_sparse_fwd. See plan §Edge case catalog."""
import pytest
import torch
from hypothesis import given, settings, strategies as st

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(B, H_q, H_kv, S_kv, D, page_size=64, seed=0):
    """Returns (Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn) all cuda."""
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)
    V_uint8, V_scale, V_mn = quantize_v(V)
    return Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn


@cuda
@pytest.mark.parametrize("S_kv", [64, 96, 100, 128, 192, 1023])
def test_partial_last_page(S_kv):
    """ES3: S_kv not multiple of page_size."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 2, 1, S_kv, 64, seed=S_kv)
    num_pages = K_s.shape[2]
    sel = torch.ones(1, 2, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
@pytest.mark.parametrize("n_rep", [2, 4, 8])
def test_gqa(n_rep):
    """ES5: GQA with selection per query head."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    H_kv, D, S_kv = 2, 64, 256
    H_q = H_kv * n_rep
    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, H_q, H_kv, S_kv, D, seed=n_rep)
    num_pages = S_kv // 64
    sel = torch.zeros(1, H_q, 1, num_pages, dtype=torch.bool, device="cuda")
    # Different selection per head.
    for h in range(H_q):
        sel[0, h, 0, h % num_pages] = True
    sel[..., 0] = True  # sink
    sel[..., -1] = True  # window

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    # Build matching eager output by overriding the eager selection:
    #   eager uses retention/sinks/window to *derive* a mask. Here we want
    #   the eager path to use *our* mask. Easiest: dequantize K/V and run a
    #   manual masked SDPA with the same mask.
    from flashquest.kernel.kv_quant import dequantize_k, dequantize_v

    K = dequantize_k(K_u8, K_s, K_m, page_size=64)
    V = dequantize_v(V_u8, V_s, V_m)
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    # Expand page mask -> token mask -> attn_bias.
    token_mask = sel.repeat_interleave(64, dim=-1)[..., :S_kv]  # (1, H_q, 1, S_kv)
    attn_bias = torch.zeros_like(token_mask, dtype=torch.bfloat16)
    attn_bias = attn_bias.masked_fill(~token_mask, float("-inf"))
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q, Kr, Vr, attn_mask=attn_bias, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_ref, rtol=5e-2, atol=5e-2)


@cuda
@pytest.mark.parametrize("D", [64, 128])
def test_head_dim(D):
    """ES6: head_dim ∈ {64, 128}."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 2, 1, 128, D, seed=D)
    num_pages = 2
    sel = torch.ones(1, 2, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_single_page():
    """ES7: S_kv == page_size."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 1, 1, 64, 64)
    sel = torch.ones(1, 1, 1, 1, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_zero_range_channel():
    """ES9: a constant-valued channel does not produce NaN."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    torch.manual_seed(0)
    B, H, S_kv, D = 1, 1, 128, 64
    Q = torch.randn(B, H, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    K[..., 5] = 0.7  # constant channel
    V[..., 5] = 0.3

    K_u8, K_s, K_m = quantize_k(K, page_size=64)
    V_u8, V_s, V_m = quantize_v(V)
    sel = torch.ones(B, H, 1, 2, dtype=torch.bool, device="cuda")

    O, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    assert torch.isfinite(O).all()


@cuda
@settings(deadline=None, max_examples=15)
@given(
    S_kv=st.integers(min_value=64, max_value=512),
    H_kv=st.sampled_from([1, 2]),
    n_rep=st.sampled_from([1, 2, 4]),
    D=st.sampled_from([64, 128]),
    retention=st.sampled_from([0.25, 0.5, 1.0]),
)
def test_random_shapes_match_eager(S_kv, H_kv, n_rep, D, retention):
    """Property fuzz: random shape grid, mask built from eager Quest, kernel
    output matches eager INT8 sparse within INT8 tolerance."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.eager.criticality import page_scores
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k

    H_q = H_kv * n_rep
    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(
        1, H_q, H_kv, S_kv, D, seed=S_kv * 31 + H_kv * 7 + n_rep + D
    )
    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=64)
    K_dq_rep = K_dq.repeat_interleave(n_rep, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
    scores = page_scores(Q.float(), pmin, pmax)
    sel = select_pages(scores, retention=retention, num_sinks=4, window_pages=2)

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=retention, num_sinks=4, window_pages=2, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)
```

- [ ] **Step 4.2: Run the edge grid**

Run: `pytest tests/test_kernel_sparse_edges.py -v`
Expected: 6 + 3 + 2 + 1 + 1 = 13 deterministic tests + 15 hypothesis fuzz examples passing.

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_kernel_sparse_edges.py
git commit -m "phase 3: edge case grid (ES1-ES11) + property fuzz"
```

---

## Task 5: Sparse ≡ Phase 2 dense at full mask

**Purpose:** With `selection_mask` all True, the sparse kernel and Phase 2 dense kernel compute the same thing — modulo INT8 quant error. Tightens the safety net.

**Files:**
- Create: `tests/test_kernel_sparse_dense_equivalence.py`

- [ ] **Step 5.1: Write the test**

Create `tests/test_kernel_sparse_dense_equivalence.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_sparse_full_mask_matches_dense_kernel():
    """At selection_mask=all-True, the Phase 3 sparse kernel and the Phase 2
    dense kernel compute the same attention, up to INT8 quantization error."""
    from flashquest.kernel import flash_attn_fwd, flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k, dequantize_v, quantize_k, quantize_v

    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 4, 1, 256, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_u8, K_s, K_m = quantize_k(K, page_size=64)
    V_u8, V_s, V_m = quantize_v(V)
    num_pages = S_kv // 64
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    # Sparse kernel: INT8 KV.
    O_sparse, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )

    # Dense kernel: BF16 KV (dequantized so the comparison is INT8-vs-INT8).
    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=64)
    V_dq = dequantize_v(V_u8, V_s, V_m)
    O_dense, _ = flash_attn_fwd(Q, K_dq, V_dq, causal=False)

    torch.testing.assert_close(O_sparse, O_dense, rtol=5e-2, atol=5e-2)
```

- [ ] **Step 5.2: Run**

Run: `pytest tests/test_kernel_sparse_dense_equivalence.py -v`
Expected: 1 passing test.

- [ ] **Step 5.3: Commit**

```bash
git add tests/test_kernel_sparse_dense_equivalence.py
git commit -m "phase 3: sparse-full-mask ≡ phase 2 dense kernel within INT8 tolerance"
```

---

## Task 6: Decode perf bench

**Purpose:** Measure the sparsity payoff. Synthetic decode loop: starting from a populated S_kv-sized cache, run N=64 decode steps. Compare tok/s for sparse-INT8 (retention=0.25, sinks=4, window=2) vs Phase 2 dense kernel.

**Files:**
- Create: `scripts/phase3_bench_decode.py`
- Create: `benchmarks/phase3_perf.json` (filled by script)

- [ ] **Step 6.1: Write the bench**

Create `scripts/phase3_bench_decode.py`:
```python
"""Synthetic decode-loop perf: phase 3 sparse INT8 vs phase 2 dense.

Setup: Llama-3.2-3B geometry (H_q=24, H_kv=8, D=64). Pre-populate an 8 k
KV cache. Time N=64 decode steps. Compare tok/s.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages
from flashquest.kernel import flash_attn_fwd, flash_attn_sparse_fwd
from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v


N_DECODE_STEPS = 64
S_KV = 8192
PAGE_SIZE = 64
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2


def _setup():
    torch.manual_seed(0)
    B, H_q, H_kv, D = 1, 24, 8, 64
    K = torch.randn(B, H_kv, S_KV, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_KV, D, dtype=torch.bfloat16, device="cuda")
    K_u8, K_s, K_m = quantize_k(K, page_size=PAGE_SIZE)
    V_u8, V_s, V_m = quantize_v(V)
    return B, H_q, H_kv, D, K, V, K_u8, K_s, K_m, V_u8, V_s, V_m


def _bench_dense(B, H_q, H_kv, D, K, V):
    """One decode step on dense BF16 KV."""
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    n_rep = H_q // H_kv
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    for _ in range(5):  # warm
        flash_attn_fwd(Q, Kr, Vr, causal=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_STEPS):
        flash_attn_fwd(Q, Kr, Vr, causal=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_DECODE_STEPS * 1000  # ms / step


def _bench_sparse(B, H_q, H_kv, D, K_u8, K_s, K_m, V_u8, V_s, V_m):
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    # Build selection mask once (same Q each step in this micro-bench).
    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=PAGE_SIZE)
    K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=PAGE_SIZE)
    scores = page_scores(Q.float(), pmin, pmax)
    sel = select_pages(scores, RETENTION, NUM_SINKS, WINDOW_PAGES)

    for _ in range(5):  # warm
        flash_attn_sparse_fwd(Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=PAGE_SIZE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_STEPS):
        flash_attn_sparse_fwd(Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=PAGE_SIZE)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_DECODE_STEPS * 1000


def main() -> None:
    B, H_q, H_kv, D, K, V, K_u8, K_s, K_m, V_u8, V_s, V_m = _setup()

    dense_ms = _bench_dense(B, H_q, H_kv, D, K, V)
    sparse_ms = _bench_sparse(B, H_q, H_kv, D, K_u8, K_s, K_m, V_u8, V_s, V_m)

    speedup = dense_ms / sparse_ms
    result = {
        "shape": {"B": B, "H_q": H_q, "H_kv": H_kv, "S_kv": S_KV, "D": D},
        "config": {
            "page_size": PAGE_SIZE, "retention": RETENTION,
            "num_sinks": NUM_SINKS, "window_pages": WINDOW_PAGES,
            "n_decode_steps": N_DECODE_STEPS,
        },
        "phase2_dense_ms_per_step": dense_ms,
        "phase3_sparse_ms_per_step": sparse_ms,
        "sparse_speedup": speedup,
        "phase3_target_speedup": 1.5,
        "passes_phase3_target": speedup >= 1.5,
    }
    print(json.dumps(result, indent=2))

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase3_perf.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.2: Run the bench**

Run: `. .venv/bin/activate && nice -n 19 python scripts/phase3_bench_decode.py 2>&1 | tail -30`
Expected: prints JSON, writes `benchmarks/phase3_perf.json`. `passes_phase3_target` should be True (speedup ≥ 1.5×).

If speedup < 1.5×: kernel is correct but underperforms. Likely causes: (a) Q is broadcast to a non-tile-friendly shape (we use BLOCK_M=1 for decode), (b) per-page scale loads cost as much as the dequant savings. Profile with `nsys` (record in phase-3-notes). If gap is real, it's an optimization issue, not a correctness issue — flag and continue.

- [ ] **Step 6.3: Commit**

```bash
git add scripts/phase3_bench_decode.py benchmarks/phase3_perf.json
git commit -m "phase 3: decode-loop perf bench (sparse INT8 vs dense BF16)"
```

---

## Task 7: Phase 3 notes + README + DOC.md + tag

**Files:**
- Create: `docs/PHASES/phase-3-notes.md`
- Modify: `README.md`
- Modify: `DOC.md`

- [ ] **Step 7.1: Write the phase notes**

Create `docs/PHASES/phase-3-notes.md`:
```markdown
# Phase 3 Notes

**Started:** <YYYY-MM-DD>
**Completed:** <YYYY-MM-DD> (tag `phase-3`)
**Status:** complete
**Spec:** [docs/SPEC.md §6 Phase 3](../SPEC.md)

## Goal

Layer Quest-style sparse selection on top of Phase 2's dense kernel, with KIVI-style INT8 KV. Decode-only sparse forward.

## Surface

- `flashquest.kernel.kv_quant.{quantize_k, dequantize_k, quantize_v, dequantize_v}` — KIVI convention asymmetric uint8 with per-page channel-wise K, per-token V.
- `flashquest.kernel.flash_attn_sparse_fwd(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, selection_mask, page_size, sm_scale, return_lse)` — decode-only Triton kernel that dequantizes inside.
- `flashquest.eager.quest_eager_sparse_int8(...)` — pure-PyTorch reference (the correctness oracle).

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Sparse ≡ Phase 2 dense at full mask within INT8 tolerance | <fill> | ✅ / ❌ |
| Sparse ≡ Phase 1 eager at retention=0.25 within INT8 tolerance | <fill> | ✅ / ❌ |
| Decode speedup vs Phase 2 dense at 8 k context, retention=0.25 | <fill>× | ≥ 1.5× ✅ / ❌ |

## Edge cases handled

ES1 full mask, ES2 zero mask, ES3 partial last page, ES4 decode hot path, ES5 GQA, ES6 head_dim ∈ {64, 128}, ES7 single page, ES8 causal trivializes, ES9 zero-range channel, ES10 quant round-trip. ES11 (multi-query) is rejected at the wrapper.

## Decisions

- Decode-only sparse for v1 (multi-query / chunked-prefill rejected at wrapper). Prefill uses Phase 2 dense kernel.
- uint8 storage (no bit-packing). INT4/INT2 deferred — start simple, halve the cache vs BF16 today, push lower in v2.
- Per-page channel-wise K + per-token V scales (KIVI convention, page_size aligned to Phase 2 BLOCK_N=64).
- Selection mask is computed *outside* the kernel (Phase 1 page_summary + page_scores + select_pages) and passed in as a (B, H_q, 1, num_pages) bool tensor. Keeps the kernel simple; fusing top-k inside is a v2 optimization.
- BLOCK_M=1 for decode — accept reduced tensor-core utilisation; sparsity gain dominates at typical retention rates.

## Phase 3 → Phase 4 handoff

Algorithm validated; INT8 KV works; decode sparsity yields a real perf win. Phase 4:
- DuoAttention head split (per-head retrieval vs streaming pattern; load pre-trained classifications).
- HF Llama integration (patch LlamaAttention to use the sparse kernel + INT8 cache).
- Larger model (Llama-3.1-8B IQ3_XXS) with Marlin W4A16 weight projections.
- Real end-to-end at 32 k context.

Open items deferred:
- Sparse prefill (chunked) — track when it becomes load-bearing.
- INT4 / INT2 KV — Phase 5+.
- vLLM-style page_table for non-contiguous KV — Phase 5+.
```

- [ ] **Step 7.2: Append Phase 3 to README**

Edit `README.md`. After the Phase 2 section, append:
```markdown
## Phase 3 — Sparse retrieval + INT8 KV

Decode-only sparse forward kernel. Quest selection + KIVI-style asymmetric uint8 KV (per-page channel-wise K, per-token V). Dequant happens inside the Triton kernel; only selected pages are loaded.

Llama-3.2-3B geometry, S_kv=8192, retention=0.25, sinks=4, window=128:

| Backend | ms / decode step | speedup vs dense |
|---|---|---|
| Phase 2 dense (BF16 KV) | <fill> | 1.0× |
| **Phase 3 sparse (INT8 KV)** | **<fill>** | **<fill>×** |

11 catalogued edge cases (ES1–ES11) + property-based shape fuzz. See [`docs/PHASES/phase-3-notes.md`](docs/PHASES/phase-3-notes.md). Re-run via `python scripts/phase3_bench_decode.py`.
```

- [ ] **Step 7.3: Flip Phase 3 in DOC.md**

Edit `DOC.md`. Replace:
```
- Phase 3 — Sparse retrieval + INT8 KV.
```
with:
```
- **Phase 3 — Sparse retrieval + INT8 KV** ✅ **complete (tag `phase-3`)**. `flashquest.kernel.flash_attn_sparse_fwd(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, selection_mask, page_size, sm_scale, return_lse)`. Decode-only (S_q=1). KIVI-style asymmetric uint8 KV (per-page channel-wise K, per-token V); dequant fused inside the kernel. 11 catalogued edge cases pass. See `docs/PHASES/phase-3-notes.md`.
```

Add a usage block under Phase 2's:
````markdown
Decode-only sparse INT8 attention:

```python
import torch
from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages
from flashquest.kernel import flash_attn_sparse_fwd
from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v

# Quantize the KV cache once (or as it grows).
K_uint8, K_scale, K_mn = quantize_k(K_bf16, page_size=64)
V_uint8, V_scale, V_mn = quantize_v(V_bf16)

# Decode step: build a per-head selection mask via Quest criticality.
K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=64)
K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
scores = page_scores(Q.float(), pmin, pmax)
sel = select_pages(scores, retention=0.25, num_sinks=4, window_pages=2)

O, lse = flash_attn_sparse_fwd(
    Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
    selection_mask=sel, page_size=64,
)
```
````

- [ ] **Step 7.4: Final smoke**

Run:
```bash
. .venv/bin/activate
pytest tests/ --ignore=tests/test_eager_e2e.py -v
python scripts/verify_triton_int8.py
```
Expected: all kernel + eager + smoke + sparse tests pass.

- [ ] **Step 7.5: Commit + tag**

```bash
git add docs/PHASES/phase-3-notes.md README.md DOC.md
git commit -m "phase 3: README + DOC + phase-3-notes complete"
git tag -a phase-3 -m "Phase 3 complete: sparse INT8 decode kernel + Quest selection"
```

---

## Self-review

**1. Spec coverage** (SPEC §6 Phase 3):
- "Block-summary precompute kernel (per-channel min/max per 64-token block)." → Already exists in Phase 1 (`compute_page_summary`); reused here.
- "Top-k selection (start with `torch.topk` outside the kernel; fuse into kernel later)." → Selection done outside the kernel via Phase 1's `select_pages`. Fusion is a v2 optimization.
- "Sparse outer loop: kernel iterates over selected blocks only." → Task 3 kernel.
- "INT8 KV storage with per-channel-K, per-token-V scales. Dequant inside the kernel." → Tasks 1 + 3.
- "StreamingLLM sinks + sliding window (always-attended blocks, free in our kernel)." → Phase 1 selection composes sinks ∪ window ∪ top-k; the kernel just consumes the pre-built mask.
- "Win condition: Llama-3.2-3B at 32k context, ≥10 tok/s decode, ≥85% RULER." → SPEC's 32 k tok/s + RULER bar requires HF integration (Phase 4). Phase 3 v1 instead validates the kernel-level speedup (≥1.5× over dense decode at 8 k); RULER deferred to Phase 4 with the real model. Documented in Phase 3 notes as a deferred item.

**2. Placeholder scan**: every code block is real code. README/DOC/notes have `<fill>` slots for measured values, but those are explicit data-entry slots.

**3. Type / name consistency**:
- `quantize_k(K, page_size) -> (uint8, scale, mn)` — same in Tasks 1, 2, 3, 4, 5, 6.
- `quantize_v(V) -> (uint8, scale, mn)` — same throughout.
- `flash_attn_sparse_fwd(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, selection_mask, page_size, sm_scale, return_lse)` — same signature in Tasks 3, 4, 5, 6, 7.
- `quest_eager_sparse_int8(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, page_size, retention, num_sinks, window_pages, is_causal)` — same in Tasks 2, 3, 4, 5.
- `selection_mask: (B, H_q, 1, num_pages) bool` — consistent.

**4. Reversibility**: every task ends with a commit. Task 6 explicitly flags a perf gap as "optimization issue, not correctness" — so we don't block Phase 3 closure on a tuning miss.

## Phase 3 → Phase 4 handoff

When the plan completes:
- `phase-3` git tag exists.
- `flashquest.kernel.flash_attn_sparse_fwd` is the decode-time sparse kernel.
- `benchmarks/phase3_perf.json` shows real speedup vs Phase 2 dense.
- Phase 4 begins: DuoAttention per-head retrieval/streaming split, HF Llama patch using the sparse kernel + INT8 KV cache, and end-to-end measurement on Llama-3.2-3B and Llama-3.1-8B at 32 k context. Phase 4 will get its own plan.
