# Phase 5 — Persistent INT8 KV Cache + AWQ + Fused DuoDispatch + 32k Eval Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the production-grade decode path for flashquest: a persistent INT8 KV cache (HF `Cache` subclass), AWQ-INT4 weight loading, a fused per-head DuoAttention dispatch (one sparse-kernel call per layer instead of Phase 4's two-paths-then-`torch.where`), and a 32 k-context passkey + decode benchmark on Llama-3.2-3B-AWQ. The 8B-at-32k SPEC win is hardware-blocked on 4 GB VRAM (Llama-3.1-8B AWQ ~4.5 GB weights alone); we run a smaller-context 8B eval as a control and document the wall explicitly.

**Architecture:** A `PersistentInt8KVCache` subclasses `transformers.cache_utils.Cache`, pre-allocating uint8 K/V buffers (max_seq_len) plus per-page channel-wise K scales/mns (Phase 3 KIVI layout) and per-token V scales/mns. A small BF16 staging buffer (one partial page per layer) holds tokens that haven't yet filled a page; once full, the page is quantized and flushed. The patched LlamaAttention forward writes K/V to the cache after RoPE, reads back the int8 views, and at decode time runs `flash_attn_sparse_fwd` over completed pages and a tiny BF16 dense attention over the partial-page tail, merging both via online softmax (LSE). At prefill (S_q > 1) we keep the existing dense BF16 path on the dequant'd cache. The fused DuoAttention dispatch sets retention=0 for streaming heads inside a per-head-aware `select_pages`, so a single sparse-kernel call serves both head classes (~50% wall-clock vs Phase 4's torch.where).

**Tech Stack:** Same as Phase 4 (torch 2.5.1+cu121, triton 3.1.0, transformers 4.57.6) plus `autoawq` for AWQ-INT4 model loading. Test model: `casperhansen/llama-3.2-3b-instruct-awq` (~2 GB weights, fits at 32k with sparse INT8 KV). Optional 8B control: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` at the largest context that fits (likely ≤8k).

**Win conditions:**
- `PersistentInt8KVCache` round-trip: BF16 K/V → cache.update → cache.get_views → dequant matches Phase 3's `quantize_k`/`dequantize_k` pair to `rtol=1e-2`.
- HF Llama with persistent cache: end-to-end logits match Phase 4 patch on Llama-3.2-1B at `S=128` to `rtol=5e-2` (INT8 quant noise vs Phase 4's BF16 eager).
- Fused DuoDispatch ≡ Phase 4 `quest_duo_eager_sdpa` to `rtol=2e-2` on (B=1, GQA 4:2, S=512), uniform mixed pattern.
- AWQ load: `casperhansen/llama-3.2-3b-instruct-awq` runs a forward pass and produces non-NaN logits.
- 32 k passkey on Llama-3.2-3B-AWQ at retention=0.25 70/30 split: ≥80 % at depth=0.5 (3 trials × 3 depths × 3 contexts {8k, 16k, 32k}).
- Decode at 32 k context: ≥4 tok/s sustained (SPEC §3 win was ≥10 tok/s for 3B at 128k; we run at 32k with full sparse INT8 path).
- Phase 5 doc: 8B reality + Marlin/EAGLE-2/INT4-KV deferral rationale.

**Phase 5 explicit non-goals (deferred to Phase 6 or v2):**
- Marlin W4A16 packing conversion (AWQ via `autoawq` already gives W4A16; explicit Marlin only pays off if the AWQ kernel becomes the decode bottleneck — measure first).
- ExLlamaV2 attention-backend integration (orthogonal ecosystem; HF + AWQ is sufficient at 3B/32k).
- EAGLE-2 speculative decoding wrapper (orthogonal optimization; layer on after kernel is stable).
- INT4 KV cache (KIVI's full target — Phase 5 stays at INT8 to keep the kernel changes contained).
- Llama-3.1-8B at 32k (SPEC Phase 4/5 8B win condition is hardware-blocked; documented and demonstrated at smaller context as a control).
- Full RULER eval (hours-long; we run passkey at 8k/16k/32k as a tractable proxy).
- Triton-side kernel changes (the existing Phase 3 sparse kernel already accepts per-head selection masks; the fused dispatch is purely a Python-side `select_pages` change).

**Hardware envelope:** Same as Phases 0-4. Working VRAM budget 3.0 GB. AWQ 3B weights ~2 GB + sparse INT8 KV at 32k ~70 MB + activations ~150 MB ≈ 2.2 GB total — comfortable.

---

## Edge case catalog — Phase 5

| ID | Case | Why it matters |
|---|---|---|
| EQ1 | Cache `update` on a fresh cache, S_new = page_size (one full page) | Bulk-quantize-and-flush exactly. |
| EQ2 | Cache `update` with S_new straddling a page boundary | Partial page → fill → flush + new partial. |
| EQ3 | Cache `update` decode step (S_new = 1) into mid-page | Partial-buffer append, no flush. |
| EQ4 | Cache `update` decode step (S_new = 1) that completes a page | Partial → full → flush, then partial empties. |
| EQ5 | Cache `get_views` when seen < page_size | num_complete_pages = 0; partial buffer = full payload. |
| EQ6 | Cache pre-allocated max_seq_len smaller than actual run | Must error clearly, not silently overwrite. |
| EQ7 | Cache layer_idx out of range | IndexError, not silent corruption. |
| EQ8 | Per-head retention scalar ≡ tensor of identical values | API back-compat with Phase 1-4 callers. |
| EQ9 | Per-head retention tensor with 0.0 entries | Streaming heads: only sinks + window selected. |
| EQ10 | Per-head retention tensor with 1.0 entries | Retrieval-everything heads: full mask. |
| EQ11 | Fused DuoDispatch all-retrieval ≡ Phase 4 (one path) | Sanity. |
| EQ12 | Fused DuoDispatch all-streaming ≡ Phase 4 streaming | Sanity. |
| EQ13 | AWQ load wrong dtype (e.g., fp32 weights) | Loader rejects with explicit error. |
| EQ14 | AWQ load model.config.quantization_config absent | Loader rejects (signals not actually quantized). |
| EQ15 | 32 k passkey with seen_tokens > max_seq_len at last decode step | Cache must error before sparse kernel reads bad memory. |
| EQ16 | Online softmax merge with empty partial buffer (S_partial = 0) | Returns sparse-kernel output unchanged. |
| EQ17 | Online softmax merge with empty completed-pages region (seen < page_size) | Returns dense BF16 attention output unchanged. |

---

## File Structure

**Created:**
- `src/flashquest/cache/persistent_int8.py` — `PersistentInt8KVCache(transformers.cache_utils.Cache)`.
- `src/flashquest/eager/llama_persistent_patch.py` — HF Llama monkeypatch using the persistent cache + fused dispatch.
- `src/flashquest/duo/fused_dispatch.py` — `quest_duo_fused_sdpa` (single sparse-kernel call with per-head retention).
- `src/flashquest/runtime/awq_load.py` — `load_awq_model(name) -> (model, tokenizer)` helper.
- `tests/test_persistent_cache.py` — cache class behaviour (EQ1-EQ7).
- `tests/test_persistent_e2e.py` — Llama-3.2-1B end-to-end equivalence (rtol=5e-2 vs Phase 4).
- `tests/test_per_head_retention.py` — `select_pages` per-head retention vector.
- `tests/test_fused_dispatch.py` — `quest_duo_fused_sdpa` ≡ Phase 4 (EQ11, EQ12) + mixed.
- `tests/test_awq_load.py` — gated by checkpoint availability; skip if offline.
- `scripts/phase5_run_passkey_32k.py` — passkey at {8k, 16k, 32k} on Llama-3.2-3B-AWQ.
- `scripts/phase5_bench_decode_32k.py` — decode tok/s at 32k.
- `scripts/phase5_run_8b_control.py` — Llama-3.1-8B-AWQ at largest fitting context (control).
- `benchmarks/phase5_passkey.json`, `benchmarks/phase5_decode.json`, `benchmarks/phase5_8b_control.json` — eval outputs.
- `docs/PHASES/phase-5-notes.md` — phase journal.

**Modified:**
- `src/flashquest/eager/selection.py` — `select_pages` accepts `retention: float | torch.Tensor`.
- `src/flashquest/cache/__init__.py` — export `PersistentInt8KVCache`.
- `src/flashquest/duo/__init__.py` — export `quest_duo_fused_sdpa`.
- `src/flashquest/runtime/__init__.py` — export `load_awq_model`.
- `pyproject.toml` — add `autoawq` to `[bench]` extras.
- `README.md` — Phase 5 row + 32k benchmark table + 8B-doesn't-fit footnote.
- `DOC.md` — flip Phase 5 status; add persistent-cache usage example.

---

## Conventions

- **Cache memory layout (per layer):**
  - `K_uint8`: `(B, H_kv, max_seq_len, D)` uint8 — completed-page region only valid up to `seen_tokens // page_size * page_size`.
  - `K_scale`, `K_mn`: `(B, H_kv, max_pages, D)` bfloat16 — valid up to `seen_tokens // page_size`.
  - `V_uint8`: `(B, H_kv, max_seq_len, D)` uint8 — valid up to `seen_tokens // page_size * page_size` (V quantizes per-token but we still flush per-page for layout uniformity with K).
  - `V_scale`, `V_mn`: `(B, H_kv, max_seq_len, 1)` bfloat16 — same validity range.
  - `K_partial`, `V_partial`: `(B, H_kv, page_size, D)` bfloat16 — staging for the current incomplete page; valid up to `seen_tokens % page_size`.
- **`Cache` subclass contract:** the patched LlamaAttention does NOT use `cache.update`'s return value. It calls `cache.update_quantized(K_new, V_new, layer_idx)` (a non-standard method) which writes to the cache and returns nothing. The forward then calls `cache.get_views(layer_idx)` for the int8 views. We implement the standard `update`/`get_seq_length`/`get_max_length` for HF compatibility but the data path is `update_quantized`/`get_views`.
- **Online-softmax merge** (decode):
  - `O_sparse, lse_sparse = flash_attn_sparse_fwd(Q, K_uint8 views, V_uint8 views, selection_mask=...)` covers completed-page region.
  - `O_partial, lse_partial = dense_bf16_attention(Q, K_partial[:S_partial], V_partial[:S_partial])` covers the BF16 partial-page tail (S_partial ≤ 63).
  - Merge: `m = max(lse_sparse, lse_partial); O = (exp(lse_sparse - m) * O_sparse + exp(lse_partial - m) * O_partial) / (exp(lse_sparse - m) + exp(lse_partial - m))`.
  - Edge: if S_partial == 0, return `O_sparse`. If seen_tokens < page_size, return `O_partial` (no completed pages).
- **Per-head retention API:** `select_pages(scores, retention, num_sinks, window_pages)` — `retention` is `float` (back-compat) OR `torch.Tensor` of shape `(H,)` aligned with the H axis of `scores`. Per-head k = `ceil(retention[h] * P)`.
- **Fused DuoDispatch:** `head_pattern: (H_kv,) bool`. Build per-q-head retention vector: `retention_per_q_head[h_q] = retention if head_pattern[h_q // n_rep] else 0.0`. One `select_pages` call → one `flash_attn_sparse_fwd` call.
- **AWQ load:** `from awq import AutoAWQForCausalLM` (autoawq pkg). We use HF's `AutoModelForCausalLM.from_pretrained` which auto-detects AWQ via `quantization_config`. Verify: `model.config.quantization_config["quant_method"] == "awq"`.

---

## Task 1: Per-head retention in `select_pages`

**Files:**
- Modify: `src/flashquest/eager/selection.py`
- Test: `tests/test_per_head_retention.py`

- [ ] **Step 1: Write failing test for tensor retention back-compat with scalar**

```python
# tests/test_per_head_retention.py
import torch
from flashquest.eager.selection import select_pages


def test_tensor_retention_uniform_matches_scalar():
    torch.manual_seed(0)
    scores = torch.randn(1, 4, 8, 16)  # B=1, H=4, S_q=8, P=16
    mask_scalar = select_pages(scores, retention=0.25, num_sinks=2, window_pages=2)

    retention_t = torch.tensor([0.25, 0.25, 0.25, 0.25])
    mask_tensor = select_pages(scores, retention=retention_t, num_sinks=2, window_pages=2)

    assert torch.equal(mask_scalar, mask_tensor)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_per_head_retention.py::test_tensor_retention_uniform_matches_scalar -v`
Expected: FAIL — `select_pages` rejects tensor input.

- [ ] **Step 3: Modify `select_pages` to accept tensor retention**

Replace the body of `src/flashquest/eager/selection.py`:

```python
"""Top-k page selection with sink + sliding-window always-attended set."""
from __future__ import annotations

import math

import torch


def select_pages(
    scores: torch.Tensor,
    retention: float | torch.Tensor,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Build a boolean mask over pages: union of top-k by score with sinks + window.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: fraction of pages to select via top-k. Scalar in [0, 1] OR
            a 1-D tensor of shape (H,) for per-head retention.
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include (recency window).

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

    mask = torch.zeros_like(scores, dtype=torch.bool)

    # Per-head top-k. Heads with retention=0 contribute nothing here.
    for h in range(H):
        r = retention_per_h[h].item()
        if r >= 1.0:
            mask[:, h] = True
            continue
        if r <= 0.0:
            continue
        k = math.ceil(r * P)
        if k > 0:
            topk_idx = scores[:, h].topk(k, dim=-1).indices
            mask[:, h].scatter_(-1, topk_idx, True)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask
```

- [ ] **Step 4: Run the new test to verify it passes**

Run: `pytest tests/test_per_head_retention.py::test_tensor_retention_uniform_matches_scalar -v`
Expected: PASS.

- [ ] **Step 5: Add a mixed-retention test**

Append to `tests/test_per_head_retention.py`:

```python
def test_tensor_retention_mixed_zero_streaming_heads():
    """Heads with retention=0 select only sinks + window (no top-k)."""
    torch.manual_seed(1)
    scores = torch.randn(1, 4, 1, 16)
    retention_t = torch.tensor([0.5, 0.0, 0.5, 0.0])
    mask = select_pages(scores, retention=retention_t, num_sinks=2, window_pages=2)

    # Heads 1 and 3 (retention=0): only first 2 + last 2 pages selected.
    expected_streaming = torch.zeros(16, dtype=torch.bool)
    expected_streaming[:2] = True
    expected_streaming[-2:] = True
    assert torch.equal(mask[0, 1, 0], expected_streaming)
    assert torch.equal(mask[0, 3, 0], expected_streaming)

    # Heads 0 and 2 (retention=0.5): 8 top-k + sinks + window. At least 8 selected.
    assert mask[0, 0, 0].sum() >= 8
    assert mask[0, 2, 0].sum() >= 8


def test_scalar_retention_existing_callers_unchanged():
    """Phase 1 callers passing float retention must still work."""
    scores = torch.randn(2, 8, 4, 32)
    mask = select_pages(scores, retention=0.1, num_sinks=4, window_pages=2)
    assert mask.shape == (2, 8, 4, 32)
    assert mask.dtype == torch.bool
```

- [ ] **Step 6: Run all `select_pages` tests**

Run: `pytest tests/test_per_head_retention.py tests/test_eager_selection.py -v`
Expected: PASS (all pre-existing scalar-retention tests still green).

- [ ] **Step 7: Commit**

```bash
git add src/flashquest/eager/selection.py tests/test_per_head_retention.py
git commit -m "phase 5: select_pages accepts per-head retention tensor"
```

---

## Task 2: PersistentInt8KVCache skeleton + allocation

**Files:**
- Create: `src/flashquest/cache/persistent_int8.py`
- Create: `tests/test_persistent_cache.py`
- Modify: `src/flashquest/cache/__init__.py`

- [ ] **Step 1: Write failing instantiation test**

```python
# tests/test_persistent_cache.py
import pytest
import torch

from flashquest.cache import PersistentInt8KVCache


def test_cache_allocation_shapes():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=4, num_kv_heads=8, head_dim=128,
        max_seq_len=2048, page_size=64, device="cuda",
    )
    assert cache.K_uint8.shape == (4, 1, 8, 2048, 128)
    assert cache.K_uint8.dtype == torch.uint8
    assert cache.K_scale.shape == (4, 1, 8, 32, 128)  # 2048 / 64 = 32 pages
    assert cache.K_scale.dtype == torch.bfloat16
    assert cache.V_scale.shape == (4, 1, 8, 2048, 1)
    assert cache.K_partial.shape == (4, 1, 8, 64, 128)
    assert cache.K_partial.dtype == torch.bfloat16
    assert cache.get_seq_length(0) == 0
    assert cache.get_max_length() == 2048
```

- [ ] **Step 2: Run test to verify it fails (module missing)**

Run: `pytest tests/test_persistent_cache.py::test_cache_allocation_shapes -v`
Expected: FAIL — `ModuleNotFoundError: flashquest.cache.PersistentInt8KVCache`.

- [ ] **Step 3: Create `cache/persistent_int8.py` with skeleton**

```python
# src/flashquest/cache/persistent_int8.py
"""Persistent INT8 KV cache for HF transformers integration.

Layout:
- Completed pages stored as KIVI-style uint8 with per-page channel-wise K
  and per-token V (Phase 3 layout).
- A small BF16 staging buffer (`K_partial`, `V_partial`) holds the current
  incomplete page; once full, it's quantized and flushed.
- The patched LlamaAttention forward calls `update_quantized(K_new, V_new,
  layer_idx)` after RoPE and reads `get_views(layer_idx)` for the int8
  views.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache


class PersistentInt8KVCache(Cache):
    def __init__(
        self,
        *,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int = 64,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        if max_seq_len % page_size != 0:
            # Allow non-multiple but round up the page allocation.
            pass
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        max_pages = (max_seq_len + page_size - 1) // page_size
        self.max_pages = max_pages
        dev = torch.device(device)

        shape_kv = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        shape_kpage = (num_layers, batch_size, num_kv_heads, max_pages, head_dim)
        shape_vtok = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_partial = (num_layers, batch_size, num_kv_heads, page_size, head_dim)

        self.K_uint8 = torch.zeros(shape_kv, dtype=torch.uint8, device=dev)
        self.V_uint8 = torch.zeros(shape_kv, dtype=torch.uint8, device=dev)
        self.K_scale = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.K_mn = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.V_scale = torch.zeros(shape_vtok, dtype=torch.bfloat16, device=dev)
        self.V_mn = torch.zeros(shape_vtok, dtype=torch.bfloat16, device=dev)
        self.K_partial = torch.zeros(shape_partial, dtype=torch.bfloat16, device=dev)
        self.V_partial = torch.zeros(shape_partial, dtype=torch.bfloat16, device=dev)

        self._seen_tokens = [0] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens[layer_idx]

    def get_max_length(self) -> int:
        return self.max_seq_len

    def update_quantized(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        layer_idx: int,
    ) -> None:
        raise NotImplementedError("filled in Task 3")

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("filled in Task 4")

    # HF Cache contract — we don't expose BF16 K/V here, but transformers may
    # call update() before our patch overrides the forward path. Return
    # empties to satisfy the API; the patched forward never uses these.
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "PersistentInt8KVCache.update should not be called directly; "
            "the flashquest patch uses update_quantized() + get_views()."
        )
```

- [ ] **Step 4: Add export to `cache/__init__.py`**

```python
# src/flashquest/cache/__init__.py
from .persistent_int8 import PersistentInt8KVCache

__all__ = ["PersistentInt8KVCache"]
```

- [ ] **Step 5: Run instantiation test to verify it passes**

Run: `pytest tests/test_persistent_cache.py::test_cache_allocation_shapes -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/flashquest/cache/persistent_int8.py src/flashquest/cache/__init__.py tests/test_persistent_cache.py
git commit -m "phase 5: PersistentInt8KVCache skeleton + allocation"
```

---

## Task 3: PersistentInt8KVCache.update_quantized

**Files:**
- Modify: `src/flashquest/cache/persistent_int8.py`
- Modify: `tests/test_persistent_cache.py`

- [ ] **Step 1: Write failing test for prefill bulk-quantize (S_new = 2 * page_size)**

Append to `tests/test_persistent_cache.py`:

```python
from flashquest.kernel.kv_quant import quantize_k, quantize_v, dequantize_k, dequantize_v


def test_update_quantized_prefill_full_pages():
    torch.manual_seed(0)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=2, num_kv_heads=4, head_dim=64,
        max_seq_len=512, page_size=64, device="cuda",
    )
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")

    cache.update_quantized(K, V, layer_idx=0)
    assert cache.get_seq_length(0) == 128

    # Two complete pages flushed; partial buffer empty.
    K_uint8_ref, K_scale_ref, K_mn_ref = quantize_k(K, page_size=64)
    V_uint8_ref, V_scale_ref, V_mn_ref = quantize_v(V)

    torch.testing.assert_close(cache.K_uint8[0, :, :, :128, :], K_uint8_ref)
    torch.testing.assert_close(cache.K_scale[0, :, :, :2, :], K_scale_ref)
    torch.testing.assert_close(cache.K_mn[0, :, :, :2, :], K_mn_ref)
    torch.testing.assert_close(cache.V_uint8[0, :, :, :128, :], V_uint8_ref)
    torch.testing.assert_close(cache.V_scale[0, :, :, :128, :], V_scale_ref)
    torch.testing.assert_close(cache.V_mn[0, :, :, :128, :], V_mn_ref)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_persistent_cache.py::test_update_quantized_prefill_full_pages -v`
Expected: FAIL — `update_quantized` raises `NotImplementedError`.

- [ ] **Step 3: Implement `update_quantized` for any S_new**

Replace the `update_quantized` body in `src/flashquest/cache/persistent_int8.py`:

```python
    def update_quantized(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Append K_new, V_new (B, H_kv, S_new, D) bf16 to layer_idx's cache.

        Completes any partial page first, then bulk-quantizes whole pages,
        then stages any remaining partial page in BF16.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        from flashquest.kernel.kv_quant import quantize_k, quantize_v

        seen = self._seen_tokens[layer_idx]
        S_new = K_new.shape[2]
        if seen + S_new > self.max_seq_len:
            raise RuntimeError(
                f"PersistentInt8KVCache: seen+new={seen + S_new} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        page_size = self.page_size

        # Step 1: combine current partial buffer + new tokens into a contiguous
        # BF16 stream we can bulk-process.
        partial_len = seen % page_size
        partial_K = self.K_partial[layer_idx, :, :, :partial_len, :]
        partial_V = self.V_partial[layer_idx, :, :, :partial_len, :]
        K_stream = torch.cat([partial_K, K_new], dim=2)
        V_stream = torch.cat([partial_V, V_new], dim=2)

        # Step 2: how many complete pages do we have starting from the page-aligned
        # slot in the cache? `seen - partial_len` is page-aligned.
        page_start_token = seen - partial_len  # page_start_token / page_size = next page idx to write
        total_stream_len = K_stream.shape[2]
        n_complete_pages = total_stream_len // page_size
        complete_len = n_complete_pages * page_size
        new_partial_len = total_stream_len - complete_len

        # Step 3: bulk-quantize the complete-page region.
        if n_complete_pages > 0:
            K_full = K_stream[:, :, :complete_len, :]
            V_full = V_stream[:, :, :complete_len, :]
            K_uint8, K_scale, K_mn = quantize_k(K_full, page_size=page_size)
            V_uint8, V_scale, V_mn = quantize_v(V_full)

            tok_start = page_start_token
            tok_end = page_start_token + complete_len
            page_idx_start = tok_start // page_size
            page_idx_end = page_idx_start + n_complete_pages

            self.K_uint8[layer_idx, :, :, tok_start:tok_end, :] = K_uint8
            self.V_uint8[layer_idx, :, :, tok_start:tok_end, :] = V_uint8
            self.K_scale[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_scale
            self.K_mn[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_mn
            self.V_scale[layer_idx, :, :, tok_start:tok_end, :] = V_scale
            self.V_mn[layer_idx, :, :, tok_start:tok_end, :] = V_mn

        # Step 4: stage the new partial page (BF16).
        if new_partial_len > 0:
            self.K_partial[layer_idx, :, :, :new_partial_len, :] = K_stream[:, :, complete_len:, :]
            self.V_partial[layer_idx, :, :, :new_partial_len, :] = V_stream[:, :, complete_len:, :]
        # Zero out beyond the new partial length to avoid stale data.
        if new_partial_len < page_size:
            self.K_partial[layer_idx, :, :, new_partial_len:, :].zero_()
            self.V_partial[layer_idx, :, :, new_partial_len:, :].zero_()

        self._seen_tokens[layer_idx] = seen + S_new
```

- [ ] **Step 4: Run prefill test to verify it passes**

Run: `pytest tests/test_persistent_cache.py::test_update_quantized_prefill_full_pages -v`
Expected: PASS.

- [ ] **Step 5: Add decode-step test (S_new = 1, mid-page)**

Append to `tests/test_persistent_cache.py`:

```python
def test_update_quantized_decode_mid_page():
    """Decode steps that don't complete a page stay in the partial buffer."""
    torch.manual_seed(2)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=256, page_size=64, device="cuda",
    )
    # 10 decode steps of S_new=1; no page completes.
    K_steps = torch.randn(10, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    V_steps = torch.randn(10, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    for K_t, V_t in zip(K_steps, V_steps):
        cache.update_quantized(K_t, V_t, layer_idx=0)

    assert cache.get_seq_length(0) == 10
    # Partial buffer holds tokens 0..10.
    K_partial_collected = cache.K_partial[0, :, :, :10, :]
    K_steps_concat = K_steps.squeeze(3).transpose(0, 1).reshape(1, 2, 10, 64)
    # K_steps was (10, 1, 2, 1, 64); reshape correctly:
    K_steps_concat = K_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 10, 64)
    torch.testing.assert_close(K_partial_collected, K_steps_concat)


def test_update_quantized_decode_completes_page():
    """A decode step that completes a page must flush to the uint8 region."""
    torch.manual_seed(3)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    # 64 decode steps complete exactly one page.
    K_steps = torch.randn(64, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    V_steps = torch.randn(64, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    for K_t, V_t in zip(K_steps, V_steps):
        cache.update_quantized(K_t, V_t, layer_idx=0)

    assert cache.get_seq_length(0) == 64
    # First page flushed; partial empty.
    K_full = K_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 64, 64)
    V_full = V_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 64, 64)
    K_uint8_ref, K_scale_ref, K_mn_ref = quantize_k(K_full, page_size=64)
    V_uint8_ref, V_scale_ref, V_mn_ref = quantize_v(V_full)
    torch.testing.assert_close(cache.K_uint8[0, :, :, :64, :], K_uint8_ref)
    torch.testing.assert_close(cache.K_scale[0, :, :, :1, :], K_scale_ref)
    torch.testing.assert_close(cache.V_uint8[0, :, :, :64, :], V_uint8_ref)
```

- [ ] **Step 6: Run decode tests to verify they pass**

Run: `pytest tests/test_persistent_cache.py -v`
Expected: PASS (all four tests).

- [ ] **Step 7: Add overflow + bad-layer tests**

Append:

```python
def test_update_overflow_raises():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="exceeds max_seq_len"):
        cache.update_quantized(K, V, layer_idx=0)


def test_update_bad_layer_raises():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(IndexError, match="layer_idx"):
        cache.update_quantized(K, V, layer_idx=5)
```

- [ ] **Step 8: Run all cache tests**

Run: `pytest tests/test_persistent_cache.py -v`
Expected: PASS (six tests).

- [ ] **Step 9: Commit**

```bash
git add src/flashquest/cache/persistent_int8.py tests/test_persistent_cache.py
git commit -m "phase 5: PersistentInt8KVCache.update_quantized (prefill + decode)"
```

---

## Task 4: PersistentInt8KVCache.get_views

**Files:**
- Modify: `src/flashquest/cache/persistent_int8.py`
- Modify: `tests/test_persistent_cache.py`

- [ ] **Step 1: Write failing round-trip test**

Append to `tests/test_persistent_cache.py`:

```python
def test_get_views_roundtrip_matches_quantize_dequantize():
    """View slices, when dequanted, match the same dequant of a fresh quantize_k/v call."""
    torch.manual_seed(7)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=192, page_size=64, device="cuda",
    )
    # 130 tokens: 2 complete pages (128 tokens) + 2 in partial.
    K = torch.randn(1, 2, 130, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 130, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)

    views = cache.get_views(0)
    assert views["K_uint8"].shape == (1, 2, 128, 64)
    assert views["K_scale"].shape == (1, 2, 2, 64)
    assert views["V_uint8"].shape == (1, 2, 128, 64)
    assert views["V_scale"].shape == (1, 2, 128, 1)
    assert views["K_partial"].shape == (1, 2, 2, 64)
    assert views["V_partial"].shape == (1, 2, 2, 64)
    assert views["seq_len"] == 130
    assert views["completed_len"] == 128
    assert views["partial_len"] == 2

    # Round-trip: dequant of complete + concat partial == original K (bf16 tolerance).
    K_dq = dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=64)
    V_dq = dequantize_v(views["V_uint8"], views["V_scale"], views["V_mn"])
    K_full_recovered = torch.cat([K_dq, views["K_partial"]], dim=2)
    V_full_recovered = torch.cat([V_dq, views["V_partial"]], dim=2)
    torch.testing.assert_close(K_full_recovered, K, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(V_full_recovered, V, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_persistent_cache.py::test_get_views_roundtrip_matches_quantize_dequantize -v`
Expected: FAIL — `NotImplementedError` from `get_views`.

- [ ] **Step 3: Implement `get_views`**

Replace the `get_views` body in `src/flashquest/cache/persistent_int8.py`:

```python
    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return slices of the cache for the current sequence length.

        Returns dict with:
            seq_len, completed_len, partial_len: ints
            K_uint8, V_uint8: (B, H_kv, completed_len, D) uint8
            K_scale, K_mn: (B, H_kv, num_complete_pages, D) bf16
            V_scale, V_mn: (B, H_kv, completed_len, 1) bf16
            K_partial, V_partial: (B, H_kv, partial_len, D) bf16
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")
        seen = self._seen_tokens[layer_idx]
        page_size = self.page_size
        partial_len = seen % page_size
        completed_len = seen - partial_len
        n_complete_pages = completed_len // page_size

        return {
            "seq_len": seen,
            "completed_len": completed_len,
            "partial_len": partial_len,
            "K_uint8": self.K_uint8[layer_idx, :, :, :completed_len, :],
            "V_uint8": self.V_uint8[layer_idx, :, :, :completed_len, :],
            "K_scale": self.K_scale[layer_idx, :, :, :n_complete_pages, :],
            "K_mn": self.K_mn[layer_idx, :, :, :n_complete_pages, :],
            "V_scale": self.V_scale[layer_idx, :, :, :completed_len, :],
            "V_mn": self.V_mn[layer_idx, :, :, :completed_len, :],
            "K_partial": self.K_partial[layer_idx, :, :, :partial_len, :],
            "V_partial": self.V_partial[layer_idx, :, :, :partial_len, :],
        }
```

- [ ] **Step 4: Run round-trip test to verify it passes**

Run: `pytest tests/test_persistent_cache.py::test_get_views_roundtrip_matches_quantize_dequantize -v`
Expected: PASS.

- [ ] **Step 5: Add empty-cache and partial-only test**

Append:

```python
def test_get_views_fresh_cache():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    views = cache.get_views(0)
    assert views["seq_len"] == 0
    assert views["completed_len"] == 0
    assert views["partial_len"] == 0
    assert views["K_uint8"].shape == (1, 2, 0, 64)
    assert views["K_partial"].shape == (1, 2, 0, 64)


def test_get_views_partial_only():
    """30 tokens: no complete pages, all in partial."""
    torch.manual_seed(11)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 30, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 30, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    assert views["completed_len"] == 0
    assert views["partial_len"] == 30
    torch.testing.assert_close(views["K_partial"], K)
    torch.testing.assert_close(views["V_partial"], V)
```

- [ ] **Step 6: Run all cache tests**

Run: `pytest tests/test_persistent_cache.py -v`
Expected: PASS (nine tests).

- [ ] **Step 7: Commit**

```bash
git add src/flashquest/cache/persistent_int8.py tests/test_persistent_cache.py
git commit -m "phase 5: PersistentInt8KVCache.get_views (kernel-shaped slices)"
```

---

## Task 5: Fused DuoAttention dispatch

**Files:**
- Create: `src/flashquest/duo/fused_dispatch.py`
- Create: `tests/test_fused_dispatch.py`
- Modify: `src/flashquest/duo/__init__.py`

- [ ] **Step 1: Write failing test for all-retrieval ≡ Phase 4 (EQ11)**

```python
# tests/test_fused_dispatch.py
"""Phase 5 fused DuoAttention dispatch — single sparse-kernel call with
per-head retention."""
import pytest
import torch

from flashquest.duo.dispatch import quest_duo_eager_sdpa
from flashquest.duo.fused_dispatch import quest_duo_fused_sdpa
from flashquest.kernel.kv_quant import quantize_k, quantize_v


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(B=1, H_q=4, H_kv=2, S_q=1, S_kv=512, D=64, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    return Q, K, V


def test_fused_all_retrieval_matches_eager():
    Q, K, V = _setup()
    head_pattern = torch.ones(2, dtype=torch.bool, device="cuda")  # all retrieval

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )

    # rtol=2e-2 because fused goes through INT8 quant; eager is BF16.
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fused_dispatch.py::test_fused_all_retrieval_matches_eager -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `quest_duo_fused_sdpa`**

```python
# src/flashquest/duo/fused_dispatch.py
"""Fused per-head DuoAttention dispatch.

Replaces Phase 4's two-paths-then-torch.where with a single sparse-kernel
call. The trick: streaming heads get retention=0 in select_pages, so their
mask reduces to sinks ∪ window — same as Phase 4's streaming_eager_sdpa
output, but at zero extra kernel cost.
"""
from __future__ import annotations

import torch

from ..eager.criticality import page_scores
from ..eager.page_summary import compute_page_summary
from ..eager.selection import select_pages
from ..kernel import flash_attn_sparse_fwd
from ..kernel.kv_quant import dequantize_k


def quest_duo_fused_sdpa(
    Q: torch.Tensor,
    K_uint8: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_uint8: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    head_pattern: torch.Tensor,
    page_size: int = 64,
    retention: float = 0.25,
    num_sinks: int = 4,
    window_pages: int = 2,
) -> torch.Tensor:
    """Single-call DuoAttention dispatch over INT8 KV.

    Args:
        Q: (B, H_q, 1, D) bf16. Decode-only.
        K_uint8, V_uint8: (B, H_kv, S_kv, D) uint8.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16.
        head_pattern: (H_kv,) bool. True = retrieval, False = streaming.

    Returns:
        Output (B, H_q, 1, D) bf16.
    """
    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError("quest_duo_fused_sdpa is decode-only (S_q=1)")
    _, H_kv, S_kv, _ = K_uint8.shape
    if head_pattern.shape != (H_kv,):
        raise ValueError(
            f"head_pattern must be ({H_kv},); got {tuple(head_pattern.shape)}"
        )

    n_rep = H_q // H_kv

    # Build per-q-head retention vector: retention for retrieval, 0.0 for streaming.
    pattern_per_q_head = head_pattern.to(Q.device).repeat_interleave(n_rep)
    retention_per_q = torch.where(
        pattern_per_q_head,
        torch.full((H_q,), retention, device=Q.device),
        torch.zeros(H_q, device=Q.device),
    )

    # Compute criticality scores against dequantized K (page summary needs BF16).
    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)  # broadcast to query heads
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    scores = page_scores(Q.float(), page_min, page_max)  # (B, H_q, 1, num_pages)

    # Per-head selection — retrieval gets top-k, streaming gets retention=0.
    sel = select_pages(
        scores, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )

    O, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=sel, page_size=page_size, return_lse=False,
    )
    return O
```

- [ ] **Step 4: Add export to `duo/__init__.py`**

Edit `src/flashquest/duo/__init__.py` to add:

```python
from .fused_dispatch import quest_duo_fused_sdpa
```

And update `__all__` to include `"quest_duo_fused_sdpa"`.

- [ ] **Step 5: Run all-retrieval test to verify it passes**

Run: `pytest tests/test_fused_dispatch.py::test_fused_all_retrieval_matches_eager -v`
Expected: PASS.

- [ ] **Step 6: Add all-streaming and mixed-pattern tests**

Append to `tests/test_fused_dispatch.py`:

```python
def test_fused_all_streaming_matches_eager():
    Q, K, V = _setup(seed=1)
    head_pattern = torch.zeros(2, dtype=torch.bool, device="cuda")  # all streaming

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)


def test_fused_mixed_pattern_matches_eager():
    """Mixed pattern must match Phase 4's torch.where path within INT8 noise."""
    Q, K, V = _setup(seed=2)
    head_pattern = torch.tensor([True, False], device="cuda")  # head 0 retrieval, 1 streaming

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)


def test_fused_rejects_prefill():
    Q = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    with pytest.raises(NotImplementedError, match="decode-only"):
        quest_duo_fused_sdpa(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            head_pattern=torch.tensor([True, True], device="cuda"),
            page_size=64, retention=0.25, num_sinks=4, window_pages=2,
        )


def test_fused_pattern_shape_mismatch_raises():
    Q, K, V = _setup()
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    with pytest.raises(ValueError, match=r"head_pattern must be \(2,\)"):
        quest_duo_fused_sdpa(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            head_pattern=torch.tensor([True, True, True], device="cuda"),
            page_size=64, retention=0.25, num_sinks=4, window_pages=2,
        )
```

- [ ] **Step 7: Run all fused-dispatch tests**

Run: `pytest tests/test_fused_dispatch.py -v`
Expected: PASS (five tests).

- [ ] **Step 8: Commit**

```bash
git add src/flashquest/duo/fused_dispatch.py src/flashquest/duo/__init__.py tests/test_fused_dispatch.py
git commit -m "phase 5: fused DuoAttention dispatch (single sparse-kernel call)"
```

---

## Task 6: HF Llama monkeypatch using persistent cache + fused dispatch

**Files:**
- Create: `src/flashquest/eager/llama_persistent_patch.py`
- Create: `tests/test_persistent_e2e.py`

- [ ] **Step 1: Write failing end-to-end test**

```python
# tests/test_persistent_e2e.py
"""End-to-end: a Quest-persistent-patched HF model produces logits within
INT8 quant tolerance of Phase 4's BF16-eager Duo patch."""
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
def test_persistent_cache_logits_within_int8_tolerance():
    """Persistent INT8 cache + fused dispatch logits should match Phase 4
    BF16-eager Duo logits to roughly INT8 quant noise (rtol=5e-2 atol=5e-2)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from flashquest.cache import PersistentInt8KVCache
    from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    cfg = AutoConfig.from_pretrained(name)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)

    # Phase 4 reference (BF16 eager).
    m4 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    patch_llama_for_quest_duo(
        m4, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    inp = tok("The capital of France is Paris. The capital of Spain is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        ref_logits = m4(**inp).logits
    del m4
    torch.cuda.empty_cache()

    # Phase 5 (persistent INT8 cache).
    m5 = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq_len=512, page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        m5, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    with torch.no_grad():
        out_logits = m5(**inp).logits

    torch.testing.assert_close(out_logits, ref_logits, rtol=5e-2, atol=5e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_persistent_e2e.py -v`
Expected: FAIL — `ModuleNotFoundError: flashquest.eager.llama_persistent_patch`.

- [ ] **Step 3: Implement the patch**

```python
# src/flashquest/eager/llama_persistent_patch.py
"""HF Llama monkeypatch with persistent INT8 KV cache + fused DuoAttention.

Prefill (S_q > 1): writes K/V to the persistent cache, dequantizes the
full cache via Phase 3's dequant pair, runs Phase 4's BF16 eager Duo path.
Decode (S_q = 1): writes K/V to cache, reads int8 views, runs the fused
dispatch over completed pages, then merges with a tiny BF16 dense
attention over the partial-page tail via online softmax (LSE).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from ..cache.persistent_int8 import PersistentInt8KVCache
from ..duo.dispatch import quest_duo_eager_sdpa
from ..duo.fused_dispatch import quest_duo_fused_sdpa
from ..kernel.kv_quant import dequantize_k, dequantize_v


def _bf16_dense_attn_with_lse(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny dense BF16 attention for the partial-page tail. Returns (O, lse)
    where lse is in nats. Q: (B, H_q, 1, D); K, V: (B, H_kv, S_partial, D)."""
    B, H_q, _, D = Q.shape
    H_kv = K.shape[1]
    n_rep = H_q // H_kv
    K_full = K.repeat_interleave(n_rep, dim=1)
    V_full = V.repeat_interleave(n_rep, dim=1)
    sm_scale = 1.0 / math.sqrt(D)
    qk = (Q.float() @ K_full.float().transpose(-1, -2)) * sm_scale  # (B, H_q, 1, S_partial)
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    l = p.sum(dim=-1, keepdim=True)
    O = (p @ V_full.float()) / l
    lse = (m + torch.log(l)).squeeze(-1)  # (B, H_q, 1)
    return O.to(torch.bfloat16), lse


def _merge_two_attentions(
    O_a: torch.Tensor, lse_a: torch.Tensor,
    O_b: torch.Tensor, lse_b: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge of two partial attention results sharing Q."""
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m).unsqueeze(-1)
    wb = torch.exp(lse_b - m).unsqueeze(-1)
    return ((wa * O_a.float() + wb * O_b.float()) / (wa + wb)).to(O_a.dtype)


def make_quest_persistent_forward(
    *,
    cache: PersistentInt8KVCache,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
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

        S_q = q.shape[2]

        # Always write to the persistent cache.
        cache.update_quantized(k, v, layer_idx=self.layer_idx)
        views = cache.get_views(self.layer_idx)

        if S_q > 1:
            # Prefill: dequantize the cache and run Phase 4 BF16 eager path.
            K_full = torch.cat(
                [
                    dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=page_size),
                    views["K_partial"],
                ],
                dim=2,
            )
            V_full = torch.cat(
                [
                    dequantize_v(views["V_uint8"], views["V_scale"], views["V_mn"]),
                    views["V_partial"],
                ],
                dim=2,
            )
            attn_output = quest_duo_eager_sdpa(
                q, K_full, V_full,
                head_pattern=head_pattern_layer,
                page_size=page_size, retention=retention,
                num_sinks=num_sinks, window_pages=window_pages,
                is_causal=True,
            )
        else:
            # Decode: split between completed-page region (sparse INT8) and
            # partial-page tail (dense BF16), merge via online softmax.
            partial_len = views["partial_len"]
            completed_len = views["completed_len"]

            if completed_len == 0:
                # Fewer than page_size tokens seen — only partial buffer exists.
                attn_output, _ = _bf16_dense_attn_with_lse(
                    q, views["K_partial"], views["V_partial"],
                )
            else:
                O_sparse_lse = quest_duo_fused_sdpa_with_lse(
                    q, views["K_uint8"], views["K_scale"], views["K_mn"],
                    views["V_uint8"], views["V_scale"], views["V_mn"],
                    head_pattern=head_pattern_layer,
                    page_size=page_size, retention=retention,
                    num_sinks=num_sinks, window_pages=window_pages,
                )
                if partial_len == 0:
                    attn_output = O_sparse_lse[0]
                else:
                    O_partial, lse_partial = _bf16_dense_attn_with_lse(
                        q, views["K_partial"], views["V_partial"],
                    )
                    # Both lses are shape (B, H_q, 1); merge broadcasts to (B, H_q, 1, D).
                    attn_output = _merge_two_attentions(
                        O_sparse_lse[0], O_sparse_lse[1],
                        O_partial, lse_partial,
                    )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def quest_duo_fused_sdpa_with_lse(
    Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
    *, head_pattern, page_size, retention, num_sinks, window_pages,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused dispatch returning (O, lse) — needed for online-softmax merge."""
    from ..eager.criticality import page_scores
    from ..eager.page_summary import compute_page_summary
    from ..eager.selection import select_pages
    from ..kernel import flash_attn_sparse_fwd
    from ..kernel.kv_quant import dequantize_k as _dq

    B, H_q, S_q, D = Q.shape
    _, H_kv, _, _ = K_uint8.shape
    n_rep = H_q // H_kv
    pattern_per_q = head_pattern.to(Q.device).repeat_interleave(n_rep)
    retention_per_q = torch.where(
        pattern_per_q,
        torch.full((H_q,), retention, device=Q.device),
        torch.zeros(H_q, device=Q.device),
    )

    K_dq = _dq(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    scores = page_scores(Q.float(), page_min, page_max)
    sel = select_pages(
        scores, retention=retention_per_q, num_sinks=num_sinks, window_pages=window_pages,
    )
    O, lse = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=sel, page_size=page_size, return_lse=True,
    )
    return O, lse


def patch_llama_for_quest_persistent(
    model: torch.nn.Module,
    *,
    cache: PersistentInt8KVCache,
    head_pattern: torch.Tensor,
    retention: float = 0.25,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward with the persistent-cache version."""
    if head_pattern.ndim != 2:
        raise ValueError(
            f"head_pattern must be 2D (num_layers, num_kv_heads); got {tuple(head_pattern.shape)}"
        )
    num_layers = head_pattern.shape[0]
    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            li = module.layer_idx
            if li >= num_layers:
                raise ValueError(
                    f"head_pattern has {num_layers} layers but model layer_idx={li}"
                )
            fwd = make_quest_persistent_forward(
                cache=cache,
                head_pattern_layer=head_pattern[li].to("cuda"),
                retention=retention,
                num_sinks=num_sinks,
                window_pages=window_pages,
                page_size=page_size,
            )
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_persistent: no LlamaAttention modules")
    if n_patched != num_layers:
        raise ValueError(
            f"head_pattern has {num_layers} layers but model has {n_patched}"
        )
```

- [ ] **Step 4: Run end-to-end test to verify it passes**

Run: `pytest tests/test_persistent_e2e.py::test_persistent_cache_logits_within_int8_tolerance -v -m slow`
Expected: PASS (rtol=5e-2 atol=5e-2 should hold).

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/eager/llama_persistent_patch.py tests/test_persistent_e2e.py
git commit -m "phase 5: HF Llama patch using PersistentInt8KVCache + fused dispatch"
```

---

## Task 7: AWQ weight loading helper

**Files:**
- Create: `src/flashquest/runtime/awq_load.py`
- Create: `tests/test_awq_load.py`
- Modify: `src/flashquest/runtime/__init__.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `autoawq` to bench extras**

Edit `pyproject.toml` line 31-36 (the `[bench]` extras block):

```toml
bench = [
  "transformers>=4.45,<5",
  "accelerate",
  "huggingface_hub[cli]",
  "datasets>=3",
  "autoawq>=0.2,<1",
]
```

- [ ] **Step 2: Install the new dep**

Run: `pip install -e ".[bench]"`
Expected: autoawq installs successfully (CPU+CUDA wheel available for torch 2.5).

- [ ] **Step 3: Write failing test (skipped unless model is available)**

```python
# tests/test_awq_load.py
"""AWQ load smoke test. Skipped when checkpoint isn't cached locally."""
import pytest
import torch

pytestmark = pytest.mark.slow


def _have_awq_3b() -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("casperhansen/llama-3.2-3b-instruct-awq")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_awq_3b(), reason="AWQ checkpoint not cached")
def test_load_awq_3b_runs_forward():
    from flashquest.runtime.awq_load import load_awq_model

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    assert model.config.quantization_config["quant_method"] == "awq"

    inp = tok("Hello, world.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    assert torch.isfinite(out.logits).all()


def test_load_awq_rejects_non_awq():
    from flashquest.runtime.awq_load import load_awq_model
    with pytest.raises(ValueError, match="not AWQ-quantized"):
        load_awq_model("unsloth/Llama-3.2-1B-Instruct")
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_awq_load.py -v`
Expected: FAIL — module missing.

- [ ] **Step 5: Implement `load_awq_model`**

```python
# src/flashquest/runtime/awq_load.py
"""AWQ-INT4 model loading helper. Wraps transformers' auto-AWQ path with a
quantization-config sanity check so we fail loudly on accidental BF16 loads."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_awq_model(
    name: str,
    *,
    attn_implementation: str = "sdpa",
    device_map: str = "cuda",
):
    """Load an AWQ-INT4 HF model. Returns (model, tokenizer)."""
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    qcfg = getattr(model.config, "quantization_config", None)
    if qcfg is None or qcfg.get("quant_method") != "awq":
        raise ValueError(
            f"Model {name} is not AWQ-quantized (quantization_config={qcfg})"
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name)
    return model, tok
```

- [ ] **Step 6: Add export to `runtime/__init__.py`**

```python
# src/flashquest/runtime/__init__.py
from .awq_load import load_awq_model

__all__ = ["load_awq_model"]
```

- [ ] **Step 7: Run AWQ tests**

Run: `pytest tests/test_awq_load.py -v -m slow`
Expected: PASS (or skipped if checkpoint not cached). The reject-non-AWQ test must run regardless.

- [ ] **Step 8: Download Llama-3.2-3B-AWQ for downstream tasks**

Run: `hf download casperhansen/llama-3.2-3b-instruct-awq --local-dir ~/models/llama-3.2-3b-awq`
Expected: ~2 GB download.

- [ ] **Step 9: Re-run AWQ smoke test now that the checkpoint is cached**

Run: `pytest tests/test_awq_load.py -v -m slow`
Expected: PASS (both tests).

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml src/flashquest/runtime/awq_load.py src/flashquest/runtime/__init__.py tests/test_awq_load.py
git commit -m "phase 5: AWQ-INT4 model loader (autoawq via transformers)"
```

---

## Task 8: 32k passkey eval on Llama-3.2-3B-AWQ

**Files:**
- Create: `scripts/phase5_run_passkey_32k.py`
- Create: `benchmarks/phase5_passkey.json`

- [ ] **Step 1: Write the eval script**

```python
# scripts/phase5_run_passkey_32k.py
"""Phase 5 passkey: Llama-3.2-3B-AWQ with persistent INT8 KV cache + fused
DuoAttention dispatch (synthetic 70/30 split). Runs at {8k, 16k, 32k}
contexts × {0.1, 0.5, 0.9} depths × 3 trials.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.passkey import make_example, score
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_LENS = [8192, 16384, 32768]
DEPTHS = [0.1, 0.5, 0.9]
N_TRIALS = 3
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2
RETRIEVAL_FRACTION = 0.7


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def synthetic_pattern(num_layers: int, num_kv: int, fraction: float, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    return torch.rand((num_layers, num_kv), generator=rng) < fraction


@torch.no_grad()
def generate_passkey_answer(model, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    pattern = synthetic_pattern(
        cfg.num_hidden_layers, cfg.num_key_value_heads, RETRIEVAL_FRACTION,
    )

    results = {
        "model": name,
        "context_lens": CONTEXT_LENS,
        "depths": DEPTHS,
        "n_trials": N_TRIALS,
        "retention": RETENTION,
        "retrieval_fraction": RETRIEVAL_FRACTION,
        "by_context": {},
    }

    for ctx_len in CONTEXT_LENS:
        print(f"\n=== context = {ctx_len} ===")
        # Fresh cache per context length to keep memory predictable.
        cache = PersistentInt8KVCache(
            batch_size=1, num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            max_seq_len=ctx_len + 64, page_size=64, device="cuda",
        )
        # Re-patch the model with this context's cache.
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=RETENTION, num_sinks=NUM_SINKS,
            window_pages=WINDOW_PAGES, page_size=64,
        )

        rng = random.Random(ctx_len)
        examples = []
        for d in DEPTHS:
            for _ in range(N_TRIALS):
                examples.append(make_example(
                    rng=rng, tokenizer=tok,
                    target_total_tokens=ctx_len, depth_pct=d,
                ))

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        correct_by_depth = {d: 0 for d in DEPTHS}
        for ex in examples:
            # Reset cache between examples (each example is a fresh sequence).
            cache._seen_tokens = [0] * cache.num_layers
            gen = generate_passkey_answer(model, tok, ex.text)
            if score(gen, ex.passkey):
                correct_by_depth[ex.depth_pct] += 1
        dt = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  accuracy: { {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS} }")
        print(f"  elapsed: {dt:.1f}s, peak VRAM: {peak_mb:.0f} MiB")

        results["by_context"][str(ctx_len)] = {
            "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
            "elapsed_s": dt,
            "peak_vram_mib": peak_mb,
        }

        del cache
        _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase5_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the eval**

Run: `python scripts/phase5_run_passkey_32k.py`
Expected: prints accuracy per context+depth. ≥80 % at depth=0.5 across all contexts; peak VRAM ≤ 3 GB.

- [ ] **Step 3: Inspect `benchmarks/phase5_passkey.json` and confirm the win condition**

Run: `cat benchmarks/phase5_passkey.json | python -m json.tool`
Expected: depth=0.5 accuracy ≥ 0.8 at all three contexts.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase5_run_passkey_32k.py benchmarks/phase5_passkey.json
git commit -m "phase 5: passkey eval on Llama-3.2-3B-AWQ at 8k/16k/32k"
```

---

## Task 9: 32k decode benchmark

**Files:**
- Create: `scripts/phase5_bench_decode_32k.py`
- Create: `benchmarks/phase5_decode.json`

- [ ] **Step 1: Write the bench script**

```python
# scripts/phase5_bench_decode_32k.py
"""Phase 5 decode benchmark: Llama-3.2-3B-AWQ at 32k context, sustained
decode tok/s with persistent INT8 KV + fused DuoAttention dispatch.

Compares against:
- Phase 4 path (eager Quest, on-the-fly quant) on the same model.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


N_PREFILL_TOKENS = 32768
N_DECODE_TOKENS = 128
N_TRIALS = 3


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    """Generate a token sequence of approximately n_tokens. Uses a repeating
    English filler tokenized to land near the target."""
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@torch.no_grad()
def measure_decode(model, tok, n_prefill: int, n_decode: int) -> dict:
    ids = synthetic_prompt_ids(tok, n_prefill)
    # Prefill (one forward pass).
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    out = model(ids, use_cache=True)
    past = out.past_key_values
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    # Decode.
    next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    for _ in range(n_decode):
        out = model(next_tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
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
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    results = {
        "model": name, "n_prefill": N_PREFILL_TOKENS, "n_decode": N_DECODE_TOKENS,
        "n_trials": N_TRIALS, "trials": {},
    }

    # Phase 5 path.
    print("=== Phase 5 (persistent INT8 + fused dispatch) ===")
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq_len=N_PREFILL_TOKENS + N_DECODE_TOKENS + 128,
        page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    p5 = []
    for trial in range(N_TRIALS):
        cache._seen_tokens = [0] * cache.num_layers
        torch.cuda.reset_peak_memory_stats()
        m = measure_decode(model, tok, N_PREFILL_TOKENS, N_DECODE_TOKENS)
        m["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  trial {trial}: prefill={m['prefill_tok_per_s']:.1f}, "
              f"decode={m['decode_tok_per_s']:.2f} tok/s, "
              f"peak VRAM={m['peak_vram_mib']:.0f} MiB")
        p5.append(m)
    results["trials"]["phase5"] = p5
    del cache
    _free()

    # Phase 4 reference path (eager Quest, on-the-fly quant — slow).
    print("=== Phase 4 reference (eager Duo, no persistent cache) ===")
    patch_llama_for_quest_duo(
        model, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    p4 = []
    for trial in range(N_TRIALS):
        torch.cuda.reset_peak_memory_stats()
        m = measure_decode(model, tok, N_PREFILL_TOKENS, N_DECODE_TOKENS)
        m["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  trial {trial}: prefill={m['prefill_tok_per_s']:.1f}, "
              f"decode={m['decode_tok_per_s']:.2f} tok/s, "
              f"peak VRAM={m['peak_vram_mib']:.0f} MiB")
        p4.append(m)
    results["trials"]["phase4"] = p4

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase5_decode.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench**

Run: `python scripts/phase5_bench_decode_32k.py`
Expected: Phase 5 decode ≥4 tok/s; Phase 4 decode noticeably slower (the win is the comparison).

- [ ] **Step 3: Inspect output**

Run: `cat benchmarks/phase5_decode.json | python -m json.tool`
Expected: phase5 decode_tok_per_s mean ≥ phase4 decode_tok_per_s; both <50 tok/s.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase5_bench_decode_32k.py benchmarks/phase5_decode.json
git commit -m "phase 5: 32k decode bench (Phase 5 vs Phase 4 reference)"
```

---

## Task 10: 8B control eval (largest fitting context)

**Files:**
- Create: `scripts/phase5_run_8b_control.py`
- Create: `benchmarks/phase5_8b_control.json`

- [ ] **Step 1: Document the 8B-doesn't-fit-at-32k reality and run smaller-context control**

```python
# scripts/phase5_run_8b_control.py
"""Phase 5 control: Llama-3.1-8B-AWQ at the largest context that fits on 4 GB.

The SPEC §6 Phase 4 win was "8B at 32k, ≥4 tok/s, ≥80 % RULER" — but
Llama-3.1-8B AWQ-INT4 weights alone are ~4.5 GB, exceeding the 4 GB
envelope before any KV cache. This script demonstrates the 8B decode
path works (passkey at 4k context) and records OOM at 32k as the
expected hardware result.
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.duo import load_duo_pattern
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.passkey import make_example, score
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_TARGETS = [4096, 8192, 16384, 32768]  # try ascending; record first OOM
DEPTHS = [0.5]
N_TRIALS = 2
RETENTION = 0.25


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    try:
        model, tok = load_awq_model(name)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        Path("benchmarks/phase5_8b_control.json").write_text(json.dumps({
            "model": name, "result": "OOM_AT_LOAD", "error": str(e),
        }, indent=2))
        print(f"8B AWQ OOMs at load on 4 GB VRAM (expected): {e}")
        return

    cfg = model.config
    duo_path = Path("vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/"
                    "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv")
    pattern = load_duo_pattern(str(duo_path))

    results = {"model": name, "by_context": {}}

    for ctx_len in CONTEXT_TARGETS:
        print(f"\n--- 8B at context={ctx_len} ---")
        try:
            cache = PersistentInt8KVCache(
                batch_size=1, num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
                max_seq_len=ctx_len + 64, page_size=64, device="cuda",
            )
            patch_llama_for_quest_persistent(
                model, cache=cache, head_pattern=pattern,
                retention=RETENTION, num_sinks=4, window_pages=2, page_size=64,
            )

            rng = random.Random(ctx_len)
            torch.cuda.reset_peak_memory_stats()
            correct = 0
            total = 0
            for d in DEPTHS:
                for _ in range(N_TRIALS):
                    ex = make_example(
                        rng=rng, tokenizer=tok,
                        target_total_tokens=ctx_len, depth_pct=d,
                    )
                    cache._seen_tokens = [0] * cache.num_layers
                    ids = tok(ex.text, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        out = model.generate(
                            **ids, max_new_tokens=10, do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
                    correct += int(score(gen, ex.passkey))
                    total += 1
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            results["by_context"][str(ctx_len)] = {
                "accuracy": correct / total, "peak_vram_mib": peak,
            }
            print(f"  accuracy: {correct}/{total}  peak VRAM: {peak:.0f} MiB")
            del cache
            _free()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            results["by_context"][str(ctx_len)] = {"result": "OOM", "error": str(e)[:200]}
            print(f"  OOM (expected at large ctx): {str(e)[:80]}")
            _free()
            break

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase5_8b_control.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Download 8B AWQ checkpoint (large — ~4.5 GB)**

Run: `hf download hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 --local-dir ~/models/llama-3.1-8b-awq`
Expected: ~4.5 GB download. OK to skip if disk-tight; the script handles missing-checkpoint gracefully.

- [ ] **Step 3: Run the 8B control**

Run: `python scripts/phase5_run_8b_control.py`
Expected: either passes at 4k and OOMs at 8k, or OOMs at load. Either way, output is recorded.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase5_run_8b_control.py benchmarks/phase5_8b_control.json
git commit -m "phase 5: 8B control — documents 4 GB hardware wall"
```

---

## Task 11: Phase 5 documentation + tag

**Files:**
- Create: `docs/PHASES/phase-5-notes.md`
- Modify: `README.md`
- Modify: `DOC.md`

- [ ] **Step 1: Write `docs/PHASES/phase-5-notes.md`**

```markdown
# Phase 5 Notes

**Started:** 2026-04-30
**Completed:** 2026-04-30 (tag `phase-5`)
**Status:** **complete (3B AWQ end-to-end with persistent INT8 KV + fused DuoAttention; 8B at 32k blocked by 4 GB VRAM, documented)**
**Spec:** [docs/SPEC.md §6 Phase 5](../SPEC.md)

## Goal

Land production-grade decode for flashquest: persistent INT8 KV cache (HF
`Cache` subclass), AWQ-INT4 weight loading, fused per-head DuoAttention
dispatch (single sparse-kernel call per layer), and 32 k passkey + decode
benchmark on Llama-3.2-3B-AWQ.

## Surface

- `flashquest.cache.PersistentInt8KVCache(batch_size, num_layers, num_kv_heads, head_dim, max_seq_len, page_size, device)` — pre-allocated uint8 cache + BF16 partial-page staging.
- `flashquest.cache.PersistentInt8KVCache.update_quantized(K, V, layer_idx)` — quantize-and-flush on page completion.
- `flashquest.cache.PersistentInt8KVCache.get_views(layer_idx)` — slices for the sparse kernel + partial buffer.
- `flashquest.duo.quest_duo_fused_sdpa(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, head_pattern, ...)` — single-call DuoAttention via per-head retention.
- `flashquest.eager.llama_persistent_patch.patch_llama_for_quest_persistent(model, *, cache, head_pattern, retention, num_sinks, window_pages, page_size)` — HF Llama monkeypatch with online-softmax merge of sparse + partial-page tail.
- `flashquest.runtime.load_awq_model(name)` — AWQ-INT4 loader with quant-config sanity check.
- `flashquest.eager.selection.select_pages(scores, retention, ...)` — `retention` is now `float | torch.Tensor` of shape `(H,)`.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Cache round-trip matches Phase 3 quant→dequant pair (rtol=1e-2) | confirmed | ✅ |
| Persistent-cache HF logits ≡ Phase 4 BF16-eager (rtol=5e-2) | confirmed on Llama-3.2-1B | ✅ |
| Fused dispatch ≡ Phase 4 dispatch (rtol=2e-2) all-retrieval / all-streaming / mixed | confirmed | ✅ |
| AWQ load smoke test (Llama-3.2-3B-AWQ forward pass) | ran cleanly | ✅ |
| 32 k passkey on Llama-3.2-3B-AWQ ≥80 % at depth=0.5 | (see benchmarks/phase5_passkey.json) | ✅/❌ |
| Decode at 32 k ≥4 tok/s | (see benchmarks/phase5_decode.json) | ✅/❌ |
| 8B reality + Marlin/EAGLE/INT4-KV deferral documented | here | ✅ |

(Final pass/fail filled in after running the eval scripts.)

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| EQ1 | Cache update with full page | ✅ |
| EQ2 | Cache update straddling page boundary | ✅ |
| EQ3 | Decode-step append into mid-page | ✅ |
| EQ4 | Decode-step that completes a page | ✅ |
| EQ5 | View on cache with seen < page_size | ✅ |
| EQ6 | Pre-allocated max_seq_len exceeded | ✅ raises RuntimeError |
| EQ7 | layer_idx out of range | ✅ raises IndexError |
| EQ8-EQ10 | Per-head retention scalar/0/1 | ✅ |
| EQ11-EQ12 | Fused dispatch all-retrieval / all-streaming | ✅ |
| EQ13-EQ14 | AWQ load wrong dtype / no quant_config | ✅ |
| EQ15 | seen > max at last decode step | ✅ raises before kernel call |
| EQ16-EQ17 | Online-softmax merge with empty partial / empty completed | ✅ |

## Decisions

- **Per-page channel-wise K + per-token V (Phase 3 KIVI layout) preserved.** Decode appends go into a small BF16 partial-page staging buffer; only complete pages flush to uint8. Quality preserved because the Phase 3 sparse kernel reads exactly the layout it expects.
- **Online-softmax merge** combines the sparse-kernel's LSE with a tiny BF16 dense attention over the partial-page tail (≤63 tokens). Decode-step cost is dominated by the sparse path; the partial-tail cost is constant.
- **Fused DuoAttention via per-head retention.** `select_pages` now accepts a `(H,)` tensor; streaming heads get retention=0 → only sinks ∪ window selected → same behaviour as Phase 4's separate streaming path, but at zero extra kernel cost.
- **AWQ via transformers + autoawq.** No explicit Marlin packing conversion — autoawq's CUDA kernel for W4A16 is already in place. Marlin-tuned packing is deferred until profiling shows the AWQ kernel is the decode bottleneck.
- **3B AWQ as the test model, not 8B.** Llama-3.1-8B AWQ-INT4 weights are ~4.5 GB — they don't fit on a 4 GB GPU before any KV cache. The SPEC §6 Phase 4/5 8B win condition is hardware-blocked on this tier; we ran an 8B control at smaller contexts to demonstrate the path works and recorded the OOM wall (`benchmarks/phase5_8b_control.json`).

## Phase 6 prerequisites (deferred work)

These were originally Phase 5 in the SPEC but defer to Phase 6 (polish & release) once the core kernel pipeline lands:

1. **Marlin W4A16 projections** — only if `nsys` shows the AWQ kernel as a decode bottleneck. Convert AWQ → Marlin packing once at load time.
2. **ExLlamaV2 backend integration** — optional second runtime adapter alongside HF; their `exllamav2_ext` C++ extension model has cleaner extension points but a different ecosystem.
3. **EAGLE-2 speculative decoding wrapper** — orthogonal optimization; ~2× decode multiplier on top of the sparse path.
4. **INT4 KV** — KIVI's full target. INT8 → INT4 needs a kernel-side dequant change (4-bit unpack into BF16 registers).
5. **Llama-3.1-8B at 32k** requires either IQ3-XXS (GGUF, llama.cpp interop) or 2-bit weights, OR PowerInfer-style hot/cold layer offload to system RAM. v2 stretch.
6. **Full RULER eval** instead of passkey. Hours-long; gates a future v1.0 release.

## Phase 5 → Phase 6 handoff

Phase 5 ships:
- `phase-5` git tag.
- Persistent INT8 KV cache + HF integration + AWQ load + fused DuoDispatch.
- Llama-3.2-3B-AWQ at 32 k passkey + decode benchmark.
- 8B control documenting the 4 GB hardware wall.

Phase 6 begins: README polish, demo script, optional Marlin / ExLlamaV2 / EAGLE-2 / INT4-KV / 7B-Mistral with CPU offload. Any of these is independently valuable; the order is profile-driven.
```

- [ ] **Step 2: Update `DOC.md` Phase 5 row + getting-started snippet**

Edit `DOC.md`:

In the Phases list, replace the line `- Phase 5 — ExLlamaV2 backend + optional EAGLE-2.` with:

```markdown
- **Phase 5 — Persistent INT8 KV cache + AWQ + fused DuoAttention** ✅ **complete (tag `phase-5`)**. `flashquest.cache.PersistentInt8KVCache` (HF `Cache` subclass) + `flashquest.runtime.load_awq_model` + `flashquest.duo.quest_duo_fused_sdpa` + `flashquest.eager.llama_persistent_patch.patch_llama_for_quest_persistent`. Validated on Llama-3.2-3B-AWQ at 32 k context with passkey + decode tok/s benchmarks. 8B at 32k blocked by 4 GB VRAM (8B AWQ ~4.5 GB weights alone) — documented in `docs/PHASES/phase-5-notes.md`.
- Phase 6 — Polish & release (optional Marlin / ExLlamaV2 / EAGLE-2 / INT4-KV / Mistral-7B + CPU offload).
```

In "To use the Phase 1 eager Quest attention..." block, append a new block:

````markdown
Phase 5 persistent-cache decode on a Llama-3.2-3B-AWQ checkpoint:

```python
import torch
from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model

model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
cfg = model.config

cache = PersistentInt8KVCache(
    batch_size=1, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
    max_seq_len=32_768 + 128, page_size=64, device="cuda",
)
pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

patch_llama_for_quest_persistent(
    model, cache=cache, head_pattern=pattern,
    retention=0.25, num_sinks=4, window_pages=2, page_size=64,
)
# model.generate(...) now uses persistent INT8 KV + fused DuoAttention.
```
````

In the "Reproduce Phase 1 evals" section, append:

```bash
python scripts/phase5_run_passkey_32k.py     # 32k passkey on Llama-3.2-3B-AWQ
python scripts/phase5_bench_decode_32k.py    # 32k decode tok/s
python scripts/phase5_run_8b_control.py      # 8B AWQ control (largest fitting ctx)
```

- [ ] **Step 3: Update `README.md` with Phase 5 results**

Add a new section after "## Phase 4 — DuoAttention head split + HF integration":

```markdown
## Phase 5 — Persistent INT8 KV cache + AWQ-INT4 + fused DuoAttention

Production-grade decode path: persistent INT8 KV cache (HF `Cache` subclass), AWQ-INT4 weight loading, single-call fused DuoAttention dispatch.

Validation on Llama-3.2-3B-AWQ (`casperhansen/llama-3.2-3b-instruct-awq`, ~2 GB weights), synthetic 70/30 retrieval/streaming split, retention=0.25, sinks=4, window=128.

**Passkey at 32 k context** (3 trials × 3 depths):

| Context | depth=0.1 | depth=0.5 | depth=0.9 | Peak VRAM |
|---|---|---|---|---|
| 8 192 | (see `benchmarks/phase5_passkey.json`) | … | … | … |
| 16 384 | … | … | … | … |
| 32 768 | … | … | … | … |

**Decode at 32 k context** (sustained, 128 new tokens × 3 trials):

| Path | Decode tok/s |
|---|---|
| Phase 4 (eager Duo, on-the-fly quant) | (see `benchmarks/phase5_decode.json`) |
| **Phase 5 (persistent INT8 + fused dispatch)** | … |

17 catalogued edge cases (EQ1–EQ17). See [`docs/PHASES/phase-5-notes.md`](docs/PHASES/phase-5-notes.md). Re-run via `python scripts/phase5_run_passkey_32k.py` and `python scripts/phase5_bench_decode_32k.py`.

**8B reality check.** Llama-3.1-8B AWQ-INT4 weights alone are ~4.5 GB — they don't fit on a 4 GB GPU before any KV cache. The SPEC §6 Phase 4/5 8B win condition is hardware-blocked on this tier; an 8B control at smaller contexts is recorded in `benchmarks/phase5_8b_control.json`. Phase 6 (or v2) is the natural place for IQ3-XXS or PowerInfer-style hot/cold layer offload to clear this wall.
```

- [ ] **Step 4: Fill in the actual benchmark numbers from the JSON files**

After running Tasks 8/9/10, replace the `…` placeholders in README.md with the actual values from:

```bash
cat benchmarks/phase5_passkey.json | python -m json.tool
cat benchmarks/phase5_decode.json | python -m json.tool
cat benchmarks/phase5_8b_control.json | python -m json.tool
```

- [ ] **Step 5: Run the full test suite as a final check**

Run: `pytest tests/ -v`
Expected: All previous tests still green; new Phase 5 tests pass. Slow tests (`-m slow`) optional — they require model checkpoints.

- [ ] **Step 6: Commit Phase 5 documentation**

```bash
git add docs/PHASES/phase-5-notes.md DOC.md README.md
git commit -m "phase 5: notes + DOC + README complete"
```

- [ ] **Step 7: Tag the phase**

```bash
git tag phase-5
git log --oneline -1
```

Expected: tag `phase-5` placed on the most recent commit.

- [ ] **Step 8: Final smoke test**

Run: `pytest tests/ -q`
Expected: 100+ tests pass; no regressions.
