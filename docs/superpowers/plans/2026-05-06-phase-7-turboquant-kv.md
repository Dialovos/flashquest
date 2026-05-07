# Phase 7 — TurboQuant K3-V2 KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new `--kv-bits 3` cache mode that stores K at 3 bits/value and V at 2 bits/value via TurboQuant (per-token Walsh-Hadamard rotation along `head_dim` + fixed Rayleigh-Lloyd-Max codebook + per-token scalar). Shrinks the KV footprint by ~34% vs current INT4 cache (980 → 644 MiB at 32k cache, 28-layer Llama-3.2-3B) while holding RULER NIAH 4k.

**Architecture:** Mirror Phase 6 task 6's playbook. Build the WHT helper + quant primitives + cache class + reference Python forward + fused Triton kernel in TDD steps. Keep `--kv-bits 4` (KIVI-INT4) as the v1 default; TurboQuant lands behind `--kv-bits 3`. Dual statistics: `K_scale_turbo, V_scale_turbo` per-token (used by kernel); `K_scale_raw, K_mn_raw` per-page channel-wise from un-rotated K (used by `page_scores_int4_fast` only — Quest top-k math unchanged).

**Tech Stack:** Triton 3.1, PyTorch 2.5.1, existing `flashquest.kernel.sparse_int4_fwd` as the structural reference for the fused kernel.

**Spec:** `docs/superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md`.

**Reference files (read first, do not modify):**
- `src/flashquest/kernel/sparse_int4_fwd.py` — the INT4 fused kernel + Python wrapper. Mirror this structure.
- `src/flashquest/kernel/kv_quant.py` — INT8/INT4 quant primitives + `_pack_int4`/`_unpack_int4`. Add TurboQuant primitives alongside.
- `src/flashquest/cache/persistent_int4.py` — INT4 cache class. Mirror this for `PersistentTurboKVCache`.
- `src/flashquest/eager/llama_persistent_patch.py` — dispatcher that branches on `cache.kv_bits`. Extend to handle `kv_bits=3`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/flashquest/kernel/wht.py` | Pure-PyTorch normalized Walsh-Hadamard along last dim. Forward = inverse. | Create |
| `src/flashquest/kernel/kv_quant.py` | Add bit-split + INT2 packing primitives, K/V codebook constants, `quantize_{k,v}_turbo` + dequant inverses | Modify |
| `src/flashquest/cache/persistent_turbo.py` | `PersistentTurboKVCache(Cache)` (kv_bits=3); seven-tensor storage layout | Create |
| `src/flashquest/kernel/sparse_turbo_fwd.py` | `@triton.jit _sparse_attn_fwd_kernel_turbo` + Python wrapper + Python reference path for equivalence test | Create |
| `src/flashquest/eager/llama_persistent_patch.py` | Extend `make_quest_persistent_forward` dispatch for `kv_bits=3` | Modify |
| `src/flashquest/runtime/chat.py` | `--kv-bits` accepts `{4, 8, 3}` | Modify |
| `scripts/bench_flashquest.py` | `--kv-bits` accepts `{4, 8, 3}` | Modify |
| `scripts/phase6_run_headtohead.py` | Pass `--kv-bits 3` to flashquest cells when running TurboQuant matrix | Modify |
| `scripts/phase7_run_ruler_4k_turbo.py` | RULER 4k runner using `PersistentTurboKVCache` | Create |
| `tests/test_wht.py` | WHT correctness + orthogonality | Create |
| `tests/test_kv_quant_turbo.py` | TurboQuant primitives, codebook lookup, bit-split | Create |
| `tests/test_persistent_turbo.py` | Cache class + dispatcher smoke | Create |
| `tests/test_sparse_turbo.py` | Sparse forward correctness, fused-vs-reference, no-BF16-intermediate | Create |
| `benchmarks/phase7_decode_turbo_32k.json` | 32k single-cell decode bench artifact | Create at run end |
| `benchmarks/phase7_headtohead_turbo.{json,md}` + `phase7_cells_turbo/` | Head-to-head re-run results | Create at run end |
| `docs/PHASES/phase-7-notes.md` | Phase 7 notes (status, results, follow-ups) | Create |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task; update §11 verdict with new throughput numbers | Modify at end |

---

## Task 1: Walsh-Hadamard transform helper

**Files:**
- Create: `src/flashquest/kernel/wht.py`
- Test: `tests/test_wht.py`

The fast Walsh-Hadamard transform (FWHT) on the last dim. We use the **normalized** convention so that `wht(wht(x)) = x` exactly (no `/N` scale at decode). The butterfly does `log2(D)` stages; at D=128 that's 7 stages of `(a, b) → (a+b, a-b)`, and a final `/sqrt(D)` divides through.

- [ ] **Step 1: Write the test file**

Create `tests/test_wht.py`:

```python
"""WHT correctness + orthogonality (used by Phase 7 TurboQuant)."""
import pytest
import torch

from flashquest.kernel.wht import wht_along_head_dim


def test_wht_inverse():
    """Normalized WHT is its own inverse: wht(wht(x)) ≈ x."""
    torch.manual_seed(7)
    for D in (64, 128):
        x = torch.randn(2, 4, 8, D, dtype=torch.bfloat16, device="cuda")
        y = wht_along_head_dim(x)
        x_back = wht_along_head_dim(y)
        err = (x - x_back).float().abs().max()
        assert err < 5e-3, f"D={D}: wht inverse err {err}"


def test_wht_orthogonal():
    """Inner products preserved: <wht(x), wht(y)> ≈ <x, y>."""
    torch.manual_seed(11)
    D = 128
    x = torch.randn(1, 1, 1, D, dtype=torch.float32, device="cuda")
    y = torch.randn(1, 1, 1, D, dtype=torch.float32, device="cuda")
    dot_raw = (x * y).sum(dim=-1)
    dot_rot = (wht_along_head_dim(x) * wht_along_head_dim(y)).sum(dim=-1)
    err = (dot_raw - dot_rot).abs().max()
    assert err < 1e-4, f"orthogonality err {err}"


def test_wht_requires_power_of_two():
    """Non-power-of-2 head_dim raises."""
    x = torch.randn(1, 1, 1, 96, device="cuda")  # 96 = 32*3, not pow2
    with pytest.raises(ValueError, match="power of 2"):
        wht_along_head_dim(x)
```

- [ ] **Step 2: Run the test — should fail with ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_wht.py -v
```
Expected: `ImportError: cannot import name 'wht_along_head_dim' from 'flashquest.kernel.wht'`.

- [ ] **Step 3: Implement the WHT helper**

Create `src/flashquest/kernel/wht.py`:

```python
"""Normalized fast Walsh-Hadamard transform along the last dimension.

Convention: H @ H.T / N = I, so wht(wht(x)) = x exactly. No /N scale needed
at decode. Used by Phase 7 TurboQuant: rotates K, V along head_dim before
quantization to Gaussianize the per-block distribution.

The butterfly does log2(D) stages of (a, b) → (a + b, a - b), then divides
the final tensor by sqrt(D) once. At D=128 this is 7 stages.
"""
from __future__ import annotations

import math

import torch


def wht_along_head_dim(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform along the last dimension.

    Args:
        x: any tensor with last-dim a power of 2.

    Returns:
        Same shape, dtype preserved. Self-inverse.
    """
    D = x.shape[-1]
    if D & (D - 1) != 0 or D == 0:
        raise ValueError(f"head_dim must be a positive power of 2, got {D}")

    out = x.contiguous()
    h = 1
    while h < D:
        # Reshape last dim into (D / (2h), 2, h) — pairs of size-h blocks
        prefix_shape = out.shape[:-1]
        out = out.reshape(*prefix_shape, D // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.stack([a + b, a - b], dim=-2)
        out = out.reshape(*prefix_shape, D)
        h *= 2

    return out / math.sqrt(D)
```

- [ ] **Step 4: Run the tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_wht.py -v
```
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/kernel/wht.py tests/test_wht.py
git commit -m "phase 7 task 1: WHT helper (normalized, self-inverse)"
```

---

## Task 2: Bit-split + INT2 packing primitives + codebook constants

**Files:**
- Modify: `src/flashquest/kernel/kv_quant.py`
- Test: `tests/test_kv_quant_turbo.py`

Bit-split packs 3-bit indices as separate 1-bit MSB plane (8/byte) + 2-bit LSB plane (4/byte). INT2 packs 4 values per byte. Codebook constants are 8 K-codepoints + 4 V-codepoints, derived from Rayleigh-Lloyd-Max for unit-variance Gaussian (the symmetric 8-level / 4-level Lloyd-Max tables). Exact values may need tuning during quality validation; the values below are widely-tabulated optimal-Lloyd-Max for unit Gaussian.

- [ ] **Step 1: Write the test file**

Create `tests/test_kv_quant_turbo.py`:

```python
"""TurboQuant primitives: codebook lookup, bit-split, INT2 packing."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    K_TURBO_CODEBOOK, V_TURBO_CODEBOOK,
    _pack_bit_split, _unpack_bit_split,
    _pack_int2, _unpack_int2,
    _quantize_to_codebook,
)


def test_codebook_shapes_and_symmetry():
    """K codebook has 8 entries; V has 4. Both symmetric around 0."""
    assert K_TURBO_CODEBOOK.shape == (8,)
    assert V_TURBO_CODEBOOK.shape == (4,)
    for cb in (K_TURBO_CODEBOOK, V_TURBO_CODEBOOK):
        # Symmetric: codepoints come in ±pairs
        sorted_cb = torch.sort(cb).values
        assert torch.allclose(sorted_cb, -sorted_cb.flip(0), atol=1e-5), (
            f"codebook not symmetric: {cb}"
        )


def test_quantize_to_codebook_picks_nearest():
    """_quantize_to_codebook returns the index of the nearest codepoint."""
    cb = torch.tensor([-1.0, -0.5, 0.5, 1.0], device="cuda")
    x = torch.tensor([-0.9, -0.4, 0.0, 0.6, 1.1], device="cuda")
    idx = _quantize_to_codebook(x, cb)
    expected = torch.tensor([0, 1, 1, 2, 3], device="cuda", dtype=torch.uint8)
    assert torch.equal(idx, expected), f"got {idx}, expected {expected}"


def test_pack_bit_split_roundtrip():
    """_unpack_bit_split(_pack_bit_split(idx)) == idx for idx in 0..7."""
    torch.manual_seed(0)
    idx = torch.randint(0, 8, (1, 4, 16, 64), dtype=torch.uint8, device="cuda")
    msb, lsb = _pack_bit_split(idx)
    assert msb.shape == (1, 4, 16, 64 // 8)
    assert lsb.shape == (1, 4, 16, 64 // 4)
    idx_back = _unpack_bit_split(msb, lsb, head_dim=64)
    assert torch.equal(idx_back, idx)


def test_pack_int2_roundtrip():
    """_unpack_int2(_pack_int2(idx)) == idx for idx in 0..3."""
    torch.manual_seed(1)
    idx = torch.randint(0, 4, (1, 4, 16, 64), dtype=torch.uint8, device="cuda")
    packed = _pack_int2(idx)
    assert packed.shape == (1, 4, 16, 64 // 4)
    idx_back = _unpack_int2(packed, head_dim=64)
    assert torch.equal(idx_back, idx)


def test_pack_int2_requires_multiple_of_4():
    """Last axis must be divisible by 4."""
    x = torch.zeros(1, 1, 1, 6, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="multiple of 4"):
        _pack_int2(x)


def test_pack_bit_split_requires_multiple_of_8():
    """Last axis must be divisible by 8 (MSB plane)."""
    x = torch.zeros(1, 1, 1, 12, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="multiple of 8"):
        _pack_bit_split(x)
```

- [ ] **Step 2: Run the tests — should fail with ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_kv_quant_turbo.py -v
```
Expected: `ImportError: cannot import name 'K_TURBO_CODEBOOK' from 'flashquest.kernel.kv_quant'`.

- [ ] **Step 3: Add the primitives + codebook to `kv_quant.py`**

Append to `src/flashquest/kernel/kv_quant.py` (after the existing INT4 section, before EOF):

```python
# === Phase 7: TurboQuant primitives (bit-split, INT2 packing, codebooks) ===

# Lloyd-Max optimal codepoints for unit-variance Gaussian. The TurboQuant
# paper derives bounds from Rayleigh quantile statistics; these are the
# widely-tabulated symmetric-Lloyd-Max levels and serve as the data-oblivious
# codebook for K (3-bit, 8 levels) and V (2-bit, 4 levels). Values may be
# refined during quality validation if needed.
K_TURBO_CODEBOOK = torch.tensor(
    [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    dtype=torch.float32, device="cuda",
)  # 8 codepoints, indices 0..7
V_TURBO_CODEBOOK = torch.tensor(
    [-1.5104, -0.4528, 0.4528, 1.5104],
    dtype=torch.float32, device="cuda",
)  # 4 codepoints, indices 0..3

# c_max for per-token scaling: s = max(|x_rot|) / c_max so that the largest
# rotated value lands on the largest codepoint.
_K_TURBO_C_MAX = 2.1519
_V_TURBO_C_MAX = 1.5104


def _quantize_to_codebook(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Round each x to nearest codepoint; return uint8 indices.

    Args:
        x: any shape, fp32 or bf16. Last-dim values are quantized independently.
        codebook: (K,) fp32 codepoints. K must be ≤ 256.

    Returns:
        uint8 tensor same shape as x, values in 0..K-1.
    """
    # Broadcast: x[..., None] - codebook[None,...] -> (..., K)
    diffs = (x.float().unsqueeze(-1) - codebook.view(*([1] * x.dim()), -1)).abs()
    return diffs.argmin(dim=-1).to(torch.uint8)


def _pack_bit_split(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split 3-bit uint8 indices into 1-bit MSB plane (8/byte) + 2-bit LSB plane (4/byte).

    Args:
        idx: uint8 with values in 0..7. Last axis must be a multiple of 8.

    Returns:
        (msb, lsb) where msb shape == idx.shape[:-1] + (D/8,),
                       lsb shape == idx.shape[:-1] + (D/4,), both uint8.
    """
    if idx.shape[-1] % 8 != 0:
        raise ValueError(f"_pack_bit_split: last axis must be multiple of 8, got {idx.shape[-1]}")
    if idx.dtype != torch.uint8:
        raise ValueError(f"_pack_bit_split: idx must be uint8, got {idx.dtype}")
    msb_bits = (idx >> 2) & 0x1   # high bit
    lsb_bits = idx & 0x3          # low 2 bits

    *prefix, D = idx.shape
    # MSB: pack 8 single-bit values per byte. Reshape to (..., D/8, 8), shift, OR.
    msb_reshaped = msb_bits.reshape(*prefix, D // 8, 8)
    shifts = torch.arange(8, device=idx.device, dtype=torch.uint8)
    msb = (msb_reshaped << shifts).sum(dim=-1, dtype=torch.int32).to(torch.uint8)

    # LSB: pack 4 two-bit values per byte. Reshape to (..., D/4, 4), shift, OR.
    lsb_reshaped = lsb_bits.reshape(*prefix, D // 4, 4)
    shifts2 = (torch.arange(4, device=idx.device, dtype=torch.uint8) * 2)
    lsb = (lsb_reshaped << shifts2).sum(dim=-1, dtype=torch.int32).to(torch.uint8)

    return msb, lsb


def _unpack_bit_split(msb: torch.Tensor, lsb: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Inverse of _pack_bit_split. Returns uint8 indices in 0..7."""
    *prefix, D_msb = msb.shape
    if D_msb * 8 != head_dim:
        raise ValueError(f"_unpack_bit_split: msb last axis {D_msb} * 8 != head_dim {head_dim}")
    if lsb.shape[-1] * 4 != head_dim:
        raise ValueError(f"_unpack_bit_split: lsb last axis {lsb.shape[-1]} * 4 != head_dim {head_dim}")

    bit_offsets = torch.arange(8, device=msb.device, dtype=torch.uint8)
    msb_bits = (msb.unsqueeze(-1) >> bit_offsets) & 0x1                # (..., D/8, 8)
    msb_full = msb_bits.reshape(*prefix, head_dim)                      # (..., D)

    lsb_offsets = (torch.arange(4, device=lsb.device, dtype=torch.uint8) * 2)
    lsb_bits = (lsb.unsqueeze(-1) >> lsb_offsets) & 0x3                # (..., D/4, 4)
    lsb_full = lsb_bits.reshape(*prefix, head_dim)                      # (..., D)

    return ((msb_full << 2) | lsb_full).to(torch.uint8)


def _pack_int2(idx: torch.Tensor) -> torch.Tensor:
    """Pack 4 × 2-bit values per byte along last axis.

    Args:
        idx: uint8 with values in 0..3. Last axis must be a multiple of 4.
    """
    if idx.shape[-1] % 4 != 0:
        raise ValueError(f"_pack_int2: last axis must be multiple of 4, got {idx.shape[-1]}")
    if idx.dtype != torch.uint8:
        raise ValueError(f"_pack_int2: idx must be uint8, got {idx.dtype}")
    *prefix, D = idx.shape
    reshaped = idx.reshape(*prefix, D // 4, 4)
    shifts = (torch.arange(4, device=idx.device, dtype=torch.uint8) * 2)
    return (reshaped << shifts).sum(dim=-1, dtype=torch.int32).to(torch.uint8)


def _unpack_int2(packed: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Inverse of _pack_int2. Returns uint8 indices in 0..3."""
    *prefix, D_packed = packed.shape
    if D_packed * 4 != head_dim:
        raise ValueError(f"_unpack_int2: packed last axis {D_packed} * 4 != head_dim {head_dim}")
    offsets = (torch.arange(4, device=packed.device, dtype=torch.uint8) * 2)
    bits = (packed.unsqueeze(-1) >> offsets) & 0x3
    return bits.reshape(*prefix, head_dim).to(torch.uint8)
```

- [ ] **Step 4: Run the tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_kv_quant_turbo.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/kernel/kv_quant.py tests/test_kv_quant_turbo.py
git commit -m "phase 7 task 2: bit-split + INT2 primitives + Lloyd-Max codebooks"
```

---

## Task 3: TurboQuant K + V quant/dequant (compose WHT + scale + pack)

**Files:**
- Modify: `src/flashquest/kernel/kv_quant.py`
- Test: extend `tests/test_kv_quant_turbo.py`

Compose the bits we built. K turbo: WHT → per-token scale → quantize-to-K-codebook → bit-split. Plus the un-rotated `K_scale_raw, K_mn_raw` per-page channel-wise (for criticality, computed by the existing `_scale_mn_per_page_channel_int4` function on raw K). V turbo: WHT → per-token scale → quantize-to-V-codebook → INT2 pack. Dequant inverts.

- [ ] **Step 1: Append tests to `tests/test_kv_quant_turbo.py`**

Append to `tests/test_kv_quant_turbo.py`:

```python
def test_quantize_k_turbo_shapes():
    """quantize_k_turbo returns five tensors with the documented shapes."""
    from flashquest.kernel.kv_quant import quantize_k_turbo
    torch.manual_seed(2)
    B, H, S, D, page_size = 1, 4, 256, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_turbo, K_scale_raw, K_mn_raw = quantize_k_turbo(K, page_size=page_size)
    assert K_msb.shape == (B, H, S, D // 8) and K_msb.dtype == torch.uint8
    assert K_lsb.shape == (B, H, S, D // 4) and K_lsb.dtype == torch.uint8
    assert K_scale_turbo.shape == (B, H, S, 1) and K_scale_turbo.dtype == torch.bfloat16
    num_pages = S // page_size
    assert K_scale_raw.shape == (B, H, num_pages, D)
    assert K_mn_raw.shape == (B, H, num_pages, D)


def test_quantize_v_turbo_shapes():
    """quantize_v_turbo returns (V_packed, V_scale_turbo)."""
    from flashquest.kernel.kv_quant import quantize_v_turbo
    torch.manual_seed(3)
    B, H, S, D = 1, 4, 256, 128
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V_packed, V_scale_turbo = quantize_v_turbo(V)
    assert V_packed.shape == (B, H, S, D // 4) and V_packed.dtype == torch.uint8
    assert V_scale_turbo.shape == (B, H, S, 1) and V_scale_turbo.dtype == torch.bfloat16


def test_dequantize_k_turbo_roundtrip():
    """K → quant → dequant ≈ K within 3-bit Lloyd-Max noise band on Gaussian inputs."""
    from flashquest.kernel.kv_quant import quantize_k_turbo, dequantize_k_turbo
    torch.manual_seed(4)
    K = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_turbo, _, _ = quantize_k_turbo(K, page_size=64)
    K_back = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=128)
    err = (K - K_back).float().abs().max()
    # Loose: 3-bit Lloyd-Max on unit Gaussian peaks around 0.4 abs error.
    assert err < 1.5, f"K dequant max abs err {err} exceeds 1.5"
    # Tighter: mean abs error should be in the Lloyd-Max band.
    mean_err = (K - K_back).float().abs().mean()
    assert mean_err < 0.4, f"K dequant mean abs err {mean_err} exceeds 0.4"


def test_dequantize_v_turbo_roundtrip():
    """V → quant → dequant ≈ V within 2-bit Lloyd-Max noise band."""
    from flashquest.kernel.kv_quant import quantize_v_turbo, dequantize_v_turbo
    torch.manual_seed(5)
    V = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16, device="cuda")
    V_packed, V_scale_turbo = quantize_v_turbo(V)
    V_back = dequantize_v_turbo(V_packed, V_scale_turbo, head_dim=128)
    mean_err = (V - V_back).float().abs().mean()
    assert mean_err < 0.7, f"V dequant mean abs err {mean_err} exceeds 0.7"
```

- [ ] **Step 2: Run the tests — should fail with ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_kv_quant_turbo.py::test_quantize_k_turbo_shapes -v
```
Expected: `ImportError: cannot import name 'quantize_k_turbo' from 'flashquest.kernel.kv_quant'`.

- [ ] **Step 3: Add the quant/dequant functions to `kv_quant.py`**

Append to `src/flashquest/kernel/kv_quant.py`:

```python
def quantize_k_turbo(K: torch.Tensor, page_size: int):
    """TurboQuant K: WHT → per-token scale → 3-bit Lloyd-Max → bit-split pack.

    Also computes un-rotated per-page channel-wise (K_scale_raw, K_mn_raw) for
    `page_scores_int4_fast` in the dispatcher. Mirrors KIVI INT4 stat shapes
    (15 → equivalent envelope) so `criticality.page_scores_int4_fast` works
    unmodified on the raw stats.

    Returns:
        K_msb       (B, H, S, D/8)        uint8
        K_lsb       (B, H, S, D/4)        uint8
        K_scale_turbo (B, H, S, 1)        bf16
        K_scale_raw   (B, H, num_pages, D) bf16  -- KIVI-INT4-equivalent
        K_mn_raw      (B, H, num_pages, D) bf16
    """
    from flashquest.kernel.wht import wht_along_head_dim

    B, H, S, D = K.shape
    if D % 8 != 0:
        raise ValueError(f"quantize_k_turbo requires head_dim multiple of 8; got {D}")

    # 1) WHT rotate along head_dim (preserves L2; Gaussianizes per-token).
    K_rot = wht_along_head_dim(K)

    # 2) Per-token scale: max(|K_rot|) / c_max. Shape (B, H, S, 1).
    K_abs_max = K_rot.float().abs().amax(dim=-1, keepdim=True)
    K_scale_turbo = (K_abs_max / _K_TURBO_C_MAX).clamp_min(_EPS)

    # 3) Quantize K_rot / scale → 3-bit indices via codebook.
    K_normalized = K_rot.float() / K_scale_turbo
    K_idx = _quantize_to_codebook(K_normalized, K_TURBO_CODEBOOK)  # (B, H, S, D) uint8 0..7

    # 4) Bit-split pack.
    K_msb, K_lsb = _pack_bit_split(K_idx)

    # 5) Raw KIVI-INT4-style stats from the un-rotated K, for criticality.
    K_scale_raw, K_mn_raw = _scale_mn_per_page_channel_int4(K, page_size)

    return (
        K_msb, K_lsb,
        K_scale_turbo.to(torch.bfloat16),
        K_scale_raw.to(torch.bfloat16),
        K_mn_raw.to(torch.bfloat16),
    )


def dequantize_k_turbo(
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Inverse: bit-split unpack → codebook lookup → multiply scale → inverse WHT → BF16."""
    from flashquest.kernel.wht import wht_along_head_dim

    K_idx = _unpack_bit_split(K_msb, K_lsb, head_dim=head_dim)         # (B, H, S, D) uint8 0..7
    K_rot = K_TURBO_CODEBOOK[K_idx.long()] * K_scale_turbo.float()    # (B, H, S, D) fp32
    K = wht_along_head_dim(K_rot)                                      # inverse WHT (self-inverse)
    return K.to(torch.bfloat16)


def quantize_v_turbo(V: torch.Tensor):
    """TurboQuant V: WHT → per-token scale → 2-bit Lloyd-Max → INT2 pack.

    Returns:
        V_packed       (B, H, S, D/4) uint8
        V_scale_turbo  (B, H, S, 1)   bf16
    """
    from flashquest.kernel.wht import wht_along_head_dim

    B, H, S, D = V.shape
    if D % 4 != 0:
        raise ValueError(f"quantize_v_turbo requires head_dim multiple of 4; got {D}")

    V_rot = wht_along_head_dim(V)
    V_abs_max = V_rot.float().abs().amax(dim=-1, keepdim=True)
    V_scale_turbo = (V_abs_max / _V_TURBO_C_MAX).clamp_min(_EPS)
    V_normalized = V_rot.float() / V_scale_turbo
    V_idx = _quantize_to_codebook(V_normalized, V_TURBO_CODEBOOK)
    V_packed = _pack_int2(V_idx)
    return V_packed, V_scale_turbo.to(torch.bfloat16)


def dequantize_v_turbo(
    V_packed: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Inverse of quantize_v_turbo."""
    from flashquest.kernel.wht import wht_along_head_dim

    V_idx = _unpack_int2(V_packed, head_dim=head_dim)
    V_rot = V_TURBO_CODEBOOK[V_idx.long()] * V_scale_turbo.float()
    V = wht_along_head_dim(V_rot)
    return V.to(torch.bfloat16)
```

- [ ] **Step 4: Run the new tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_kv_quant_turbo.py -v
```
Expected: 10 PASSED (6 from task 2 + 4 new). If `test_dequantize_k_turbo_roundtrip` fails with `mean_err > 0.4`, the codebook scaling is off — check that `K_TURBO_C_MAX` matches the largest codepoint magnitude.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/kernel/kv_quant.py tests/test_kv_quant_turbo.py
git commit -m "phase 7 task 3: quantize_{k,v}_turbo (compose WHT + scale + pack)"
```

---

## Task 4: PersistentTurboKVCache class

**Files:**
- Create: `src/flashquest/cache/persistent_turbo.py`
- Test: `tests/test_persistent_turbo.py`

Mirror `PersistentInt4KVCache` but with seven storage tensors instead of six (split `K_packed` into `K_msb` + `K_lsb`, plus the raw stats live alongside the rotated `K_scale_turbo`).

- [ ] **Step 1: Write the test file**

Create `tests/test_persistent_turbo.py`:

```python
"""PersistentTurboKVCache: shapes, roundtrip, dispatcher smoke."""
import pytest
import torch

from flashquest.cache.persistent_turbo import PersistentTurboKVCache


def _make_cache(S=256, page_size=64):
    return PersistentTurboKVCache(
        batch_size=1, num_layers=2, num_kv_heads=4, head_dim=64,
        max_seq_len=S, page_size=page_size, device="cuda",
    )


def test_kv_bits_attribute():
    cache = _make_cache()
    assert cache.kv_bits == 3


def test_storage_shapes():
    cache = _make_cache(S=256, page_size=64)
    L, B, H, S, D, P = 2, 1, 4, 256, 64, 256 // 64
    assert cache.K_msb.shape == (L, B, H, S, D // 8)
    assert cache.K_lsb.shape == (L, B, H, S, D // 4)
    assert cache.K_scale_turbo.shape == (L, B, H, S, 1)
    assert cache.K_scale_raw.shape == (L, B, H, P, D)
    assert cache.K_mn_raw.shape == (L, B, H, P, D)
    assert cache.V_packed.shape == (L, B, H, S, D // 4)
    assert cache.V_scale_turbo.shape == (L, B, H, S, 1)


def test_update_quantized_writes_one_full_page():
    """Write 64 tokens (one page); verify _seen_tokens advances and views shapes."""
    torch.manual_seed(0)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    assert cache.get_seq_length(0) == 64
    views = cache.get_views(0)
    assert views["seq_len"] == 64
    assert views["completed_len"] == 64
    assert views["partial_len"] == 0
    assert views["K_msb"].shape == (1, 4, 64, 64 // 8)
    assert views["K_lsb"].shape == (1, 4, 64, 64 // 4)
    assert views["V_packed"].shape == (1, 4, 64, 64 // 4)
    assert views["K_scale_turbo"].shape == (1, 4, 64, 1)
    assert views["V_scale_turbo"].shape == (1, 4, 64, 1)
    assert views["K_scale_raw"].shape == (1, 4, 1, 64)
    assert views["K_mn_raw"].shape == (1, 4, 1, 64)


def test_partial_page_staging():
    """Write 32 tokens (half page); should land in K_partial / V_partial."""
    torch.manual_seed(1)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    assert views["completed_len"] == 0
    assert views["partial_len"] == 32
    assert views["K_partial"].shape == (1, 4, 32, 64)


def test_roundtrip_through_cache():
    """Write K, V → read views → dequant → ≈ K, V within 3-bit/2-bit noise."""
    from flashquest.kernel.kv_quant import dequantize_k_turbo, dequantize_v_turbo
    torch.manual_seed(7)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    K_back = dequantize_k_turbo(views["K_msb"], views["K_lsb"], views["K_scale_turbo"], head_dim=64)
    V_back = dequantize_v_turbo(views["V_packed"], views["V_scale_turbo"], head_dim=64)
    assert (K - K_back).float().abs().mean() < 0.4
    assert (V - V_back).float().abs().mean() < 0.7


def test_requires_head_dim_multiple_of_8():
    """head_dim must be ≥8 and divisible by 8 (MSB plane requires 8/byte)."""
    with pytest.raises(ValueError, match="multiple of 8"):
        PersistentTurboKVCache(
            batch_size=1, num_layers=1, num_kv_heads=1, head_dim=4,
            max_seq_len=64, page_size=64, device="cuda",
        )
```

- [ ] **Step 2: Run the tests — should fail with ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_persistent_turbo.py -v
```
Expected: `ModuleNotFoundError: No module named 'flashquest.cache.persistent_turbo'`.

- [ ] **Step 3: Implement the cache class**

Create `src/flashquest/cache/persistent_turbo.py`:

```python
"""Phase 7 — Persistent TurboQuant KV cache (K=3-bit bit-split, V=2-bit).

Mirrors `PersistentInt4KVCache` but with seven storage tensors:
  K_msb, K_lsb       : 3-bit K split into 1-bit MSB plane + 2-bit LSB plane
  K_scale_turbo      : per-token scalar for kernel dequant
  K_scale_raw, K_mn_raw : per-page channel-wise from un-rotated K, for criticality
  V_packed           : 2-bit V (4 values per byte)
  V_scale_turbo      : per-token scalar for kernel dequant

`kv_bits = 3` is read by the dispatcher in `eager/llama_persistent_patch.py`.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache


class PersistentTurboKVCache(Cache):
    kv_bits = 3

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
        if head_dim % 8 != 0:
            raise ValueError(
                f"PersistentTurboKVCache requires head_dim multiple of 8 "
                f"(MSB plane packs 8/byte); got {head_dim}"
            )
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(
                f"PersistentTurboKVCache requires head_dim power of 2 (WHT); got {head_dim}"
            )
        self.layers: list = []
        self.layer_class_to_replicate = None
        self.offloading = False
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        max_pages = (max_seq_len + page_size - 1) // page_size
        self.max_pages = max_pages
        dev = torch.device(device)

        D_msb = head_dim // 8
        D_lsb = head_dim // 4
        D_v = head_dim // 4
        shape_msb = (num_layers, batch_size, num_kv_heads, max_seq_len, D_msb)
        shape_lsb = (num_layers, batch_size, num_kv_heads, max_seq_len, D_lsb)
        shape_vpk = (num_layers, batch_size, num_kv_heads, max_seq_len, D_v)
        shape_kscale_t = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_vscale_t = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_kpage = (num_layers, batch_size, num_kv_heads, max_pages, head_dim)
        shape_partial = (num_layers, batch_size, num_kv_heads, page_size, head_dim)

        self.K_msb = torch.zeros(shape_msb, dtype=torch.uint8, device=dev)
        self.K_lsb = torch.zeros(shape_lsb, dtype=torch.uint8, device=dev)
        self.K_scale_turbo = torch.zeros(shape_kscale_t, dtype=torch.bfloat16, device=dev)
        self.K_scale_raw = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.K_mn_raw = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.V_packed = torch.zeros(shape_vpk, dtype=torch.uint8, device=dev)
        self.V_scale_turbo = torch.zeros(shape_vscale_t, dtype=torch.bfloat16, device=dev)
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
        """Append K_new, V_new (B, H_kv, S_new, D) bf16 to layer_idx's cache."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        from flashquest.kernel.kv_quant import quantize_k_turbo, quantize_v_turbo

        seen = self._seen_tokens[layer_idx]
        S_new = K_new.shape[2]
        if seen + S_new > self.max_seq_len:
            raise RuntimeError(
                f"PersistentTurboKVCache: seen+new={seen + S_new} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        page_size = self.page_size

        partial_len = seen % page_size
        if partial_len > 0:
            head_K = self.K_partial[layer_idx, :, :, :partial_len, :]
            K_full = torch.cat([head_K, K_new], dim=2)
            V_full = torch.cat(
                [self.V_partial[layer_idx, :, :, :partial_len, :], V_new], dim=2,
            )
        else:
            K_full = K_new
            V_full = V_new

        total_stream_len = K_full.shape[2]
        n_complete_pages = total_stream_len // page_size
        complete_len = n_complete_pages * page_size

        if n_complete_pages > 0:
            K_complete = K_full[:, :, :complete_len, :]
            V_complete = V_full[:, :, :complete_len, :]
            K_msb, K_lsb, K_scale_t, K_scale_r, K_mn_r = quantize_k_turbo(
                K_complete, page_size=page_size,
            )
            V_packed, V_scale_t = quantize_v_turbo(V_complete)

            tok_start = seen - partial_len
            tok_end = tok_start + complete_len
            page_idx_start = tok_start // page_size
            page_idx_end = page_idx_start + n_complete_pages

            self.K_msb[layer_idx, :, :, tok_start:tok_end, :] = K_msb
            self.K_lsb[layer_idx, :, :, tok_start:tok_end, :] = K_lsb
            self.K_scale_turbo[layer_idx, :, :, tok_start:tok_end, :] = K_scale_t
            self.K_scale_raw[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_scale_r
            self.K_mn_raw[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_mn_r
            self.V_packed[layer_idx, :, :, tok_start:tok_end, :] = V_packed
            self.V_scale_turbo[layer_idx, :, :, tok_start:tok_end, :] = V_scale_t

        new_partial_len = total_stream_len - complete_len
        if new_partial_len < page_size:
            self.K_partial[layer_idx, :, :, :new_partial_len, :] = K_full[:, :, complete_len:, :]
            self.V_partial[layer_idx, :, :, :new_partial_len, :] = V_full[:, :, complete_len:, :]

        self._seen_tokens[layer_idx] = seen + S_new

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        seen = self._seen_tokens[layer_idx]
        page_size = self.page_size
        partial_len = seen % page_size
        completed_len = seen - partial_len
        n_complete_pages = completed_len // page_size

        return {
            "seq_len": seen,
            "completed_len": completed_len,
            "partial_len": partial_len,
            "K_msb": self.K_msb[layer_idx, :, :, :completed_len, :],
            "K_lsb": self.K_lsb[layer_idx, :, :, :completed_len, :],
            "K_scale_turbo": self.K_scale_turbo[layer_idx, :, :, :completed_len, :],
            "K_scale_raw": self.K_scale_raw[layer_idx, :, :, :n_complete_pages, :],
            "K_mn_raw": self.K_mn_raw[layer_idx, :, :, :n_complete_pages, :],
            "V_packed": self.V_packed[layer_idx, :, :, :completed_len, :],
            "V_scale_turbo": self.V_scale_turbo[layer_idx, :, :, :completed_len, :],
            "K_partial": self.K_partial[layer_idx, :, :, :partial_len, :],
            "V_partial": self.V_partial[layer_idx, :, :, :partial_len, :],
        }

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "PersistentTurboKVCache.update is not used; the patched forward "
            "calls update_quantized directly."
        )
```

- [ ] **Step 4: Run the tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_persistent_turbo.py -v
```
Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/cache/persistent_turbo.py tests/test_persistent_turbo.py
git commit -m "phase 7 task 4: PersistentTurboKVCache (kv_bits=3)"
```

---

## Task 5: Reference Python sparse forward (no kernel)

**Files:**
- Create: `src/flashquest/kernel/sparse_turbo_fwd.py` (reference path only — kernel added in task 7)

The reference path mirrors what the kernel will do but in pure PyTorch: dequant whole pages of K/V, run dense attention over the selected pages. This is what the kernel-fused-vs-reference test will compare against.

- [ ] **Step 1: Create the module skeleton with the reference path**

Create `src/flashquest/kernel/sparse_turbo_fwd.py`:

```python
"""Phase 7 — sparse-attention forward with TurboQuant KV (K=3-bit, V=2-bit).

This module ships two callables:
  - `_flash_attn_sparse_turbo_fwd_reference` — pure-PyTorch reference path
    (used by the equivalence test; never on the hot path).
  - `flash_attn_sparse_turbo_fwd` — fused Triton kernel + Python wrapper
    (added in task 7). The wrapper applies WHT to Q (single vector),
    calls the kernel, and applies inverse-WHT to the output (because V
    was stored rotated). Kernel docstring describes the bit-plane unpack
    + codebook lookup + online-softmax math.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from flashquest.kernel.kv_quant import (
    K_TURBO_CODEBOOK, V_TURBO_CODEBOOK,
    dequantize_k_turbo, dequantize_v_turbo,
)
from flashquest.kernel.wht import wht_along_head_dim


def _flash_attn_sparse_turbo_fwd_reference(
    Q: torch.Tensor,
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference path — Python, used for kernel equivalence testing.

    Args mirror `flash_attn_sparse_turbo_fwd` (added in task 7). Algorithm:
    dequant the entire K, V cache to BF16, mask out un-selected pages,
    run dense attention. Output O is in raw basis (V dequant applies
    inverse WHT).
    """
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")
    if K_msb.dtype != torch.uint8 or K_lsb.dtype != torch.uint8:
        raise ValueError(f"K_msb/K_lsb must be uint8")
    if V_packed.dtype != torch.uint8:
        raise ValueError(f"V_packed must be uint8")

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"decode-only (S_q={S_q})")

    Bk, H_kv, S_kv, _ = K_msb.shape
    n_rep = H_q // H_kv
    num_pages = selection_mask.shape[-1]

    # Dequant: returns raw (un-rotated) BF16 K, V because both quantizers
    # apply inverse WHT in their dequant.
    K_full = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=D)
    V_full = dequantize_v_turbo(V_packed, V_scale_turbo, head_dim=D)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Build a (B, H_q, 1, S_kv) attention mask from per-page selection.
    # Tokens in un-selected pages get -inf score.
    page_idx = torch.arange(S_kv, device=Q.device) // page_size  # (S_kv,)
    sel = selection_mask[..., page_idx]                          # (B, H_q, 1, S_kv) bool
    attn_bias = torch.where(sel, 0.0, float("-inf")).to(Q.dtype)

    # Replicate K, V from H_kv → H_q for direct matmul.
    K_rep = K_full.repeat_interleave(n_rep, dim=1)
    V_rep = V_full.repeat_interleave(n_rep, dim=1)

    qk = (Q.float() @ K_rep.float().transpose(-1, -2)) * sm_scale + attn_bias.float()
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    l = p.sum(dim=-1, keepdim=True)
    O = (p @ V_rep.float()) / l

    lse = None
    if return_lse:
        lse = (m + torch.log(l)).squeeze(-1).to(torch.float32)

    return O.to(torch.bfloat16), lse
```

- [ ] **Step 2: Smoke-test the reference path imports**

Run:
```
source .venv/bin/activate && python -c "from flashquest.kernel.sparse_turbo_fwd import _flash_attn_sparse_turbo_fwd_reference; print('reference OK')"
```
Expected: `reference OK`.

- [ ] **Step 3: Commit**

```bash
git add src/flashquest/kernel/sparse_turbo_fwd.py
git commit -m "phase 7 task 5: reference Python sparse_turbo forward (for kernel eq test)"
```

---

## Task 6: Failing fused kernel + reference test

**Files:**
- Create: `tests/test_sparse_turbo.py`

Write the test that the kernel must pass before the kernel exists.

- [ ] **Step 1: Write the test file**

Create `tests/test_sparse_turbo.py`:

```python
"""Sparse forward correctness for TurboQuant KV (K=3-bit, V=2-bit)."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    quantize_k_turbo, quantize_v_turbo,
    dequantize_k_turbo, dequantize_v_turbo,
)


def test_sparse_turbo_matches_dense():
    """All-pages-selected ≡ dense attention on dequant'd KV (loose tol for K3+V2)."""
    from flashquest.kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd

    torch.manual_seed(13)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 4, 2, 1, 256, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_packed, V_scale_t = quantize_v_turbo(V)

    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    out, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_packed, V_scale_t,
        selection_mask=sel, page_size=page_size, sm_scale=D ** -0.5, return_lse=True,
    )

    # Reference: dequant + dense SDPA on raw (un-rotated) K, V.
    K_deq = dequantize_k_turbo(K_msb, K_lsb, K_scale_t, head_dim=D)
    V_deq = dequantize_v_turbo(V_packed, V_scale_t, head_dim=D)
    n_rep = H_q // H_kv
    K_deq_q = K_deq.repeat_interleave(n_rep, dim=1)
    V_deq_q = V_deq.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_deq_q.float(), V_deq_q.float(), is_causal=False,
    ).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs().max()
    # K=3-bit + V=2-bit Lloyd-Max + WHT round-trip — wider tolerance than INT4's 5e-2.
    assert err < 1.5e-1, f"max abs diff {err}"


def test_fused_matches_reference():
    """Fused Triton kernel ≡ reference Python path on small fixtures."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd,
        _flash_attn_sparse_turbo_fwd_reference,
    )

    torch.manual_seed(17)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 2, 1, 1, 64, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_packed, V_scale_t = quantize_v_turbo(V)
    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")

    kw = dict(
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )
    O_ref, lse_ref = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_packed, V_scale_t, **kw,
    )
    O_fused, lse_fused = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_packed, V_scale_t, **kw,
    )

    err_O = (O_fused.float() - O_ref.float()).abs().max()
    err_lse = (lse_fused.float() - lse_ref.float()).abs().max()
    # Tighter than the dense test — both paths use the SAME quant data;
    # only kernel implementation differs. Allow some FP-order noise.
    assert err_O < 5e-2, f"fused vs reference O max abs err {err_O}"
    assert err_lse < 5e-2, f"fused vs reference lse max abs err {err_lse}"
```

- [ ] **Step 2: Run the tests — should fail with ImportError on `flash_attn_sparse_turbo_fwd`**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_turbo.py -v
```
Expected: `ImportError: cannot import name 'flash_attn_sparse_turbo_fwd' from 'flashquest.kernel.sparse_turbo_fwd'`.

- [ ] **Step 3: Do not commit yet — the test goes green in task 7**

Skip — this test will be committed alongside the kernel in task 7.

---

## Task 7: Fused Triton kernel + Python wrapper

**Files:**
- Modify: `src/flashquest/kernel/sparse_turbo_fwd.py`

Mirror `src/flashquest/kernel/sparse_int4_fwd.py:_sparse_attn_fwd_kernel_int4`. The differences from INT4: (a) two K planes (MSB 1-bit, LSB 2-bit) instead of one packed nibble byte; (b) codebook lookup against an 8-entry K table and a 4-entry V table instead of `* scale + mn`; (c) wrapper applies WHT to Q and inverse-WHT to output (V was stored rotated, so the kernel's `softmax · V` accumulation lives in the rotated basis).

- [ ] **Step 1: Append the kernel + wrapper to `sparse_turbo_fwd.py`**

Append to `src/flashquest/kernel/sparse_turbo_fwd.py`:

```python
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (64, 128)


@triton.jit
def _sparse_attn_fwd_kernel_turbo(
    Q_rot_ptr, K_msb_ptr, K_lsb_ptr, V_packed_ptr,
    O_rot_ptr, L_ptr,
    K_scale_t_ptr, V_scale_t_ptr,
    K_codebook_ptr, V_codebook_ptr,
    sel_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kmb, stride_kmh, stride_kms, stride_kmd,
    stride_klb, stride_klh, stride_kls, stride_kld,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_kstb, stride_ksth, stride_ksts,
    stride_vstb, stride_vsth, stride_vsts,
    stride_selb, stride_selh, stride_selp,
    H_q, H_kv, S_kv, NUM_PAGES,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_MSB: tl.constexpr,    # HEAD_DIM // 8
    HEAD_DIM_LSB: tl.constexpr,    # HEAD_DIM // 4
    PAGE_SIZE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """Decode-only sparse forward, TurboQuant K3-V2. One CTA per (batch, query head).

    Tile-load layout per page (PAGE_SIZE rows):
      K_msb:    (PAGE_SIZE, HEAD_DIM_MSB) uint8  -- 8 1-bit values per byte
      K_lsb:    (PAGE_SIZE, HEAD_DIM_LSB) uint8  -- 4 2-bit values per byte
      V_packed: (PAGE_SIZE, HEAD_DIM_LSB) uint8  -- same shape as K_lsb
    Reconstruct K_idx ∈ [0..7] via (msb << 2) | lsb, then K_codebook gather.
    Reconstruct V_idx ∈ [0..3] via 2-bit unpack, then V_codebook gather.
    """
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_d_msb = tl.arange(0, HEAD_DIM_MSB)
    offs_d_lsb = tl.arange(0, HEAD_DIM_LSB)

    q_ptrs = Q_rot_ptr + b * stride_qb + h_q * stride_qh + offs_d * stride_qd
    q = tl.load(q_ptrs)

    NEG_INF: tl.constexpr = float("-inf")
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for p in range(0, NUM_PAGES):
        sel_p = tl.load(sel_ptr + b * stride_selb + h_q * stride_selh + p * stride_selp)
        if sel_p:
            page_start = p * PAGE_SIZE
            n_idx = page_start + offs_n
            valid_kv = n_idx < S_kv

            # === Load K_msb and K_lsb tiles, reconstruct 3-bit indices, codebook gather. ===
            k_msb_byte = tl.load(
                K_msb_ptr + b * stride_kmb + h_kv * stride_kmh
                + n_idx[:, None] * stride_kms + offs_d_msb[None, :] * stride_kmd,
                mask=valid_kv[:, None], other=0,
            )  # (PAGE_SIZE, HEAD_DIM_MSB) uint8

            k_lsb_byte = tl.load(
                K_lsb_ptr + b * stride_klb + h_kv * stride_klh
                + n_idx[:, None] * stride_kls + offs_d_lsb[None, :] * stride_kld,
                mask=valid_kv[:, None], other=0,
            )  # (PAGE_SIZE, HEAD_DIM_LSB) uint8

            # 1-bit unpack on MSB plane: each byte → 8 single bits.
            # tl.join in 3 levels: (PAGE_SIZE, HEAD_DIM_MSB) -> ... -> (PAGE_SIZE, HEAD_DIM)
            k_b0 = (k_msb_byte >> 0) & 0x1
            k_b1 = (k_msb_byte >> 1) & 0x1
            k_b2 = (k_msb_byte >> 2) & 0x1
            k_b3 = (k_msb_byte >> 3) & 0x1
            k_b4 = (k_msb_byte >> 4) & 0x1
            k_b5 = (k_msb_byte >> 5) & 0x1
            k_b6 = (k_msb_byte >> 6) & 0x1
            k_b7 = (k_msb_byte >> 7) & 0x1
            k_msb_4 = tl.join(tl.join(k_b0, k_b1), tl.join(k_b2, k_b3))   # (..., HEAD_DIM_MSB, 4)
            k_msb_4 = tl.reshape(k_msb_4, (PAGE_SIZE, HEAD_DIM_MSB * 4))  # (PAGE_SIZE, D/2)
            k_msb_8 = tl.join(
                tl.reshape(k_msb_4, (PAGE_SIZE, HEAD_DIM_MSB * 4)),
                tl.reshape(
                    tl.join(tl.join(k_b4, k_b5), tl.join(k_b6, k_b7)),
                    (PAGE_SIZE, HEAD_DIM_MSB * 4),
                ),
            )
            k_msb_full = tl.reshape(k_msb_8, (PAGE_SIZE, HEAD_DIM))  # 0/1

            # 2-bit unpack on LSB plane: each byte → 4 × 2-bit values.
            k_l0 = (k_lsb_byte >> 0) & 0x3
            k_l1 = (k_lsb_byte >> 2) & 0x3
            k_l2 = (k_lsb_byte >> 4) & 0x3
            k_l3 = (k_lsb_byte >> 6) & 0x3
            k_lsb_full = tl.join(tl.join(k_l0, k_l1), tl.join(k_l2, k_l3))  # (PAGE_SIZE, HEAD_DIM_LSB, 4)
            k_lsb_full = tl.reshape(k_lsb_full, (PAGE_SIZE, HEAD_DIM))  # 0..3

            # Combine: idx = (msb << 2) | lsb  ∈  0..7
            k_idx = ((k_msb_full.to(tl.int32) << 2) | k_lsb_full.to(tl.int32))

            # Codebook gather: K_codebook is an 8-entry fp32 tensor.
            k_rot = tl.load(K_codebook_ptr + k_idx)  # (PAGE_SIZE, HEAD_DIM) fp32

            # Per-token scale.
            k_scale_t = tl.load(
                K_scale_t_ptr + b * stride_kstb + h_kv * stride_ksth + n_idx * stride_ksts,
                mask=valid_kv, other=0.0,
            ).to(tl.float32)
            k = k_rot * k_scale_t[:, None]

            # Online softmax (mirrors INT4 kernel exactly).
            qk = tl.sum(q[None, :].to(tl.float32) * k, axis=1)
            qk = tl.where(valid_kv, qk, NEG_INF)

            qk_max = tl.max(qk * qk_scale, axis=0)
            m_ij = tl.maximum(m_i, qk_max)
            m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
            p_softmax = tl.math.exp2(qk * qk_scale - m_ij_safe)
            p_softmax = tl.where(m_ij == NEG_INF, 0.0, p_softmax)

            alpha = tl.math.exp2(m_i - m_ij_safe)
            if m_i == NEG_INF:
                alpha = 0.0

            l_i = l_i * alpha + tl.sum(p_softmax, axis=0)
            acc = acc * alpha

            # === Load V_packed tile, 2-bit unpack, codebook gather, accumulate. ===
            v_byte = tl.load(
                V_packed_ptr + b * stride_vb + h_kv * stride_vh
                + n_idx[:, None] * stride_vs + offs_d_lsb[None, :] * stride_vd,
                mask=valid_kv[:, None], other=0,
            )
            v_l0 = (v_byte >> 0) & 0x3
            v_l1 = (v_byte >> 2) & 0x3
            v_l2 = (v_byte >> 4) & 0x3
            v_l3 = (v_byte >> 6) & 0x3
            v_idx = tl.join(tl.join(v_l0, v_l1), tl.join(v_l2, v_l3))
            v_idx = tl.reshape(v_idx, (PAGE_SIZE, HEAD_DIM)).to(tl.int32)
            v_rot = tl.load(V_codebook_ptr + v_idx)

            v_scale_t = tl.load(
                V_scale_t_ptr + b * stride_vstb + h_kv * stride_vsth + n_idx * stride_vsts,
                mask=valid_kv, other=0.0,
            ).to(tl.float32)
            v = v_rot * v_scale_t[:, None]

            acc += tl.sum(p_softmax[:, None] * v, axis=0)

            m_i = m_ij

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l

    o_ptrs = O_rot_ptr + b * stride_ob + h_q * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, acc.to(O_rot_ptr.dtype.element_ty))

    if WRITE_LSE:
        lse_val = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse_val = tl.where(l_i == 0.0, NEG_INF, lse_val)
        tl.store(L_ptr + b * stride_lb + h_q * stride_lh, lse_val)


def flash_attn_sparse_turbo_fwd(
    Q: torch.Tensor,
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode-only fused TurboQuant sparse forward.

    Wrapper applies WHT to Q (single vector per head, ~1k FP ops total) and
    inverse-WHT to the output (V was stored rotated). Kernel handles tile
    loads, bit-plane unpack, codebook gather, online softmax.

    Args:
        Q: (B, H_q, 1, D) bf16 cuda. RAW basis (wrapper rotates).
        K_msb: (B, H_kv, S_kv, D/8) uint8 cuda.
        K_lsb: (B, H_kv, S_kv, D/4) uint8 cuda.
        K_scale_turbo: (B, H_kv, S_kv, 1) bf16.
        V_packed: (B, H_kv, S_kv, D/4) uint8.
        V_scale_turbo: (B, H_kv, S_kv, 1) bf16.
        selection_mask: (B, H_q, 1, num_pages) bool.

    Returns:
        (O (B, H_q, 1, D) bf16, lse (B, H_q, 1) fp32 or None). Both in raw basis.
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_msb.dtype == torch.uint8 and K_lsb.dtype == torch.uint8
    assert V_packed.dtype == torch.uint8

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"flash_attn_sparse_turbo_fwd: decode-only (S_q={S_q})")
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")

    Bk, H_kv, S_kv, _ = K_msb.shape
    assert B == Bk
    assert H_q % H_kv == 0
    num_pages = selection_mask.shape[-1]
    assert selection_mask.shape == (B, H_q, 1, num_pages)
    assert selection_mask.dtype == torch.bool

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Wrapper-side WHT: rotate Q into the same basis as the stored K, V.
    Q_rot = wht_along_head_dim(Q)
    Q_2d = Q_rot.squeeze(2).contiguous()
    O_rot_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    sl_b, sl_h = (L.stride() if L is not None else (0, 0))

    sel_2d = selection_mask.squeeze(2)

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_turbo[grid](
        Q_2d, K_msb, K_lsb, V_packed,
        O_rot_2d, L_ptr,
        K_scale_turbo, V_scale_turbo,
        K_TURBO_CODEBOOK, V_TURBO_CODEBOOK,
        sel_2d,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_msb.stride(0), K_msb.stride(1), K_msb.stride(2), K_msb.stride(3),
        K_lsb.stride(0), K_lsb.stride(1), K_lsb.stride(2), K_lsb.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O_rot_2d.stride(0), O_rot_2d.stride(1), O_rot_2d.stride(2),
        sl_b, sl_h,
        K_scale_turbo.stride(0), K_scale_turbo.stride(1), K_scale_turbo.stride(2),
        V_scale_turbo.stride(0), V_scale_turbo.stride(1), V_scale_turbo.stride(2),
        sel_2d.stride(0), sel_2d.stride(1), sel_2d.stride(2),
        H_q, H_kv, S_kv, num_pages,
        HEAD_DIM=D,
        HEAD_DIM_MSB=D // 8,
        HEAD_DIM_LSB=D // 4,
        PAGE_SIZE=page_size,
        WRITE_LSE=bool(return_lse),
        num_warps=4,
        num_stages=2,
    )

    # Inverse WHT on the output (V was rotated → acc is in rotated basis).
    O_rot = O_rot_2d.unsqueeze(2)
    O = wht_along_head_dim(O_rot)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out
```

- [ ] **Step 2: Run the equivalence + dense tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_turbo.py -v
```
Expected: 2 PASSED.

Common failure modes:
- `tl.join` dimension mismatch: print intermediate shapes inside the test by enabling Triton interpreter mode (`@triton.jit(interpret=True)`); the MSB 8-way unpack has the trickiest reshape sequence.
- Numerical mismatch >5e-2 in fused-vs-reference: likely a stride mismatch or codebook indexing issue — check that `K_codebook_ptr + k_idx` returns the right shape (Triton infers from the index shape).
- `mismatch >1.5e-1` on `test_sparse_turbo_matches_dense`: codebook constants are off (re-check `K_TURBO_CODEBOOK` values vs the `_K_TURBO_C_MAX` scaling rule).

- [ ] **Step 3: Commit**

```bash
git add src/flashquest/kernel/sparse_turbo_fwd.py tests/test_sparse_turbo.py
git commit -m "phase 7 task 7: fused TurboQuant Triton kernel + Python wrapper"
```

---

## Task 8: No-K/V-shaped-BF16-intermediate test

**Files:**
- Modify: `tests/test_sparse_turbo.py`

Phase 6 task 6 introduced this gate to catch regressions to the reference (BF16-materializing) path. Same pattern here: fused kernel must allocate at most the output BF16 tensor + LSE, nothing K/V-shaped.

- [ ] **Step 1: Append the test**

Append to `tests/test_sparse_turbo.py`:

```python
def test_no_kv_shaped_bf16_intermediate(monkeypatch):
    """Fused kernel must not allocate any BF16 tensor with K/V shape.

    Tracks every torch.empty / torch.zeros call during flash_attn_sparse_turbo_fwd;
    if any allocation has size matching B*H_kv*S_kv*head_dim*2 bytes (BF16
    K/V intermediate), we've regressed to the reference path.
    """
    from flashquest.kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd

    torch.manual_seed(19)
    B, H_q, H_kv, S_kv, D, page_size = 1, 4, 2, 1024, 64, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_packed, V_scale_t = quantize_v_turbo(V)
    sel = torch.ones(B, H_q, 1, S_kv // page_size, dtype=torch.bool, device="cuda")

    kv_bytes_threshold = B * H_kv * S_kv * D * 2  # BF16 K (or V) intermediate size
    allocations: list[tuple[tuple[int, ...], torch.dtype, int]] = []
    orig_empty = torch.empty
    orig_zeros = torch.zeros

    def _track(fn):
        def inner(*args, **kwargs):
            t = fn(*args, **kwargs)
            try:
                allocations.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
            except Exception:
                pass
            return t
        return inner

    monkeypatch.setattr(torch, "empty", _track(orig_empty))
    monkeypatch.setattr(torch, "zeros", _track(orig_zeros))

    flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_packed, V_scale_t,
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    bf16_kv_intermediates = [
        (shape, dt, nbytes)
        for (shape, dt, nbytes) in allocations
        if dt == torch.bfloat16 and nbytes >= kv_bytes_threshold
    ]
    assert not bf16_kv_intermediates, (
        f"fused turbo kernel allocated K/V-shaped BF16 intermediates "
        f"(threshold {kv_bytes_threshold} bytes): {bf16_kv_intermediates}"
    )
```

- [ ] **Step 2: Run the new test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_turbo.py::test_no_kv_shaped_bf16_intermediate -v
```
Expected: PASS.

If it fails: the wrapper is dequant-ing somewhere it shouldn't. Verify that `flash_attn_sparse_turbo_fwd` does NOT call `dequantize_k_turbo` / `dequantize_v_turbo` (those are reserved for the prefill path in the dispatcher).

- [ ] **Step 3: Run the full test_sparse_turbo.py file**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_turbo.py -v
```
Expected: 3 PASSED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_sparse_turbo.py
git commit -m "phase 7 task 8: assert fused turbo kernel allocates no BF16 K/V intermediate"
```

---

## Task 9: Dispatcher integration + CLI plumbing

**Files:**
- Modify: `src/flashquest/eager/llama_persistent_patch.py`
- Modify: `src/flashquest/runtime/chat.py`
- Modify: `scripts/bench_flashquest.py`
- Modify: `scripts/phase6_run_headtohead.py`

Wire `kv_bits=3` through the dispatcher. The dispatcher reads `cache.kv_bits`, picks the right view keys, dequant fns, and sparse-fwd entry. CLI scripts gain `--kv-bits 3` as a valid choice.

- [ ] **Step 1: Read the current dispatcher to find the exact insertion point**

Run:
```
grep -n "elif kv_bits == 8\|elif kv_bits == 4\|kv_bits == 4" src/flashquest/eager/llama_persistent_patch.py
```
Expected: shows lines around `make_quest_persistent_forward` where `kv_bits` is branched on. Note the line numbers.

- [ ] **Step 2: Extend the dispatcher in `make_quest_persistent_forward`**

In `src/flashquest/eager/llama_persistent_patch.py`, find:

```python
    kv_bits = getattr(cache, "kv_bits", 8)
    if kv_bits == 4:
        K_view_key = "K_packed"
        V_view_key = "V_packed"

        def _dequant_k(k_storage, k_scale, k_mn):
            return dequantize_k_int4(k_storage, k_scale, k_mn, page_size=page_size)

        def _dequant_v(v_storage, v_scale, v_mn):
            return dequantize_v_int4(v_storage, v_scale, v_mn)
    elif kv_bits == 8:
```

Insert a new branch for `kv_bits == 3` BEFORE the INT4 branch (or after — order doesn't matter, just keep them mutually exclusive). The TurboQuant branch is shape-different from INT4/INT8: K is *two* views (msb + lsb), so the dispatcher needs a different protocol.

The simplest implementation: instead of single `K_view_key`, store a tuple `K_view_keys`, and the dequant closures take the views dict directly.

Refactor the existing dispatcher to take views-dict-based closures. Replace the entire `make_quest_persistent_forward` body's kv_bits selection block with:

```python
    kv_bits = getattr(cache, "kv_bits", 8)
    if kv_bits == 3:
        from flashquest.kernel.kv_quant import dequantize_k_turbo, dequantize_v_turbo
        from flashquest.kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd

        def _dequant_k_from_views(views):
            return dequantize_k_turbo(
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                head_dim=cache.head_dim,
            )

        def _dequant_v_from_views(views):
            return dequantize_v_turbo(
                views["V_packed"], views["V_scale_turbo"],
                head_dim=cache.head_dim,
            )

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_turbo_fwd(
                q,
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                views["V_packed"], views["V_scale_turbo"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )

        def _criticality_scores(q, views):
            from flashquest.eager.criticality import page_scores_int4_fast
            return page_scores_int4_fast(q, views["K_scale_raw"], views["K_mn_raw"])
    elif kv_bits == 4:
        from flashquest.kernel.kv_quant import dequantize_k_int4, dequantize_v_int4
        from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd

        def _dequant_k_from_views(views):
            return dequantize_k_int4(
                views["K_packed"], views["K_scale"], views["K_mn"], page_size=page_size,
            )

        def _dequant_v_from_views(views):
            return dequantize_v_int4(views["V_packed"], views["V_scale"], views["V_mn"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_int4_fwd(
                q,
                views["K_packed"], views["K_scale"], views["K_mn"],
                views["V_packed"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )

        def _criticality_scores(q, views):
            from flashquest.eager.criticality import page_scores_int4_fast
            return page_scores_int4_fast(q, views["K_scale"], views["K_mn"])
    elif kv_bits == 8:
        from flashquest.kernel.kv_quant import dequantize_k, dequantize_v
        from flashquest.kernel.sparse_fwd import flash_attn_sparse_fwd

        def _dequant_k_from_views(views):
            return dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=page_size)

        def _dequant_v_from_views(views):
            return dequantize_v(views["V_uint8"], views["V_scale"], views["V_mn"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_fwd(
                q,
                views["K_uint8"], views["K_scale"], views["K_mn"],
                views["V_uint8"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )

        def _criticality_scores(q, views):
            from flashquest.eager.criticality import page_scores_int8_fast
            return page_scores_int8_fast(q, views["K_scale"], views["K_mn"])
    else:
        raise ValueError(f"unsupported cache.kv_bits={kv_bits!r}")
```

Then update the body of `forward` to use `_dequant_k_from_views(views)`, `_dequant_v_from_views(views)`, and `_sparse_fwd_call(q, views, sel)`. Also replace the old call to `_quest_duo_fused_with_lse` with a small inline pattern that uses `_criticality_scores(q, views)` for selection and `_sparse_fwd_call` for the kernel — the existing helper can be deleted or refactored.

Specifically, in the `forward` function inside `make_quest_persistent_forward`:

Replace:
```python
            K_full = torch.cat(
                [
                    _dequant_k(views[K_view_key], views["K_scale"], views["K_mn"]),
                    views["K_partial"],
                ],
                dim=2,
            )
            V_full = torch.cat(
                [
                    _dequant_v(views[V_view_key], views["V_scale"], views["V_mn"]),
                    views["V_partial"],
                ],
                dim=2,
            )
```

With:
```python
            K_full = torch.cat([_dequant_k_from_views(views), views["K_partial"]], dim=2)
            V_full = torch.cat([_dequant_v_from_views(views), views["V_partial"]], dim=2)
```

And replace the decode-path `_quest_duo_fused_with_lse(...)` call with:
```python
            scores = _criticality_scores(q, views)
            sel = select_pages_vectorized(
                scores,
                retention=retention_per_q,
                num_sinks=num_sinks,
                window_pages=window_pages,
            )
            O_sparse, lse_sparse = _sparse_fwd_call(q, views, sel)
```

(Move the `retention_per_q` / `head_pattern` setup that was inside `_quest_duo_fused_with_lse` into this branch; the helper can be removed or thinned to share the GQA replication.)

- [ ] **Step 3: Run the regression suite to confirm INT4 + INT8 paths still work**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -m "not slow"
```
Expected: 200+ passed (existing 204 + ~15 new TurboQuant tests). All INT4 / INT8 tests must still pass.

If `test_int4_dispatcher_smoke_llama_1b` fails: re-check the dispatcher refactor — keys must still match `views["K_packed"]` for INT4 etc. Easy mistake: typo'd `views["K_uint8"]` for the INT8 branch.

- [ ] **Step 4: Plumb `--kv-bits 3` into CLI scripts**

In `scripts/bench_flashquest.py`, find:
```python
    p.add_argument("--kv-bits", type=int, choices=[4, 8], default=4, ...)
```
Change to:
```python
    p.add_argument("--kv-bits", type=int, choices=[4, 8, 3], default=4,
                   help="KV cache bit width. 4 = KIVI-INT4 (default, RULER 100/100/100). "
                        "3 = TurboQuant K3-V2 (Phase 7). 8 = KIVI-INT8.")
```

In the same file, find the cache class import block:
```python
        if args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls
```
Replace with:
```python
        if args.kv_bits == 3:
            from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls
        elif args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls
```

In the `quant_label` build, add a TurboQuant case:
```python
    quant_label = {
        3: "AWQ-INT4 + TurboQuant K3-V2 paged KV + Quest top-k retention=0.25",
        4: "AWQ-INT4 + INT4 paged KV + Quest top-k retention=0.25",
        8: "AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25",
    }[args.kv_bits]
```

In `src/flashquest/runtime/chat.py`, locate the `--kv-bits` argparse line and apply the same `choices=[4, 8, 3]` change. Update the cache-import block analogously.

In `scripts/phase6_run_headtohead.py`, find the section that runs `bench_flashquest.py --kv-bits 4 ...` and add a way to override via env var `KV_BITS` (or a new CLI flag). Default stays 4. For Phase 7 head-to-head we'll set `KV_BITS=3` when launching.

The minimal change in `phase6_run_headtohead.py`: read `kv_bits = int(os.environ.get("KV_BITS", "4"))` near the top, pass `--kv-bits {kv_bits}` to flashquest cells, and tag output paths with `_turbo` suffix when `kv_bits == 3`.

- [ ] **Step 5: Smoke-test the CLI**

Run a tiny prefill+decode at small ctx with TurboQuant:
```
source .venv/bin/activate && python -u scripts/bench_flashquest.py --ctx-len 1024 --n-decode 4 --kv-bits 3 --out /tmp/turbo_smoke.json
cat /tmp/turbo_smoke.json
```
Expected: `decode_tok_s` is a real number (not null), `oom: false`, `error: null`. Quant label string contains "TurboQuant K3-V2".

If decoding produces NaN logits or token outputs are gibberish: that's a *runtime* check that confirms quality before the formal RULER eval. Diagnose by re-running the unit tests; almost always means a kernel bug (wrong codebook lookup or stride mismatch) that the small fixtures didn't catch.

- [ ] **Step 6: Commit**

```bash
git add src/flashquest/eager/llama_persistent_patch.py \
        src/flashquest/runtime/chat.py \
        scripts/bench_flashquest.py \
        scripts/phase6_run_headtohead.py
git commit -m "phase 7 task 9: dispatcher + CLI plumbing for --kv-bits 3 (TurboQuant)"
```

---

## Task 10: Phase 1-7 fast-suite regression gate

**Files:** none (run-only)

- [ ] **Step 1: Run the full fast suite**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -m "not slow"
```
Expected: 220+ passed (204 from end of Phase 6 task 6 + ~16 new Phase 7 tests). No regressions in any phase.

- [ ] **Step 2: Run the slow Llama-1B dispatcher smoke for TurboQuant**

This integration test verifies the full forward path on a real model. Append to `tests/test_persistent_turbo.py`:

```python
@pytest.mark.slow
def test_turbo_dispatcher_smoke_llama_1b():
    """Patch Llama-3.2-1B with PersistentTurboKVCache, run a forward, assert finite logits."""
    pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent

    model_id = "meta-llama/Llama-3.2-1B"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda",
        )
        tok = AutoTokenizer.from_pretrained(model_id)
    except Exception as e:
        pytest.skip(f"skipping (model not available): {e}")

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    cache = PersistentTurboKVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=128, page_size=64, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    ids = tok("The quick brown fox", return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    assert torch.isfinite(out.logits).all(), "non-finite logits"
```

Run:
```
source .venv/bin/activate && python -m pytest tests/test_persistent_turbo.py::test_turbo_dispatcher_smoke_llama_1b -v -m slow
```
Expected: PASS in <60s on a card that has the model cached locally.

- [ ] **Step 3: Commit the test (no code change beyond the test)**

```bash
git add tests/test_persistent_turbo.py
git commit -m "phase 7 task 10: dispatcher smoke on Llama-3.2-1B for TurboQuant"
```

---

## Task 11: RULER NIAH 4k @ TurboQuant quality gate

**Files:**
- Create: `scripts/phase7_run_ruler_4k_turbo.py`

Mirror `scripts/phase6_run_ruler_4k_int4.py` but route through `PersistentTurboKVCache`. Wall: ~50 min.

- [ ] **Step 1: Create the runner script**

Create `scripts/phase7_run_ruler_4k_turbo.py` by copying `scripts/phase6_run_ruler_4k_int4.py` and changing:
1. The cache import: `from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls`.
2. The output JSON: `benchmarks/phase7_ruler_4k_turbo.json` and `kv_bits: 3` in the metadata.
3. The "patched" run label: `"patched_turbo_hits"` instead of `"patched_int4_hits"`.

The rest of the script (model loading, RULER NIAH 4k task generation, scoring) is identical.

- [ ] **Step 2: Run a smoke test at n_samples=2**

Run:
```
source .venv/bin/activate && python -u scripts/phase7_run_ruler_4k_turbo.py --n-samples 2 --out /tmp/turbo_ruler_smoke.json 2>&1 | tail -20
```
Expected: completes in <2 min, prints per-task hit counts, writes JSON with `all_pass: true|false`.

If `all_pass: false` even at n=2: investigate before launching the 50-min full run.

- [ ] **Step 3: Run the full 20-sample eval in background**

Run via Bash `run_in_background`:
```
source .venv/bin/activate && nice -n 19 python -u scripts/phase7_run_ruler_4k_turbo.py 2>&1 | tee /tmp/phase7_ruler_4k_turbo.log
```

Monitor via `Monitor`:
```
tail -F /tmp/phase7_ruler_4k_turbo.log 2>/dev/null | grep --line-buffered -E "===|hits=|All-pass|Wrote|PASS|FAIL|all_pass|Error|Traceback|OOM"
```

- [ ] **Step 4: Verify the gate clears**

Run:
```
cat benchmarks/phase7_ruler_4k_turbo.json | python -m json.tool | head -40
```
Expected: `"all_pass": true` and per-task ratios ≥0.85.

If any category < 85%: the data-oblivious codebook is mis-calibrated for Llama-3.2-3B's K, V distribution. Two paths:
1. Re-derive `K_TURBO_CODEBOOK` / `V_TURBO_CODEBOOK` from a calibration dump (Approach C from the spec). Capture WHT-rotated K, V from a Llama-3.2-3B forward on RULER prompts; compute Lloyd-Max levels from the empirical distribution. Update `kv_quant.py` constants.
2. Fall back to `--kv-bits 4` and document Phase 7 as gated. Phase 7.5 = calibrated codebook.

- [ ] **Step 5: Commit (only if gate passes)**

```bash
git add scripts/phase7_run_ruler_4k_turbo.py benchmarks/phase7_ruler_4k_turbo.json
git commit -m "phase 7 task 11: RULER 4k @ TurboQuant — quality gate result"
```

---

## Task 12: Single-cell decode bench at 32k

**Files:**
- Create: `benchmarks/phase7_decode_turbo_32k.json`

- [ ] **Step 1: Run the bench**

Run via Bash `run_in_background` (~5–10 min wall):
```
source .venv/bin/activate && nice -n 19 python -u scripts/bench_flashquest.py \
    --ctx-len 32768 --n-decode 32 --kv-bits 3 \
    --out benchmarks/phase7_decode_turbo_32k.json 2>&1 | tee /tmp/phase7_decode_turbo_32k.log
```

Monitor via `Monitor`:
```
tail -F /tmp/phase7_decode_turbo_32k.log 2>/dev/null | grep --line-buffered -E "decode_tok_s|prefill_tok_s|peak_vram|oom|wall_s|^\\}|Error|Traceback"
```

- [ ] **Step 2: Verify the throughput gate**

Run:
```
cat benchmarks/phase7_decode_turbo_32k.json | python -m json.tool
```

Read `decode_tok_s`. Expected ranges:
- **Pass: ≥6 tok/s** — hits the spec's primary throughput gate.
- **Floor: ≥3.88 tok/s** — at least matches Phase 6 task 6 INT4 fused. Anything slower is a kernel regression.
- **Stretch: ≥8 tok/s** — strong result.

If `decode_tok_s < 3.88`: profile the kernel.
- Most likely: 1-bit MSB unpack (8 successive shifts + 3-level `tl.join` chain) is slower than INT4's nibble unpack.
- Try replacing the 8-shift MSB unpack with a single `tl.gather` over a precomputed lookup table (256-entry table from byte → 8 expanded bits).
- Profile with `nsys` or Triton's built-in profiler to confirm the hot spot.

- [ ] **Step 3: Commit**

```bash
git add benchmarks/phase7_decode_turbo_32k.json
git commit -m "phase 7 task 12: 32k decode bench under TurboQuant kernel"
```

---

## Task 13: Head-to-head re-run + docs + tag

**Files:**
- Create: `benchmarks/phase7_headtohead_turbo.{json,md}` + `benchmarks/phase7_cells_turbo/`
- Create: `docs/PHASES/phase-7-notes.md`
- Modify: `DOC.md`, `README.md`, `docs/SPEC.md`

- [ ] **Step 1: Run the head-to-head with TurboQuant flashquest cells**

The orchestrator in `scripts/phase6_run_headtohead.py` was updated in task 9 to read `KV_BITS` from env. Run:

```
mkdir -p benchmarks/phase7_cells_turbo
source .venv/bin/activate && KV_BITS=3 OUT_DIR=benchmarks/phase7_cells_turbo \
    OUT_JSON=benchmarks/phase7_headtohead_turbo.json \
    OUT_MD=benchmarks/phase7_headtohead_turbo.md \
    nice -n 19 python -u scripts/phase6_run_headtohead.py 2>&1 \
    | tee /tmp/phase7_headtohead_turbo.log
```

(If `phase6_run_headtohead.py` doesn't read `OUT_DIR`/`OUT_JSON`/`OUT_MD` env vars: just run it as-is, then rename the `phase6_*` outputs to `phase7_*_turbo` afterward — same approach Phase 6 task 6 used.)

Monitor with the same regex as task 12.

- [ ] **Step 2: Inspect the markdown**

Run:
```
cat benchmarks/phase7_headtohead_turbo.md
```

Expected shape (numbers depend on measured runs):

| Backend | Quant | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest | AWQ-INT4 + TurboQuant K3-V2 | ≥6 (target) | ≥6 (target) | ✗ |
| llama.cpp | Q4_K_M, FP16 KV | ~39 | ✗ (abort) | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | ✗ | ✗ |

- [ ] **Step 3: Write `docs/PHASES/phase-7-notes.md`**

Create the file with this skeleton (replace `<fill>` with measured numbers from tasks 11, 12, 13):

```markdown
# Phase 7 Notes — TurboQuant K3-V2 KV

**Started:** 2026-05-06
**Completed:** <fill date>
**Status:** **complete (tag `phase-7`); fused TurboQuant kernel ships behind --kv-bits 3; SPEC §11.4 verdict update: <CLEARED at throughput axis if ≥6 tok/s clears the 5× capability ratio in head-to-head; STILL GATED at 8 k throughput vs llama.cpp>.**
**Plan:** [../superpowers/plans/2026-05-06-phase-7-turboquant-kv.md](../superpowers/plans/2026-05-06-phase-7-turboquant-kv.md)
**Spec:** [../superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md](../superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md)

## Summary

TurboQuant ships behind `--kv-bits 3`. Per-token Walsh-Hadamard rotation
along `head_dim` Gaussianizes the per-block distribution; a fixed
Lloyd-Max codebook (8 K codepoints, 4 V codepoints) handles the
quantization. K stored as bit-split (1-bit MSB plane + 2-bit LSB plane);
V stored as packed INT2 (4 values per byte). Storage shrink: 34% vs INT4
(980 → 644 MiB at 32 k). Quality preserved by orthogonality of WHT
combined with the unbiased Lloyd-Max codebook. Dual statistics:
`K_scale_turbo` (per-token, kernel) + `K_scale_raw, K_mn_raw`
(per-page channel-wise from un-rotated K, used by `page_scores_int4_fast`)
keep Quest top-k math unchanged.

## Result — quality (RULER NIAH 4k @ TurboQuant)

`benchmarks/phase7_ruler_4k_turbo.json`:

| task | dense | patched (TurboQuant) | ratio | gate ≥85 % |
|---|---|---|---|---|
| niah_single | <fill> | <fill> | <fill> | <fill> |
| niah_multikey | <fill> | <fill> | <fill> | <fill> |
| niah_multivalue | <fill> | <fill> | <fill> | <fill> |

## Result — single-cell 32 k decode

`benchmarks/phase7_decode_turbo_32k.json`:

- decode_tok_s = `<fill>` (target ≥6; INT4 baseline 3.88)
- prefill_tok_s = `<fill>`
- peak_vram_mib = `<fill>` (target < INT4's 5478)
- wall_s = `<fill>`

## Result — head-to-head re-test

`benchmarks/phase7_headtohead_turbo.json`:

| backend | quant + KV | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest TurboQuant | AWQ-INT4 + K3-V2 paged | <fill> | <fill> | <fill> |
| flashquest INT4 (Phase 6) | AWQ-INT4 + INT4 paged | 4.94 | 3.88 | ✗ |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 39.16 | ✗ (abort) | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | ✗ | ✗ |

## v2 follow-ups

- **Calibrated codebook** if RULER multivalue regresses below 90 % —
  recompute K_TURBO_CODEBOOK from Llama-3.2-3B WHT-rotated K samples on
  RULER prompts. Approach C in the spec.
- **1-bit QJL residual on K** — push K to effective 4-bit with negligible
  storage growth. Re-evaluate after the v1 ships.
- **Prefill TurboQuant kernel** — current kernel is decode-only; prefill
  still goes through dense BF16 SDPA after dequant + inverse-WHT.
- **TransMLA** (GQA → MLA conversion) — separate phase; needs ~6 B-token
  fine-tune. Composes with TurboQuant.
- **xKV (cross-layer KV sharing)** — orthogonal compression axis; ICLR
  2025. Composes with TurboQuant.
```

- [ ] **Step 4: Update DOC.md, README.md, docs/SPEC.md**

In `DOC.md`, find:
```
- Phase 6 task 7+ — TurboQuant, EAGLE-2, Marlin, ExLlamaV2.
```
Replace with:
```
- **Phase 7 — TurboQuant K3-V2 KV** ✅ **complete (tag `phase-7`); SPEC §11.4 verdict update: <fill>**. `flashquest.kernel.sparse_turbo_fwd` ships a fused Triton kernel that reads bit-split-packed K (1-bit MSB plane + 2-bit LSB plane) and INT2 V tiles directly, applies Walsh-Hadamard to Q at decode, gathers from fixed Rayleigh-Lloyd-Max codebooks (8 K codepoints, 4 V codepoints), and runs the same online-softmax math as INT4 fused. `flashquest.cache.PersistentTurboKVCache` (kv_bits=3) shrinks the KV footprint by 34 % vs INT4 (980 → 644 MiB at 32 k full cache). Quest criticality unchanged via dual statistics (`K_scale_raw, K_mn_raw` per-page channel-wise from un-rotated K). RULER 4k @ TurboQuant: <fill>/<fill>/<fill>. 32k decode: `<fill>` tok/s, peak VRAM `<fill>` MiB. Reference: `docs/PHASES/phase-7-notes.md`.
- Phase 7+ — TurboQuant calibrated codebook (if needed), TransMLA, xKV, EAGLE-2.
```

In `README.md`, after the Phase 6 task 6 section (`## Phase 6 task 6 — fused INT4 Triton kernel`), insert a new section:
```markdown
## Phase 7 — TurboQuant K3-V2 KV

Per-token Walsh-Hadamard rotation along `head_dim` + fixed
Rayleigh-Lloyd-Max codebook (8 K codepoints, 4 V codepoints). K stored
as bit-split planes (1-bit MSB + 2-bit LSB); V stored as packed INT2.
Shrinks the cache by 34 % vs Phase 6 INT4. Quest criticality unchanged
via dual statistics (un-rotated `K_scale_raw, K_mn_raw` per-page
channel-wise).

### Decode at 32 k under TurboQuant

`benchmarks/phase7_decode_turbo_32k.json`:

- decode_tok_s = `<fill>` (Phase 6 INT4 baseline: 3.88)
- peak_vram_mib = `<fill>` (Phase 6 INT4: 5478)
- wall_s = `<fill>`

### Throughput re-test — `benchmarks/phase7_headtohead_turbo.json`

| backend | quant + KV | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest TurboQuant | AWQ-INT4 + K3-V2 paged | **<fill>** | **<fill>** | <fill> |
| flashquest INT4 (Phase 6) | AWQ-INT4 + INT4 paged | 4.94 | 3.88 | ✗ |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 39.16 | ✗ | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | ✗ | ✗ |

(Fill placeholders with measured numbers.)

**SPEC §11.4 ≥5× verdict (post-Phase 7):** <CLEARED on capability + throughput
axes / partial — fill in based on measurement>.

Re-run via:
```bash
python scripts/bench_flashquest.py --ctx-len 32768 --kv-bits 3 \
    --out benchmarks/phase7_decode_turbo_32k.json
python scripts/phase7_run_ruler_4k_turbo.py
KV_BITS=3 python scripts/phase6_run_headtohead.py
```
```

In `docs/SPEC.md`, find the §6 task 6 line (added in Phase 6 task 6) and insert task 7:
```
7. **TurboQuant K3-V2 KV.** ✅ **DONE (<fill date>, tag `phase-7`); RULER NIAH 4k @ TurboQuant <fill>; SPEC §11.4 verdict: <fill>.** `flashquest.cache.PersistentTurboKVCache` (kv_bits=3) + `flashquest.kernel.sparse_turbo_fwd` (fused Triton kernel: bit-split K + INT2 V tile loads, in-kernel codebook gather, WHT applied to Q via the wrapper). Storage 34 % smaller than INT4 (980 → 644 MiB at 32 k full cache). Quest criticality unchanged via dual `K_scale_raw, K_mn_raw` from un-rotated K. CLI: `--kv-bits {3, 4, 8}` (default still 4 until Phase 7 ships its own ≥5× capability + throughput verdict). 32 k decode: <fill> tok/s under TurboQuant; capability ratio at 32 k stays ∞× over llama.cpp/vLLM. See `docs/PHASES/phase-7-notes.md`.
```

Renumber the subsequent tasks (EAGLE-2, Marlin, ExLlamaV2) so they become 8, 9, 10.

Update the §11 acceptance bullet 4 status with the new throughput numbers, replacing the Phase 6 task 6 status text.

- [ ] **Step 5: Commit and tag**

```bash
git add docs/PHASES/phase-7-notes.md DOC.md README.md docs/SPEC.md \
        benchmarks/phase7_headtohead_turbo.json benchmarks/phase7_headtohead_turbo.md \
        benchmarks/phase7_cells_turbo/
git commit -m "phase 7: complete — TurboQuant K3-V2 KV (verdict: <CLEARED|GATED>)"
git tag phase-7
git log --oneline phase-6-task-6..HEAD
```

---

## Self-review (post-write)

**1. Spec coverage** — every requirement in the spec maps to a task:
- §Goal: ship TurboQuant behind `--kv-bits 3`, ~34% shrink, hold RULER → covered by Tasks 1–13 end-to-end.
- §Architecture: WHT + dual-stat + bit-split + codebook → Tasks 1, 3, 4, 7.
- §Data flow (write/decode/prefill) → Tasks 3 (write), 5+7 (decode), 9 (prefill via dispatcher).
- §Storage layout → Task 4 (cache class shape).
- §Codebook constants → Task 2.
- §Bit-plane unpack → Task 7 (kernel).
- §Test strategy → Tasks 1–8 unit tests + Tasks 10–12 system gates.
- §Acceptance gates → Tasks 11 (RULER), 12 (32k decode), 10 (regression), 13 (head-to-head).
- §Out of scope → noted in plan; Tasks do NOT build QJL, calibrated codebook, prefill kernel, TransMLA, xKV.
- §Rollback plan → noted in Task 11 (RULER fail) and Task 12 (throughput fail).

**2. Placeholder scan** — Task 13's docs use `<fill>` deliberately for numbers that come from measured runs (Tasks 11, 12). Same pattern Phase 6 task 6 used. Step 4 of Task 13 explicitly tells the engineer where to read the JSON to fill placeholders. No "TBD" / "implement later" / "similar to Task N" elsewhere. ✓

**3. Type consistency** — `flash_attn_sparse_turbo_fwd` (kernel wrapper), `_flash_attn_sparse_turbo_fwd_reference` (Python ref), `_sparse_attn_fwd_kernel_turbo` (kernel JIT), `quantize_k_turbo` / `quantize_v_turbo` / `dequantize_k_turbo` / `dequantize_v_turbo`, `K_TURBO_CODEBOOK` / `V_TURBO_CODEBOOK`, `_pack_bit_split` / `_unpack_bit_split`, `_pack_int2` / `_unpack_int2`, `PersistentTurboKVCache` — all names used consistently across Tasks 2, 3, 4, 5, 6, 7, 8, 9. View dictionary keys (`K_msb`, `K_lsb`, `K_scale_turbo`, `K_scale_raw`, `K_mn_raw`, `V_packed`, `V_scale_turbo`) consistent across cache class (Task 4), reference path (Task 5), tests (Tasks 6, 8), and dispatcher (Task 9). ✓

**4. Risk + rollback** — Task 11 step 4 lists explicit RULER-fail recovery (Approach C calibrated codebook OR drop to `--kv-bits 4`). Task 12 step 2 lists throughput-fail recovery (profile and replace MSB unpack with lookup table). INT4 stays as v1 default — TurboQuant is opt-in via `--kv-bits 3` until Phase 7 ships its own verdict. ✓

No issues found.
