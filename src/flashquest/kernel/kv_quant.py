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

    Pads the last (partial) page with +inf for min and -inf for max so the
    padding doesn't influence the per-page extrema.
    """
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    pad = num_pages * page_size - S

    if pad > 0:
        K_for_min = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("inf"))
        K_for_max = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("-inf"))
    else:
        K_for_min = K
        K_for_max = K
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
        K: (B, H, S, D) bf16 / fp16 / fp32.
        page_size: tokens per page.

    Returns:
        (K_uint8 (B, H, S, D), scale (B, H, num_pages, D) bf16, mn (B, H, num_pages, D) bf16).
    """
    B, H, S, D = K.shape
    scale, mn = _scale_mn_per_page_channel(K, page_size)

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
        (V_uint8 (B, H, S, D), scale (B, H, S, 1) bf16, mn (B, H, S, 1) bf16).
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


# === Phase 6 task 5: INT4 KV (KIVI-style asymmetric, packed 2-per-byte) ===


def _pack_int4(x: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values per byte along the last axis.

    Caller contract: x is uint8 with values in 0..15 and even last-axis size.
    Each output byte's low nibble = x[..., 2k], high nibble = x[..., 2k+1].
    Out-of-range high nibbles in the input are silently masked off.
    """
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"_pack_int4 requires even last axis; got {x.shape[-1]}")
    lo = x[..., 0::2] & 0x0F
    hi = x[..., 1::2] & 0x0F
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4. Returns uint8 with values in 0..15, last axis 2× input."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2,
        dtype=torch.uint8, device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def _scale_mn_per_page_channel_int4(K: torch.Tensor, page_size: int):
    """Same shape contract as _scale_mn_per_page_channel, but scale = (mx - mn) / 15."""
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    pad = num_pages * page_size - S

    if pad > 0:
        K_for_min = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("inf"))
        K_for_max = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("-inf"))
    else:
        K_for_min = K
        K_for_max = K
    K_for_min = K_for_min.view(B, H, num_pages, page_size, D)
    K_for_max = K_for_max.view(B, H, num_pages, page_size, D)
    mn = K_for_min.min(dim=3).values
    mx = K_for_max.max(dim=3).values
    scale = (mx - mn) / 15.0
    scale = scale.clamp_min(_EPS)
    return scale, mn


def quantize_k_int4(K: torch.Tensor, page_size: int):
    """KIVI-style asymmetric INT4 K quant, packed 2-per-byte along head_dim.

    Returns:
        (K_packed (B, H, S, D//2) uint8,
         scale   (B, H, num_pages, D) bf16,
         mn      (B, H, num_pages, D) bf16)
    """
    B, H, S, D = K.shape
    if D % 2 != 0:
        raise ValueError(f"quantize_k_int4 requires even head_dim; got {D}")
    scale, mn = _scale_mn_per_page_channel_int4(K, page_size)

    scale_per_token = scale.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    mn_per_token = mn.repeat_interleave(page_size, dim=2)[:, :, :S, :]

    K_norm = (K.float() - mn_per_token.float()) / scale_per_token.float()
    K_q4 = K_norm.round().clamp(0, 15).to(torch.uint8)
    K_packed = _pack_int4(K_q4)
    return K_packed, scale.to(torch.bfloat16), mn.to(torch.bfloat16)


def dequantize_k_int4(
    K_packed: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Inverse of quantize_k_int4. Returns bf16."""
    K_q4 = _unpack_int4(K_packed)
    B, H, S, D = K_q4.shape
    scale_per_token = scale.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    mn_per_token = mn.repeat_interleave(page_size, dim=2)[:, :, :S, :]
    out = K_q4.to(torch.float32) * scale_per_token.float() + mn_per_token.float()
    return out.to(torch.bfloat16)


def quantize_v_int4(V: torch.Tensor):
    """Per-token V quant, packed 2-per-byte along head_dim.

    Returns:
        (V_packed (B, H, S, D//2) uint8,
         scale   (B, H, S, 1) bf16,
         mn      (B, H, S, 1) bf16)
    """
    if V.shape[-1] % 2 != 0:
        raise ValueError(f"quantize_v_int4 requires even head_dim; got {V.shape[-1]}")
    mn = V.float().amin(dim=-1, keepdim=True)
    mx = V.float().amax(dim=-1, keepdim=True)
    scale = (mx - mn) / 15.0
    scale = scale.clamp_min(_EPS)
    V_q4 = ((V.float() - mn) / scale).round().clamp(0, 15).to(torch.uint8)
    V_packed = _pack_int4(V_q4)
    return V_packed, scale.to(torch.bfloat16), mn.to(torch.bfloat16)


def dequantize_v_int4(
    V_packed: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
) -> torch.Tensor:
    V_q4 = _unpack_int4(V_packed)
    out = V_q4.to(torch.float32) * scale.float() + mn.float()
    return out.to(torch.bfloat16)


# === Phase 7: TurboQuant primitives (bit-split, INT2 packing, codebooks) ===

# Lloyd-Max optimal codepoints for unit-variance Gaussian. The TurboQuant
# paper derives bounds from Rayleigh quantile statistics; these are the
# widely-tabulated symmetric-Lloyd-Max levels and serve as the data-oblivious
# codebook for K (3-bit, 8 levels) and V (2-bit, 4 levels). Values may be
# refined during quality validation if needed.
K_TURBO_CODEBOOK = torch.tensor(
    [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    dtype=torch.float32, device="cuda",
)  # 8 codepoints, indices 0..7 (3-bit Lloyd-Max for unit Gaussian)
V_TURBO_CODEBOOK = K_TURBO_CODEBOOK.clone()  # K3-V3: V also 8 levels (Phase 7 task 11b)

# Per-token RMS scale: s = sqrt(mean(x_rot**2)). Lloyd-Max codebook is optimal
# for unit-variance Gaussian, so RMS scaling puts the data at the right
# variance for the codebook. Outliers beyond the largest codepoint clip
# naturally via argmin (~0.1% of values at D=128 Gaussian).
_K_TURBO_C_MAX = 2.1519  # retained for reference / docs
_V_TURBO_C_MAX = 2.1519  # K3-V3 — V codebook now matches K


def _quantize_to_codebook(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """Round each x to nearest codepoint; return uint8 indices.

    Args:
        x: any shape, fp32 or bf16. Last-dim values are quantized independently.
        codebook: (K,) fp32 codepoints. K must be ≤ 256.

    Returns:
        uint8 tensor same shape as x, values in 0..K-1.
    """
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
    msb_bits = (idx >> 2) & 0x1
    lsb_bits = idx & 0x3

    *prefix, D = idx.shape
    msb_reshaped = msb_bits.reshape(*prefix, D // 8, 8)
    shifts = torch.arange(8, device=idx.device, dtype=torch.uint8)
    msb = (msb_reshaped << shifts).sum(dim=-1, dtype=torch.int32).to(torch.uint8)

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
    msb_bits = (msb.unsqueeze(-1) >> bit_offsets) & 0x1
    msb_full = msb_bits.reshape(*prefix, head_dim)

    lsb_offsets = (torch.arange(4, device=lsb.device, dtype=torch.uint8) * 2)
    lsb_bits = (lsb.unsqueeze(-1) >> lsb_offsets) & 0x3
    lsb_full = lsb_bits.reshape(*prefix, head_dim)

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


def quantize_k_turbo(K: torch.Tensor, page_size: int):
    """TurboQuant K: WHT → per-token scale → 3-bit Lloyd-Max → bit-split pack.

    Also computes un-rotated per-page channel-wise (K_scale_raw, K_mn_raw) for
    `page_scores_int4_fast` in the dispatcher.

    Returns:
        K_msb         (B, H, S, D/8)        uint8
        K_lsb         (B, H, S, D/4)        uint8
        K_scale_turbo (B, H, S, 1)          bf16
        K_scale_raw   (B, H, num_pages, D)  bf16  -- KIVI-INT4-equivalent
        K_mn_raw      (B, H, num_pages, D)  bf16
    """
    from flashquest.kernel.wht import wht_along_head_dim

    B, H, S, D = K.shape
    if D % 8 != 0:
        raise ValueError(f"quantize_k_turbo requires head_dim multiple of 8; got {D}")

    K_rot = wht_along_head_dim(K)

    K_rms = K_rot.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    K_scale_turbo = K_rms.clamp_min(_EPS)

    K_normalized = K_rot.float() / K_scale_turbo
    K_idx = _quantize_to_codebook(K_normalized, K_TURBO_CODEBOOK)

    K_msb, K_lsb = _pack_bit_split(K_idx)

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

    K_idx = _unpack_bit_split(K_msb, K_lsb, head_dim=head_dim)
    K_rot = K_TURBO_CODEBOOK[K_idx.long()] * K_scale_turbo.float()
    K = wht_along_head_dim(K_rot)
    return K.to(torch.bfloat16)


def quantize_v_turbo(V: torch.Tensor):
    """TurboQuant V: WHT → per-token RMS → 3-bit Lloyd-Max → bit-split pack.

    K3-V3 (Phase 7 task 11b): bumped V from 2-bit to 3-bit because the
    2-bit V codebook couldn't pass RULER multivalue. V now uses the same
    8-codepoint codebook + bit-split layout as K.

    Returns:
        V_msb          (B, H, S, D/8) uint8
        V_lsb          (B, H, S, D/4) uint8
        V_scale_turbo  (B, H, S, 1)   bf16
    """
    from flashquest.kernel.wht import wht_along_head_dim

    B, H, S, D = V.shape
    if D % 8 != 0:
        raise ValueError(f"quantize_v_turbo requires head_dim multiple of 8; got {D}")

    V_rot = wht_along_head_dim(V)
    V_rms = V_rot.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    V_scale_turbo = V_rms.clamp_min(_EPS)
    V_normalized = V_rot.float() / V_scale_turbo
    V_idx = _quantize_to_codebook(V_normalized, V_TURBO_CODEBOOK)
    V_msb, V_lsb = _pack_bit_split(V_idx)
    return V_msb, V_lsb, V_scale_turbo.to(torch.bfloat16)


def dequantize_v_turbo(
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """Inverse of quantize_v_turbo (K3-V3)."""
    from flashquest.kernel.wht import wht_along_head_dim

    V_idx = _unpack_bit_split(V_msb, V_lsb, head_dim=head_dim)
    V_rot = V_TURBO_CODEBOOK[V_idx.long()] * V_scale_turbo.float()
    V = wht_along_head_dim(V_rot)
    return V.to(torch.bfloat16)
