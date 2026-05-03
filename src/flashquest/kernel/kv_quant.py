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
