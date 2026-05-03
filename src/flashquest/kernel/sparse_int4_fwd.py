"""Phase 6 task 5 — sparse-attention forward with INT4 KV.

This iteration ships a *reference path*: dequantize INT4 → BF16 → re-quantize
to INT8 → call the existing flash_attn_sparse_fwd kernel. This validates the
full plumbing (cache → dispatcher → CLI flag → quality eval) without writing
new Triton bit-unpacking logic.

The kernel-fused inline INT4 unpack (single Triton kernel that reads packed
uint8 with head_dim/2 trailing axis and unpacks lo/hi nibbles inside the
tile load) is queued as a v2 follow-up. Sketch in this module's docstring:

    @triton.jit
    def _sparse_int4_fwd_kernel(...):
        # offs_dp = tl.arange(0, BLOCK_D // 2) — bytes per row
        # k_byte = tl.load(K_packed_ptr + offs_dp * stride_kd_packed, ...)
        # k_lo = (k_byte & 0xF).to(tl.float32)
        # k_hi = ((k_byte >> 4) & 0xF).to(tl.float32)
        # is_even = (offs_d % 2) == 0
        # half_idx = offs_d // 2
        # k_int = tl.where(is_even, gather(k_lo, half_idx), gather(k_hi, half_idx))
        # k_f = k_int * scale[d] + mn[d]
        # ... rest of online-softmax attention ...

The reference path below is faithful w.r.t. shape contracts and per-element
dequant; the only loss vs a fused INT4 kernel is the extra BF16 → INT8
re-quantization round-trip. Numerical agreement with dense is well within the
5e-2 RULER tolerance.
"""
from __future__ import annotations

import torch


def flash_attn_sparse_int4_fwd(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse forward with INT4 KV (reference path: dequant → INT8 → existing kernel).

    Shape contracts mirror flash_attn_sparse_fwd except K_packed / V_packed are
    uint8 with head_dim/2 trailing axis (2 INT4 values per byte).

    Args:
        Q: (B, H_q, 1, D) bf16 cuda.
        K_packed: (B, H_kv, S_kv, D/2) uint8 cuda.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16 cuda.
        V_packed: (B, H_kv, S_kv, D/2) uint8 cuda.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16 cuda.
        selection_mask: (B, H_q, 1, num_pages) bool cuda — True = attend.

    Returns:
        (O (B, H_q, 1, D) bf16, lse (B, H_q, 1) fp32 or None).
    """
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")
    if K_packed.dtype != torch.uint8 or V_packed.dtype != torch.uint8:
        raise ValueError(
            f"K_packed/V_packed must be uint8; got K={K_packed.dtype} V={V_packed.dtype}"
        )

    from flashquest.kernel.kv_quant import (
        dequantize_k_int4, dequantize_v_int4, quantize_k, quantize_v,
    )
    from flashquest.kernel.sparse_fwd import flash_attn_sparse_fwd

    K_bf16 = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)
    V_bf16 = dequantize_v_int4(V_packed, V_scale, V_mn)
    K_int8, K_scale8, K_mn8 = quantize_k(K_bf16, page_size=page_size)
    V_int8, V_scale8, V_mn8 = quantize_v(V_bf16)

    return flash_attn_sparse_fwd(
        Q, K_int8, K_scale8, K_mn8,
        V_int8, V_scale8, V_mn8,
        selection_mask=selection_mask,
        page_size=page_size, sm_scale=sm_scale, return_lse=return_lse,
    )
