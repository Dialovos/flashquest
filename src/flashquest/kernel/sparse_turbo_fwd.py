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
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference path — Python, used for kernel equivalence testing (K3-V3).

    Algorithm: dequant the entire K, V cache to BF16 (raw basis), build a
    per-token attention mask from the page selection, run dense attention.
    Output O is in raw basis.
    """
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")
    if K_msb.dtype != torch.uint8 or K_lsb.dtype != torch.uint8:
        raise ValueError("K_msb/K_lsb must be uint8")
    if V_msb.dtype != torch.uint8 or V_lsb.dtype != torch.uint8:
        raise ValueError("V_msb/V_lsb must be uint8")

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"decode-only (S_q={S_q})")

    Bk, H_kv, S_kv, _ = K_msb.shape
    n_rep = H_q // H_kv

    K_full = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=D)
    V_full = dequantize_v_turbo(V_msb, V_lsb, V_scale_turbo, head_dim=D)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    page_idx = torch.arange(S_kv, device=Q.device) // page_size  # (S_kv,)
    sel = selection_mask[..., page_idx]                          # (B, H_q, 1, S_kv) bool
    attn_bias = torch.where(sel, 0.0, float("-inf")).float()

    K_rep = K_full.repeat_interleave(n_rep, dim=1)
    V_rep = V_full.repeat_interleave(n_rep, dim=1)

    qk = (Q.float() @ K_rep.float().transpose(-1, -2)) * sm_scale + attn_bias
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    l = p.sum(dim=-1, keepdim=True)
    O = (p @ V_rep.float()) / l

    lse = None
    if return_lse:
        lse = (m + torch.log(l)).squeeze(-1).to(torch.float32)

    return O.to(torch.bfloat16), lse


import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (64, 128)


@triton.jit
def _codebook_lookup_3bit(idx):
    """Map uint8 idx ∈ {0..7} → fp32 codebook value via SELP chain.

    Codebook: 8 Lloyd-Max levels for unit-variance Gaussian, identical for K and V
    in the K3-V3 design. Inlining as tl.where avoids the GMEM scatter-gather
    pattern that dominated the kernel time.
    """
    return tl.where(idx == 0, -2.1519,
           tl.where(idx == 1, -1.3439,
           tl.where(idx == 2, -0.7560,
           tl.where(idx == 3, -0.2451,
           tl.where(idx == 4, 0.2451,
           tl.where(idx == 5, 0.7560,
           tl.where(idx == 6, 1.3439,
                              2.1519)))))))


@triton.jit
def _sparse_attn_fwd_kernel_turbo(
    Q_rot_ptr,
    K_msb_ptr, K_lsb_ptr, V_msb_ptr, V_lsb_ptr,
    O_rot_ptr, L_ptr,
    K_scale_t_ptr, V_scale_t_ptr,
    sel_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kmb, stride_kmh, stride_kms, stride_kmd,
    stride_klb, stride_klh, stride_kls, stride_kld,
    stride_vmb, stride_vmh, stride_vms, stride_vmd,
    stride_vlb, stride_vlh, stride_vls, stride_vld,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_kstb, stride_ksth, stride_ksts,
    stride_vstb, stride_vsth, stride_vsts,
    stride_selb, stride_selh, stride_selp,
    H_q, H_kv, S_kv, NUM_PAGES,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_MSB: tl.constexpr,
    HEAD_DIM_LSB: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """Decode-only sparse forward, TurboQuant K3-V3. One CTA per (batch, query head)."""
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

            # === Load K_msb (PAGE_SIZE, D/8) and K_lsb (PAGE_SIZE, D/4) tiles. ===
            k_msb_byte = tl.load(
                K_msb_ptr + b * stride_kmb + h_kv * stride_kmh
                + n_idx[:, None] * stride_kms + offs_d_msb[None, :] * stride_kmd,
                mask=valid_kv[:, None], other=0,
            )
            k_lsb_byte = tl.load(
                K_lsb_ptr + b * stride_klb + h_kv * stride_klh
                + n_idx[:, None] * stride_kls + offs_d_lsb[None, :] * stride_kld,
                mask=valid_kv[:, None], other=0,
            )

            # 1-bit MSB unpack via broadcast-shift: byte k bits 0..7 → output
            # positions [k*8..k*8+7] in D-dim (matches Python pack via shift=arange(8)).
            bit_offsets_8 = tl.arange(0, 8)
            k_msb_expanded = (k_msb_byte[:, :, None] >> bit_offsets_8[None, None, :]) & 0x1
            k_msb_full = tl.reshape(k_msb_expanded, (PAGE_SIZE, HEAD_DIM))

            # 2-bit LSB unpack via broadcast-shift: byte k 2-bit values 0..3 → output
            # positions [k*4..k*4+3] in D-dim.
            lsb_offsets = tl.arange(0, 4) * 2
            k_lsb_expanded = (k_lsb_byte[:, :, None] >> lsb_offsets[None, None, :]) & 0x3
            k_lsb_full = tl.reshape(k_lsb_expanded, (PAGE_SIZE, HEAD_DIM))

            # Combine: idx = (msb << 2) | lsb  ∈  0..7
            k_idx = (k_msb_full.to(tl.int32) << 2) | k_lsb_full.to(tl.int32)
            k_rot = _codebook_lookup_3bit(k_idx)

            k_scale_t = tl.load(
                K_scale_t_ptr + b * stride_kstb + h_kv * stride_ksth + n_idx * stride_ksts,
                mask=valid_kv, other=0.0,
            ).to(tl.float32)
            k = k_rot * k_scale_t[:, None]

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

            # === Load V_msb (PAGE_SIZE, D/8) + V_lsb (PAGE_SIZE, D/4), 3-bit unpack. ===
            v_msb_byte = tl.load(
                V_msb_ptr + b * stride_vmb + h_kv * stride_vmh
                + n_idx[:, None] * stride_vms + offs_d_msb[None, :] * stride_vmd,
                mask=valid_kv[:, None], other=0,
            )
            v_lsb_byte = tl.load(
                V_lsb_ptr + b * stride_vlb + h_kv * stride_vlh
                + n_idx[:, None] * stride_vls + offs_d_lsb[None, :] * stride_vld,
                mask=valid_kv[:, None], other=0,
            )
            v_msb_expanded = (v_msb_byte[:, :, None] >> bit_offsets_8[None, None, :]) & 0x1
            v_msb_full = tl.reshape(v_msb_expanded, (PAGE_SIZE, HEAD_DIM))
            v_lsb_expanded = (v_lsb_byte[:, :, None] >> lsb_offsets[None, None, :]) & 0x3
            v_lsb_full = tl.reshape(v_lsb_expanded, (PAGE_SIZE, HEAD_DIM))
            v_idx = (v_msb_full.to(tl.int32) << 2) | v_lsb_full.to(tl.int32)
            v_rot = _codebook_lookup_3bit(v_idx)

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
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode-only fused TurboQuant sparse forward (K3-V3).

    Wrapper applies WHT to Q (single vector per head) and inverse-WHT to the
    output (V was stored rotated). Kernel handles tile loads, bit-plane unpack,
    codebook gather, online softmax.

    Args:
        Q: (B, H_q, 1, D) bf16 cuda. RAW basis (wrapper rotates).
        K_msb / K_lsb: (B, H_kv, S_kv, D/8) / (B, H_kv, S_kv, D/4) uint8.
        K_scale_turbo: (B, H_kv, S_kv, 1) bf16.
        V_msb / V_lsb: (B, H_kv, S_kv, D/8) / (B, H_kv, S_kv, D/4) uint8.
        V_scale_turbo: (B, H_kv, S_kv, 1) bf16.
        selection_mask: (B, H_q, 1, num_pages) bool.

    Returns:
        (O (B, H_q, 1, D) bf16, lse (B, H_q, 1) fp32 or None). Both in raw basis.
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_msb.dtype == torch.uint8 and K_lsb.dtype == torch.uint8
    assert V_msb.dtype == torch.uint8 and V_lsb.dtype == torch.uint8

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

    Q_rot = wht_along_head_dim(Q)
    Q_2d = Q_rot.squeeze(2).contiguous()
    O_rot_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    sl_b, sl_h = (L.stride() if L is not None else (0, 0))

    sel_2d = selection_mask.squeeze(2)

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_turbo[grid](
        Q_2d,
        K_msb, K_lsb, V_msb, V_lsb,
        O_rot_2d, L_ptr,
        K_scale_turbo, V_scale_turbo,
        sel_2d,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_msb.stride(0), K_msb.stride(1), K_msb.stride(2), K_msb.stride(3),
        K_lsb.stride(0), K_lsb.stride(1), K_lsb.stride(2), K_lsb.stride(3),
        V_msb.stride(0), V_msb.stride(1), V_msb.stride(2), V_msb.stride(3),
        V_lsb.stride(0), V_lsb.stride(1), V_lsb.stride(2), V_lsb.stride(3),
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

    O_rot = O_rot_2d.unsqueeze(2)
    O = wht_along_head_dim(O_rot)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out
