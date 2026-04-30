"""Decode-only sparse forward kernel with INT8 KV. Phase 3.

Algorithm:
  - One program per (b, h_q). Q is a single row.
  - Outer loop over pages where selection_mask[b, h_q, 0, p] == True.
  - For each selected page:
      - Load K_uint8 page (page_size, D) — dequant to bf16 with K_scale, K_mn.
      - QK^T -> (page_size,).
      - Online softmax update.
      - Load V_uint8 page (page_size, D) — dequant to bf16 with V_scale, V_mn.
      - softmax · V -> (D,), accumulate.
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
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh + offs_d * stride_qd
    )
    q = tl.load(q_ptrs)

    NEG_INF: tl.constexpr = float("-inf")
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for p in range(0, NUM_PAGES):
        sel_ptr_p = sel_ptr + b * stride_selb + h_q * stride_selh + p * stride_selp
        is_sel = tl.load(sel_ptr_p)

        if is_sel:
            page_start = p * PAGE_SIZE
            n_idx = page_start + offs_n
            valid_kv = n_idx < S_kv

            k_ptrs = (
                K_ptr + b * stride_kb + h_kv * stride_kh
                + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd
            )
            k_u8 = tl.load(k_ptrs, mask=valid_kv[:, None], other=0)
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
            k = k_u8.to(tl.float32) * k_scale[None, :] + k_mn[None, :]

            qk = tl.sum(q[None, :].to(tl.float32) * k, axis=1)
            qk = tl.where(valid_kv, qk, NEG_INF)

            qk_max = tl.max(qk * qk_scale, axis=0)
            m_ij = tl.maximum(m_i, qk_max)
            m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
            p_softmax = tl.math.exp2(qk * qk_scale - m_ij_safe)
            row_all_neg_inf = m_ij == NEG_INF
            p_softmax = tl.where(row_all_neg_inf, 0.0, p_softmax)

            alpha = tl.math.exp2(m_i - m_ij_safe)
            if m_i == NEG_INF:
                alpha = 0.0

            l_i = l_i * alpha + tl.sum(p_softmax, axis=0)
            acc = acc * alpha

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

            acc += tl.sum(p_softmax[:, None] * v, axis=0)

            m_i = m_ij

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l

    o_ptrs = O_ptr + b * stride_ob + h_q * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty))

    if WRITE_LSE:
        lse_val = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse_val = tl.where(l_i == 0.0, NEG_INF, lse_val)
        l_ptr_bh = L_ptr + b * stride_lb + h_q * stride_lh
        tl.store(l_ptr_bh, lse_val)


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

    Q_2d = Q.squeeze(2)
    O_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    if L is not None:
        sl_b, sl_h = L.stride()
    else:
        sl_b = sl_h = 0

    sel_2d = selection_mask.squeeze(2)

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
