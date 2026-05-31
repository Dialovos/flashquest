"""Phase 8a — sparse-attention forward with INT4 KV (compact-list variant).

Replaces the bool-mask kernel's `for p in range(0, NUM_PAGES)` loop with
`for i in range(0, BUCKET_MAX)` over a compact int32 list of selected page
IDs (B, H_q, BUCKET_MAX). Sentinel -1 marks unused slots; the kernel skips
their contribution via load-time masking + downstream qk = -inf.

ABI is per-H_q (not per-H_kv) — Quest selection is per-H_q in selection.py
and we preserve that to avoid quality risk from head unioning.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (64, 128)
_SQ_MAX_COMPACT = 16


@triton.jit
def _sparse_attn_fwd_kernel_int4_compact(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_seli,
    H_q, H_kv, S_kv,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PACKED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUCKET_MAX: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """One CTA per (batch, query head). Iterates BUCKET_MAX compact slots."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED)

    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh + offs_d * stride_qd
    )
    q = tl.load(q_ptrs)

    NEG_INF: tl.constexpr = float("-inf")
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for i in range(0, BUCKET_MAX):
        sel_off = b * stride_selb + h_q * stride_selh + i * stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0
        p_safe = tl.where(page_valid, p, 0)

        page_start = p_safe * PAGE_SIZE
        n_idx = page_start + offs_n
        valid_kv = (n_idx < S_kv) & page_valid

        # === Per-token K loads (masked with valid_kv) ===
        k_byte_ptrs = (
            K_packed_ptr + b * stride_kb + h_kv * stride_kh
            + n_idx[:, None] * stride_ks + offs_dp[None, :] * stride_kdp
        )
        k_byte = tl.load(k_byte_ptrs, mask=valid_kv[:, None], other=0)
        k_lo = (k_byte & 0xF).to(tl.uint8)
        k_hi = ((k_byte >> 4) & 0xF).to(tl.uint8)
        k_int_2 = tl.join(k_lo, k_hi)
        k_int = tl.reshape(k_int_2, (PAGE_SIZE, HEAD_DIM))

        # === Per-page K scale/mn loads (use p_safe; result discarded via qk = -inf) ===
        ks_ptrs = (
            K_scale_ptr + b * stride_ksb + h_kv * stride_ksh
            + p_safe * stride_ksp + offs_d * stride_ksd
        )
        km_ptrs = (
            K_mn_ptr + b * stride_kmb + h_kv * stride_kmh
            + p_safe * stride_kmp + offs_d * stride_kmd
        )
        k_scale = tl.load(ks_ptrs).to(tl.float32)
        k_mn = tl.load(km_ptrs).to(tl.float32)
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]

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

        # === Per-token V loads (masked with valid_kv) ===
        v_byte_ptrs = (
            V_packed_ptr + b * stride_vb + h_kv * stride_vh
            + n_idx[:, None] * stride_vs + offs_dp[None, :] * stride_vdp
        )
        v_byte = tl.load(v_byte_ptrs, mask=valid_kv[:, None], other=0)
        v_lo = (v_byte & 0xF).to(tl.uint8)
        v_hi = ((v_byte >> 4) & 0xF).to(tl.uint8)
        v_int_2 = tl.join(v_lo, v_hi)
        v_int = tl.reshape(v_int_2, (PAGE_SIZE, HEAD_DIM))

        vs_ptrs = V_scale_ptr + b * stride_vsb + h_kv * stride_vsh + n_idx * stride_vss
        vm_ptrs = V_mn_ptr + b * stride_vmb + h_kv * stride_vmh + n_idx * stride_vms
        v_scale = tl.load(vs_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
        v_mn = tl.load(vm_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]

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


def flash_attn_sparse_int4_fwd_compact(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selected_page_ids: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode-only sparse forward with INT4 KV — compact-list variant.

    Args mirror flash_attn_sparse_int4_fwd, EXCEPT selection_mask is replaced
    with selected_page_ids (compact int32 list per H_q with -1 sentinel padding).

    selected_page_ids shape: (B, H_q, S_q=1, BUCKET_MAX) OR (B, H_q, BUCKET_MAX)
    (S_q axis is squeezed if present).
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_packed.dtype == torch.uint8 and V_packed.dtype == torch.uint8
    assert selected_page_ids.dtype == torch.int32

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"compact INT4: decode-only (S_q={S_q})")
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")

    Bk, H_kv, S_kv, Dp = K_packed.shape
    assert B == Bk
    assert Dp == D // 2, f"K_packed last axis {Dp} != D/2={D // 2}"
    assert H_q % H_kv == 0

    if selected_page_ids.dim() == 4:
        sel_3d = selected_page_ids.squeeze(2)  # (B, H_q, BUCKET_MAX)
    elif selected_page_ids.dim() == 3:
        sel_3d = selected_page_ids
    else:
        raise ValueError(
            f"selected_page_ids must be 3D or 4D; got {selected_page_ids.dim()}D"
        )
    Bs, H_qs, BUCKET_MAX = sel_3d.shape
    assert Bs == B and H_qs == H_q

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

    sel_3d_contig = sel_3d.contiguous()

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_int4_compact[grid](
        Q_2d, K_packed, V_packed, O_2d, L_ptr,
        K_scale, K_mn, V_scale, V_mn,
        sel_3d_contig,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O_2d.stride(0), O_2d.stride(1), O_2d.stride(2),
        sl_b, sl_h,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel_3d_contig.stride(0), sel_3d_contig.stride(1), sel_3d_contig.stride(2),
        H_q, H_kv, S_kv,
        HEAD_DIM=D,
        HEAD_DIM_PACKED=D // 2,
        PAGE_SIZE=page_size,
        BUCKET_MAX=BUCKET_MAX,
        WRITE_LSE=bool(return_lse),
        num_warps=4,
        num_stages=2,
    )

    O = O_2d.unsqueeze(2)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out


# === Phase 12 task 1: S_q>1 verify variant (non-causal over completed pages) ===


@triton.jit
def _sparse_attn_fwd_kernel_int4_compact_sq_gt_1(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr, sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_seli,
    H_q, H_kv, S_kv, S_q,
    HEAD_DIM: tl.constexpr, HEAD_DIM_PACKED: tl.constexpr, PAGE_SIZE: tl.constexpr,
    BUCKET_MAX: tl.constexpr, SQ_MAX: tl.constexpr, WRITE_LSE: tl.constexpr,
):
    """One CTA per (batch, query head). S_q>1 non-causal verify variant; iterates BUCKET_MAX compact slots, sq_mask gates padding rows."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q; h_q = pid_bh % H_q
    n_rep = H_q // H_kv; h_kv = h_q // n_rep
    offs_n = tl.arange(0, PAGE_SIZE); offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED); offs_sq = tl.arange(0, SQ_MAX)
    sq_mask = offs_sq < S_q
    NEG_INF: tl.constexpr = float("-inf")
    q_ptrs = (Q_ptr + b*stride_qb + h_q*stride_qh
              + offs_sq[:, None]*stride_qs + offs_d[None, :]*stride_qd)
    q = tl.load(q_ptrs, mask=sq_mask[:, None], other=0.0)              # (SQ_MAX, HEAD_DIM)
    m_i = tl.full((SQ_MAX,), NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((SQ_MAX,), dtype=tl.float32)
    acc = tl.zeros((SQ_MAX, HEAD_DIM), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    for i in range(0, BUCKET_MAX):
        sel_off = b*stride_selb + h_q*stride_selh + i*stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0; p_safe = tl.where(page_valid, p, 0)
        n_idx = p_safe*PAGE_SIZE + offs_n
        valid_kv = (n_idx < S_kv) & page_valid
        k_byte = tl.load(K_packed_ptr + b*stride_kb + h_kv*stride_kh
                         + n_idx[:, None]*stride_ks + offs_dp[None, :]*stride_kdp,
                         mask=valid_kv[:, None], other=0)
        k_int = tl.reshape(tl.join((k_byte & 0xF).to(tl.uint8), ((k_byte >> 4) & 0xF).to(tl.uint8)),
                           (PAGE_SIZE, HEAD_DIM))
        k_scale = tl.load(K_scale_ptr + b*stride_ksb + h_kv*stride_ksh + p_safe*stride_ksp + offs_d*stride_ksd).to(tl.float32)
        k_mn = tl.load(K_mn_ptr + b*stride_kmb + h_kv*stride_kmh + p_safe*stride_kmp + offs_d*stride_kmd).to(tl.float32)
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]    # (PAGE_SIZE, HEAD_DIM)
        qk = tl.dot(q, tl.trans(k.to(tl.bfloat16)))  # bf16 tensor-core dot, fp32 accum -> (SQ_MAX, PAGE_SIZE)
        qk = tl.where(sq_mask[:, None] & valid_kv[None, :], qk, NEG_INF)
        qk_max = tl.max(qk * qk_scale, axis=1)
        m_ij = tl.maximum(m_i, qk_max)
        m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
        p_sm = tl.math.exp2(qk * qk_scale - m_ij_safe[:, None])
        p_sm = tl.where(qk == NEG_INF, 0.0, p_sm)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2(m_i - m_ij_safe))
        l_i = l_i * alpha + tl.sum(p_sm, axis=1)
        acc = acc * alpha[:, None]
        v_byte = tl.load(V_packed_ptr + b*stride_vb + h_kv*stride_vh
                         + n_idx[:, None]*stride_vs + offs_dp[None, :]*stride_vdp,
                         mask=valid_kv[:, None], other=0)
        v_int = tl.reshape(tl.join((v_byte & 0xF).to(tl.uint8), ((v_byte >> 4) & 0xF).to(tl.uint8)),
                           (PAGE_SIZE, HEAD_DIM))
        v_scale = tl.load(V_scale_ptr + b*stride_vsb + h_kv*stride_vsh + n_idx*stride_vss, mask=valid_kv, other=0.0).to(tl.float32)
        v_mn = tl.load(V_mn_ptr + b*stride_vmb + h_kv*stride_vmh + n_idx*stride_vms, mask=valid_kv, other=0.0).to(tl.float32)
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]
        acc += tl.dot(p_sm.to(tl.bfloat16), v.to(tl.bfloat16))  # bf16 tensor-core dot, fp32 accum
        m_i = m_ij
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]
    o_ptrs = O_ptr + b*stride_ob + h_q*stride_oh + offs_sq[:, None]*stride_os + offs_d[None, :]*stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=sq_mask[:, None])
    if WRITE_LSE:
        lse_val = tl.where(l_i == 0.0, NEG_INF, (m_i + tl.math.log2(safe_l)) * 0.69314718)
        tl.store(L_ptr + b*stride_lb + h_q*stride_lh + offs_sq*stride_ls, lse_val, mask=sq_mask)


def flash_attn_sparse_int4_fwd_compact_sq(
    Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
    *, selected_page_ids, page_size=64, sm_scale=None, return_lse=True,
):
    """S_q>1 verify variant of flash_attn_sparse_int4_fwd_compact. Non-causal
    over selected completed pages. selected_page_ids: (B, H_q, BUCKET_MAX) int32."""
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_packed.dtype == torch.uint8 and V_packed.dtype == torch.uint8
    assert selected_page_ids.dtype == torch.int32 and selected_page_ids.dim() == 3
    B, H_q, S_q, D = Q.shape
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")
    if S_q > _SQ_MAX_COMPACT:
        raise NotImplementedError(f"S_q={S_q} > SQ_MAX={_SQ_MAX_COMPACT}")
    Bk, H_kv, S_kv, Dp = K_packed.shape
    assert B == Bk, f"batch mismatch: Q {B} vs K_packed {Bk}"
    assert Dp == D // 2, f"K_packed last axis {Dp} != D/2={D // 2}"
    assert H_q % H_kv == 0, f"H_q={H_q} not divisible by H_kv={H_kv}"
    _, _, BUCKET_MAX = selected_page_ids.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q)
    L = torch.empty(B, H_q, S_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    sl_b, sl_h, sl_s = (L.stride() if L is not None else (0, 0, 0))
    sel = selected_page_ids.contiguous()
    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_int4_compact_sq_gt_1[grid](
        Q, K_packed, V_packed, O, L_ptr, K_scale, K_mn, V_scale, V_mn, sel, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sl_b, sl_h, sl_s,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel.stride(0), sel.stride(1), sel.stride(2),
        H_q, H_kv, S_kv, S_q,
        HEAD_DIM=D, HEAD_DIM_PACKED=D // 2, PAGE_SIZE=page_size,
        BUCKET_MAX=BUCKET_MAX, SQ_MAX=_SQ_MAX_COMPACT, WRITE_LSE=bool(return_lse),
        # Kernel uses bf16 tensor-core tl.dot with fp32 accumulation (resolves
        # the prior fp32-emulation slowness that made verify ~7x too slow on
        # sm_86). num_warps=8, num_stages=1.
        num_warps=8, num_stages=1,
    )
    return O, L
