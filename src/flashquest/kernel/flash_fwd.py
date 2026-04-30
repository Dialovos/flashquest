"""Dense FA-2 forward kernel for sm_86. Phase 2.

Algorithm: standard FlashAttention-2 online softmax, ported from the Triton
06-fused-attention tutorial with sm_86-friendly tile shapes. No sparsity
(Phase 3), no backward (out of scope).

Shapes (B=batch, H_q=query heads, H_kv=KV heads, S=seq, D=head_dim):
    Q: (B, H_q,  S_q,  D) bf16
    K: (B, H_kv, S_kv, D) bf16
    V: (B, H_kv, S_kv, D) bf16
    O: (B, H_q,  S_q,  D) bf16
    L: (B, H_q,  S_q)     fp32  (logsumexp; LSE output added in Task 4)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._autotune import FORWARD_CONFIGS


@triton.autotune(configs=FORWARD_CONFIGS, key=["S_q", "S_kv", "HEAD_DIM"])
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H_q, H_kv, S_q, S_kv,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)        # which BLOCK_M of queries
    pid_bh = tl.program_id(1)       # batch * H_q
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to Q tile (BLOCK_M, HEAD_DIM)
    q_ptrs = (
        Q_ptr
        + b * stride_qb + h_q * stride_qh
        + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < S_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e), so we use exp2 internally

    # Causal: queries at offs_m attend to keys at offs_n where offs_n <= offs_m + (S_kv - S_q).
    # For S_q == S_kv this is the standard lower triangle. For S_q == 1 (decode) and IS_CAUSAL=True,
    # this still attends to all S_kv keys (single query at the last position).
    if IS_CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M + (S_kv - S_q)
        kv_end = tl.minimum(kv_end, S_kv)
    else:
        kv_end = S_kv

    for start_n in range(0, kv_end, BLOCK_N):
        n_idx = start_n + offs_n
        k_mask = n_idx[:, None] < S_kv

        k_ptrs = (
            K_ptr
            + b * stride_kb + h_kv * stride_kh
            + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # qk: (BLOCK_M, BLOCK_N) accumulated in fp32
        qk = tl.dot(q, tl.trans(k))

        # Causal mask: q_pos = (S_kv - S_q) + offs_m; attend to k_pos <= q_pos.
        if IS_CAUSAL:
            q_pos = offs_m[:, None] + (S_kv - S_q)
            causal_mask = q_pos >= n_idx[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))

        # Out-of-range KV (when S_kv is not multiple of BLOCK_N): mask
        kv_oob_mask = n_idx[None, :] < S_kv
        qk = tl.where(kv_oob_mask, qk, -float("inf"))

        # Online softmax with safe handling of all-(-inf) rows (E10).
        m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, axis=1))
        m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.math.exp2(qk * qk_scale - m_ij_safe[:, None])
        p = tl.where(m_ij[:, None] == -float("inf"), 0.0, p)

        alpha = tl.math.exp2(m_i - m_ij_safe)
        alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V load + accumulate
        v_ptrs = (
            V_ptr
            + b * stride_vb + h_kv * stride_vh
            + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    # Normalize (avoid 0/0 when row was entirely masked).
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]

    # Store output
    o_ptrs = (
        O_ptr
        + b * stride_ob + h_q * stride_oh
        + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)

    if WRITE_LSE:
        # LSE = max + ln(sum exp(qk - max)). Kernel uses base-2 internally, so
        # accumulated state is in log2 space; convert to nats at write time.
        lse = (m_i + tl.math.log2(safe_l)) * 0.69314718  # ln(2)
        lse = tl.where(l_i == 0.0, -float("inf"), lse)
        l_ptrs = (
            L_ptr
            + b * stride_lb + h_q * stride_lh
            + offs_m * stride_ls
        )
        tl.store(l_ptrs, lse, mask=offs_m < S_q)


_SUPPORTED_HEAD_DIMS = (64, 128)


def flash_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dense FA-2 forward on sm_86 via Triton.

    Args:
        Q: (B, H_q, S_q, D) bfloat16 on cuda.
        K: (B, H_kv, S_kv, D) bfloat16 on cuda.
        V: (B, H_kv, S_kv, D) bfloat16 on cuda.
        causal: whether to apply a causal mask. The query at row i attends to
            keys at columns j where j <= i + (S_kv - S_q). In particular, for
            S_q == 1 (decode), the single query is at virtual position S_kv-1
            and attends to all S_kv keys — causal becomes a no-op. (Note: this
            is the FA-2 / Phase 1 eager convention. PyTorch SDPA's
            `is_causal=True` instead anchors the triangle at the top-left,
            which is wrong for decode — use `is_causal=False` against this
            kernel's `causal=True` decode output as the reference.) For
            S_q != S_kv with S_q > 1 (chunked prefill, E13), raises.
        sm_scale: softmax scale; defaults to 1/sqrt(D).
        return_lse: when True, also return per-(b,h,q) logsumexp.

    Returns:
        (O, lse) where lse is None if return_lse=False.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Q/K/V must be CUDA tensors"
    assert Q.dtype == K.dtype == V.dtype == torch.bfloat16, "Q/K/V must be bfloat16"

    B, H_q, S_q, D = Q.shape
    Bk, H_kv, S_kv, Dk = K.shape
    Bv, H_kvv, S_v, Dv = V.shape
    assert (B, D) == (Bk, Dk) == (Bv, Dv), f"shape mismatch Q/K/V: {Q.shape} vs {K.shape} vs {V.shape}"
    assert H_kv == H_kvv, "K and V must have the same number of heads"
    assert S_kv == S_v, "K and V must have the same sequence length"
    assert H_q % H_kv == 0, f"GQA: H_q ({H_q}) must be a multiple of H_kv ({H_kv})"

    if S_q == 0 or S_kv == 0:
        raise ValueError("flash_attn_fwd: zero-length input")
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            f"flash_attn_fwd: head_dim={D} not in {_SUPPORTED_HEAD_DIMS}"
        )
    if causal and S_q != S_kv and S_q != 1:
        raise NotImplementedError(
            f"flash_attn_fwd: causal with S_q={S_q} S_kv={S_kv} (E13: chunked prefill) not supported in Phase 2"
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    L = torch.empty(B, H_q, S_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)

    if L is not None:
        sl_b, sl_h, sl_s = L.stride()
    else:
        sl_b = sl_h = sl_s = 0

    grid = lambda META: (
        triton.cdiv(S_q, META["BLOCK_M"]),
        B * H_q,
    )

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, L_ptr,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sl_b, sl_h, sl_s,
        B, H_q, H_kv, S_q, S_kv,
        HEAD_DIM=D,
        IS_CAUSAL=bool(causal),
        WRITE_LSE=bool(return_lse),
    )

    return O, L
