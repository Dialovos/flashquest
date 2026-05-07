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

    Algorithm: dequant the entire K, V cache to BF16 (raw basis), build a
    per-token attention mask from the page selection, run dense attention.
    Output O is in raw basis.
    """
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")
    if K_msb.dtype != torch.uint8 or K_lsb.dtype != torch.uint8:
        raise ValueError("K_msb/K_lsb must be uint8")
    if V_packed.dtype != torch.uint8:
        raise ValueError("V_packed must be uint8")

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"decode-only (S_q={S_q})")

    Bk, H_kv, S_kv, _ = K_msb.shape
    n_rep = H_q // H_kv

    K_full = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=D)
    V_full = dequantize_v_turbo(V_packed, V_scale_turbo, head_dim=D)

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
