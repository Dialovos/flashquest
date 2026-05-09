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
    retention: float = 0.20,                 # Phase 10 default (was 0.25)
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

    pattern_per_q_head = head_pattern.to(Q.device).repeat_interleave(n_rep)
    retention_per_q = torch.where(
        pattern_per_q_head,
        torch.full((H_q,), retention, device=Q.device),
        torch.zeros(H_q, device=Q.device),
    )

    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    scores = page_scores(Q.float(), page_min, page_max)

    sel = select_pages(
        scores, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )

    O, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=sel, page_size=page_size, return_lse=False,
    )
    return O
