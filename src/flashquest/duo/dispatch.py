"""DuoAttention per-head dispatch.

Reuses Phase 1's quest_eager_sdpa for retrieval heads and the streaming
helper for streaming heads. Eager Python — Phase 4 v1 does not require
kernel-side dispatch since the Phase 3 kernel already accepts arbitrary
per-query-head selection masks; running everything once per path and
selecting per head is correct (just slower than a fused dispatch).
"""
from __future__ import annotations

import torch

from ..eager.attention import quest_eager_sdpa
from ..eager.streaming import streaming_eager_sdpa


def quest_duo_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    head_pattern: torch.Tensor,
    page_size: int = 64,
    retention: float = 0.20,                 # Phase 10 default (was 0.25)
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Per-head DuoAttention dispatch.

    Args:
        Q: (B, H_q, S_q, D).
        K, V: (B, H_kv, S_kv, D).
        head_pattern: (H_kv,) bool tensor. True = retrieval, False = streaming.
            Per-KV-head; broadcast to all query heads in the GQA group.

    Returns:
        Output (B, H_q, S_q, D).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, _, _ = K.shape
    if head_pattern.shape != (H_kv,):
        raise ValueError(
            f"head_pattern must be shape ({H_kv},) (one entry per KV head); got {tuple(head_pattern.shape)}"
        )

    O_retrieval = quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
    O_streaming = streaming_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )

    n_rep = H_q // H_kv
    pattern_per_q_head = head_pattern.to(Q.device).repeat_interleave(n_rep)
    sel = pattern_per_q_head.view(1, H_q, 1, 1)

    return torch.where(sel, O_retrieval, O_streaming)
