"""Streaming-only attention: sink tokens + sliding window, no top-k.

Equivalent to Phase 1's quest_eager_sdpa with retention=0.0, num_sinks=N,
window_pages=W. Exposed as a standalone function so DuoAttention dispatch
reads more clearly.
"""
from __future__ import annotations

import torch

from .attention import quest_eager_sdpa


def streaming_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    page_size: int = 64,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Streaming-only attention: only sinks + recency window are attended.

    Equivalent to StreamingLLM's behavior: keep first num_sinks tokens plus
    last window_pages*page_size tokens, drop everything else.
    """
    return quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=0.0,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
