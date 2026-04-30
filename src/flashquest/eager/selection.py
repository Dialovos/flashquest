"""Top-k page selection with sink + sliding-window always-attended set."""
from __future__ import annotations

import math

import torch


def select_pages(
    scores: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Build a boolean mask over pages: union of top-k by score with sinks + window.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: fraction of pages to select via top-k. 1.0 = all, 0.0 = none
            (only sinks + window). Always rounds up: at least 1 page if retention > 0.
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include (recency window).

    Returns:
        Boolean mask shaped (B, H, S_q, P).
    """
    B, H, S_q, P = scores.shape
    assert 0.0 <= retention <= 1.0

    if retention >= 1.0:
        return torch.ones_like(scores, dtype=torch.bool)

    k = math.ceil(retention * P) if retention > 0 else 0
    mask = torch.zeros_like(scores, dtype=torch.bool)

    if k > 0:
        topk_idx = scores.topk(k, dim=-1).indices
        mask.scatter_(-1, topk_idx, True)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask
