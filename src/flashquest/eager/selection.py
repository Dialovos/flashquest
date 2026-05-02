"""Top-k page selection with sink + sliding-window always-attended set."""
from __future__ import annotations

import math

import torch


def select_pages(
    scores: torch.Tensor,
    retention: float | torch.Tensor,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Build a boolean mask over pages: union of top-k by score with sinks + window.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: fraction of pages to select via top-k. Scalar in [0, 1] OR
            a 1-D tensor of shape (H,) for per-head retention.
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include (recency window).

    Returns:
        Boolean mask shaped (B, H, S_q, P).
    """
    B, H, S_q, P = scores.shape

    if isinstance(retention, torch.Tensor):
        if retention.shape != (H,):
            raise ValueError(
                f"per-head retention must be shape ({H},); got {tuple(retention.shape)}"
            )
        retention_per_h = retention.to(scores.device).float()
    else:
        if not (0.0 <= retention <= 1.0):
            raise ValueError(f"scalar retention must be in [0, 1]; got {retention}")
        retention_per_h = torch.full((H,), float(retention), device=scores.device)

    mask = torch.zeros_like(scores, dtype=torch.bool)

    # Per-head top-k. Heads with retention=0 contribute nothing here.
    for h in range(H):
        r = retention_per_h[h].item()
        if r >= 1.0:
            mask[:, h] = True
            continue
        if r <= 0.0:
            continue
        k = math.ceil(r * P)
        if k > 0:
            topk_idx = scores[:, h].topk(k, dim=-1).indices
            mask[:, h].scatter_(-1, topk_idx, True)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask


def select_pages_vectorized(
    scores: torch.Tensor,
    retention: float | torch.Tensor,
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:
    """Vectorized equivalent of select_pages — single batched topk + scatter,
    no Python per-head loop, no `.item()` per head.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: scalar in [0, 1] or 1-D tensor of shape (H,).
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include.

    Returns:
        Boolean mask shaped (B, H, S_q, P).
    """
    B, H, S_q, P = scores.shape

    if isinstance(retention, torch.Tensor):
        if retention.shape != (H,):
            raise ValueError(
                f"per-head retention must be shape ({H},); got {tuple(retention.shape)}"
            )
        retention_per_h = retention.to(scores.device).float()
    else:
        if not (0.0 <= retention <= 1.0):
            raise ValueError(f"scalar retention must be in [0, 1]; got {retention}")
        retention_per_h = torch.full((H,), float(retention), device=scores.device)

    k_per_h = (retention_per_h * P).ceil().long().clamp(min=0, max=P)  # (H,)

    mask = torch.zeros_like(scores, dtype=torch.bool)

    k_max = int(k_per_h.max().item())

    if k_max > 0:
        topk_idx = scores.topk(k_max, dim=-1).indices  # (B, H, S_q, k_max)
        ranks = torch.arange(k_max, device=scores.device).view(1, 1, 1, k_max)
        keep = ranks < k_per_h.view(1, H, 1, 1)
        src = keep.expand_as(topk_idx)
        mask.scatter_(-1, topk_idx, src)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True

    return mask
