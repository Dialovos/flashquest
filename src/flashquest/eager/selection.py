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
    *,
    k_max_static: int | None = None,
) -> torch.Tensor:
    """Vectorized equivalent of select_pages — single batched topk + scatter,
    no Python per-head loop, no `.item()` per head.

    Args:
        scores: (B, H, S_q, P) per-query per-page criticality scores.
        retention: scalar in [0, 1] or 1-D tensor of shape (H,).
        num_sinks: number of leading pages to always include.
        window_pages: number of trailing pages to always include.
        k_max_static: optional precomputed upper bound on `k_per_h.max()`.
            If provided, eliminates the per-step `.item()` sync. The caller
            must guarantee `k_max_static >= ceil(max(retention) * P_max)`.
            Clamped to current P at runtime (Phase 8a codex r3 finding #1
            — early decode has P < P_max).

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

    if k_max_static is None:
        k_max = int(k_per_h.max().item())
    else:
        k_max = min(int(k_max_static), P)

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


def build_compact_selection(
    mask: torch.Tensor,
    BUCKET_MAX: int,
) -> torch.Tensor:
    """Convert (B, H, S_q, P) bool mask -> (B, H, S_q, BUCKET_MAX) int32.

    Selected page indices are placed first (sorted descending by index — order
    inside the bucket doesn't matter for softmax); remaining slots are -1
    sentinels. GPU-resident, no `.item()`. Bool mask handles dedup naturally
    (each page is True or False, no duplicates).

    Args:
        mask: (B, H, S_q, P) bool — output of select_pages_vectorized.
        BUCKET_MAX: int >= 1 — fixed length of the output's last axis.

    Returns:
        (B, H, S_q, BUCKET_MAX) int32 with values in [-1, P).
    """
    if BUCKET_MAX < 1:
        raise ValueError(f"BUCKET_MAX must be >= 1, got {BUCKET_MAX}")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")

    B, H, S_q, P = mask.shape
    positions = torch.arange(P, device=mask.device, dtype=torch.int32)
    positions = positions.expand(B, H, S_q, P)
    pos_or_neg1 = torch.where(mask, positions, torch.full_like(positions, -1))
    sorted_pos, _ = pos_or_neg1.sort(dim=-1, descending=True)

    if P >= BUCKET_MAX:
        return sorted_pos[..., :BUCKET_MAX].contiguous()
    out = torch.full(
        (B, H, S_q, BUCKET_MAX), -1, dtype=torch.int32, device=mask.device,
    )
    out[..., :P] = sorted_pos
    return out.contiguous()
