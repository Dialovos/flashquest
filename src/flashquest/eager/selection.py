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


def build_compact_union_selection(
    sel_per_q: torch.Tensor,
    scores: torch.Tensor,
    *,
    num_sinks: int,
    window_pages: int,
    completed_len: int,
    page_size: int,
    BUCKET_MAX_UNION: int,
) -> torch.Tensor:
    """Score-prioritized UNION over the S_q axis, sinks+window force-included.

    Takes the per-query boolean page masks (one per speculative query row) and
    collapses them into a single compact int32 page list per (B, H) pair. Sinks
    and the recency window are always included; on overflow the highest-scoring
    pages win.

    Args:
        sel_per_q: bool (B, H_q, S_q, P) — per-query selection masks.
        scores: float (B, H_q, S_q, P) — per-query per-page criticality scores.
        num_sinks: number of leading pages to force-include.
        window_pages: number of trailing pages (recency window) to force-include.
        completed_len: total number of completed KV tokens (determines n_pages).
        page_size: tokens per page.
        BUCKET_MAX_UNION: fixed output width (last axis); overflow drops
            lowest-score pages, underflow pads with -1 sentinels.

    Returns:
        int32 tensor shaped (B, H_q, BUCKET_MAX_UNION) with -1 sentinels.
        GPU-resident, no .item() calls.
    """
    if sel_per_q.dtype != torch.bool:
        raise ValueError(f"sel_per_q must be bool, got {sel_per_q.dtype}")

    union = sel_per_q.any(dim=2)                        # (B, H_q, P)
    max_scores = scores.amax(dim=2).float()             # (B, H_q, P)

    P = union.shape[-1]
    n_pages = completed_len // page_size

    forced = torch.zeros_like(union)
    ns = min(num_sinks, P)
    if ns > 0:
        forced[..., :ns] = True
    nw = min(window_pages, n_pages)
    if nw > 0:
        start = max(0, min(n_pages - nw, P - 1))
        end = min(n_pages, P)
        if start < end:
            forced[..., start:end] = True

    NEG = torch.finfo(torch.float32).min
    POS = torch.finfo(torch.float32).max

    priority = torch.where(union | forced, max_scores, torch.full_like(max_scores, NEG))
    priority = torch.where(forced, torch.full_like(priority, POS), priority)

    k = min(BUCKET_MAX_UNION, P)
    top = priority.topk(k, dim=-1).indices              # (B, H_q, k)
    top_pri = priority.gather(-1, top)
    out_k = torch.where(top_pri <= NEG, torch.full_like(top, -1), top).to(torch.int32)

    if k == BUCKET_MAX_UNION:
        return out_k.contiguous()

    out = torch.full((*out_k.shape[:-1], BUCKET_MAX_UNION), -1,
                     dtype=torch.int32, device=sel_per_q.device)
    out[..., :k] = out_k
    return out.contiguous()


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
