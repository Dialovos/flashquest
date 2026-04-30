"""Per-page approximate-attention score from page summaries (Quest criticality)."""
from __future__ import annotations

import torch


def page_scores(
    Q: torch.Tensor,
    page_min: torch.Tensor,
    page_max: torch.Tensor,
) -> torch.Tensor:
    """Quest criticality: per-page upper bound on Q . K.

    For each (b, h, q, p), returns sum_d max(Q_d * page_max[p, d], Q_d * page_min[p, d]).
    This upper-bounds max_t (Q . K[t]) over tokens t in page p.

    Args:
        Q: (B, H, S_q, D) queries.
        page_min: (B, H, num_pages, D) per-page channel min of K.
        page_max: (B, H, num_pages, D) per-page channel max of K.

    Returns:
        (B, H, S_q, num_pages) page-criticality scores.
    """
    Q_e = Q.unsqueeze(3)
    pmin_e = page_min.unsqueeze(2)
    pmax_e = page_max.unsqueeze(2)
    cand_max = Q_e * pmax_e
    cand_min = Q_e * pmin_e
    scores = torch.maximum(cand_max, cand_min).sum(dim=-1)
    return scores
