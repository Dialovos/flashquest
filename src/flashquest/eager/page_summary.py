"""Per-page channel-wise min/max statistics of K, the criticality signal Quest uses."""
from __future__ import annotations

import torch


def compute_page_summary(
    K: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-page per-channel min and max of K.

    Args:
        K: (B, H, S, D) keys (head-major).
        page_size: tokens per page (must be > 0).

    Returns:
        (page_min, page_max) each shaped (B, H, num_pages, D), where
        num_pages = ceil(S / page_size). Tail pages with fewer than
        page_size valid tokens summarise only the valid prefix.
    """
    assert page_size > 0
    B, H, S, D = K.shape
    num_pages = (S + page_size - 1) // page_size
    pad = num_pages * page_size - S

    if pad > 0:
        K_min_padded = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("inf"))
        K_max_padded = torch.nn.functional.pad(K, (0, 0, 0, pad), value=float("-inf"))
    else:
        K_min_padded = K
        K_max_padded = K

    K_min_pages = K_min_padded.view(B, H, num_pages, page_size, D)
    K_max_pages = K_max_padded.view(B, H, num_pages, page_size, D)

    page_min = K_min_pages.min(dim=3).values
    page_max = K_max_pages.max(dim=3).values
    return page_min, page_max
