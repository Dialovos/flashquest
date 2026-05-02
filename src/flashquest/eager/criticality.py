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


def page_scores_int8(
    Q: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
) -> torch.Tensor:
    """Quest criticality computed directly from INT8 quant params.

    Identity: under per-page channel-wise asymmetric uint8 quantization
    (kv_quant._scale_mn_per_page_channel), K_mn is the per-page per-channel
    minimum and K_mn + 255 * K_scale is the maximum (exact, modulo the
    eps-clamp on constant channels). So:

        score(b, h_q, q, p) = sum_d max(Q[d] * K_mn[p, d], Q[d] * (K_mn[p, d] + 255 * K_scale[p, d]))

    GQA broadcast happens on the small (P, D) summary, not on the (S, D) cache.

    Args:
        Q: (B, H_q, S_q, D) bf16/fp16/fp32.
        K_scale: (B, H_kv, P, D) bf16 — per-page per-channel quant scale.
        K_mn: (B, H_kv, P, D) bf16 — per-page per-channel quant min.

    Returns:
        (B, H_q, S_q, P) fp32 page-criticality scores.
    """
    B, H_q, S_q, D = Q.shape
    H_kv = K_scale.shape[1]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
    n_rep = H_q // H_kv

    Kmn_f = K_mn.float()
    Kmx_f = Kmn_f + 255.0 * K_scale.float()
    if n_rep > 1:
        Kmn_f = Kmn_f.repeat_interleave(n_rep, dim=1)
        Kmx_f = Kmx_f.repeat_interleave(n_rep, dim=1)
    Q_e = Q.float().unsqueeze(3)
    Kmn_e = Kmn_f.unsqueeze(2)
    Kmx_e = Kmx_f.unsqueeze(2)
    cand_mx = Q_e * Kmx_e
    cand_mn = Q_e * Kmn_e
    return torch.maximum(cand_mx, cand_mn).sum(dim=-1)
