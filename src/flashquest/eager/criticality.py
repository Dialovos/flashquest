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


def page_scores_int8_fast(
    Q: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
) -> torch.Tensor:
    """Fast equivalent of page_scores_int8 — two batched matmuls instead of
    materializing the (B, H_q, S_q, P, D) intermediate.

    Identity (since K_scale ≥ 0 elementwise after kv_quant's eps clamp):

        max(Q[d]·K_mn[p,d], Q[d]·(K_mn[p,d] + 255·K_scale[p,d]))
          = Q[d]·K_mn[p,d] + relu(255·Q[d]·K_scale[p,d])
          = Q[d]·K_mn[p,d] + 255·relu(Q[d])·K_scale[p,d]

    Sum over D:

        score(p) = Q · K_mn[p]  +  255 · relu(Q) · K_scale[p]

    Two batched matmuls + one clamp(min=0). Memory traffic drops ~12× vs
    page_scores_int8 (no (B, H_q, S_q, P, D) fp32 intermediate).

    Args:
        Q: (B, H_q, S_q, D) bf16/fp16/fp32.
        K_scale: (B, H_kv, P, D) bf16 — per-page per-channel quant scale (≥0).
        K_mn: (B, H_kv, P, D) bf16 — per-page per-channel quant min.

    Returns:
        (B, H_q, S_q, P) fp32 page-criticality scores.
    """
    B, H_q, S_q, D = Q.shape
    H_kv, P = K_scale.shape[1], K_scale.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
    n_rep = H_q // H_kv

    # Group GQA in the matmul instead of materializing repeated K — view as
    # (B, H_kv, n_rep * S_q, D) and let bmm broadcast.
    Q_f = Q.float()
    Q_g = Q_f.view(B, H_kv, n_rep * S_q, D)
    Q_pos_g = Q_g.clamp(min=0)

    Kmn_f = K_mn.float()
    Kscale_f = K_scale.float()

    # (B, H_kv, n_rep*S_q, D) @ (B, H_kv, D, P) -> (B, H_kv, n_rep*S_q, P)
    term1 = torch.matmul(Q_g, Kmn_f.transpose(-1, -2))
    term2 = torch.matmul(Q_pos_g, Kscale_f.transpose(-1, -2)) * 255.0
    scores_g = term1 + term2  # (B, H_kv, n_rep * S_q, P)
    return scores_g.view(B, H_q, S_q, P)


def page_scores_int4_fast(
    Q: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
) -> torch.Tensor:
    """Algebraic Quest criticality for INT4 KV.

    Identity: max(Q·K_mn, Q·K_mx) = Q·K_mn + 15·relu(Q)·K_scale,
    since K_scale ≥ 0 and page_max ≡ K_mn + 15·K_scale (asymmetric INT4 KIVI).
    Two-matmul reformulation; same shape contracts as page_scores_int8_fast,
    only the constant changes (15 vs 255).

    Args:
        Q: (B, H_q, S_q, D) bf16/fp16/fp32.
        K_scale: (B, H_kv, P, D) bf16 — per-page channel-wise scale (≥0).
        K_mn:    (B, H_kv, P, D) bf16 — per-page channel-wise minimum.

    Returns:
        (B, H_q, S_q, P) fp32 page-criticality scores.
    """
    B, H_q, S_q, D = Q.shape
    H_kv, P = K_scale.shape[1], K_scale.shape[2]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q={H_q} must be divisible by H_kv={H_kv}")
    n_rep = H_q // H_kv

    Q_f = Q.float()
    Q_g = Q_f.view(B, H_kv, n_rep * S_q, D)
    Q_pos_g = Q_g.clamp(min=0)
    Kmn_f = K_mn.float()
    Kscale_f = K_scale.float()

    term1 = torch.matmul(Q_g, Kmn_f.transpose(-1, -2))
    term2 = torch.matmul(Q_pos_g, Kscale_f.transpose(-1, -2)) * 15.0
    scores_g = term1 + term2
    return scores_g.view(B, H_q, S_q, P)
