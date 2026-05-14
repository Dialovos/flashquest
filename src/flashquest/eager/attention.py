"""Pure-PyTorch Quest-style sparse attention. Composes page-summary, criticality,
selection, and a masked-SDPA call. Decode-only sparse-selection logic; for
multi-query (prefill) inputs, retention=1.0 is assumed and we just call SDPA."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .criticality import page_scores
from .page_summary import compute_page_summary
from .selection import select_pages


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def quest_eager_sdpa(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    page_size: int = 64,
    retention: float = 1.0,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Quest-style sparse causal SDPA.

    Args:
        Q: (B, H_q, S_q, D)
        K: (B, H_kv, S_kv, D)
        V: (B, H_kv, S_kv, D)
        page_size, retention, num_sinks, window_pages: Quest knobs.
        is_causal: standard causal mask.

    Returns:
        Output (B, H_q, S_q, D).
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, S_kv, _ = K.shape
    assert H_q % H_kv == 0, "GQA: H_q must be a multiple of H_kv"
    n_rep = H_q // H_kv

    Kr = _repeat_kv(K, n_rep)
    Vr = _repeat_kv(V, n_rep)

    if retention >= 1.0 and num_sinks == 0 and window_pages == 0:
        return F.scaled_dot_product_attention(Q, Kr, Vr, is_causal=is_causal)

    page_min, page_max = compute_page_summary(Kr.float(), page_size)
    scores = page_scores(Q.float(), page_min, page_max)
    page_mask = select_pages(scores, retention, num_sinks, window_pages)

    P = page_mask.shape[-1]
    token_mask = (
        page_mask.unsqueeze(-1)
        .expand(B, H_q, S_q, P, page_size)
        .reshape(B, H_q, S_q, P * page_size)
    )
    token_mask = token_mask[..., :S_kv]

    if is_causal and S_q > 1:
        causal = torch.ones(S_q, S_kv, dtype=torch.bool, device=Q.device).tril(
            diagonal=S_kv - S_q
        )
        token_mask = token_mask & causal

    attn_bias = torch.zeros_like(token_mask, dtype=Q.dtype)
    attn_bias = attn_bias.masked_fill(~token_mask, float("-inf"))

    return F.scaled_dot_product_attention(
        Q, Kr, Vr,
        attn_mask=attn_bias,
        is_causal=False,
    )
