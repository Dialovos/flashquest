"""Sliding-window negative-log-likelihood / perplexity over a corpus."""
from __future__ import annotations

import math

import torch


@torch.no_grad()
def perplexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    window: int,
    stride: int,
    device: str = "cuda",
) -> float:
    """Sliding-window perplexity following the standard HF recipe.

    For long sequences, slide a window of `window` tokens with `stride` step,
    computing NLL only over the *new* tokens at each step (so we don't
    double-count overlap). Returns ppl = exp(mean NLL).
    """
    assert input_ids.ndim == 1, "pass a single 1D token tensor"
    n = input_ids.shape[0]
    nlls: list[torch.Tensor] = []
    counts = 0
    prev_end = 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        target_len = end - prev_end
        ids = input_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, : -target_len] = -100
        out = model(input_ids=ids, labels=labels)
        nlls.append(out.loss * target_len)
        counts += target_len
        prev_end = end
        if end == n:
            break
    avg_nll = torch.stack(nlls).sum() / counts
    return math.exp(avg_nll.item())
