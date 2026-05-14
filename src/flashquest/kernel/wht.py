"""Normalized fast Walsh-Hadamard transform along the last dimension.

Convention: H @ H.T / N = I, so wht(wht(x)) = x exactly. No /N scale needed
at decode. Used by Phase 7 TurboQuant: rotates K, V along head_dim before
quantization to Gaussianize the per-block distribution.

The butterfly does log2(D) stages of (a, b) → (a + b, a - b), then divides
the final tensor by sqrt(D) once. At D=128 this is 7 stages.
"""
from __future__ import annotations

import math

import torch


def wht_along_head_dim(x: torch.Tensor) -> torch.Tensor:
    """Walsh-Hadamard transform along the last dimension.

    Args:
        x: any tensor with last-dim a power of 2.

    Returns:
        Same shape, dtype preserved. Self-inverse.
    """
    D = x.shape[-1]
    if D & (D - 1) != 0 or D == 0:
        raise ValueError(f"head_dim must be a positive power of 2, got {D}")

    out = x.contiguous()
    h = 1
    while h < D:
        prefix_shape = out.shape[:-1]
        out = out.reshape(*prefix_shape, D // (2 * h), 2, h)
        a = out[..., 0, :]
        b = out[..., 1, :]
        out = torch.stack([a + b, a - b], dim=-2)
        out = out.reshape(*prefix_shape, D)
        h *= 2

    return out / math.sqrt(D)
