"""Eager Quest-style sparse attention with INT8 KV.

Composes:
  1. Dequantize K and V from uint8 to bf16.
  2. Use the existing quest_eager_sdpa over the dequantized tensors.

Splitting like this keeps the algorithm identical to Phase 1 — the only
difference is the quantization round-trip on K/V before the attention.
This is the *correctness oracle* the Triton sparse kernel must match.
"""
from __future__ import annotations

import torch

from ..kernel.kv_quant import dequantize_k, dequantize_v
from .attention import quest_eager_sdpa


def quest_eager_sparse_int8(
    Q: torch.Tensor,
    K_uint8: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_uint8: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    page_size: int = 64,
    retention: float = 1.0,
    num_sinks: int = 4,
    window_pages: int = 2,
    is_causal: bool = True,
) -> torch.Tensor:
    """Sparse attention over INT8-quantized KV.

    Args:
        Q: (B, H_q, S_q, D) bf16.
        K_uint8: (B, H_kv, S_kv, D) uint8.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16.
        V_uint8: (B, H_kv, S_kv, D) uint8.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16.

    Returns:
        Output (B, H_q, S_q, D) bf16.
    """
    K = dequantize_k(K_uint8, K_scale, K_mn, page_size)
    V = dequantize_v(V_uint8, V_scale, V_mn)
    return quest_eager_sdpa(
        Q, K, V,
        page_size=page_size,
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        is_causal=is_causal,
    )
