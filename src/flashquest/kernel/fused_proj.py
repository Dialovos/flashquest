"""Phase 8a — fused projections wrapper.

Calls 3 (or 2) AutoAWQ Linear forward passes in one Python function, returning
sliced output views. The win is launch-overhead reduction in the per-step
hot path — not a true GEMM fusion (AutoAWQ's INT4 GEMM is already hand-tuned;
re-implementing in Triton would not beat it at decode batch=1; Phase 6 notes
line 148 already established this).

Reads source weight tensors in place — no stacked tensor materialization,
zero extra VRAM (codex r3 finding #3).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def fused_qkv_proj(
    hidden: torch.Tensor,
    q_proj: nn.Module,
    k_proj: nn.Module,
    v_proj: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run q_proj, k_proj, v_proj in sequence; return (Q, K, V).

    Each linear is an AutoAWQ-quantized nn.Linear (q_proj.qweight, .scales, .qzeros).
    Their forward passes call the AutoAWQ INT4 GEMM kernel.

    Args:
        hidden: (B, S, in_features) fp16 or bf16
        q_proj, k_proj, v_proj: AWQ Linear modules

    Returns:
        Q (B, S, N_q), K (B, S, N_k), V (B, S, N_v) — same dtype as hidden
    """
    q = q_proj(hidden)
    k = k_proj(hidden)
    v = v_proj(hidden)
    return q, k, v


def fused_gate_up_proj(
    hidden: torch.Tensor,
    gate_proj: nn.Module,
    up_proj: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run gate_proj, up_proj in sequence; return (gate, up). Caller applies SwiGLU.

    Same launch-reduction win as fused_qkv_proj.
    """
    gate = gate_proj(hidden)
    up = up_proj(hidden)
    return gate, up
