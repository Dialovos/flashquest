"""Drop-in monkeypatch of HF LlamaAttention.forward to use Quest-eager SDPA.

Tracked against transformers 4.57.6's LlamaAttention.forward signature.
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from .attention import quest_eager_sdpa


def make_quest_eager_forward(
    *,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
    """Build a forward function bound to the given Quest knobs."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[object] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Decode (q_len == 1) needs no causal mask vs the cache; multi-query input is causal.
        is_causal = q.shape[2] > 1

        attn_output = quest_eager_sdpa(
            q, k, v,
            page_size=page_size,
            retention=retention,
            num_sinks=num_sinks,
            window_pages=window_pages,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def patch_llama_for_quest_eager(
    model: torch.nn.Module,
    *,
    retention: float,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward in `model` with the Quest-eager version."""
    fwd = make_quest_eager_forward(
        retention=retention,
        num_sinks=num_sinks,
        window_pages=window_pages,
        page_size=page_size,
    )
    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_eager: no LlamaAttention modules found")
