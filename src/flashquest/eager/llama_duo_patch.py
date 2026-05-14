"""HF Llama monkeypatch: per-layer DuoAttention dispatch.

Each layer reads its own row from the head_pattern tensor and dispatches
through quest_duo_eager_sdpa. Tracked against transformers 4.57.6's
LlamaAttention.forward signature (same as Phase 1's patch).
"""
from __future__ import annotations

from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from ..duo.dispatch import quest_duo_eager_sdpa


def make_quest_duo_forward(
    *,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
    """Build a forward function bound to one layer's head pattern + Quest knobs.

    head_pattern_layer: (H_kv,) bool tensor for THIS layer.
    """

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

        is_causal = q.shape[2] > 1

        attn_output = quest_duo_eager_sdpa(
            q, k, v,
            head_pattern=head_pattern_layer,
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


def patch_llama_for_quest_duo(
    model: torch.nn.Module,
    *,
    head_pattern: torch.Tensor,
    retention: float = 0.20,                 # Phase 10 default (was 0.25)
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward in `model` with the Duo version.

    Args:
        head_pattern: (num_layers, num_kv_heads) bool tensor.
    """
    if head_pattern.ndim != 2:
        raise ValueError(
            f"head_pattern must be 2D (num_layers, num_kv_heads); got {tuple(head_pattern.shape)}"
        )
    num_layers, num_kv = head_pattern.shape

    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            li = module.layer_idx
            if li >= num_layers:
                raise ValueError(
                    f"head_pattern has {num_layers} layers but model layer_idx={li}"
                )
            layer_pattern = head_pattern[li]
            fwd = make_quest_duo_forward(
                head_pattern_layer=layer_pattern,
                retention=retention,
                num_sinks=num_sinks,
                window_pages=window_pages,
                page_size=page_size,
            )
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1

    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_duo: no LlamaAttention modules found")
    if n_patched != num_layers:
        raise ValueError(
            f"head_pattern has {num_layers} layers but model has {n_patched} attention modules"
        )
