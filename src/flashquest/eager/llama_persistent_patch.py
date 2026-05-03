"""HF Llama monkeypatch with persistent INT8 KV cache + fused DuoAttention.

Prefill (S_q > 1): writes K/V to the persistent cache, dequantizes the
full cache via Phase 3's dequant pair, runs Phase 4's BF16 eager Duo path.
Decode (S_q = 1): writes K/V to cache, reads int8 views, runs the fused
dispatch over completed pages, then merges with a tiny BF16 dense
attention over the partial-page tail via online softmax (LSE).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

from ..cache.persistent_int8 import PersistentInt8KVCache
from ..eager.criticality import page_scores_int4_fast, page_scores_int8_fast
from ..eager.selection import select_pages_vectorized
from ..kernel import flash_attn_sparse_fwd
from ..kernel.kv_quant import (
    dequantize_k, dequantize_k_int4, dequantize_v, dequantize_v_int4,
)
from ..kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd


def _bf16_dense_attn_with_lse(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tiny dense BF16 attention for the partial-page tail. Returns (O, lse)
    where lse is in nats. Q: (B, H_q, 1, D); K, V: (B, H_kv, S_partial, D)."""
    B, H_q, _, D = Q.shape
    H_kv = K.shape[1]
    n_rep = H_q // H_kv
    K_full = K.repeat_interleave(n_rep, dim=1)
    V_full = V.repeat_interleave(n_rep, dim=1)
    sm_scale = 1.0 / math.sqrt(D)
    qk = (Q.float() @ K_full.float().transpose(-1, -2)) * sm_scale  # (B, H_q, 1, S_partial)
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    l = p.sum(dim=-1, keepdim=True)
    O = (p @ V_full.float()) / l
    lse = (m + torch.log(l)).squeeze(-1)  # (B, H_q, 1)
    return O.to(torch.bfloat16), lse


def _merge_two_attentions(
    O_a: torch.Tensor, lse_a: torch.Tensor,
    O_b: torch.Tensor, lse_b: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge of two partial attention results sharing Q.

    lse_a, lse_b shape: (B, H_q, 1). O_a, O_b shape: (B, H_q, 1, D).
    """
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m).unsqueeze(-1)  # (B, H_q, 1, 1)
    wb = torch.exp(lse_b - m).unsqueeze(-1)
    return ((wa * O_a.float() + wb * O_b.float()) / (wa + wb)).to(O_a.dtype)


def _quest_duo_fused_with_lse(
    Q, K_storage, K_scale, K_mn, V_storage, V_scale, V_mn,
    *, head_pattern, page_size, retention, num_sinks, window_pages, kv_bits,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused dispatch returning (O, lse) — needed for online-softmax merge.

    kv_bits ∈ {4, 8} selects the criticality fast path and the sparse kernel.
    K_storage / V_storage are uint8 tensors:
        kv_bits=8 → shape (B, H_kv, S_kv, D)        (full uint8)
        kv_bits=4 → shape (B, H_kv, S_kv, D//2)     (packed 2-per-byte)
    """
    B, H_q, S_q, D = Q.shape
    _, H_kv, _, _ = K_storage.shape
    n_rep = H_q // H_kv
    pattern_per_q = head_pattern.to(Q.device).repeat_interleave(n_rep)
    retention_per_q = torch.where(
        pattern_per_q,
        torch.full((H_q,), retention, device=Q.device),
        torch.zeros(H_q, device=Q.device),
    )

    if kv_bits == 4:
        scores = page_scores_int4_fast(Q, K_scale, K_mn)
    elif kv_bits == 8:
        scores = page_scores_int8_fast(Q, K_scale, K_mn)
    else:
        raise ValueError(f"unsupported kv_bits={kv_bits!r}")

    sel = select_pages_vectorized(
        scores, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )

    if kv_bits == 4:
        O, lse = flash_attn_sparse_int4_fwd(
            Q, K_storage, K_scale, K_mn, V_storage, V_scale, V_mn,
            selection_mask=sel, page_size=page_size, return_lse=True,
        )
    else:
        O, lse = flash_attn_sparse_fwd(
            Q, K_storage, K_scale, K_mn, V_storage, V_scale, V_mn,
            selection_mask=sel, page_size=page_size, return_lse=True,
        )
    return O, lse


def make_quest_persistent_forward(
    *,
    cache,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
):
    kv_bits = getattr(cache, "kv_bits", 8)
    if kv_bits == 4:
        K_view_key = "K_packed"
        V_view_key = "V_packed"

        def _dequant_k(k_storage, k_scale, k_mn):
            return dequantize_k_int4(k_storage, k_scale, k_mn, page_size=page_size)

        def _dequant_v(v_storage, v_scale, v_mn):
            return dequantize_v_int4(v_storage, v_scale, v_mn)
    elif kv_bits == 8:
        K_view_key = "K_uint8"
        V_view_key = "V_uint8"

        def _dequant_k(k_storage, k_scale, k_mn):
            return dequantize_k(k_storage, k_scale, k_mn, page_size=page_size)

        def _dequant_v(v_storage, v_scale, v_mn):
            return dequantize_v(v_storage, v_scale, v_mn)
    else:
        raise ValueError(f"unsupported cache.kv_bits={kv_bits!r}")
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

        # Sparse Triton kernel + cache require bf16. AWQ models are fp16;
        # cast in, then cast back before o_proj.
        model_dtype = q.dtype
        if model_dtype != torch.bfloat16:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        S_q = q.shape[2]

        cache.update_quantized(k, v, layer_idx=self.layer_idx)
        views = cache.get_views(self.layer_idx)

        if S_q > 1:
            # Prefill: dense attention over the dequant'd cache.
            # Sparse selection at long S_q would balloon criticality intermediates;
            # we keep prefill dense (SPEC win condition is decode tok/s, not prefill).
            K_full = torch.cat(
                [
                    _dequant_k(views[K_view_key], views["K_scale"], views["K_mn"]),
                    views["K_partial"],
                ],
                dim=2,
            )
            V_full = torch.cat(
                [
                    _dequant_v(views[V_view_key], views["V_scale"], views["V_mn"]),
                    views["V_partial"],
                ],
                dim=2,
            )
            n_rep = q.shape[1] // K_full.shape[1]
            K_rep = K_full.repeat_interleave(n_rep, dim=1)
            V_rep = V_full.repeat_interleave(n_rep, dim=1)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, K_rep, V_rep, is_causal=True,
            )
        else:
            partial_len = views["partial_len"]
            completed_len = views["completed_len"]

            if completed_len == 0:
                attn_output, _ = _bf16_dense_attn_with_lse(
                    q, views["K_partial"], views["V_partial"],
                )
            else:
                O_sparse, lse_sparse = _quest_duo_fused_with_lse(
                    q, views[K_view_key], views["K_scale"], views["K_mn"],
                    views[V_view_key], views["V_scale"], views["V_mn"],
                    head_pattern=head_pattern_layer,
                    page_size=page_size, retention=retention,
                    num_sinks=num_sinks, window_pages=window_pages,
                    kv_bits=kv_bits,
                )
                if partial_len == 0:
                    attn_output = O_sparse
                else:
                    O_partial, lse_partial = _bf16_dense_attn_with_lse(
                        q, views["K_partial"], views["V_partial"],
                    )
                    attn_output = _merge_two_attentions(
                        O_sparse, lse_sparse,
                        O_partial, lse_partial,
                    )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1)
        if attn_output.dtype != model_dtype:
            attn_output = attn_output.to(model_dtype)
        attn_output = self.o_proj(attn_output)
        return attn_output, None

    return forward


def patch_llama_for_quest_persistent(
    model: torch.nn.Module,
    *,
    cache,
    head_pattern: torch.Tensor,
    retention: float = 0.25,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
) -> None:
    """Replace every LlamaAttention.forward with the persistent-cache version."""
    if head_pattern.ndim != 2:
        raise ValueError(
            f"head_pattern must be 2D (num_layers, num_kv_heads); got {tuple(head_pattern.shape)}"
        )
    num_layers = head_pattern.shape[0]
    n_patched = 0
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            li = module.layer_idx
            if li >= num_layers:
                raise ValueError(
                    f"head_pattern has {num_layers} layers but model layer_idx={li}"
                )
            fwd = make_quest_persistent_forward(
                cache=cache,
                head_pattern_layer=head_pattern[li].to("cuda"),
                retention=retention,
                num_sinks=num_sinks,
                window_pages=window_pages,
                page_size=page_size,
            )
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_persistent: no LlamaAttention modules")
    if n_patched != num_layers:
        raise ValueError(
            f"head_pattern has {num_layers} layers but model has {n_patched}"
        )
