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
from ..eager.selection import build_compact_selection, select_pages_vectorized
from ..kernel import flash_attn_sparse_fwd
from ..kernel.kv_quant import (
    dequantize_k, dequantize_k_int4, dequantize_k_turbo,
    dequantize_v, dequantize_v_int4, dequantize_v_turbo,
)
from ..kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd
from ..kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact
from ..kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd


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


def make_quest_persistent_forward(
    *,
    cache,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
    use_compact_kernel: bool = False,
):
    kv_bits = getattr(cache, "kv_bits", 8)
    head_dim = cache.head_dim

    # Phase 8a: precompute static k_max + BUCKET_MAX (no per-step .item())
    max_seq_len = getattr(cache, "max_seq_len", 32768)
    P_max = max_seq_len // page_size
    if isinstance(retention, float):
        retention_max = float(retention)
    else:
        retention_max = float(retention.max().item())
    k_max_static = math.ceil(retention_max * P_max)
    bucket_max_static = k_max_static + num_sinks + window_pages

    if use_compact_kernel and kv_bits != 4:
        raise NotImplementedError(
            f"Phase 8a: use_compact_kernel only supports kv_bits=4; got {kv_bits}. "
            f"INT8/Turbo compact kernels deferred to Phase 8b."
        )

    if kv_bits == 3:
        def _dequant_k_from_views(views):
            return dequantize_k_turbo(
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                head_dim=head_dim,
            )

        def _dequant_v_from_views(views):
            return dequantize_v_turbo(
                views["V_msb"], views["V_lsb"], views["V_scale_turbo"],
                head_dim=head_dim,
            )

        def _criticality_scores(q, views):
            return page_scores_int4_fast(q, views["K_scale_raw"], views["K_mn_raw"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_turbo_fwd(
                q,
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                views["V_msb"], views["V_lsb"], views["V_scale_turbo"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )
    elif kv_bits == 4:
        def _dequant_k_from_views(views):
            return dequantize_k_int4(
                views["K_packed"], views["K_scale"], views["K_mn"], page_size=page_size,
            )

        def _dequant_v_from_views(views):
            return dequantize_v_int4(views["V_packed"], views["V_scale"], views["V_mn"])

        def _criticality_scores(q, views):
            return page_scores_int4_fast(q, views["K_scale"], views["K_mn"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_int4_fwd(
                q,
                views["K_packed"], views["K_scale"], views["K_mn"],
                views["V_packed"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )

        def _sparse_fwd_call_compact(q, views, sel_compact):
            return flash_attn_sparse_int4_fwd_compact(
                q,
                views["K_packed"], views["K_scale"], views["K_mn"],
                views["V_packed"], views["V_scale"], views["V_mn"],
                selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
            )
    elif kv_bits == 8:
        def _dequant_k_from_views(views):
            return dequantize_k(
                views["K_uint8"], views["K_scale"], views["K_mn"], page_size=page_size,
            )

        def _dequant_v_from_views(views):
            return dequantize_v(views["V_uint8"], views["V_scale"], views["V_mn"])

        def _criticality_scores(q, views):
            return page_scores_int8_fast(q, views["K_scale"], views["K_mn"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_fwd(
                q,
                views["K_uint8"], views["K_scale"], views["K_mn"],
                views["V_uint8"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=page_size, return_lse=True,
            )
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
            K_full = torch.cat([_dequant_k_from_views(views), views["K_partial"]], dim=2)
            V_full = torch.cat([_dequant_v_from_views(views), views["V_partial"]], dim=2)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, K_full, V_full, is_causal=True, enable_gqa=True,
            )
        else:
            partial_len = views["partial_len"]
            completed_len = views["completed_len"]

            if completed_len == 0:
                attn_output, _ = _bf16_dense_attn_with_lse(
                    q, views["K_partial"], views["V_partial"],
                )
            else:
                B, H_q, _, _ = q.shape
                H_kv = cache.num_kv_heads
                n_rep = H_q // H_kv
                pattern_per_q = head_pattern_layer.to(q.device).repeat_interleave(n_rep)
                retention_per_q = torch.where(
                    pattern_per_q,
                    torch.full((H_q,), retention, device=q.device),
                    torch.zeros(H_q, device=q.device),
                )
                scores = _criticality_scores(q, views)
                sel = select_pages_vectorized(
                    scores, retention=retention_per_q,
                    num_sinks=num_sinks, window_pages=window_pages,
                    k_max_static=k_max_static,
                )
                if use_compact_kernel:
                    sel_compact = build_compact_selection(
                        sel, BUCKET_MAX=bucket_max_static,
                    )
                    O_sparse, lse_sparse = _sparse_fwd_call_compact(
                        q, views, sel_compact,
                    )
                else:
                    O_sparse, lse_sparse = _sparse_fwd_call(q, views, sel)

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
    retention: float = 0.20,
    num_sinks: int = 4,
    window_pages: int = 2,
    page_size: int = 64,
    use_compact_kernel: bool = False,
) -> None:
    """Replace every LlamaAttention.forward with the persistent-cache version.

    Phase 10 (2026-05-09): retention default bumped 0.25 → 0.20 after the
    retention sweep + RULER quality test. retention=0.20 yields ~1.05× decode
    speedup at 32k vs the prior 0.25 default while holding RULER NIAH single
    100/100 + multivalue 19/20 (95%) — same quality as the 0.25 baseline.
    See docs/PHASES/phase-10-notes.md for the full quality/speed curve.
    """
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
                use_compact_kernel=use_compact_kernel,
            )
            module.forward = fwd.__get__(module, type(module))
            n_patched += 1
    if n_patched == 0:
        raise RuntimeError("patch_llama_for_quest_persistent: no LlamaAttention modules")
    if n_patched != num_layers:
        raise ValueError(
            f"head_pattern has {num_layers} layers but model has {n_patched}"
        )
