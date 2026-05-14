"""Persistent INT8 KV cache for HF transformers integration.

Layout:
- Completed pages stored as KIVI-style uint8 with per-page channel-wise K
  and per-token V (Phase 3 layout).
- A small BF16 staging buffer (`K_partial`, `V_partial`) holds the current
  incomplete page; once full, it's quantized and flushed.
- The patched LlamaAttention forward calls `update_quantized(K_new, V_new,
  layer_idx)` after RoPE and reads `get_views(layer_idx)` for the int8
  views.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from transformers.cache_utils import Cache


class PersistentInt8KVCache(Cache):
    kv_bits = 8

    def __init__(
        self,
        *,
        batch_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        page_size: int = 64,
        device: str | torch.device = "cuda",
    ):
        # Skip Cache.__init__ — its layers/layer_class_to_replicate API is
        # incompatible with our pre-allocated uint8 buffers. We satisfy the
        # contract attributes manually below.
        self.layers: list = []
        self.layer_class_to_replicate = None
        self.offloading = False
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.page_size = page_size
        max_pages = (max_seq_len + page_size - 1) // page_size
        self.max_pages = max_pages
        dev = torch.device(device)

        shape_kv = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim)
        shape_kpage = (num_layers, batch_size, num_kv_heads, max_pages, head_dim)
        shape_vtok = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_partial = (num_layers, batch_size, num_kv_heads, page_size, head_dim)

        self.K_uint8 = torch.zeros(shape_kv, dtype=torch.uint8, device=dev)
        self.V_uint8 = torch.zeros(shape_kv, dtype=torch.uint8, device=dev)
        self.K_scale = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.K_mn = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.V_scale = torch.zeros(shape_vtok, dtype=torch.bfloat16, device=dev)
        self.V_mn = torch.zeros(shape_vtok, dtype=torch.bfloat16, device=dev)
        self.K_partial = torch.zeros(shape_partial, dtype=torch.bfloat16, device=dev)
        self.V_partial = torch.zeros(shape_partial, dtype=torch.bfloat16, device=dev)

        self._seen_tokens = [0] * num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._seen_tokens[layer_idx]

    def get_max_length(self) -> int:
        return self.max_seq_len

    def update_quantized(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Append K_new, V_new (B, H_kv, S_new, D) bf16 to layer_idx's cache.

        Completes any partial page first, then bulk-quantizes whole pages,
        then stages any remaining partial page in BF16.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        from flashquest.kernel.kv_quant import quantize_k, quantize_v

        seen = self._seen_tokens[layer_idx]
        S_new = K_new.shape[2]
        if seen + S_new > self.max_seq_len:
            raise RuntimeError(
                f"PersistentInt8KVCache: seen+new={seen + S_new} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        page_size = self.page_size

        partial_len = seen % page_size
        partial_K = self.K_partial[layer_idx, :, :, :partial_len, :]
        partial_V = self.V_partial[layer_idx, :, :, :partial_len, :]
        K_stream = torch.cat([partial_K, K_new], dim=2)
        V_stream = torch.cat([partial_V, V_new], dim=2)

        page_start_token = seen - partial_len
        total_stream_len = K_stream.shape[2]
        n_complete_pages = total_stream_len // page_size
        complete_len = n_complete_pages * page_size
        new_partial_len = total_stream_len - complete_len

        if n_complete_pages > 0:
            K_full = K_stream[:, :, :complete_len, :]
            V_full = V_stream[:, :, :complete_len, :]
            K_uint8, K_scale, K_mn = quantize_k(K_full, page_size=page_size)
            V_uint8, V_scale, V_mn = quantize_v(V_full)

            tok_start = page_start_token
            tok_end = page_start_token + complete_len
            page_idx_start = tok_start // page_size
            page_idx_end = page_idx_start + n_complete_pages

            self.K_uint8[layer_idx, :, :, tok_start:tok_end, :] = K_uint8
            self.V_uint8[layer_idx, :, :, tok_start:tok_end, :] = V_uint8
            self.K_scale[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_scale
            self.K_mn[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_mn
            self.V_scale[layer_idx, :, :, tok_start:tok_end, :] = V_scale
            self.V_mn[layer_idx, :, :, tok_start:tok_end, :] = V_mn

        if new_partial_len > 0:
            self.K_partial[layer_idx, :, :, :new_partial_len, :] = K_stream[:, :, complete_len:, :]
            self.V_partial[layer_idx, :, :, :new_partial_len, :] = V_stream[:, :, complete_len:, :]
        if new_partial_len < page_size:
            self.K_partial[layer_idx, :, :, new_partial_len:, :].zero_()
            self.V_partial[layer_idx, :, :, new_partial_len:, :].zero_()

        self._seen_tokens[layer_idx] = seen + S_new

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return slices of the cache for the current sequence length.

        Returns dict with:
            seq_len, completed_len, partial_len: ints
            K_uint8, V_uint8: (B, H_kv, completed_len, D) uint8
            K_scale, K_mn: (B, H_kv, num_complete_pages, D) bf16
            V_scale, V_mn: (B, H_kv, completed_len, 1) bf16
            K_partial, V_partial: (B, H_kv, partial_len, D) bf16
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        seen = self._seen_tokens[layer_idx]
        page_size = self.page_size
        partial_len = seen % page_size
        completed_len = seen - partial_len
        n_complete_pages = completed_len // page_size

        return {
            "seq_len": seen,
            "completed_len": completed_len,
            "partial_len": partial_len,
            "K_uint8": self.K_uint8[layer_idx, :, :, :completed_len, :],
            "V_uint8": self.V_uint8[layer_idx, :, :, :completed_len, :],
            "K_scale": self.K_scale[layer_idx, :, :, :n_complete_pages, :],
            "K_mn": self.K_mn[layer_idx, :, :, :n_complete_pages, :],
            "V_scale": self.V_scale[layer_idx, :, :, :completed_len, :],
            "V_mn": self.V_mn[layer_idx, :, :, :completed_len, :],
            "K_partial": self.K_partial[layer_idx, :, :, :partial_len, :],
            "V_partial": self.V_partial[layer_idx, :, :, :partial_len, :],
        }

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError(
            "PersistentInt8KVCache.update should not be called directly; "
            "the flashquest patch uses update_quantized() + get_views()."
        )
