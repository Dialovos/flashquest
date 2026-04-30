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
        raise NotImplementedError("filled in Task 3")

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError("filled in Task 4")

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
