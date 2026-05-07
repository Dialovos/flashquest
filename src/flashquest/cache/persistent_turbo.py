"""Phase 7 — Persistent TurboQuant KV cache (K3-V3, both 3-bit bit-split).

Mirrors `PersistentInt4KVCache` but with eight storage tensors:
  K_msb, K_lsb       : 3-bit K split into 1-bit MSB plane + 2-bit LSB plane
  K_scale_turbo      : per-token scalar for kernel dequant
  K_scale_raw, K_mn_raw : per-page channel-wise from un-rotated K, for criticality
  V_msb, V_lsb       : 3-bit V same bit-split layout as K (Phase 7 task 11b
                       upgrade from 2-bit; cleared RULER multivalue ≥85 %)
  V_scale_turbo      : per-token scalar for kernel dequant

`kv_bits = 3` is read by the dispatcher in `eager/llama_persistent_patch.py`.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache


class PersistentTurboKVCache(Cache):
    kv_bits = 3

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
        if head_dim % 8 != 0:
            raise ValueError(
                f"PersistentTurboKVCache requires head_dim multiple of 8 "
                f"(MSB plane packs 8/byte); got {head_dim}"
            )
        if head_dim & (head_dim - 1) != 0:
            raise ValueError(
                f"PersistentTurboKVCache requires head_dim power of 2 (WHT); got {head_dim}"
            )
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

        D_msb = head_dim // 8
        D_lsb = head_dim // 4
        shape_msb = (num_layers, batch_size, num_kv_heads, max_seq_len, D_msb)
        shape_lsb = (num_layers, batch_size, num_kv_heads, max_seq_len, D_lsb)
        shape_kscale_t = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_vscale_t = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_kpage = (num_layers, batch_size, num_kv_heads, max_pages, head_dim)
        shape_partial = (num_layers, batch_size, num_kv_heads, page_size, head_dim)

        self.K_msb = torch.zeros(shape_msb, dtype=torch.uint8, device=dev)
        self.K_lsb = torch.zeros(shape_lsb, dtype=torch.uint8, device=dev)
        self.K_scale_turbo = torch.zeros(shape_kscale_t, dtype=torch.bfloat16, device=dev)
        self.K_scale_raw = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.K_mn_raw = torch.zeros(shape_kpage, dtype=torch.bfloat16, device=dev)
        self.V_msb = torch.zeros(shape_msb, dtype=torch.uint8, device=dev)
        self.V_lsb = torch.zeros(shape_lsb, dtype=torch.uint8, device=dev)
        self.V_scale_turbo = torch.zeros(shape_vscale_t, dtype=torch.bfloat16, device=dev)
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
        """Append K_new, V_new (B, H_kv, S_new, D) bf16 to layer_idx's cache."""
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        from flashquest.kernel.kv_quant import quantize_k_turbo, quantize_v_turbo

        seen = self._seen_tokens[layer_idx]
        S_new = K_new.shape[2]
        if seen + S_new > self.max_seq_len:
            raise RuntimeError(
                f"PersistentTurboKVCache: seen+new={seen + S_new} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        page_size = self.page_size

        partial_len = seen % page_size
        if partial_len > 0:
            head_K = self.K_partial[layer_idx, :, :, :partial_len, :]
            K_full = torch.cat([head_K, K_new], dim=2)
            V_full = torch.cat(
                [self.V_partial[layer_idx, :, :, :partial_len, :], V_new], dim=2,
            )
        else:
            K_full = K_new
            V_full = V_new

        total_stream_len = K_full.shape[2]
        n_complete_pages = total_stream_len // page_size
        complete_len = n_complete_pages * page_size

        if n_complete_pages > 0:
            K_complete = K_full[:, :, :complete_len, :]
            V_complete = V_full[:, :, :complete_len, :]
            K_msb, K_lsb, K_scale_t, K_scale_r, K_mn_r = quantize_k_turbo(
                K_complete, page_size=page_size,
            )
            V_msb, V_lsb, V_scale_t = quantize_v_turbo(V_complete)

            tok_start = seen - partial_len
            tok_end = tok_start + complete_len
            page_idx_start = tok_start // page_size
            page_idx_end = page_idx_start + n_complete_pages

            self.K_msb[layer_idx, :, :, tok_start:tok_end, :] = K_msb
            self.K_lsb[layer_idx, :, :, tok_start:tok_end, :] = K_lsb
            self.K_scale_turbo[layer_idx, :, :, tok_start:tok_end, :] = K_scale_t
            self.K_scale_raw[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_scale_r
            self.K_mn_raw[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_mn_r
            self.V_msb[layer_idx, :, :, tok_start:tok_end, :] = V_msb
            self.V_lsb[layer_idx, :, :, tok_start:tok_end, :] = V_lsb
            self.V_scale_turbo[layer_idx, :, :, tok_start:tok_end, :] = V_scale_t

        new_partial_len = total_stream_len - complete_len
        if new_partial_len < page_size:
            self.K_partial[layer_idx, :, :, :new_partial_len, :] = K_full[:, :, complete_len:, :]
            self.V_partial[layer_idx, :, :, :new_partial_len, :] = V_full[:, :, complete_len:, :]

        self._seen_tokens[layer_idx] = seen + S_new

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
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
            "K_msb": self.K_msb[layer_idx, :, :, :completed_len, :],
            "K_lsb": self.K_lsb[layer_idx, :, :, :completed_len, :],
            "K_scale_turbo": self.K_scale_turbo[layer_idx, :, :, :completed_len, :],
            "K_scale_raw": self.K_scale_raw[layer_idx, :, :, :n_complete_pages, :],
            "K_mn_raw": self.K_mn_raw[layer_idx, :, :, :n_complete_pages, :],
            "V_msb": self.V_msb[layer_idx, :, :, :completed_len, :],
            "V_lsb": self.V_lsb[layer_idx, :, :, :completed_len, :],
            "V_scale_turbo": self.V_scale_turbo[layer_idx, :, :, :completed_len, :],
            "K_partial": self.K_partial[layer_idx, :, :, :partial_len, :],
            "V_partial": self.V_partial[layer_idx, :, :, :partial_len, :],
        }

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "PersistentTurboKVCache.update is not used; the patched forward "
            "calls update_quantized directly."
        )
