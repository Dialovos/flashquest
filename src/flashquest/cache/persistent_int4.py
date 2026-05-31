"""Phase 6 task 5 — Persistent INT4 KV cache (KIVI-style, packed 2-per-byte).

Mirrors persistent_int8.PersistentInt8KVCache. Differences:
- K_packed / V_packed have head_dim/2 storage (2 INT4 values per uint8 byte);
- update_quantized calls quantize_k_int4 / quantize_v_int4;
- kv_bits = 4 (read by the dispatcher in llama_persistent_patch).
"""
from __future__ import annotations

import torch
from transformers.cache_utils import Cache


class PersistentInt4KVCache(Cache):
    kv_bits = 4

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
        if head_dim % 2 != 0:
            raise ValueError(
                f"PersistentInt4KVCache requires even head_dim "
                f"(2 INT4 per byte); got {head_dim}"
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

        head_dim_packed = head_dim // 2
        shape_kv_packed = (num_layers, batch_size, num_kv_heads, max_seq_len, head_dim_packed)
        shape_kpage = (num_layers, batch_size, num_kv_heads, max_pages, head_dim)
        shape_vtok = (num_layers, batch_size, num_kv_heads, max_seq_len, 1)
        shape_partial = (num_layers, batch_size, num_kv_heads, page_size, head_dim)

        self.K_packed = torch.zeros(shape_kv_packed, dtype=torch.uint8, device=dev)
        self.V_packed = torch.zeros(shape_kv_packed, dtype=torch.uint8, device=dev)
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

        Mirrors PersistentInt8KVCache.update_quantized but uses INT4 quant primitives.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4

        seen = self._seen_tokens[layer_idx]
        S_new = K_new.shape[2]
        if seen + S_new > self.max_seq_len:
            raise RuntimeError(
                f"PersistentInt4KVCache: seen+new={seen + S_new} exceeds "
                f"max_seq_len={self.max_seq_len}"
            )
        page_size = self.page_size

        partial_len = seen % page_size
        if partial_len > 0:
            head = self.K_partial[layer_idx, :, :, :partial_len, :]
            K_full = torch.cat([head, K_new], dim=2)
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
            K_packed, K_scale, K_mn = quantize_k_int4(K_complete, page_size=page_size)
            V_packed, V_scale, V_mn = quantize_v_int4(V_complete)

            tok_start = seen - partial_len
            tok_end = tok_start + complete_len
            page_idx_start = tok_start // page_size
            page_idx_end = page_idx_start + n_complete_pages

            self.K_packed[layer_idx, :, :, tok_start:tok_end, :] = K_packed
            self.V_packed[layer_idx, :, :, tok_start:tok_end, :] = V_packed
            self.K_scale[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_scale
            self.K_mn[layer_idx, :, :, page_idx_start:page_idx_end, :] = K_mn
            self.V_scale[layer_idx, :, :, tok_start:tok_end, :] = V_scale
            self.V_mn[layer_idx, :, :, tok_start:tok_end, :] = V_mn

        new_partial_len = total_stream_len - complete_len
        if new_partial_len < page_size:
            self.K_partial[layer_idx, :, :, :new_partial_len, :] = K_full[:, :, complete_len:, :]
            self.V_partial[layer_idx, :, :, :new_partial_len, :] = V_full[:, :, complete_len:, :]

        self._seen_tokens[layer_idx] = seen + S_new

    def get_views(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Mirror PersistentInt8KVCache.get_views — same keys (incl. seq_len /
        completed_len / partial_len) but K_uint8/V_uint8 → K_packed/V_packed."""
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
            "K_packed": self.K_packed[layer_idx, :, :, :completed_len, :],
            "V_packed": self.V_packed[layer_idx, :, :, :completed_len, :],
            "K_scale": self.K_scale[layer_idx, :, :, :n_complete_pages, :],
            "K_mn": self.K_mn[layer_idx, :, :, :n_complete_pages, :],
            "V_scale": self.V_scale[layer_idx, :, :, :completed_len, :],
            "V_mn": self.V_mn[layer_idx, :, :, :completed_len, :],
            "K_partial": self.K_partial[layer_idx, :, :, :partial_len, :],
            "V_partial": self.V_partial[layer_idx, :, :, :partial_len, :],
        }

    def update(self, *args, **kwargs):
        """HF Cache.update is unused on the patched path; raise to surface bugs."""
        raise NotImplementedError(
            "PersistentInt4KVCache.update is not used; the patched forward "
            "calls update_quantized directly."
        )

    # ------------------------------------------------------------------
    # Phase 12 — speculative-decoding sandbox + two-phase commit API
    # ------------------------------------------------------------------

    MAX_DRAFT = 8

    def _ensure_sandbox(self):
        if hasattr(self, "K_sandbox"):
            return
        shp = (self.num_layers, self.batch_size, self.num_kv_heads, self.MAX_DRAFT, self.head_dim)
        dev = self.K_partial.device
        self.K_sandbox = torch.zeros(shp, dtype=torch.bfloat16, device=dev)
        self.V_sandbox = torch.zeros(shp, dtype=torch.bfloat16, device=dev)
        self._sandbox_count = [0] * self.num_layers

    def add_draft(self, K_new: torch.Tensor, V_new: torch.Tensor, layer_idx: int) -> None:
        """Stage draft K/V (B, H_kv, S_new, D) in the sandbox WITHOUT advancing _seen_tokens."""
        self._ensure_sandbox()
        S = K_new.shape[2]
        if S > self.MAX_DRAFT:
            raise ValueError(f"S_new={S} > MAX_DRAFT={self.MAX_DRAFT}")
        self.K_sandbox[layer_idx, :, :, :S, :] = K_new
        self.V_sandbox[layer_idx, :, :, :S, :] = V_new
        self._sandbox_count[layer_idx] = S

    def get_views_with_sandbox(self, layer_idx: int) -> dict:
        """Return get_views() augmented with sandbox tensors and count."""
        self._ensure_sandbox()
        v = self.get_views(layer_idx)
        s = self._sandbox_count[layer_idx]
        v["K_sandbox"] = self.K_sandbox[layer_idx, :, :, :s, :]
        v["V_sandbox"] = self.V_sandbox[layer_idx, :, :, :s, :]
        v["sandbox_count"] = s
        return v

    def preflight_commit(self, accept_count: int, layer_idx: int) -> None:
        """Validate without mutating. Raises on bad input or a page-crossing commit."""
        self._ensure_sandbox()
        if accept_count < 0:
            raise ValueError(f"accept_count={accept_count} negative")
        if accept_count > self._sandbox_count[layer_idx]:
            raise ValueError(
                f"accept_count={accept_count} > sandbox_count={self._sandbox_count[layer_idx]}"
            )
        seen = self._seen_tokens[layer_idx]
        if accept_count > 0 and (seen + accept_count) // self.page_size != seen // self.page_size:
            raise RuntimeError(
                f"commit_draft({accept_count}) on layer {layer_idx} crosses page boundary "
                f"(seen={seen}, page_size={self.page_size})"
            )

    def commit_draft(self, accept_count: int, layer_idx: int) -> None:
        """Commit the first accept_count sandbox slots via update_quantized. Caller must
        preflight ALL layers first (commit_draft_all_layers does)."""
        if accept_count == 0:
            self._sandbox_count[layer_idx] = 0
            return
        self.update_quantized(
            self.K_sandbox[layer_idx, :, :, :accept_count, :],
            self.V_sandbox[layer_idx, :, :, :accept_count, :],
            layer_idx,
        )
        self._sandbox_count[layer_idx] = 0

    def commit_draft_all_layers(self, accept_count: int) -> None:
        """Two-phase atomic commit: preflight all layers, then commit all (raises
        before any mutation if any layer's preflight fails)."""
        for li in range(self.num_layers):
            self.preflight_commit(accept_count, li)
        for li in range(self.num_layers):
            self.commit_draft(accept_count, li)
