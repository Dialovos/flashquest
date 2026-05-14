"""Phase 6 task 5 — eager-path INT4 sparse attention.

Thin wrapper around flashquest.kernel.sparse_int4_fwd; mirrors
flashquest.eager.sparse_int8 so the dispatcher in llama_persistent_patch
can choose between the two by reading cache.kv_bits.
"""
from __future__ import annotations

from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd


def quest_sparse_int4_fwd(
    Q, K_packed, K_scale, K_mn,
    V_packed, V_scale, V_mn,
    *, selection_mask, page_size, sm_scale, return_lse: bool = False,
):
    """Public eager-path entry. Same shape contracts as the INT8 wrapper
    except K_packed / V_packed are uint8 with head_dim/2 trailing axis."""
    return flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn,
        V_packed, V_scale, V_mn,
        selection_mask=selection_mask,
        page_size=page_size, sm_scale=sm_scale, return_lse=return_lse,
    )
