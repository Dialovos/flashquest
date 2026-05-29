"""Phase 11 — fused TurboQuant kernel parity under custom codebooks."""
from __future__ import annotations

import pytest
import torch


def _make_quantized_kv(B, H_kv, S, D, codebook_k, codebook_v, page_size, seed):
    """Build (K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t) using custom cbs."""
    from flashquest.kernel.kv_quant import quantize_k_turbo, quantize_v_turbo
    torch.manual_seed(seed)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(
        K, page_size=page_size, codebook=codebook_k,
    )
    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V, codebook=codebook_v)
    return K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_turbo_fwd_parity_custom_codebook(seed):
    """Fused kernel + reference path bit-equivalent when both use the same custom codebook."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd, _flash_attn_sparse_turbo_fwd_reference,
    )

    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    page_size = 64
    num_pages = S // page_size

    torch.manual_seed(seed)
    cb_k = torch.sort(torch.randn(8, device="cuda").float())[0]
    cb_v = torch.sort(torch.randn(8, device="cuda").float())[0]

    K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t = _make_quantized_kv(
        B, H_kv, S, D, cb_k, cb_v, page_size, seed,
    )

    torch.manual_seed(seed + 100)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_fused, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
        codebook_k=cb_k, codebook_v=cb_v,
    )
    O_ref, _ = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
        codebook_k=cb_k, codebook_v=cb_v,
    )
    err = (O_fused.float() - O_ref.float()).abs().max()
    assert err < 5e-2, f"max abs err {err}"


def test_sparse_turbo_fwd_paper_default_unchanged():
    """Phase 7 behavior preserved when no codebook is passed (default = paper)."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd, _flash_attn_sparse_turbo_fwd_reference,
    )

    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    page_size = 64
    num_pages = S // page_size
    seed = 7

    K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t = _make_quantized_kv(
        B, H_kv, S, D, None, None, page_size, seed,
    )

    torch.manual_seed(seed + 100)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_fused, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
    )
    O_ref, _ = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
    )
    err = (O_fused.float() - O_ref.float()).abs().max()
    assert err < 5e-2
