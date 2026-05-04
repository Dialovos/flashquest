"""Phase 6 task 5 — sparse INT4 forward: validates against dense reference."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    quantize_k_int4, quantize_v_int4,
    dequantize_k_int4, dequantize_v_int4,
)
from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd


def test_sparse_int4_matches_dense():
    """All-pages-selected ≡ dense attention on the dequantized KV (loose tol for INT4)."""
    torch.manual_seed(13)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 4, 2, 1, 256, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)

    selection_mask = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")

    out, _ = flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn,
        V_packed, V_scale, V_mn,
        selection_mask=selection_mask,
        page_size=page_size, sm_scale=D ** -0.5, return_lse=True,
    )

    K_deq = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)
    V_deq = dequantize_v_int4(V_packed, V_scale, V_mn)
    n_rep = H_q // H_kv
    K_deq_q = K_deq.repeat_interleave(n_rep, dim=1)
    V_deq_q = V_deq.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_deq_q.float(), V_deq_q.float(), is_causal=False,
    ).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs().max()
    assert err < 5e-2, f"max abs diff {err}"


def test_fused_matches_reference():
    """Fused Triton kernel ≡ reference path on small fixtures (FP rtol=1e-3).

    Fixture is intentionally tiny so that any divergence surfaces at the per-element
    level, not buried under attention-scale variance. retention=1.0 (all pages
    selected) so we test every dequant path, not the page-selection logic.
    """
    from flashquest.kernel.sparse_int4_fwd import (
        flash_attn_sparse_int4_fwd,
        _flash_attn_sparse_int4_fwd_reference,
    )
    torch.manual_seed(17)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 2, 1, 1, 64, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)

    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    kw = dict(
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    O_ref, lse_ref = _flash_attn_sparse_int4_fwd_reference(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, **kw,
    )
    O_fused, lse_fused = flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, **kw,
    )

    err_O = (O_fused.float() - O_ref.float()).abs().max()
    err_lse = (lse_fused.float() - lse_ref.float()).abs().max()
    assert err_O < 1e-2, f"fused vs reference O max abs err {err_O}"
    assert err_lse < 1e-2, f"fused vs reference lse max abs err {err_lse}"


def test_no_kv_shaped_bf16_intermediate(monkeypatch):
    """Fused kernel must not allocate any BF16 tensor with K/V-shape (B, H_kv, S_kv, *).

    Tracks every torch.empty / torch.zeros call during flash_attn_sparse_int4_fwd;
    if any such allocation has size matching B*H_kv*S_kv*head_dim*2 bytes (BF16
    K/V intermediate), we've regressed to the reference path.
    """
    from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd
    torch.manual_seed(19)
    B, H_q, H_kv, S_kv, D, page_size = 1, 4, 2, 1024, 64, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    sel = torch.ones(B, H_q, 1, S_kv // page_size, dtype=torch.bool, device="cuda")

    kv_bytes_threshold = B * H_kv * S_kv * D * 2  # BF16 K (or V) intermediate.
    allocations: list[tuple[tuple[int, ...], torch.dtype, int]] = []
    orig_empty = torch.empty
    orig_zeros = torch.zeros

    def _track(name, fn):
        def inner(*args, **kwargs):
            t = fn(*args, **kwargs)
            try:
                nbytes = t.numel() * t.element_size()
                allocations.append((tuple(t.shape), t.dtype, nbytes))
            except Exception:
                pass
            return t
        inner.__wrapped_name__ = name
        return inner

    monkeypatch.setattr(torch, "empty", _track("empty", orig_empty))
    monkeypatch.setattr(torch, "zeros", _track("zeros", orig_zeros))

    flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    bf16_kv_intermediates = [
        (shape, dt, nbytes)
        for (shape, dt, nbytes) in allocations
        if dt == torch.bfloat16 and nbytes >= kv_bytes_threshold
    ]
    assert not bf16_kv_intermediates, (
        f"fused kernel allocated K/V-shaped BF16 intermediates "
        f"(threshold {kv_bytes_threshold} bytes): {bf16_kv_intermediates}"
    )
