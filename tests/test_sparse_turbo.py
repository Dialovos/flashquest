"""Sparse forward correctness for TurboQuant KV (K=3-bit, V=2-bit)."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    quantize_k_turbo, quantize_v_turbo,
    dequantize_k_turbo, dequantize_v_turbo,
)


def test_sparse_turbo_matches_dense():
    """All-pages-selected ≡ dense attention on dequant'd KV (loose tol for K3+V2)."""
    from flashquest.kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd

    torch.manual_seed(13)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 4, 2, 1, 256, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V)

    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    out, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size, sm_scale=D ** -0.5, return_lse=True,
    )

    K_deq = dequantize_k_turbo(K_msb, K_lsb, K_scale_t, head_dim=D)
    V_deq = dequantize_v_turbo(V_msb, V_lsb, V_scale_t, head_dim=D)
    n_rep = H_q // H_kv
    K_deq_q = K_deq.repeat_interleave(n_rep, dim=1)
    V_deq_q = V_deq.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_deq_q.float(), V_deq_q.float(), is_causal=False,
    ).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs().max()
    assert err < 1.5e-1, f"max abs diff {err}"


def test_fused_matches_reference():
    """Fused Triton kernel ≡ reference Python path on small fixtures."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd,
        _flash_attn_sparse_turbo_fwd_reference,
    )

    torch.manual_seed(17)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 2, 1, 1, 64, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V)
    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")

    kw = dict(
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )
    O_ref, lse_ref = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t, **kw,
    )
    O_fused, lse_fused = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t, **kw,
    )

    err_O = (O_fused.float() - O_ref.float()).abs().max()
    err_lse = (lse_fused.float() - lse_ref.float()).abs().max()
    assert err_O < 5e-2, f"fused vs reference O max abs err {err_O}"
    assert err_lse < 5e-2, f"fused vs reference lse max abs err {err_lse}"


def test_no_kv_shaped_bf16_intermediate(monkeypatch):
    """Fused kernel must not allocate any BF16 tensor with K/V shape.

    Tracks every torch.empty / torch.zeros call during flash_attn_sparse_turbo_fwd;
    if any allocation has size matching B*H_kv*S_kv*head_dim*2 bytes (BF16
    K/V intermediate), we've regressed to the reference path.
    """
    from flashquest.kernel.sparse_turbo_fwd import flash_attn_sparse_turbo_fwd

    torch.manual_seed(19)
    B, H_q, H_kv, S_kv, D, page_size = 1, 4, 2, 1024, 64, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size)
    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V)
    sel = torch.ones(B, H_q, 1, S_kv // page_size, dtype=torch.bool, device="cuda")

    kv_bytes_threshold = B * H_kv * S_kv * D * 2
    allocations: list[tuple[tuple[int, ...], torch.dtype, int]] = []
    orig_empty = torch.empty
    orig_zeros = torch.zeros

    def _track(fn):
        def inner(*args, **kwargs):
            t = fn(*args, **kwargs)
            try:
                allocations.append((tuple(t.shape), t.dtype, t.numel() * t.element_size()))
            except Exception:
                pass
            return t
        return inner

    monkeypatch.setattr(torch, "empty", _track(orig_empty))
    monkeypatch.setattr(torch, "zeros", _track(orig_zeros))

    flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    bf16_kv_intermediates = [
        (shape, dt, nbytes)
        for (shape, dt, nbytes) in allocations
        if dt == torch.bfloat16 and nbytes >= kv_bytes_threshold
    ]
    assert not bf16_kv_intermediates, (
        f"fused turbo kernel allocated K/V-shaped BF16 intermediates "
        f"(threshold {kv_bytes_threshold} bytes): {bf16_kv_intermediates}"
    )
