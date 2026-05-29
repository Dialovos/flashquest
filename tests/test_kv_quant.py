import torch

from flashquest.kernel.kv_quant import (
    dequantize_k,
    dequantize_v,
    quantize_k,
    quantize_v,
)


def test_quantize_k_shapes():
    B, H, S, D = 1, 2, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    assert K_uint8.shape == (B, H, S, D)
    assert K_uint8.dtype == torch.uint8
    num_pages = S // 64
    assert scale.shape == (B, H, num_pages, D)
    assert mn.shape == (B, H, num_pages, D)
    assert scale.dtype == torch.bfloat16


def test_quantize_v_shapes():
    B, H, S, D = 1, 2, 128, 64
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V_uint8, scale, mn = quantize_v(V)
    assert V_uint8.shape == (B, H, S, D)
    assert V_uint8.dtype == torch.uint8
    assert scale.shape == (B, H, S, 1)
    assert mn.shape == (B, H, S, 1)


def test_k_round_trip_within_int8_tolerance():
    """ES10: dequant(quant(K)) within (max_range / 255) per element."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)

    expected_max_err = scale.abs().amax().item()
    actual_max_err = (K_rec.float() - K.float()).abs().amax().item()
    assert actual_max_err <= expected_max_err * 1.01, (
        f"round-trip err {actual_max_err} > expected {expected_max_err}"
    )


def test_v_round_trip_within_int8_tolerance():
    torch.manual_seed(1)
    B, H, S, D = 1, 2, 128, 64
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V_uint8, scale, mn = quantize_v(V)
    V_rec = dequantize_v(V_uint8, scale, mn)

    expected_max_err = scale.abs().amax().item()
    actual_max_err = (V_rec.float() - V.float()).abs().amax().item()
    assert actual_max_err <= expected_max_err * 1.01


def test_k_partial_last_page():
    """ES3: S_kv not multiple of page_size — last page handled."""
    B, H, S, D = 1, 1, 100, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    assert scale.shape == (B, H, 2, D)
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)
    assert K_rec.shape == K.shape


def test_zero_range_channel_safe():
    """ES9: a channel with all-equal values must not divide by 0."""
    B, H, S, D = 1, 1, 64, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K[..., 5] = 0.7
    K_uint8, scale, mn = quantize_k(K, page_size=64)
    assert torch.isfinite(scale).all()
    assert torch.isfinite(mn).all()
    K_rec = dequantize_k(K_uint8, scale, mn, page_size=64)
    assert torch.isfinite(K_rec).all()
    torch.testing.assert_close(K_rec[..., 5], K[..., 5], rtol=0, atol=1e-2)


# --- Phase 11: parameterized codebook -----------------------------------

def test_quantize_k_turbo_accepts_custom_codebook():
    """quantize_k_turbo(K, codebook=cb) round-trips through dequant with cb."""
    from flashquest.kernel.kv_quant import (
        quantize_k_turbo, dequantize_k_turbo,
    )

    torch.manual_seed(7)
    K = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device="cuda")
    cb = torch.tensor(
        [-2.0, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 2.0],
        dtype=torch.float32, device="cuda",
    )

    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=64, codebook=cb)
    K_rt = dequantize_k_turbo(K_msb, K_lsb, K_scale_t, head_dim=64, codebook=cb)

    paper_K_msb, paper_K_lsb, paper_K_scale_t, _, _ = quantize_k_turbo(K, page_size=64)
    paper_K_rt = dequantize_k_turbo(paper_K_msb, paper_K_lsb, paper_K_scale_t, head_dim=64)

    custom_err = (K.float() - K_rt.float()).abs().mean()
    paper_err = (K.float() - paper_K_rt.float()).abs().mean()
    assert custom_err < 2 * paper_err, f"custom={custom_err:.4f} paper={paper_err:.4f}"


def test_quantize_v_turbo_accepts_custom_codebook():
    """Symmetric to K case."""
    from flashquest.kernel.kv_quant import (
        quantize_v_turbo, dequantize_v_turbo,
    )

    torch.manual_seed(7)
    V = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device="cuda")
    cb = torch.tensor(
        [-2.0, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 2.0],
        dtype=torch.float32, device="cuda",
    )

    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V, codebook=cb)
    V_rt = dequantize_v_turbo(V_msb, V_lsb, V_scale_t, head_dim=64, codebook=cb)

    paper_V_msb, paper_V_lsb, paper_V_scale_t = quantize_v_turbo(V)
    paper_V_rt = dequantize_v_turbo(paper_V_msb, paper_V_lsb, paper_V_scale_t, head_dim=64)

    custom_err = (V.float() - V_rt.float()).abs().mean()
    paper_err = (V.float() - paper_V_rt.float()).abs().mean()
    assert custom_err < 2 * paper_err
