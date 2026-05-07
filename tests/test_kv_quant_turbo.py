"""TurboQuant primitives: codebook lookup, bit-split, INT2 packing."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    K_TURBO_CODEBOOK, V_TURBO_CODEBOOK,
    _pack_bit_split, _unpack_bit_split,
    _pack_int2, _unpack_int2,
    _quantize_to_codebook,
)


def test_codebook_shapes_and_symmetry():
    """K and V codebooks: 8 entries each (K3-V3); both symmetric around 0."""
    assert K_TURBO_CODEBOOK.shape == (8,)
    assert V_TURBO_CODEBOOK.shape == (8,)
    for cb in (K_TURBO_CODEBOOK, V_TURBO_CODEBOOK):
        sorted_cb = torch.sort(cb).values
        assert torch.allclose(sorted_cb, -sorted_cb.flip(0), atol=1e-5), (
            f"codebook not symmetric: {cb}"
        )


def test_quantize_to_codebook_picks_nearest():
    """_quantize_to_codebook returns the index of the nearest codepoint."""
    cb = torch.tensor([-1.0, -0.5, 0.5, 1.0], device="cuda")
    x = torch.tensor([-0.9, -0.4, 0.0, 0.6, 1.1], device="cuda")
    idx = _quantize_to_codebook(x, cb)
    expected = torch.tensor([0, 1, 1, 2, 3], device="cuda", dtype=torch.uint8)
    assert torch.equal(idx, expected), f"got {idx}, expected {expected}"


def test_pack_bit_split_roundtrip():
    """_unpack_bit_split(_pack_bit_split(idx)) == idx for idx in 0..7."""
    torch.manual_seed(0)
    idx = torch.randint(0, 8, (1, 4, 16, 64), dtype=torch.uint8, device="cuda")
    msb, lsb = _pack_bit_split(idx)
    assert msb.shape == (1, 4, 16, 64 // 8)
    assert lsb.shape == (1, 4, 16, 64 // 4)
    idx_back = _unpack_bit_split(msb, lsb, head_dim=64)
    assert torch.equal(idx_back, idx)


def test_pack_int2_roundtrip():
    """_unpack_int2(_pack_int2(idx)) == idx for idx in 0..3."""
    torch.manual_seed(1)
    idx = torch.randint(0, 4, (1, 4, 16, 64), dtype=torch.uint8, device="cuda")
    packed = _pack_int2(idx)
    assert packed.shape == (1, 4, 16, 64 // 4)
    idx_back = _unpack_int2(packed, head_dim=64)
    assert torch.equal(idx_back, idx)


def test_pack_int2_requires_multiple_of_4():
    """Last axis must be divisible by 4."""
    x = torch.zeros(1, 1, 1, 6, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="multiple of 4"):
        _pack_int2(x)


def test_pack_bit_split_requires_multiple_of_8():
    """Last axis must be divisible by 8 (MSB plane)."""
    x = torch.zeros(1, 1, 1, 12, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="multiple of 8"):
        _pack_bit_split(x)


def test_quantize_k_turbo_shapes():
    """quantize_k_turbo returns five tensors with the documented shapes."""
    from flashquest.kernel.kv_quant import quantize_k_turbo
    torch.manual_seed(2)
    B, H, S, D, page_size = 1, 4, 256, 128, 64
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_turbo, K_scale_raw, K_mn_raw = quantize_k_turbo(K, page_size=page_size)
    assert K_msb.shape == (B, H, S, D // 8) and K_msb.dtype == torch.uint8
    assert K_lsb.shape == (B, H, S, D // 4) and K_lsb.dtype == torch.uint8
    assert K_scale_turbo.shape == (B, H, S, 1) and K_scale_turbo.dtype == torch.bfloat16
    num_pages = S // page_size
    assert K_scale_raw.shape == (B, H, num_pages, D)
    assert K_mn_raw.shape == (B, H, num_pages, D)


def test_quantize_v_turbo_shapes():
    """quantize_v_turbo returns (V_msb, V_lsb, V_scale_turbo) — K3-V3."""
    from flashquest.kernel.kv_quant import quantize_v_turbo
    torch.manual_seed(3)
    B, H, S, D = 1, 4, 256, 128
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V_msb, V_lsb, V_scale_turbo = quantize_v_turbo(V)
    assert V_msb.shape == (B, H, S, D // 8) and V_msb.dtype == torch.uint8
    assert V_lsb.shape == (B, H, S, D // 4) and V_lsb.dtype == torch.uint8
    assert V_scale_turbo.shape == (B, H, S, 1) and V_scale_turbo.dtype == torch.bfloat16


def test_dequantize_k_turbo_roundtrip():
    """K → quant → dequant ≈ K within 3-bit Lloyd-Max noise band on Gaussian inputs."""
    from flashquest.kernel.kv_quant import quantize_k_turbo, dequantize_k_turbo
    torch.manual_seed(4)
    K = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_turbo, _, _ = quantize_k_turbo(K, page_size=64)
    K_back = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=128)
    err = (K - K_back).float().abs().max()
    assert err < 1.5, f"K dequant max abs err {err} exceeds 1.5"
    mean_err = (K - K_back).float().abs().mean()
    assert mean_err < 0.4, f"K dequant mean abs err {mean_err} exceeds 0.4"


def test_dequantize_v_turbo_roundtrip():
    """V → quant → dequant ≈ V within 3-bit Lloyd-Max noise band (K3-V3)."""
    from flashquest.kernel.kv_quant import quantize_v_turbo, dequantize_v_turbo
    torch.manual_seed(5)
    V = torch.randn(1, 4, 256, 128, dtype=torch.bfloat16, device="cuda")
    V_msb, V_lsb, V_scale_turbo = quantize_v_turbo(V)
    V_back = dequantize_v_turbo(V_msb, V_lsb, V_scale_turbo, head_dim=128)
    mean_err = (V - V_back).float().abs().mean()
    assert mean_err < 0.4, f"V dequant mean abs err {mean_err} exceeds 0.4"
