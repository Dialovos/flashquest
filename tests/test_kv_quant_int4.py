"""Phase 6 task 5 — INT4 KV quant primitives."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    _pack_int4,
    _unpack_int4,
    quantize_k_int4,
    dequantize_k_int4,
    quantize_v_int4,
    dequantize_v_int4,
)


def test_pack_unpack_roundtrip():
    """Pack two 4-bit values per byte, then unpack — must be identical."""
    torch.manual_seed(0)
    x = torch.randint(0, 16, (2, 3, 4, 8), dtype=torch.uint8)
    packed = _pack_int4(x)
    assert packed.shape == (2, 3, 4, 4)
    assert packed.dtype == torch.uint8
    unpacked = _unpack_int4(packed)
    assert torch.equal(unpacked, x)


def test_pack_masks_high_nibble():
    """Documented semantic: high nibble of input bytes is dropped (caller responsibility)."""
    x = torch.tensor([[0, 15, 16, 255]], dtype=torch.uint8)
    packed = _pack_int4(x)
    unpacked = _unpack_int4(packed)
    assert unpacked.tolist() == [[0, 15, 0, 15]]


def test_quantize_k_int4_shape():
    """K (B, H, S, D) → K_packed (B, H, S, D/2), scale + mn (B, H, P, D)."""
    torch.manual_seed(0)
    K = torch.randn(2, 4, 128, 64, dtype=torch.bfloat16)
    K_packed, scale, mn = quantize_k_int4(K, page_size=64)
    assert K_packed.shape == (2, 4, 128, 32)
    assert K_packed.dtype == torch.uint8
    assert scale.shape == (2, 4, 2, 64)
    assert mn.shape == (2, 4, 2, 64)
    assert scale.dtype == torch.bfloat16
    assert mn.dtype == torch.bfloat16


def test_quantize_k_int4_roundtrip_within_quant_error():
    """Round-trip K → quant → dequant should be within the INT4 step size."""
    torch.manual_seed(42)
    K = torch.randn(2, 4, 128, 64, dtype=torch.bfloat16)
    K_packed, scale, mn = quantize_k_int4(K, page_size=64)
    K_recon = dequantize_k_int4(K_packed, scale, mn, page_size=64)
    scale_per_token = scale.repeat_interleave(64, dim=2)[:, :, :128, :]
    err = (K_recon.float() - K.float()).abs()
    bound = scale_per_token.float() + 1e-2
    assert (err <= bound).all(), f"max err {err.max()}, max bound {bound.max()}"


def test_page_max_identity_int4():
    """Algebraic: per-page max channel value ≡ K_mn + 15 × K_scale."""
    torch.manual_seed(7)
    K = torch.randn(1, 2, 64, 32, dtype=torch.bfloat16)
    K_packed, scale, mn = quantize_k_int4(K, page_size=64)
    page_max_ref = K.float().view(1, 2, 1, 64, 32).max(dim=3).values
    page_max_alg = mn.float() + 15.0 * scale.float()
    assert torch.allclose(page_max_alg, page_max_ref, atol=1e-2, rtol=1e-2)


def test_quantize_v_int4_shape():
    """V (B, H, S, D) → V_packed (B, H, S, D/2), scale + mn (B, H, S, 1)."""
    V = torch.randn(2, 4, 32, 64, dtype=torch.bfloat16)
    V_packed, scale, mn = quantize_v_int4(V)
    assert V_packed.shape == (2, 4, 32, 32)
    assert V_packed.dtype == torch.uint8
    assert scale.shape == (2, 4, 32, 1)
    assert mn.shape == (2, 4, 32, 1)


def test_quantize_v_int4_roundtrip():
    """V round-trip stays within scale of the original."""
    torch.manual_seed(1)
    V = torch.randn(2, 4, 32, 64, dtype=torch.bfloat16)
    V_packed, scale, mn = quantize_v_int4(V)
    V_recon = dequantize_v_int4(V_packed, scale, mn)
    err = (V_recon.float() - V.float()).abs()
    bound = scale.float() + 1e-2
    assert (err <= bound).all()


def test_quantize_k_int4_odd_head_dim_rejected():
    """head_dim must be even for INT4 packing along that axis."""
    K = torch.randn(1, 2, 64, 65, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="even head_dim"):
        quantize_k_int4(K, page_size=64)
