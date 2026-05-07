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
    """K codebook has 8 entries; V has 4. Both symmetric around 0."""
    assert K_TURBO_CODEBOOK.shape == (8,)
    assert V_TURBO_CODEBOOK.shape == (4,)
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
