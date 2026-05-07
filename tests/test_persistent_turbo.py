"""PersistentTurboKVCache: shapes, roundtrip, dispatcher smoke."""
import pytest
import torch

from flashquest.cache.persistent_turbo import PersistentTurboKVCache


def _make_cache(S=256, page_size=64):
    return PersistentTurboKVCache(
        batch_size=1, num_layers=2, num_kv_heads=4, head_dim=64,
        max_seq_len=S, page_size=page_size, device="cuda",
    )


def test_kv_bits_attribute():
    cache = _make_cache()
    assert cache.kv_bits == 3


def test_storage_shapes():
    cache = _make_cache(S=256, page_size=64)
    L, B, H, S, D, P = 2, 1, 4, 256, 64, 256 // 64
    assert cache.K_msb.shape == (L, B, H, S, D // 8)
    assert cache.K_lsb.shape == (L, B, H, S, D // 4)
    assert cache.K_scale_turbo.shape == (L, B, H, S, 1)
    assert cache.K_scale_raw.shape == (L, B, H, P, D)
    assert cache.K_mn_raw.shape == (L, B, H, P, D)
    assert cache.V_packed.shape == (L, B, H, S, D // 4)
    assert cache.V_scale_turbo.shape == (L, B, H, S, 1)


def test_update_quantized_writes_one_full_page():
    """Write 64 tokens (one page); verify _seen_tokens advances and views shapes."""
    torch.manual_seed(0)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    assert cache.get_seq_length(0) == 64
    views = cache.get_views(0)
    assert views["seq_len"] == 64
    assert views["completed_len"] == 64
    assert views["partial_len"] == 0
    assert views["K_msb"].shape == (1, 4, 64, 64 // 8)
    assert views["K_lsb"].shape == (1, 4, 64, 64 // 4)
    assert views["V_packed"].shape == (1, 4, 64, 64 // 4)
    assert views["K_scale_turbo"].shape == (1, 4, 64, 1)
    assert views["V_scale_turbo"].shape == (1, 4, 64, 1)
    assert views["K_scale_raw"].shape == (1, 4, 1, 64)
    assert views["K_mn_raw"].shape == (1, 4, 1, 64)


def test_partial_page_staging():
    """Write 32 tokens (half page); should land in K_partial / V_partial."""
    torch.manual_seed(1)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 32, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    assert views["completed_len"] == 0
    assert views["partial_len"] == 32
    assert views["K_partial"].shape == (1, 4, 32, 64)


def test_roundtrip_through_cache():
    """Write K, V → read views → dequant → ≈ K, V within 3-bit/2-bit noise."""
    from flashquest.kernel.kv_quant import dequantize_k_turbo, dequantize_v_turbo
    torch.manual_seed(7)
    cache = _make_cache(S=256, page_size=64)
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    K_back = dequantize_k_turbo(views["K_msb"], views["K_lsb"], views["K_scale_turbo"], head_dim=64)
    V_back = dequantize_v_turbo(views["V_packed"], views["V_scale_turbo"], head_dim=64)
    assert (K - K_back).float().abs().mean() < 0.4
    assert (V - V_back).float().abs().mean() < 0.7


def test_requires_head_dim_multiple_of_8():
    """head_dim must be ≥8 and divisible by 8 (MSB plane requires 8/byte)."""
    with pytest.raises(ValueError, match="multiple of 8"):
        PersistentTurboKVCache(
            batch_size=1, num_layers=1, num_kv_heads=1, head_dim=4,
            max_seq_len=64, page_size=64, device="cuda",
        )
