import pytest
import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.kernel.kv_quant import (
    quantize_k, quantize_v, dequantize_k, dequantize_v,
)


def test_cache_allocation_shapes():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=4, num_kv_heads=8, head_dim=128,
        max_seq_len=2048, page_size=64, device="cuda",
    )
    assert cache.K_uint8.shape == (4, 1, 8, 2048, 128)
    assert cache.K_uint8.dtype == torch.uint8
    assert cache.K_scale.shape == (4, 1, 8, 32, 128)  # 2048 / 64 = 32 pages
    assert cache.K_scale.dtype == torch.bfloat16
    assert cache.V_scale.shape == (4, 1, 8, 2048, 1)
    assert cache.K_partial.shape == (4, 1, 8, 64, 128)
    assert cache.K_partial.dtype == torch.bfloat16
    assert cache.get_seq_length(0) == 0
    assert cache.get_max_length() == 2048


def test_update_quantized_prefill_full_pages():
    torch.manual_seed(0)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=2, num_kv_heads=4, head_dim=64,
        max_seq_len=512, page_size=64, device="cuda",
    )
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16, device="cuda")

    cache.update_quantized(K, V, layer_idx=0)
    assert cache.get_seq_length(0) == 128

    K_uint8_ref, K_scale_ref, K_mn_ref = quantize_k(K, page_size=64)
    V_uint8_ref, V_scale_ref, V_mn_ref = quantize_v(V)

    torch.testing.assert_close(cache.K_uint8[0, :, :, :128, :], K_uint8_ref)
    torch.testing.assert_close(cache.K_scale[0, :, :, :2, :], K_scale_ref)
    torch.testing.assert_close(cache.K_mn[0, :, :, :2, :], K_mn_ref)
    torch.testing.assert_close(cache.V_uint8[0, :, :, :128, :], V_uint8_ref)
    torch.testing.assert_close(cache.V_scale[0, :, :, :128, :], V_scale_ref)
    torch.testing.assert_close(cache.V_mn[0, :, :, :128, :], V_mn_ref)


def test_update_quantized_decode_mid_page():
    """Decode steps that don't complete a page stay in the partial buffer."""
    torch.manual_seed(2)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=256, page_size=64, device="cuda",
    )
    K_steps = torch.randn(10, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    V_steps = torch.randn(10, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    for K_t, V_t in zip(K_steps, V_steps):
        cache.update_quantized(K_t, V_t, layer_idx=0)

    assert cache.get_seq_length(0) == 10
    K_partial_collected = cache.K_partial[0, :, :, :10, :]
    K_steps_concat = K_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 10, 64)
    torch.testing.assert_close(K_partial_collected, K_steps_concat)


def test_update_quantized_decode_completes_page():
    """A decode step that completes a page must flush to the uint8 region."""
    torch.manual_seed(3)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K_steps = torch.randn(64, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    V_steps = torch.randn(64, 1, 2, 1, 64, dtype=torch.bfloat16, device="cuda")
    for K_t, V_t in zip(K_steps, V_steps):
        cache.update_quantized(K_t, V_t, layer_idx=0)

    assert cache.get_seq_length(0) == 64
    K_full = K_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 64, 64)
    V_full = V_steps.permute(1, 2, 0, 3, 4).reshape(1, 2, 64, 64)
    K_uint8_ref, K_scale_ref, K_mn_ref = quantize_k(K_full, page_size=64)
    V_uint8_ref, V_scale_ref, V_mn_ref = quantize_v(V_full)
    torch.testing.assert_close(cache.K_uint8[0, :, :, :64, :], K_uint8_ref)
    torch.testing.assert_close(cache.K_scale[0, :, :, :1, :], K_scale_ref)
    torch.testing.assert_close(cache.V_uint8[0, :, :, :64, :], V_uint8_ref)


def test_update_overflow_raises():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="exceeds max_seq_len"):
        cache.update_quantized(K, V, layer_idx=0)


def test_update_bad_layer_raises():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 32, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(IndexError, match="layer_idx"):
        cache.update_quantized(K, V, layer_idx=5)


def test_get_views_roundtrip_matches_quantize_dequantize():
    """View slices, when dequanted, match the same dequant of a fresh quantize_k/v call."""
    torch.manual_seed(7)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=192, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 130, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 130, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)

    views = cache.get_views(0)
    assert views["K_uint8"].shape == (1, 2, 128, 64)
    assert views["K_scale"].shape == (1, 2, 2, 64)
    assert views["V_uint8"].shape == (1, 2, 128, 64)
    assert views["V_scale"].shape == (1, 2, 128, 1)
    assert views["K_partial"].shape == (1, 2, 2, 64)
    assert views["V_partial"].shape == (1, 2, 2, 64)
    assert views["seq_len"] == 130
    assert views["completed_len"] == 128
    assert views["partial_len"] == 2

    K_dq = dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=64)
    V_dq = dequantize_v(views["V_uint8"], views["V_scale"], views["V_mn"])
    K_full_recovered = torch.cat([K_dq, views["K_partial"]], dim=2)
    V_full_recovered = torch.cat([V_dq, views["V_partial"]], dim=2)
    # INT8 round-trip has up to ~0.013 abs error on bf16 — well-known KIVI bound.
    torch.testing.assert_close(K_full_recovered, K, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(V_full_recovered, V, rtol=2e-2, atol=2e-2)


def test_get_views_fresh_cache():
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    views = cache.get_views(0)
    assert views["seq_len"] == 0
    assert views["completed_len"] == 0
    assert views["partial_len"] == 0
    assert views["K_uint8"].shape == (1, 2, 0, 64)
    assert views["K_partial"].shape == (1, 2, 0, 64)


def test_get_views_partial_only():
    """30 tokens: no complete pages, all in partial."""
    torch.manual_seed(11)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=1, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    K = torch.randn(1, 2, 30, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 30, 64, dtype=torch.bfloat16, device="cuda")
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)
    assert views["completed_len"] == 0
    assert views["partial_len"] == 30
    torch.testing.assert_close(views["K_partial"], K)
    torch.testing.assert_close(views["V_partial"], V)
