"""Phase 6 task 5 — PersistentInt4KVCache."""
import pytest
import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.cache.persistent_int8 import PersistentInt8KVCache


def _make(int4: bool, **overrides):
    kw = dict(
        batch_size=1, num_layers=2, num_kv_heads=2, head_dim=64,
        max_seq_len=256, page_size=64, device="cuda",
    )
    kw.update(overrides)
    cls = PersistentInt4KVCache if int4 else PersistentInt8KVCache
    return cls(**kw)


def test_kv_bits_attributes():
    """Both caches expose kv_bits as a class attribute."""
    assert PersistentInt4KVCache.kv_bits == 4
    assert PersistentInt8KVCache.kv_bits == 8


def test_packed_shape():
    """K_packed.shape[-1] == head_dim // 2 (2 INT4 per byte)."""
    c = _make(int4=True, head_dim=128)
    assert c.K_packed.shape[-1] == 64
    assert c.V_packed.shape[-1] == 64
    assert c.K_packed.dtype == torch.uint8
    assert c.V_packed.dtype == torch.uint8


def test_odd_head_dim_rejected():
    """INT4 cache rejects odd head_dim (cannot pack 2-per-byte)."""
    with pytest.raises(ValueError, match="even head_dim"):
        _make(int4=True, head_dim=65)


def test_update_quantized_int4_basic():
    """update_quantized at INT4 grows _seen_tokens; views expose K_packed."""
    c = _make(int4=True)
    K = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    c.update_quantized(K, V, layer_idx=0)
    assert c.get_seq_length(0) == 64

    views = c.get_views(0)
    assert "K_packed" in views
    assert "K_scale" in views
    assert "K_mn" in views
    assert views["K_packed"].shape == (1, 2, 64, 32)


def test_int4_int8_coexist():
    """Both caches can be constructed in the same process; dispatcher reads kv_bits."""
    c4 = _make(int4=True)
    c8 = _make(int4=False)
    assert c4.kv_bits == 4
    assert c8.kv_bits == 8
    K = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    c4.update_quantized(K, V, layer_idx=0)
    c8.update_quantized(K, V, layer_idx=0)
    assert c4.get_seq_length(0) == 64
    assert c8.get_seq_length(0) == 64


def test_max_seq_len_overflow_rejected():
    """update_quantized refuses to grow past max_seq_len."""
    c = _make(int4=True, max_seq_len=128)
    K = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 200, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="exceeds.*max_seq_len"):
        c.update_quantized(K, V, layer_idx=0)
