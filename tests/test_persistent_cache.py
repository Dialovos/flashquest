import pytest
import torch

from flashquest.cache import PersistentInt8KVCache


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
