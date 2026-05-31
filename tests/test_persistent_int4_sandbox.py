import pytest, torch
from flashquest.cache.persistent_int4 import PersistentInt4KVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def _cache(num_layers=1):
    return PersistentInt4KVCache(batch_size=1, num_layers=num_layers, num_kv_heads=8,
                                 head_dim=128, max_seq_len=4096, page_size=64, device="cuda")
def _kv(S):
    return (torch.randn(1, 8, S, 128, dtype=torch.bfloat16, device="cuda"),
            torch.randn(1, 8, S, 128, dtype=torch.bfloat16, device="cuda"))

def test_add_draft_does_not_advance_seen():
    c = _cache(); c.update_quantized(*_kv(64), 0)            # seen=64 (one full page)
    c.add_draft(*_kv(4), 0)
    assert c._seen_tokens[0] == 64
    v = c.get_views_with_sandbox(0)
    assert v["sandbox_count"] == 4 and v["K_sandbox"].shape[2] == 4 and v["V_sandbox"].shape[2] == 4

def test_commit_advances_by_accept_count():
    c = _cache(); c.update_quantized(*_kv(64), 0)
    c.add_draft(*_kv(4), 0)
    c.commit_draft_all_layers(3)
    assert c._seen_tokens[0] == 67 and c._sandbox_count[0] == 0

def test_commit_zero_clears_sandbox_without_advancing():
    c = _cache(); c.update_quantized(*_kv(64), 0)
    c.add_draft(*_kv(4), 0)
    c.commit_draft_all_layers(0)
    assert c._seen_tokens[0] == 64 and c._sandbox_count[0] == 0

def test_page_boundary_guard_raises():
    c = _cache(); c.update_quantized(*_kv(62), 0)            # seen=62
    c.add_draft(*_kv(4), 0)
    with pytest.raises(RuntimeError, match="page boundary"):
        c.commit_draft_all_layers(4)                         # 62->66 crosses 64

def test_accept_count_exceeds_sandbox_raises():
    c = _cache(); c.update_quantized(*_kv(64), 0)
    c.add_draft(*_kv(2), 0)
    with pytest.raises(ValueError):
        c.commit_draft_all_layers(3)
