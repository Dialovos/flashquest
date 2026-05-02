"""EQ21-EQ25: select_pages_vectorized — single batched topk + scatter,
equivalent to the Phase 5 per-head loop in select_pages."""
import torch
import pytest

from flashquest.eager.selection import select_pages, select_pages_vectorized


def _scores(B=1, H=4, S_q=1, P=16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(B, H, S_q, P, generator=g, device="cuda", dtype=torch.float32)


@pytest.mark.parametrize("retention", [0.0, 0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("P", [8, 16, 64, 128])
def test_eq21_scalar_retention_equiv(retention, P):
    """EQ21: vectorized output ≡ loop output for scalar retention across (P, k)."""
    s = _scores(B=2, H=6, S_q=1, P=P, seed=hash((retention, P)) & 0xFFFF)
    ref = select_pages(s, retention=retention, num_sinks=2, window_pages=1)
    out = select_pages_vectorized(s, retention=retention, num_sinks=2, window_pages=1)
    assert torch.equal(ref, out)


def test_eq22_per_head_retention_tensor():
    """EQ22: per-head retention vector — same selections as the loop."""
    B, H, P = 1, 6, 32
    retention = torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    s = _scores(B=B, H=H, S_q=1, P=P, seed=42)
    ref = select_pages(s, retention=retention, num_sinks=2, window_pages=1)
    out = select_pages_vectorized(s, retention=retention, num_sinks=2, window_pages=1)
    assert torch.equal(ref, out)


def test_eq23_retention_zero_only_sinks_window():
    """EQ23: retention=0 head — only sinks + window are set."""
    B, H, P = 1, 4, 16
    s = _scores(B=B, H=H, S_q=1, P=P, seed=7)
    out = select_pages_vectorized(s, retention=0.0, num_sinks=2, window_pages=1)
    expect = torch.zeros_like(s, dtype=torch.bool)
    expect[..., :2] = True
    expect[..., -1:] = True
    assert torch.equal(out, expect)


def test_eq24_retention_one_all_pages():
    """EQ24: retention=1 head — every page selected."""
    B, H, P = 1, 4, 16
    s = _scores(B=B, H=H, S_q=1, P=P, seed=11)
    out = select_pages_vectorized(s, retention=1.0, num_sinks=0, window_pages=0)
    assert out.all()


def test_eq25_sinks_window_clamp():
    """EQ25: num_sinks > P or window_pages > P clamps to P."""
    B, H, P = 1, 2, 4
    s = _scores(B=B, H=H, S_q=1, P=P, seed=99)
    out = select_pages_vectorized(s, retention=0.0, num_sinks=10, window_pages=10)
    assert out.all()
