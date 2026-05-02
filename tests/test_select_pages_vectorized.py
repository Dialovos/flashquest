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
