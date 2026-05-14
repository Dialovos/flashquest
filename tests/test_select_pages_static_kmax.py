"""select_pages_vectorized with precomputed k_max_static must produce
the same bool mask as the current `.item()`-based version."""
import math

import pytest
import torch

from flashquest.eager.selection import select_pages_vectorized


@pytest.mark.parametrize("retention", [0.25, 0.5])
@pytest.mark.parametrize("P", [16, 64, 512])
def test_static_kmax_parity_with_dynamic(retention, P):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    B, H, S_q = 1, 8, 1
    scores = torch.randn(B, H, S_q, P, device="cuda")

    mask_dynamic = select_pages_vectorized(
        scores, retention=retention, num_sinks=4, window_pages=2,
    )

    P_max = max(P, 512)
    k_max_static = math.ceil(retention * P_max)
    mask_static = select_pages_vectorized(
        scores, retention=retention, num_sinks=4, window_pages=2,
        k_max_static=k_max_static,
    )

    assert torch.equal(mask_dynamic, mask_static)


def test_static_kmax_per_head_retention():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    B, H, S_q, P = 1, 4, 1, 32
    scores = torch.randn(B, H, S_q, P, device="cuda")
    retention = torch.tensor([0.0, 0.25, 0.5, 1.0], device="cuda")

    mask_dynamic = select_pages_vectorized(
        scores, retention=retention, num_sinks=4, window_pages=2,
    )
    mask_static = select_pages_vectorized(
        scores, retention=retention, num_sinks=4, window_pages=2,
        k_max_static=32,  # max retention 1.0 * P=32 → 32
    )
    assert torch.equal(mask_dynamic, mask_static)
