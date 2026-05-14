import torch
from flashquest.eager.selection import select_pages


def test_tensor_retention_uniform_matches_scalar():
    torch.manual_seed(0)
    scores = torch.randn(1, 4, 8, 16)  # B=1, H=4, S_q=8, P=16
    mask_scalar = select_pages(scores, retention=0.25, num_sinks=2, window_pages=2)

    retention_t = torch.tensor([0.25, 0.25, 0.25, 0.25])
    mask_tensor = select_pages(scores, retention=retention_t, num_sinks=2, window_pages=2)

    assert torch.equal(mask_scalar, mask_tensor)


def test_tensor_retention_mixed_zero_streaming_heads():
    """Heads with retention=0 select only sinks + window (no top-k)."""
    torch.manual_seed(1)
    scores = torch.randn(1, 4, 1, 16)
    retention_t = torch.tensor([0.5, 0.0, 0.5, 0.0])
    mask = select_pages(scores, retention=retention_t, num_sinks=2, window_pages=2)

    # Heads 1 and 3 (retention=0): only first 2 + last 2 pages selected.
    expected_streaming = torch.zeros(16, dtype=torch.bool)
    expected_streaming[:2] = True
    expected_streaming[-2:] = True
    assert torch.equal(mask[0, 1, 0], expected_streaming)
    assert torch.equal(mask[0, 3, 0], expected_streaming)

    # Heads 0 and 2 (retention=0.5): 8 top-k + sinks + window. At least 8 selected.
    assert mask[0, 0, 0].sum() >= 8
    assert mask[0, 2, 0].sum() >= 8


def test_scalar_retention_existing_callers_unchanged():
    """Phase 1 callers passing float retention must still work."""
    scores = torch.randn(2, 8, 4, 32)
    mask = select_pages(scores, retention=0.1, num_sinks=4, window_pages=2)
    assert mask.shape == (2, 8, 4, 32)
    assert mask.dtype == torch.bool
