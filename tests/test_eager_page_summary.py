import torch

from flashquest.eager.page_summary import compute_page_summary


def test_basic_shape_and_values():
    B, H, S, D = 1, 2, 12, 8
    page_size = 4
    K = torch.arange(B * H * S * D, dtype=torch.float32).view(B, H, S, D)

    page_min, page_max = compute_page_summary(K, page_size)

    assert page_min.shape == (B, H, 3, D)
    assert page_max.shape == (B, H, 3, D)
    torch.testing.assert_close(page_min[0, 0, 0], K[0, 0, 0])
    torch.testing.assert_close(page_max[0, 0, 0], K[0, 0, 3])


def test_handles_partial_tail_page():
    B, H, S, D = 1, 1, 10, 4
    page_size = 4
    K = torch.randn(B, H, S, D)

    page_min, page_max = compute_page_summary(K, page_size)

    assert page_min.shape == (B, H, 3, D)
    expected_max = K[0, 0, 8:10].max(dim=0).values
    torch.testing.assert_close(page_max[0, 0, 2], expected_max)


def test_full_retention_summary_equals_pointwise():
    B, H, S, D = 1, 1, 5, 3
    K = torch.randn(B, H, S, D)
    page_min, page_max = compute_page_summary(K, page_size=1)
    torch.testing.assert_close(page_min.squeeze(2), K)
    torch.testing.assert_close(page_max.squeeze(2), K)
