import torch

from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary


def test_score_shape():
    B, H, S_q, S_kv, D = 1, 2, 1, 8, 4
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)
    assert scores.shape == (B, H, S_q, S_kv // page_size)


def test_score_upper_bounds_qk_dot():
    torch.manual_seed(0)
    B, H, S_q, S_kv, D = 1, 1, 1, 16, 8
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)

    qk = (Q @ K.transpose(-2, -1)).squeeze(2)
    qk_pages = qk.view(B, H, 4, page_size).max(dim=-1).values

    assert torch.all(scores.squeeze(2) >= qk_pages - 1e-5)


def test_score_handles_multi_query():
    B, H, S_q, S_kv, D = 1, 2, 3, 8, 4
    page_size = 4
    Q = torch.randn(B, H, S_q, D)
    K = torch.randn(B, H, S_kv, D)
    page_min, page_max = compute_page_summary(K, page_size)
    scores = page_scores(Q, page_min, page_max)
    assert scores.shape == (B, H, S_q, 2)
