import torch

from flashquest.eager.selection import select_pages


def test_full_retention_selects_all():
    B, H, S_q, P = 1, 2, 1, 8
    scores = torch.randn(B, H, S_q, P)
    mask = select_pages(scores, retention=1.0, num_sinks=0, window_pages=0)
    assert mask.shape == (B, H, S_q, P)
    assert mask.all()


def test_zero_retention_keeps_only_sinks_and_window():
    B, H, S_q, P = 1, 1, 1, 8
    scores = torch.zeros(B, H, S_q, P)
    mask = select_pages(scores, retention=0.0, num_sinks=1, window_pages=2)
    expected = torch.tensor([True, False, False, False, False, False, True, True])
    assert torch.equal(mask.squeeze(0).squeeze(0).squeeze(0), expected)


def test_topk_picks_highest_scoring():
    B, H, S_q, P = 1, 1, 1, 6
    scores = torch.tensor([[[[0.1, 0.9, 0.2, 0.8, 0.3, 0.7]]]])
    mask = select_pages(scores, retention=0.5, num_sinks=0, window_pages=0)
    expected = torch.tensor([False, True, False, True, False, True])
    assert torch.equal(mask.squeeze(0).squeeze(0).squeeze(0), expected)


def test_per_head_independent():
    head0 = torch.tensor([[[0.9, 0.0, 0.0, 0.1]]])
    head1 = torch.tensor([[[0.0, 0.0, 0.9, 0.1]]])
    scores = torch.stack([head0, head1], dim=1)  # (1, 2, 1, 4)
    mask = select_pages(scores, retention=0.25, num_sinks=0, window_pages=0)
    assert mask[0, 0, 0, 0] and not mask[0, 0, 0, 2]
    assert mask[0, 1, 0, 2] and not mask[0, 1, 0, 0]


def test_retention_rounds_up_to_at_least_one():
    B, H, S_q, P = 1, 1, 1, 4
    scores = torch.tensor([[[[0.1, 0.4, 0.2, 0.3]]]])
    mask = select_pages(scores, retention=0.1, num_sinks=0, window_pages=0)
    assert mask.sum().item() == 1
    assert mask[0, 0, 0, 1].item()
