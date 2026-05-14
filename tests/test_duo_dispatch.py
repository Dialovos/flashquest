"""Tests for DuoAttention per-head dispatch."""
import torch

from flashquest.duo.dispatch import quest_duo_eager_sdpa
from flashquest.eager import quest_eager_sdpa, streaming_eager_sdpa


def _make_qkv(B=1, H_q=4, H_kv=1, S=256, D=64, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    return Q, K, V


def test_all_retrieval_matches_pure_quest():
    """EP1: all-retrieval pattern ≡ Phase 1 quest_eager_sdpa with same knobs."""
    Q, K, V = _make_qkv()
    H_kv = K.shape[1]
    pattern = torch.ones(H_kv, dtype=torch.bool)

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_quest = quest_eager_sdpa(
        Q, K, V,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_quest, rtol=1e-3, atol=1e-3)


def test_all_streaming_matches_streaming_only():
    """EP2: all-streaming pattern ≡ streaming_eager_sdpa with same knobs."""
    Q, K, V = _make_qkv(seed=1)
    H_kv = K.shape[1]
    pattern = torch.zeros(H_kv, dtype=torch.bool)

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_stream = streaming_eager_sdpa(
        Q, K, V,
        page_size=64, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_stream, rtol=1e-3, atol=1e-3)


def test_mixed_pattern_runs():
    """EP3: mixed pattern — both code paths exercised, output is finite."""
    Q, K, V = _make_qkv(H_q=4, H_kv=2, seed=2)
    pattern = torch.tensor([True, False])

    O = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    assert O.shape == Q.shape
    assert torch.isfinite(O).all()


def test_mixed_pattern_per_head_correctness():
    """EP3: heads with retrieval pattern produce the retrieval result;
    heads with streaming pattern produce the streaming result."""
    Q, K, V = _make_qkv(H_q=4, H_kv=2, seed=3)
    pattern = torch.tensor([True, False])

    O_quest_all = quest_eager_sdpa(
        Q, K, V,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    O_stream_all = streaming_eager_sdpa(
        Q, K, V,
        page_size=64, num_sinks=4, window_pages=2, is_causal=True,
    )
    n_rep = Q.shape[1] // K.shape[1]
    pattern_per_q_head = pattern.repeat_interleave(n_rep)

    O_expected = torch.where(
        pattern_per_q_head.view(1, -1, 1, 1),
        O_quest_all,
        O_stream_all,
    )

    O_duo = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
    )
    torch.testing.assert_close(O_duo, O_expected, rtol=1e-3, atol=1e-3)


def test_pattern_shape_validation():
    """EP4/EP5: pattern length mismatch raises clearly."""
    import pytest

    Q, K, V = _make_qkv(H_q=4, H_kv=2)
    bad_pattern = torch.tensor([True, False, True])
    with pytest.raises(ValueError, match="head_pattern"):
        quest_duo_eager_sdpa(
            Q, K, V, head_pattern=bad_pattern,
            page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=True,
        )
