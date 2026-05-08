"""Overlap test: when topk + sinks + window all touch the same page,
the bool mask deduplicates and the compact output has no duplicate IDs."""
import pytest
import torch

from flashquest.eager.selection import build_compact_selection, select_pages_vectorized


def test_no_duplicate_when_topk_overlaps_sink():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Force topk to pick pages 0 and 1 (sink range = [0..3]) by making them
    # the highest-scoring. The bool mask sets pages 0/1 True via topk AND via sinks.
    # Compact list must have at most one entry per page.
    B, H, S_q, P = 1, 1, 1, 16
    scores = torch.zeros(B, H, S_q, P, device="cuda")
    scores[0, 0, 0, 0] = 10.0  # highest
    scores[0, 0, 0, 1] = 9.0
    scores[0, 0, 0, 14] = 1.0  # window range = [14..15]
    mask = select_pages_vectorized(
        scores, retention=0.25, num_sinks=4, window_pages=2,
        k_max_static=4,  # ceil(0.25*16) = 4
    )
    # Expected mask: sinks {0,1,2,3} ∪ window {14,15} ∪ topk {0,1,14,X}
    assert mask[0, 0, 0, 0] and mask[0, 0, 0, 1]  # sinks
    assert mask[0, 0, 0, 14] and mask[0, 0, 0, 15]  # window

    out = build_compact_selection(mask, BUCKET_MAX=8)
    real_ids = sorted(int(x) for x in out[0, 0, 0].tolist() if int(x) != -1)
    # Verify: no duplicates
    assert len(real_ids) == len(set(real_ids)), \
        f"Duplicate page IDs in compact list: {real_ids}"
    # Verify: matches mask population
    expected = torch.where(mask[0, 0, 0])[0].sort().values.tolist()
    assert real_ids == expected
