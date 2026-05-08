"""build_compact_selection converts a bool mask to a compact int32 list."""
import pytest
import torch

from flashquest.eager.selection import build_compact_selection


@pytest.mark.parametrize("BUCKET_MAX", [4, 8, 16])
def test_compact_basic(BUCKET_MAX):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # P = 8 pages; select pages [1, 3, 5]
    B, H, S_q, P = 1, 1, 1, 8
    mask = torch.zeros(B, H, S_q, P, dtype=torch.bool, device="cuda")
    mask[0, 0, 0, 1] = True
    mask[0, 0, 0, 3] = True
    mask[0, 0, 0, 5] = True

    out = build_compact_selection(mask, BUCKET_MAX=BUCKET_MAX)
    assert out.shape == (B, H, S_q, BUCKET_MAX)
    assert out.dtype == torch.int32

    real = sorted(out[0, 0, 0, :3].tolist())
    assert real == [1, 3, 5]
    if BUCKET_MAX > 3:
        assert (out[0, 0, 0, 3:] == -1).all()


def test_compact_all_false():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    B, H, S_q, P = 1, 1, 1, 8
    mask = torch.zeros(B, H, S_q, P, dtype=torch.bool, device="cuda")
    out = build_compact_selection(mask, BUCKET_MAX=4)
    assert (out == -1).all(), "all-False mask should produce all -1 sentinels"


def test_compact_all_true():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    B, H, S_q, P = 1, 1, 1, 4
    mask = torch.ones(B, H, S_q, P, dtype=torch.bool, device="cuda")
    out = build_compact_selection(mask, BUCKET_MAX=4)
    assert sorted(out[0, 0, 0].tolist()) == [0, 1, 2, 3]
    assert (out >= 0).all()


def test_compact_p_smaller_than_bucket():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # P=4 < BUCKET_MAX=8 — output is padded with -1 to BUCKET_MAX
    B, H, S_q, P = 1, 1, 1, 4
    mask = torch.tensor([[[[True, False, True, True]]]], device="cuda")
    out = build_compact_selection(mask, BUCKET_MAX=8)
    assert out.shape == (B, H, S_q, 8)
    real = sorted(int(x) for x in out[0, 0, 0].tolist() if int(x) != -1)
    assert real == [0, 2, 3]
    assert (out[0, 0, 0, 3:] == -1).all()


def test_compact_per_head_independent():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    # Each head has different selections — verify independence
    B, H, S_q, P = 1, 2, 1, 8
    mask = torch.zeros(B, H, S_q, P, dtype=torch.bool, device="cuda")
    mask[0, 0, 0, [1, 4]] = True
    mask[0, 1, 0, [2, 6, 7]] = True
    out = build_compact_selection(mask, BUCKET_MAX=4)
    assert sorted(x for x in out[0, 0, 0, :2].tolist()) == [1, 4]
    assert sorted(x for x in out[0, 1, 0, :3].tolist()) == [2, 6, 7]
    assert (out[0, 0, 0, 2:] == -1).all()
    assert (out[0, 1, 0, 3:] == -1).all()
