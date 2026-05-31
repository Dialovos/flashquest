import torch
from flashquest.eager.selection import build_compact_union_selection

def test_union_forces_sinks_and_window_and_includes_members():
    B, Hq, Sq, P = 1, 1, 2, 10
    sel = torch.zeros(B, Hq, Sq, P, dtype=torch.bool)
    sel[0, 0, 0, 5] = True; sel[0, 0, 1, 6] = True          # union {5, 6}
    scores = torch.zeros(B, Hq, Sq, P)
    scores[0, 0, 0, 5] = 9.0; scores[0, 0, 1, 6] = 8.0
    out = build_compact_union_selection(sel, scores, num_sinks=1, window_pages=1,
        completed_len=P * 64, page_size=64, BUCKET_MAX_UNION=4)
    got = {int(x) for x in out[0, 0].tolist() if x >= 0}
    assert 0 in got and (P - 1) in got and 5 in got and 6 in got
    assert out.shape == (B, Hq, 4) and out.dtype == torch.int32

def test_union_overflow_keeps_highest_score():
    B, Hq, Sq, P = 1, 1, 1, 10
    sel = torch.zeros(B, Hq, Sq, P, dtype=torch.bool)
    sel[0, 0, 0, [2, 3, 4, 7]] = True
    scores = torch.zeros(B, Hq, Sq, P)
    scores[0, 0, 0, [2, 3, 4, 7]] = torch.tensor([1.0, 5.0, 2.0, 9.0])
    out = build_compact_union_selection(sel, scores, num_sinks=0, window_pages=0,
        completed_len=P * 64, page_size=64, BUCKET_MAX_UNION=2)
    assert {int(x) for x in out[0, 0].tolist() if x >= 0} == {3, 7}

def test_underflow_pads_with_sentinels():
    B, Hq, Sq, P = 1, 1, 1, 10
    sel = torch.zeros(B, Hq, Sq, P, dtype=torch.bool); sel[0, 0, 0, 4] = True
    scores = torch.zeros(B, Hq, Sq, P); scores[0, 0, 0, 4] = 1.0
    out = build_compact_union_selection(sel, scores, num_sinks=0, window_pages=0,
        completed_len=P * 64, page_size=64, BUCKET_MAX_UNION=5)
    assert out.shape == (B, Hq, 5)
    assert (out[0, 0] == -1).sum().item() == 4 and 4 in out[0, 0].tolist()
