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


def test_page_scores_int4_fast_matches_slow():
    """page_scores_int4_fast ≡ Σ_d max(Q[d]·K_mn[p,d], Q[d]·K_mx[p,d]) per the
    Quest criticality definition (sum-of-per-element-max upper bound).

    Algebraic identity (since K_scale ≥ 0): per-element
        max(Q[d]·K_mn, Q[d]·(K_mn + 15·K_scale)) = Q[d]·K_mn + 15·relu(Q[d])·K_scale
    summed over D.
    """
    from flashquest.eager.criticality import page_scores_int4_fast

    torch.manual_seed(11)
    B, H_q, H_kv, S_q, P, D = 1, 4, 2, 1, 8, 64
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.float32)
    K_mn = torch.randn(B, H_kv, P, D, dtype=torch.float32)
    K_scale = torch.randn(B, H_kv, P, D, dtype=torch.float32).abs() + 1e-3
    K_mx = K_mn + 15.0 * K_scale

    # Slow reference: Σ_d max(Q[d]·K_mn[p,d], Q[d]·K_mx[p,d]).
    n_rep = H_q // H_kv
    Kmn_e = K_mn.repeat_interleave(n_rep, dim=1).unsqueeze(2)
    Kmx_e = K_mx.repeat_interleave(n_rep, dim=1).unsqueeze(2)
    Q_e = Q.unsqueeze(3)
    cand_mn = Q_e * Kmn_e
    cand_mx = Q_e * Kmx_e
    expected = torch.maximum(cand_mn, cand_mx).sum(dim=-1)

    got = page_scores_int4_fast(Q, K_scale.to(torch.bfloat16), K_mn.to(torch.bfloat16))
    assert torch.allclose(got, expected, rtol=1e-3, atol=1e-3), \
        f"max abs diff {(got - expected).abs().max()}"
