"""EQ18-EQ20: page_scores_int8 — algebraic Quest criticality direct from
quantization params. Validated against the Phase 5 path
(dequantize_k -> compute_page_summary -> page_scores).
"""
import torch
import pytest

from flashquest.eager.criticality import page_scores, page_scores_int8
from flashquest.eager.page_summary import compute_page_summary
from flashquest.kernel.kv_quant import dequantize_k, quantize_k


def _make_kv(B=1, H_kv=2, S=256, D=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    K = torch.randn(B, H_kv, S, D, generator=g, device="cuda", dtype=dtype)
    return K


def test_eq18_int8_scores_match_dequant_path():
    """EQ18: page_scores_int8 ≡ page_scores(compute_page_summary(dequantize_k(...)))."""
    page_size = 64
    B, H_kv, H_q, S, D = 1, 2, 4, 256, 64  # n_rep = 2
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)

    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    # Phase 5 path:
    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    n_rep = H_q // H_kv
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    ref = page_scores(Q.float(), page_min, page_max)

    # New path:
    out = page_scores_int8(Q, K_scale, K_mn)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("H_kv,n_rep", [(2, 1), (2, 4), (8, 3)])
def test_eq19_gqa_broadcast(H_kv, n_rep):
    """EQ19: each query head sees its corresponding KV head's K_scale/K_mn."""
    page_size = 64
    B, S, D = 1, 256, 64
    H_q = H_kv * n_rep
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D, seed=H_kv * 7 + n_rep)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)

    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    ref = page_scores(Q.float(), page_min, page_max)

    out = page_scores_int8(Q, K_scale, K_mn)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


def test_eq20_shape_and_dtype_contract():
    """EQ20: output shape (B, H_q, S_q, P) and fp32 dtype."""
    page_size = 64
    B, H_kv, H_q, S, D = 1, 2, 4, 192, 64  # 192/64 = 3 pages
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D)
    _, K_scale, K_mn = quantize_k(K, page_size=page_size)
    Q = torch.randn(B, H_q, 5, D, device="cuda", dtype=torch.bfloat16)
    out = page_scores_int8(Q, K_scale, K_mn)
    assert out.shape == (B, H_q, 5, 3)
    assert out.dtype == torch.float32


def test_eq20b_h_q_not_divisible_by_h_kv_raises():
    """EQ20: ValueError when GQA group is invalid."""
    page_size = 64
    B, H_kv, S, D = 1, 3, 64, 64
    _, K_scale, K_mn = quantize_k(_make_kv(B, H_kv, S, D), page_size=page_size)
    Q = torch.randn(B, 4, 1, D, device="cuda", dtype=torch.bfloat16)  # 4 not div by 3
    with pytest.raises(ValueError, match="divisible"):
        page_scores_int8(Q, K_scale, K_mn)


@pytest.mark.parametrize("H_kv,n_rep", [(2, 1), (2, 4), (8, 3)])
@pytest.mark.parametrize("S", [128, 256, 1024])
def test_eq26_fast_matches_int8(H_kv, n_rep, S):
    """EQ26: page_scores_int8_fast (two-matmul) ≡ page_scores_int8 (broadcast)."""
    from flashquest.eager.criticality import page_scores_int8_fast

    page_size = 64
    B, D = 1, 64
    H_q = H_kv * n_rep
    K = _make_kv(B=B, H_kv=H_kv, S=S, D=D, seed=H_kv * 17 + n_rep + S)
    _, K_scale, K_mn = quantize_k(K, page_size=page_size)
    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    ref = page_scores_int8(Q, K_scale, K_mn)
    out = page_scores_int8_fast(Q, K_scale, K_mn)
    assert out.shape == ref.shape
    # fp32 reduction reorder + matmul accumulator differences. Tight tolerance.
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
