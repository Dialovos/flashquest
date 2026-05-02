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
