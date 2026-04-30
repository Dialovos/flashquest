"""Edge case grid for flash_attn_sparse_fwd. See plan §Edge case catalog."""
import pytest
import torch
from hypothesis import given, settings, strategies as st

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(B, H_q, H_kv, S_kv, D, page_size=64, seed=0):
    """Returns (Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn) all cuda."""
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=page_size)
    V_uint8, V_scale, V_mn = quantize_v(V)
    return Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn


@cuda
@pytest.mark.parametrize("S_kv", [64, 96, 100, 128, 192, 1023])
def test_partial_last_page(S_kv):
    """ES3: S_kv not multiple of page_size."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 2, 1, S_kv, 64, seed=S_kv)
    num_pages = K_s.shape[2]
    sel = torch.ones(1, 2, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
@pytest.mark.parametrize("n_rep", [2, 4, 8])
def test_gqa(n_rep):
    """ES5: GQA with selection per query head."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k, dequantize_v

    H_kv, D, S_kv = 2, 64, 256
    H_q = H_kv * n_rep
    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, H_q, H_kv, S_kv, D, seed=n_rep)
    num_pages = S_kv // 64
    sel = torch.zeros(1, H_q, 1, num_pages, dtype=torch.bool, device="cuda")
    for h in range(H_q):
        sel[0, h, 0, h % num_pages] = True
    sel[..., 0] = True
    sel[..., -1] = True

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    K = dequantize_k(K_u8, K_s, K_m, page_size=64)
    V = dequantize_v(V_u8, V_s, V_m)
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    token_mask = sel.repeat_interleave(64, dim=-1)[..., :S_kv]
    attn_bias = torch.zeros_like(token_mask, dtype=torch.bfloat16)
    attn_bias = attn_bias.masked_fill(~token_mask, float("-inf"))
    O_ref = torch.nn.functional.scaled_dot_product_attention(
        Q, Kr, Vr, attn_mask=attn_bias, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_ref, rtol=5e-2, atol=5e-2)


@cuda
@pytest.mark.parametrize("D", [64, 128])
def test_head_dim(D):
    """ES6: head_dim ∈ {64, 128}."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 2, 1, 128, D, seed=D)
    num_pages = 2
    sel = torch.ones(1, 2, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_single_page():
    """ES7: S_kv == page_size."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd

    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(1, 1, 1, 64, 64)
    sel = torch.ones(1, 1, 1, 1, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_zero_range_channel():
    """ES9: a constant-valued channel does not produce NaN."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    torch.manual_seed(0)
    B, H, S_kv, D = 1, 1, 128, 64
    Q = torch.randn(B, H, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    K[..., 5] = 0.7
    V[..., 5] = 0.3

    K_u8, K_s, K_m = quantize_k(K, page_size=64)
    V_u8, V_s, V_m = quantize_v(V)
    sel = torch.ones(B, H, 1, 2, dtype=torch.bool, device="cuda")

    O, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    assert torch.isfinite(O).all()


@cuda
@settings(deadline=None, max_examples=15)
@given(
    S_kv=st.integers(min_value=64, max_value=512),
    H_kv=st.sampled_from([1, 2]),
    n_rep=st.sampled_from([1, 2, 4]),
    D=st.sampled_from([64, 128]),
    retention=st.sampled_from([0.25, 0.5, 1.0]),
)
def test_random_shapes_match_eager(S_kv, H_kv, n_rep, D, retention):
    """Property fuzz: random shape grid, mask built from eager Quest, kernel
    output matches eager INT8 sparse within INT8 tolerance."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.eager.criticality import page_scores
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k

    H_q = H_kv * n_rep
    Q, K_u8, K_s, K_m, V_u8, V_s, V_m = _make_inputs(
        1, H_q, H_kv, S_kv, D, seed=S_kv * 31 + H_kv * 7 + n_rep + D
    )
    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=64)
    K_dq_rep = K_dq.repeat_interleave(n_rep, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
    scores = page_scores(Q.float(), pmin, pmax)
    sel = select_pages(scores, retention=retention, num_sinks=4, window_pages=2)

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )
    O_eager = quest_eager_sparse_int8(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m,
        page_size=64, retention=retention, num_sinks=4, window_pages=2, is_causal=False,
    )
    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)
