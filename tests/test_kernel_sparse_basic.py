import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_inputs(B, H_q, H_kv, S_kv, D, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    return Q, K, V


@cuda
def test_full_mask_matches_eager_int8():
    """ES1: selection_mask all True (full retention) ≡ eager INT8 sparse @ retention=1.0."""
    from flashquest.eager import quest_eager_sparse_int8
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 4, 1, 256, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    num_pages = S_kv // 64
    selection_mask = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )

    O_eager = quest_eager_sparse_int8(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_partial_mask_matches_eager_int8():
    """selection_mask with sinks + window + a couple top-k pages."""
    from flashquest.eager.sparse_int8 import quest_eager_sparse_int8
    from flashquest.eager.criticality import page_scores
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 4, 1, 1024, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D, seed=1)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)

    K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=64)
    K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
    scores = page_scores(Q.float(), pmin, pmax)
    selection_mask = select_pages(scores, retention=0.25, num_sinks=4, window_pages=2)

    O_kernel, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )

    O_eager = quest_eager_sparse_int8(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=5e-2, atol=5e-2)


@cuda
def test_no_pages_selected_returns_zero():
    """ES2: all-False mask -> output is zero."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    B, H_q, H_kv, S_kv, D = 1, 1, 1, 128, 64
    Q, K, V = _make_inputs(B, H_q, H_kv, S_kv, D)

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    num_pages = 2
    selection_mask = torch.zeros(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O, _ = flash_attn_sparse_fwd(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        selection_mask=selection_mask, page_size=64,
    )
    assert torch.equal(O, torch.zeros_like(O))


@cuda
def test_rejects_multi_query():
    """ES11: S_q > 1 is rejected at the wrapper (Phase 3 v1 is decode-only)."""
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import quantize_k, quantize_v

    Q = torch.randn(1, 1, 8, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    selection_mask = torch.ones(1, 1, 8, 1, dtype=torch.bool, device="cuda")
    with pytest.raises(NotImplementedError, match="decode-only"):
        flash_attn_sparse_fwd(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            selection_mask=selection_mask, page_size=64,
        )
