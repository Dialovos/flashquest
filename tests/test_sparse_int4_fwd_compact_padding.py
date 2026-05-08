"""Compact INT4 kernel handles -1 sentinel padding correctly:
output identical to a tighter BUCKET_MAX with no padding,
and an all-sentinel input produces O=0, lse=-inf."""
import pytest
import torch


@pytest.fixture
def kernel_inputs():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    device = "cuda"
    D, page_size = 128, 64
    B, H_q, H_kv, P = 1, 4, 2, 16
    S_kv = P * page_size

    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device=device)
    K_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    V_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    K_scale = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device).abs()
    K_mn = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device)
    V_scale = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device).abs()
    V_mn = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device)
    return dict(Q=Q, K_packed=K_packed, K_scale=K_scale, K_mn=K_mn,
                V_packed=V_packed, V_scale=V_scale, V_mn=V_mn,
                B=B, H_q=H_q, P=P, page_size=page_size)


def test_padding_does_not_change_output(kernel_inputs):
    from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact
    inp = kernel_inputs
    sel_tight = torch.tensor([[[[1, 5, 9, 13]] for _ in range(inp["H_q"])]],
                              dtype=torch.int32, device="cuda")
    sel_padded = torch.tensor(
        [[[[1, 5, 9, 13, -1, -1, -1, -1]] for _ in range(inp["H_q"])]],
        dtype=torch.int32, device="cuda",
    )
    O_tight, lse_tight = flash_attn_sparse_int4_fwd_compact(
        inp["Q"], inp["K_packed"], inp["K_scale"], inp["K_mn"],
        inp["V_packed"], inp["V_scale"], inp["V_mn"],
        selected_page_ids=sel_tight, page_size=inp["page_size"], return_lse=True,
    )
    O_pad, lse_pad = flash_attn_sparse_int4_fwd_compact(
        inp["Q"], inp["K_packed"], inp["K_scale"], inp["K_mn"],
        inp["V_packed"], inp["V_scale"], inp["V_mn"],
        selected_page_ids=sel_padded, page_size=inp["page_size"], return_lse=True,
    )
    torch.testing.assert_close(O_pad, O_tight, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lse_pad, lse_tight, atol=1e-4, rtol=1e-4)


def test_all_sentinel_zero_output_neg_inf_lse(kernel_inputs):
    """All-sentinel input: O = 0, lse = -inf (codex r3 confirmation)."""
    from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact
    inp = kernel_inputs
    BUCKET_MAX = 8
    sel_all_neg = torch.full(
        (inp["B"], inp["H_q"], 1, BUCKET_MAX), -1,
        dtype=torch.int32, device="cuda",
    )
    O, lse = flash_attn_sparse_int4_fwd_compact(
        inp["Q"], inp["K_packed"], inp["K_scale"], inp["K_mn"],
        inp["V_packed"], inp["V_scale"], inp["V_mn"],
        selected_page_ids=sel_all_neg, page_size=inp["page_size"], return_lse=True,
    )
    assert torch.equal(O, torch.zeros_like(O))
    assert (lse == float("-inf")).all()
