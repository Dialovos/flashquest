"""All -1 sentinel + small cache: kernel must not OOB on negative
page index. Validates p_safe gating before address arithmetic."""
import pytest
import torch


def test_neg1_does_not_oob_int4():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact

    device = "cuda"
    D, page_size = 128, 64
    B, H_q, H_kv, P = 1, 1, 1, 2
    S_kv = P * page_size

    Q = torch.zeros(B, H_q, 1, D, dtype=torch.bfloat16, device=device)
    K_packed = torch.zeros(B, H_kv, S_kv, D // 2, dtype=torch.uint8, device=device)
    V_packed = torch.zeros(B, H_kv, S_kv, D // 2, dtype=torch.uint8, device=device)
    K_scale = torch.ones(B, H_kv, P, D, dtype=torch.bfloat16, device=device)
    K_mn = torch.zeros(B, H_kv, P, D, dtype=torch.bfloat16, device=device)
    V_scale = torch.ones(B, H_kv, S_kv, dtype=torch.bfloat16, device=device)
    V_mn = torch.zeros(B, H_kv, S_kv, dtype=torch.bfloat16, device=device)

    sel = torch.full((B, H_q, 1, 4), -1, dtype=torch.int32, device=device)
    O, lse = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    torch.cuda.synchronize()
    assert O.shape == (B, H_q, 1, D)
    assert (O == 0).all()
    assert (lse == float("-inf")).all()
