"""Compact INT4 kernel must match the bool-mask kernel on identical inputs."""
import pytest
import torch

from flashquest.eager.selection import build_compact_selection


@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compact_int4_parity_random_selection(D, seed):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd
    from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact

    torch.manual_seed(seed)
    device = "cuda"

    B, H_q, H_kv, P, page_size = 1, 8, 4, 32, 64
    S_kv = P * page_size  # 2048

    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device=device)

    K_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    V_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    K_scale = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device).abs()
    K_mn = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device)
    V_scale = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device).abs()
    V_mn = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device)

    sel_mask = torch.zeros(B, H_q, 1, P, dtype=torch.bool, device=device)
    for h in range(H_q):
        idx = torch.randperm(P, device=device)[:8]
        sel_mask[0, h, 0, idx] = True

    O_ref, lse_ref = flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selection_mask=sel_mask, page_size=page_size, return_lse=True,
    )

    sel_compact = build_compact_selection(sel_mask, BUCKET_MAX=8)
    O_compact, lse_compact = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
    )

    torch.testing.assert_close(O_compact, O_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_compact, lse_ref, atol=1e-3, rtol=1e-3)
