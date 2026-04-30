import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_sparse_full_mask_matches_dense_kernel():
    """At selection_mask=all-True, the Phase 3 sparse kernel and the Phase 2
    dense kernel compute the same attention, up to INT8 quantization error."""
    from flashquest.kernel import flash_attn_fwd, flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k, dequantize_v, quantize_k, quantize_v

    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 4, 1, 256, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_u8, K_s, K_m = quantize_k(K, page_size=64)
    V_u8, V_s, V_m = quantize_v(V)
    num_pages = S_kv // 64
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_sparse, _ = flash_attn_sparse_fwd(
        Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=64,
    )

    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=64)
    V_dq = dequantize_v(V_u8, V_s, V_m)
    O_dense, _ = flash_attn_fwd(Q, K_dq, V_dq, causal=False)

    torch.testing.assert_close(O_sparse, O_dense, rtol=5e-2, atol=5e-2)
