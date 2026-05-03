"""Phase 6 task 5 — sparse INT4 forward: validates against dense reference."""
import pytest
import torch

from flashquest.kernel.kv_quant import (
    quantize_k_int4, quantize_v_int4,
    dequantize_k_int4, dequantize_v_int4,
)
from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd


def test_sparse_int4_matches_dense():
    """All-pages-selected ≡ dense attention on the dequantized KV (loose tol for INT4)."""
    torch.manual_seed(13)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 4, 2, 1, 256, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)

    selection_mask = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")

    out, _ = flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn,
        V_packed, V_scale, V_mn,
        selection_mask=selection_mask,
        page_size=page_size, sm_scale=D ** -0.5, return_lse=True,
    )

    K_deq = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)
    V_deq = dequantize_v_int4(V_packed, V_scale, V_mn)
    n_rep = H_q // H_kv
    K_deq_q = K_deq.repeat_interleave(n_rep, dim=1)
    V_deq_q = V_deq.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        Q.float(), K_deq_q.float(), V_deq_q.float(), is_causal=False,
    ).to(torch.bfloat16)

    err = (out.float() - ref.float()).abs().max()
    assert err < 5e-2, f"max abs diff {err}"
