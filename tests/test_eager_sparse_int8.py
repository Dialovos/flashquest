import torch

from flashquest.eager import quest_eager_sdpa
from flashquest.eager.sparse_int8 import quest_eager_sparse_int8
from flashquest.kernel.kv_quant import quantize_k, quantize_v


def test_full_retention_close_to_bf16_eager():
    """retention=1.0: INT8 path within INT8 quant tolerance of BF16 path."""
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 4, 1, 256, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)

    O_bf16 = quest_eager_sdpa(
        Q, K, V, page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_int8 = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=1.0, num_sinks=0, window_pages=0, is_causal=False,
    )

    torch.testing.assert_close(O_int8, O_bf16, rtol=5e-2, atol=5e-2)


def test_decode_step_int8():
    """ES4: S_q=1 decode case."""
    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 4, 1, 1024, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)

    O_bf16 = quest_eager_sdpa(
        Q, K, V, page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_int8 = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )
    rel_err = (O_int8 - O_bf16).norm() / O_bf16.norm()
    assert rel_err < 0.1, f"rel_err={rel_err.item():.4f}"


def test_no_pages_selected_returns_zero():
    """ES2: when retention=0 and no sinks/window, output should be zero
    (no information attended)."""
    torch.manual_seed(0)
    B, H_q, H_kv, S_kv, D = 1, 1, 1, 128, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16)
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O = quest_eager_sparse_int8(
        Q,
        K_uint8, K_scale, K_mn,
        V_uint8, V_scale, V_mn,
        page_size=64, retention=0.0, num_sinks=0, window_pages=0, is_causal=False,
    )
    assert torch.equal(O, torch.zeros_like(O)), "no-pages-selected must produce zero"
