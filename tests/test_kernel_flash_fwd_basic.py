import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_basic_non_causal_matches_sdpa():
    """Single full tile, no causal, no GQA: kernel == torch SDPA."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _lse = flash_attn_fwd(Q, K, V, causal=False)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_gqa_two_kv_heads():
    """GQA n_rep=4 (mimics Llama-3.2-1B 32:8): kernel == SDPA on repeated K/V."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=False)

    n_rep = H_q // H_kv
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=False)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_head_dim_128():
    """head_dim=128 (Llama-3.1 family) supported."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 128
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=False)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


def test_rejects_zero_length():
    """E14: empty inputs are a Python-level error, not a kernel one."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)
