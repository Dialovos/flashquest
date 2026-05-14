import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_causal_prefill_matches_sdpa():
    """E5: standard prefill with S_q == S_kv, causal=True."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 256, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_causal_decode_step():
    """E6: S_q=1 against S_kv=512. Causal trivializes; output is dense over all kv."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S_q, S_kv, D = 1, 4, 1, 512, 64
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")

    O_causal, _ = flash_attn_fwd(Q, K, V, causal=True)
    O_dense, _ = flash_attn_fwd(Q, K, V, causal=False)
    torch.testing.assert_close(O_causal, O_dense, rtol=0, atol=0)


@cuda
def test_causal_first_token_no_nan():
    """E10: q at position 0 attends only to k at position 0 (one term in softmax).
    Must not NaN."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)
    assert torch.isfinite(O).all(), "causal first-row produced NaN/Inf"

    torch.testing.assert_close(O[0, 0, 0], V[0, 0, 0], rtol=1e-2, atol=1e-2)


@cuda
def test_chunked_prefill_rejected():
    """E13: causal with S_q != S_kv, both > 1, must raise NotImplementedError."""
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="chunked prefill"):
        flash_attn_fwd(Q, K, V, causal=True)
