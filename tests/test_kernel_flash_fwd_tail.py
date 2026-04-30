import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("S", [1, 17, 32, 63, 64, 65, 100, 127, 128, 129, 255])
@pytest.mark.parametrize("causal", [False, True])
def test_arbitrary_seq_lengths(S, causal):
    """E1, E2, E3: kernel handles S not multiple of BLOCK_M / BLOCK_N."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(S * 7 + int(causal))
    B, H, D = 1, 2, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=causal)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_decode_against_arbitrary_kv_length():
    """E6: S_q=1 decode against S_kv ∈ {1, 31, 64, 100, 1023, 1024}.

    For decode, the single query is conceptually at virtual position S_kv-1
    and attends to all S_kv prior tokens. Our kernel and Phase 1 eager both
    encode this. PyTorch's `is_causal=True` SDPA, however, anchors the
    triangle at the top-left and would mask everything past column 0 —
    that's wrong for decode. So the reference uses `is_causal=False`
    (which is equivalent to "attend to everything you can see").
    """
    from flashquest.kernel import flash_attn_fwd

    for S_kv in [1, 31, 64, 100, 1023, 1024]:
        torch.manual_seed(S_kv)
        B, H, S_q, D = 1, 4, 1, 64
        Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")

        O, _ = flash_attn_fwd(Q, K, V, causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
        assert torch.allclose(O, ref, rtol=1e-2, atol=1e-2), f"S_kv={S_kv} mismatch"
