import torch

from flashquest.eager.attention import quest_eager_sdpa


def test_full_retention_matches_dense_sdpa():
    """retention=1.0 with no sinks/window must match torch SDPA modulo numerics."""
    torch.manual_seed(0)
    B, H, S, D = 1, 4, 128, 64
    Q = torch.randn(B, H, S, D, dtype=torch.float32)
    K = torch.randn(B, H, S, D, dtype=torch.float32)
    V = torch.randn(B, H, S, D, dtype=torch.float32)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    out = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=1.0,
        num_sinks=0,
        window_pages=0,
        is_causal=True,
    )
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)


def test_gqa_repeat_kv():
    """Q heads > K/V heads is supported (Llama-3.2-1B is 32:8 GQA)."""
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.float32)
    K = torch.randn(B, H_kv, S, D, dtype=torch.float32)
    V = torch.randn(B, H_kv, S, D, dtype=torch.float32)

    out = quest_eager_sdpa(Q, K, V, page_size=64, retention=1.0, is_causal=True)
    assert out.shape == (B, H_q, S, D)


def test_partial_retention_close_to_dense_on_low_entropy():
    """When K mass is concentrated on a single page, top-k must capture it
    AND the resulting sparse output must approximate dense (because softmax
    is also concentrated on that page)."""
    torch.manual_seed(0)
    B, H, S, D = 1, 2, 256, 32
    K = torch.randn(B, H, S, D) * 0.01
    V = torch.randn(B, H, S, D)
    # Scale K by 1000x on page 1 so softmax is overwhelmingly concentrated there.
    K[..., 64:128, :] *= 1000.0
    Q = torch.randn(B, H, 1, D)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    out = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=0.25,  # 1 of 4 pages -> must be page 1
        num_sinks=0,
        window_pages=0,
        is_causal=False,
    )
    rel_err = (out - ref).norm() / ref.norm()
    assert rel_err < 0.05, f"rel_err={rel_err.item():.4f}"


def test_excluded_pages_truly_dropped():
    """Tokens in non-selected pages must contribute zero to the output."""
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 16
    Q = torch.randn(B, H, 1, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    out_sink = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=0.0,
        num_sinks=1,
        window_pages=0,
        is_causal=False,
    )

    ref = torch.nn.functional.scaled_dot_product_attention(
        Q, K[:, :, :64], V[:, :, :64], is_causal=False
    )
    torch.testing.assert_close(out_sink, ref, rtol=1e-4, atol=1e-4)
