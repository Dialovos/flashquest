import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_kernel_matches_eager_quest_at_full_retention():
    """Phase 2 dense kernel must equal Phase 1 eager Quest at retention=1.0
    (with no sinks, no window) — both compute the same dense attention."""
    from flashquest.eager import quest_eager_sdpa
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O_kernel, _ = flash_attn_fwd(Q, K, V, causal=True)

    O_eager = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=1.0,
        num_sinks=0,
        window_pages=0,
        is_causal=True,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=1e-2, atol=1e-2)
