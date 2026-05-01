"""Phase 5 fused DuoAttention dispatch — single sparse-kernel call with
per-head retention."""
import pytest
import torch

from flashquest.duo.dispatch import quest_duo_eager_sdpa
from flashquest.duo.fused_dispatch import quest_duo_fused_sdpa
from flashquest.kernel.kv_quant import quantize_k, quantize_v


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _setup(B=1, H_q=4, H_kv=2, S_q=1, S_kv=512, D=64, seed=0):
    torch.manual_seed(seed)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    return Q, K, V


def test_fused_all_retrieval_matches_eager():
    Q, K, V = _setup()
    head_pattern = torch.ones(2, dtype=torch.bool, device="cuda")  # all retrieval

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )

    # rtol=2e-2 because fused goes through INT8 quant; eager is BF16.
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)


def test_fused_all_streaming_matches_eager():
    Q, K, V = _setup(seed=1)
    head_pattern = torch.zeros(2, dtype=torch.bool, device="cuda")  # all streaming

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)


def test_fused_mixed_pattern_matches_eager():
    """Mixed pattern must match Phase 4's torch.where path within INT8 noise."""
    Q, K, V = _setup(seed=2)
    head_pattern = torch.tensor([True, False], device="cuda")  # head 0 retrieval, 1 streaming

    O_eager = quest_duo_eager_sdpa(
        Q, K, V, head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2, is_causal=False,
    )

    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    O_fused = quest_duo_fused_sdpa(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        head_pattern=head_pattern,
        page_size=64, retention=0.25, num_sinks=4, window_pages=2,
    )
    torch.testing.assert_close(O_fused, O_eager, rtol=2e-2, atol=2e-2)


def test_fused_rejects_prefill():
    Q = torch.randn(1, 4, 8, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    with pytest.raises(NotImplementedError, match="decode-only"):
        quest_duo_fused_sdpa(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            head_pattern=torch.tensor([True, True], device="cuda"),
            page_size=64, retention=0.25, num_sinks=4, window_pages=2,
        )


def test_fused_pattern_shape_mismatch_raises():
    Q, K, V = _setup()
    K_uint8, K_scale, K_mn = quantize_k(K, page_size=64)
    V_uint8, V_scale, V_mn = quantize_v(V)
    with pytest.raises(ValueError, match=r"head_pattern must be \(2,\)"):
        quest_duo_fused_sdpa(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            head_pattern=torch.tensor([True, True, True], device="cuda"),
            page_size=64, retention=0.25, num_sinks=4, window_pages=2,
        )
