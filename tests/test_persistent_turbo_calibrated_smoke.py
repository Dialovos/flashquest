"""Phase 11 — PersistentTurboKVCache calibration plumbing smoke (slow)."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
def test_persistent_turbo_calibrated_default_loads_codebook_for_3b():
    """When initialized with model_id of a calibrated model, cache loads the artifact."""
    from flashquest.cache.persistent_turbo import PersistentTurboKVCache

    cache = PersistentTurboKVCache(
        batch_size=1, num_layers=28, num_kv_heads=8, head_dim=64,
        max_seq_len=512, page_size=64, device="cuda",
        model_id="casperhansen/llama-3.2-3b-instruct-awq",
    )
    assert cache.codebook_k.shape == (28, 8)
    assert cache.codebook_v.shape == (28, 8)
    # Calibrated codebooks should NOT match paper (otherwise calibration was a no-op).
    from flashquest.turbo.codebook import PAPER_CODEBOOK
    paper = PAPER_CODEBOOK.to(cache.codebook_k.device).expand(28, 8)
    assert not torch.allclose(cache.codebook_k, paper, atol=1e-4)


@pytest.mark.slow
def test_persistent_turbo_calibrated_falls_back_to_paper_for_unknown_model():
    """Unknown model_id: cache warns and falls back to paper codebook."""
    from flashquest.cache.persistent_turbo import PersistentTurboKVCache
    from flashquest.turbo.codebook import PAPER_CODEBOOK

    with pytest.warns(UserWarning, match="paper"):
        cache = PersistentTurboKVCache(
            batch_size=1, num_layers=4, num_kv_heads=2, head_dim=64,
            max_seq_len=128, page_size=64, device="cuda",
            model_id="nonexistent/model-id",
        )

    expected = PAPER_CODEBOOK.to(cache.codebook_k.device).expand(4, 8)
    assert torch.allclose(cache.codebook_k, expected)
    assert torch.allclose(cache.codebook_v, expected)


@pytest.mark.slow
def test_persistent_turbo_calibrated_quant_uses_per_layer_codebook():
    """update_quantized routes the per-layer codebook into quantize_k/v_turbo."""
    from flashquest.cache.persistent_turbo import PersistentTurboKVCache
    from flashquest.kernel.kv_quant import dequantize_k_turbo

    # No model_id: this test injects its own per-layer codebooks below, so it must
    # not load the shipped 28-layer artifact into a 2-layer cache (a deliberate
    # ValueError). Default construction broadcasts the paper codebook to (2,2,8).
    cache = PersistentTurboKVCache(
        batch_size=1, num_layers=2, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
    )
    # Inject hand-picked codebooks for layers 0 and 1 to verify per-layer routing.
    layer0_cb = torch.tensor([-2.0, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 2.0],
                             dtype=torch.float32, device="cuda")
    layer1_cb = torch.tensor([-1.8, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.8],
                             dtype=torch.float32, device="cuda")
    cache.codebook_k = torch.stack([layer0_cb, layer1_cb])
    cache.codebook_v = torch.stack([layer0_cb, layer1_cb])

    torch.manual_seed(0)
    K_new = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V_new = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")

    cache.update_quantized(K_new, V_new, layer_idx=0)
    K_rt = dequantize_k_turbo(
        cache.K_msb[0], cache.K_lsb[0], cache.K_scale_turbo[0],
        head_dim=64, codebook=cache.codebook_k[0],
    )
    # K_msb[0] is the full pre-allocated buffer (max_seq_len); compare only the
    # 64 tokens just written to layer 0.
    K_rt = K_rt[:, :, : K_new.shape[2], :]
    err = (K_new.float() - K_rt.float()).abs().mean()
    assert err < 0.6, f"layer-0 round-trip err {err:.3f} > 0.6"
