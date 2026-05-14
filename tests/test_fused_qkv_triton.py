"""Fused Triton QKV must match (q_proj + k_proj + v_proj) AWQ Linear outputs."""
import pytest
import torch


@pytest.mark.slow
def test_fused_qkv_matches_separate_awq():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashquest.kernel.fused_proj import fused_qkv_proj
    from flashquest.runtime.awq_load import load_awq_model

    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    layer0 = None
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            layer0 = module
            break
    assert layer0 is not None

    q_proj = layer0.q_proj
    k_proj = layer0.k_proj
    v_proj = layer0.v_proj

    torch.manual_seed(0)
    hidden = torch.randn(1, 1, 3072, dtype=torch.float16, device="cuda")

    q_ref = q_proj(hidden)
    k_ref = k_proj(hidden)
    v_ref = v_proj(hidden)

    q_f, k_f, v_f = fused_qkv_proj(hidden, q_proj, k_proj, v_proj)

    torch.testing.assert_close(q_f, q_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(k_f, k_ref, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(v_f, v_ref, atol=5e-2, rtol=5e-2)
