"""Fused gate+up + SwiGLU must match separate gate/up + SwiGLU."""
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.slow
def test_fused_gate_up_matches_separate():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    from flashquest.kernel.fused_proj import fused_gate_up_proj
    from flashquest.runtime.awq_load import load_awq_model

    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    mlp0 = None
    for module in model.modules():
        if module.__class__.__name__ == "LlamaMLP":
            mlp0 = module
            break
    assert mlp0 is not None

    gate_proj = mlp0.gate_proj
    up_proj = mlp0.up_proj

    torch.manual_seed(0)
    hidden = torch.randn(1, 1, 3072, dtype=torch.float16, device="cuda")

    gate_ref = gate_proj(hidden)
    up_ref = up_proj(hidden)
    swiglu_ref = F.silu(gate_ref) * up_ref

    gate_f, up_f = fused_gate_up_proj(hidden, gate_proj, up_proj)
    swiglu_f = F.silu(gate_f) * up_f

    torch.testing.assert_close(swiglu_f, swiglu_ref, atol=5e-2, rtol=5e-2)
