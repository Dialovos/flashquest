"""Verify AWQLayout shapes match a real Llama-3.2-3B-AWQ Linear (slow).

Marked slow because it loads ~2 GB of model weights.
"""
import pytest

torch = pytest.importorskip("torch")


@pytest.mark.slow
def test_awq_layout_matches_real_llama32_3b():
    from flashquest.runtime.awq_load import load_awq_model
    from flashquest.quant.awq_layout import assert_awq_layout

    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")

    # Walk the model; find the first LlamaAttention's q_proj.
    found_q = None
    found_k = None
    found_v = None
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            found_q = module.q_proj
            found_k = module.k_proj
            found_v = module.v_proj
            break
    assert found_q is not None, "No LlamaAttention found in loaded model"

    # Llama-3.2-3B: hidden=3072, q_proj→3072, k_proj/v_proj→1024 (GQA)
    assert_awq_layout(found_q, in_features=3072, out_features=3072, name="q_proj")
    assert_awq_layout(found_k, in_features=3072, out_features=1024, name="k_proj")
    assert_awq_layout(found_v, in_features=3072, out_features=1024, name="v_proj")
