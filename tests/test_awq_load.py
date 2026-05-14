"""AWQ load smoke test. Skipped when checkpoint isn't cached locally."""
import pytest
import torch

pytestmark = pytest.mark.slow


def _have_awq_3b() -> bool:
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("casperhansen/llama-3.2-3b-instruct-awq")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_awq_3b(), reason="AWQ checkpoint not cached")
def test_load_awq_3b_runs_forward():
    from flashquest.runtime.awq_load import load_awq_model

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    qcfg = model.config.quantization_config
    quant_method = qcfg.get("quant_method") if isinstance(qcfg, dict) else getattr(qcfg, "quant_method", None)
    assert quant_method == "awq"

    inp = tok("Hello, world.", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inp)
    assert torch.isfinite(out.logits).all()


def test_load_awq_rejects_non_awq():
    from flashquest.runtime.awq_load import load_awq_model
    with pytest.raises(ValueError, match="not AWQ-quantized"):
        load_awq_model("unsloth/Llama-3.2-1B-Instruct")
