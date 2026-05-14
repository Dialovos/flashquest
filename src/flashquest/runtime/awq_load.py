"""AWQ-INT4 model loading helper.

Wraps transformers' auto-AWQ path with two compatibility shims:
1. Aliases `transformers.activations.PytorchGELUTanh` to the renamed
   `GELUTanh` so autoawq 0.2.9 imports cleanly against transformers 4.57+.
2. AWQ CUDA kernels require fp16; we load in fp16 even though the rest of
   flashquest defaults to bf16.
"""
from __future__ import annotations

import torch


def _ensure_autoawq_compat() -> None:
    """Restore the `PytorchGELUTanh` alias autoawq 0.2.9 still imports."""
    import transformers.activations as _a

    if not hasattr(_a, "PytorchGELUTanh") and hasattr(_a, "GELUTanh"):
        _a.PytorchGELUTanh = _a.GELUTanh


def load_awq_model(
    name: str,
    *,
    attn_implementation: str = "sdpa",
    device_map: str = "cuda",
):
    """Load an AWQ-INT4 HF model. Returns (model, tokenizer)."""
    _ensure_autoawq_compat()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    qcfg = getattr(model.config, "quantization_config", None)
    if qcfg is None:
        raise ValueError(f"Model {name} is not AWQ-quantized (no quantization_config)")
    quant_method = qcfg.get("quant_method") if isinstance(qcfg, dict) else getattr(qcfg, "quant_method", None)
    if quant_method != "awq":
        raise ValueError(
            f"Model {name} is not AWQ-quantized (quant_method={quant_method})"
        )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name)
    return model, tok
