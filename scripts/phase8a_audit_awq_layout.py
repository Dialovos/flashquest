"""Print AutoAWQ-loaded Llama-3.2-3B Linear layout. Run once to verify
the AWQLayout constants in src/flashquest/quant/awq_layout.py.

Usage: python scripts/phase8a_audit_awq_layout.py
"""
from __future__ import annotations

import sys


def main():
    from flashquest.runtime.awq_load import load_awq_model

    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")

    layers = []
    for module in model.modules():
        if module.__class__.__name__ == "LlamaAttention":
            layers.append(module)
    if not layers:
        print("ERROR: No LlamaAttention found", file=sys.stderr)
        sys.exit(1)

    layer0 = layers[0]
    print(f"Found {len(layers)} LlamaAttention layers (Llama-3.2-3B has 28).")
    print()
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        m = getattr(layer0, name)
        print(f"  layer0.{name}:")
        print(f"    qweight: shape={tuple(m.qweight.shape)} dtype={m.qweight.dtype}")
        print(f"    scales:  shape={tuple(m.scales.shape)}  dtype={m.scales.dtype}")
        print(f"    qzeros:  shape={tuple(m.qzeros.shape)}  dtype={m.qzeros.dtype}")
        if hasattr(m, "group_size"):
            print(f"    group_size: {m.group_size}")
        print()

    # MLP
    mlp = None
    for module in model.modules():
        if module.__class__.__name__ == "LlamaMLP":
            mlp = module
            break
    if mlp is not None:
        print("  layer0.mlp:")
        for name in ["gate_proj", "up_proj", "down_proj"]:
            m = getattr(mlp, name)
            print(f"    {name}: qweight={tuple(m.qweight.shape)} scales={tuple(m.scales.shape)} qzeros={tuple(m.qzeros.shape)}")


if __name__ == "__main__":
    main()
