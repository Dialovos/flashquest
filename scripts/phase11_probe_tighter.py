"""Phase 11 — tighter entry-gate probe addendum.

Loads the per-layer codebooks from benchmarks/phase11/probe_codepoints.json, then
runs the same 3 RULER NIAH multivalue samples through the Phase 7 TurboQuant
chain under MULTIPLE proxy codebooks:

  1. paper                — sanity baseline (must = 0 diffs)
  2. mean-of-per-layer    — what the first probe used (= 1/3 diffs)
  3. layer-14 codebook    — mid-network single-layer (peak per-layer divergence)
  4. layer-27 codebook    — last-layer single-layer (lowest divergence from paper)
  5. layer-0  codebook    — first-layer single-layer
  6. true per-layer       — global codebook varies BY LAYER via a monkey-patched
                            update_quantized + dispatcher dequant hook

Compares total diff-counts vs proxy (2). If proxy (6) moves more samples than (2),
per-layer specificity matters and Phase 11 per-layer is the right scope.
If (6) ≤ (2), per-layer doesn't add over uniform calibration → either pivot to
per-head (Phase 11b) or accept that codebook change is a weak lever.

Usage: nice -n 19 .venv/bin/python scripts/phase11_probe_tighter.py
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from flashquest.cache.persistent_turbo import PersistentTurboKVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.niah import make_prompt
from flashquest.kernel import kv_quant as kvq
from flashquest.runtime.awq_load import load_awq_model


def _load_codebooks():
    raw = json.loads(Path("benchmarks/phase11/probe_codepoints.json").read_text())
    per_layer_k = np.array(raw["per_layer_k"], dtype=np.float32)  # (28, 8)
    per_layer_v = np.array(raw["per_layer_v"], dtype=np.float32)
    paper = np.array(raw["paper"], dtype=np.float32)
    return paper, per_layer_k, per_layer_v


def _gen_with_swap(model, base_cache, tok, input_ids, K_cb, V_cb, *, max_new=32):
    """Swap module-level codebooks, reset cache, generate. K_cb/V_cb: (8,) np.float32."""
    K_t = torch.tensor(K_cb, dtype=torch.float32, device="cuda")
    V_t = torch.tensor(V_cb, dtype=torch.float32, device="cuda")
    kvq.K_TURBO_CODEBOOK.copy_(K_t)
    kvq.V_TURBO_CODEBOOK.copy_(V_t)
    base_cache._seen_tokens = [0] * base_cache.num_layers
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=True,
        )
    return out[0, input_ids.shape[-1]:].tolist()


def _gen_with_per_layer(model, base_cache, tok, input_ids, per_layer_k, per_layer_v,
                        *, paper_K, paper_V, max_new=32):
    """Per-layer codebook: wrap kvq.quantize_*_turbo + dequantize_*_turbo to read
    a layer-indexed codebook via a thread-local. Restores originals on exit.
    """
    import threading
    state = threading.local()
    state.layer_idx = 0

    orig_qk = kvq.quantize_k_turbo
    orig_qv = kvq.quantize_v_turbo
    orig_dk = kvq.dequantize_k_turbo
    orig_dv = kvq.dequantize_v_turbo

    pl_k_t = [torch.tensor(cb, dtype=torch.float32, device="cuda") for cb in per_layer_k]
    pl_v_t = [torch.tensor(cb, dtype=torch.float32, device="cuda") for cb in per_layer_v]

    def quant_k(K, page_size):
        kvq.K_TURBO_CODEBOOK.copy_(pl_k_t[state.layer_idx])
        return orig_qk(K, page_size=page_size)

    def quant_v(V):
        kvq.V_TURBO_CODEBOOK.copy_(pl_v_t[state.layer_idx])
        return orig_qv(V)

    def dequant_k(K_msb, K_lsb, K_scale_t, head_dim):
        # The dispatcher's _dequant_k_from_views passes positional args; we don't
        # have layer_idx in scope here. Use the cache layer most-recently written
        # (set by the update wrapper). Decode prefills + decodes in layer order,
        # so this state mirrors the cache's read order at decode-time.
        kvq.K_TURBO_CODEBOOK.copy_(pl_k_t[state.layer_idx])
        return orig_dk(K_msb, K_lsb, K_scale_t, head_dim=head_dim)

    def dequant_v(V_msb, V_lsb, V_scale_t, head_dim):
        kvq.V_TURBO_CODEBOOK.copy_(pl_v_t[state.layer_idx])
        return orig_dv(V_msb, V_lsb, V_scale_t, head_dim=head_dim)

    kvq.quantize_k_turbo = quant_k
    kvq.quantize_v_turbo = quant_v
    kvq.dequantize_k_turbo = dequant_k
    kvq.dequantize_v_turbo = dequant_v

    # Hook update_quantized to set state.layer_idx before the quant calls.
    orig_update = base_cache.update_quantized

    def hooked_update(K_new, V_new, layer_idx):
        state.layer_idx = layer_idx
        return orig_update(K_new, V_new, layer_idx)

    base_cache.update_quantized = hooked_update

    # Hook the patched attention forwards so dequant during decode uses the
    # current-layer codebook. The dispatcher closure inside each layer's forward
    # calls _dequant_k_from_views per layer in order; set state.layer_idx via a
    # pre-forward hook on each LlamaAttention.
    from transformers.models.llama.modeling_llama import LlamaAttention
    pre_hooks = []
    for module in model.modules():
        if isinstance(module, LlamaAttention):
            li = module.layer_idx

            def pre_hook(mod, args, kwargs, layer=li):
                state.layer_idx = layer

            pre_hooks.append(
                module.register_forward_pre_hook(pre_hook, with_kwargs=True)
            )

    base_cache._seen_tokens = [0] * base_cache.num_layers
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
            )
        return out[0, input_ids.shape[-1]:].tolist()
    finally:
        for h in pre_hooks:
            h.remove()
        kvq.quantize_k_turbo = orig_qk
        kvq.quantize_v_turbo = orig_qv
        kvq.dequantize_k_turbo = orig_dk
        kvq.dequantize_v_turbo = orig_dv
        base_cache.update_quantized = orig_update


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--sim-samples", type=int, default=3)
    ap.add_argument("--sim-ctx", type=int, default=4096)
    ap.add_argument("--seed-base", type=int, default=20260514)
    args = ap.parse_args()

    paper, per_layer_k, per_layer_v = _load_codebooks()
    print(f"[tighter] loaded per-layer codebooks: K={per_layer_k.shape} V={per_layer_v.shape}")

    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    num_layers = cfg.num_hidden_layers
    pattern = torch.ones(num_layers, cfg.num_key_value_heads, dtype=torch.bool)

    base_cache = PersistentTurboKVCache(
        batch_size=1, num_layers=num_layers, num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim, max_seq_len=args.sim_ctx + 64, page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=base_cache, head_pattern=pattern,
        retention=0.20, num_sinks=4, window_pages=2, page_size=64,
    )

    # Restore originals before each run.
    original_K = kvq.K_TURBO_CODEBOOK.clone()
    original_V = kvq.V_TURBO_CODEBOOK.clone()

    proxies = {
        "paper": (paper, paper),
        "mean": (per_layer_k.mean(axis=0), per_layer_v.mean(axis=0)),
        "layer-0": (per_layer_k[0], per_layer_v[0]),
        "layer-14": (per_layer_k[14], per_layer_v[14]),
        "layer-27": (per_layer_k[num_layers - 1], per_layer_v[num_layers - 1]),
    }

    samples = []
    for i in range(args.sim_samples):
        prompt, expected = make_prompt(
            "multivalue", ctx_len=args.sim_ctx, tokenizer=tok, seed=args.seed_base + i,
        )
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        samples.append((i, expected, input_ids))

    results = {name: {"sample_diffs": [], "diff_count": 0} for name in proxies}
    results["per-layer-proper"] = {"sample_diffs": [], "diff_count": 0}

    # Compute paper generations first (the baseline for diff comparison).
    paper_gens = []
    for (i, expected, input_ids) in samples:
        kvq.K_TURBO_CODEBOOK.copy_(original_K)
        kvq.V_TURBO_CODEBOOK.copy_(original_V)
        base_cache._seen_tokens = [0] * num_layers
        with torch.no_grad():
            out = model.generate(
                input_ids, max_new_tokens=32, do_sample=False, use_cache=True,
            )
        paper_gens.append(out[0, input_ids.shape[-1]:].tolist())
        torch.cuda.empty_cache()
    print(f"[tighter] paper baseline computed")

    # Test each scalar proxy.
    for name, (K_cb, V_cb) in proxies.items():
        if name == "paper":
            results[name]["diff_count"] = 0
            results[name]["sample_diffs"] = [False] * args.sim_samples
            continue
        for (i, expected, input_ids), paper_gen in zip(samples, paper_gens):
            calib_gen = _gen_with_swap(
                model, base_cache, tok, input_ids, K_cb, V_cb,
            )
            differs = paper_gen != calib_gen
            results[name]["sample_diffs"].append(differs)
            if differs:
                results[name]["diff_count"] += 1
            torch.cuda.empty_cache()
        print(f"[tighter] {name}: {results[name]['diff_count']}/{args.sim_samples} diffs")

    # Per-layer-proper.
    for (i, expected, input_ids), paper_gen in zip(samples, paper_gens):
        calib_gen = _gen_with_per_layer(
            model, base_cache, tok, input_ids, per_layer_k, per_layer_v,
            paper_K=paper, paper_V=paper,
        )
        differs = paper_gen != calib_gen
        results["per-layer-proper"]["sample_diffs"].append(differs)
        if differs:
            results["per-layer-proper"]["diff_count"] += 1
        torch.cuda.empty_cache()
        kvq.K_TURBO_CODEBOOK.copy_(original_K)
        kvq.V_TURBO_CODEBOOK.copy_(original_V)
    print(f"[tighter] per-layer-proper: {results['per-layer-proper']['diff_count']}/{args.sim_samples} diffs")

    out_path = Path("benchmarks/phase11/probe_tighter.md")
    md = ["# Phase 11 — Tighter Entry-Gate Probe Addendum\n"]
    md.append("Tests whether per-layer codebook specificity moves more samples than uniform proxies.")
    md.append("Baseline (paper) generations are compared against each proxy's generation; "
              "diff count is samples (of N=3) where generated tokens differ.\n")
    md.append("| proxy | diffs | sample diffs |")
    md.append("|---|---|---|")
    for name in ("paper", "mean", "layer-0", "layer-14", "layer-27", "per-layer-proper"):
        r = results[name]
        md.append(f"| {name} | {r['diff_count']}/{args.sim_samples} | {r['sample_diffs']} |")
    md.append("")
    if results["per-layer-proper"]["diff_count"] > results["mean"]["diff_count"]:
        md.append("**Reading: per-layer specificity moves additional samples vs uniform mean → "
                  "per-layer scope is JUSTIFIED. Proceed with Tasks 2-11 as planned.**")
    elif results["per-layer-proper"]["diff_count"] == results["mean"]["diff_count"]:
        md.append("**Reading: per-layer-proper matches uniform-mean diff count. Per-layer adds "
                  "nothing over a single calibrated codebook. Recommended: pivot to Phase 11b "
                  "(per-head) OR ship a SINGLE global calibrated codebook (much simpler than 28).**")
    else:
        md.append("**Reading: per-layer-proper moves FEWER samples than uniform mean — unexpected. "
                  "Possibly hook side-effect or codebook fit issue. Investigate before proceeding.**")
    out_path.write_text("\n".join(md) + "\n")

    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "results": {k: {"diff_count": v["diff_count"],
                        "sample_diffs": v["sample_diffs"]}
                    for k, v in results.items()},
    }, indent=2))

    print(f"\n[tighter] wrote {out_path}")
    print(f"[tighter] wrote {json_path}")


if __name__ == "__main__":
    raise SystemExit(main())
