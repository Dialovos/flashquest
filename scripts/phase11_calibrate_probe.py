"""Phase 11 Task 1 — calibration entry-gate probe.

Loads Llama-3.2-3B-AWQ, captures K/V tensors per layer at 8k prefill on 3 PG
essays (~24 k tokens), fits per-layer + per-head Lloyd-Max codebooks (k=8,
warm-start from paper), then runs RULER NIAH multivalue (3 samples, ctx=4 k)
twice via the dense reference path: once with paper K/V_TURBO_CODEBOOK,
once with the per-layer-mean proxy. Outputs:

  - benchmarks/phase11/probe.md            (one-page verdict + tables)
  - benchmarks/phase11/probe_codepoints.json  (raw per-layer codepoints)

Gate (must clear ALL to proceed to Task 2+):
  - codepoint_divergence_max  ≥ 5%   (per-layer-K and per-layer-V each)
  - quality_simulator_delta   ≥ 1 sample differs in generated tokens (of 3)
Granularity pre-signal R is recorded as an advisory number; not a gate.

Usage: nice -n 19 .venv/bin/python scripts/phase11_calibrate_probe.py
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from flashquest.eval.niah import make_prompt
from flashquest.kernel import kv_quant as kvq
from flashquest.kernel.wht import wht_along_head_dim
from flashquest.runtime.awq_load import load_awq_model


class _CapturingTurboKVCache:
    """Captures the (K_new, V_new) passed to update_quantized by sub-classing
    PersistentTurboKVCache. K_new is POST-RoPE (matches what quantize_k_turbo
    sees in steady state); V_new is unrotated (V is never RoPE'd).
    """
    def __init__(self, base_cache):
        self._base = base_cache
        self.captured_K = {li: [] for li in range(base_cache.num_layers)}
        self.captured_V = {li: [] for li in range(base_cache.num_layers)}

    def install(self):
        """Monkey-patch base_cache.update_quantized to also record inputs."""
        self._original = self._base.update_quantized

        def hooked_update(K_new, V_new, layer_idx):
            self.captured_K[layer_idx].append(K_new.detach().to(torch.bfloat16).cpu())
            self.captured_V[layer_idx].append(V_new.detach().to(torch.bfloat16).cpu())
            return self._original(K_new, V_new, layer_idx)

        self._base.update_quantized = hooked_update

    def uninstall(self):
        """Restore the un-hooked update_quantized so simulator calls don't grow our dicts."""
        if hasattr(self, "_original"):
            self._base.update_quantized = self._original


def capture_activations_via_cache(model, prompts, num_layers, head_dim, sim_max_ctx):
    """Apply Phase 7 TurboQuant patch + subclassed cache; capture POST-RoPE K + V
    on CPU as bf16. Returns (acts_dict, base_cache) — caller keeps base_cache alive
    so the simulator phase can reuse the patched chain (avoids two-cache OOM).

    Memory: 3 prompts × 8k × 8 H_kv × 64 D × 2 bytes × 2 (K+V) × 28 layers ≈ 1.5 GB CPU.
    """
    from flashquest.cache.persistent_turbo import PersistentTurboKVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent

    cfg = model.config
    pattern = torch.ones(num_layers, cfg.num_key_value_heads, dtype=torch.bool)

    longest_ctx = max(max(p.shape[-1] for p in prompts), sim_max_ctx)
    base_cache = PersistentTurboKVCache(
        batch_size=1,
        num_layers=num_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=longest_ctx + 64,
        page_size=64,
        device="cuda",
    )
    capturer = _CapturingTurboKVCache(base_cache)
    capturer.install()

    patch_llama_for_quest_persistent(
        model, cache=base_cache, head_pattern=pattern,
        retention=0.20, num_sinks=4, window_pages=2, page_size=64,
    )

    with torch.no_grad():
        for prompt_ids in prompts:
            # Reset cache between prompts so each contributes its own KV history.
            base_cache._seen_tokens = [0] * num_layers
            model(prompt_ids, use_cache=True)

    # Uninstall capture hook so the simulator's generate calls don't grow the
    # captured_K/V dicts; the cache itself stays alive + patched.
    capturer.uninstall()

    acts = {li: {"K": capturer.captured_K[li], "V": capturer.captured_V[li]}
            for li in range(num_layers)}
    return acts, base_cache


def lloyd_max_1d(samples: np.ndarray, init: np.ndarray, max_iter: int = 300,
                 atol: float = 1e-5) -> tuple[np.ndarray, float]:
    """1-D Lloyd-Max k-means via bisector midpoints (fully vectorized).

    Args:
        samples: (N,) float32. Calibration data.
        init: (k,) float32. Initial codepoints; warm-start from paper for stability.
        max_iter: cap iterations.
        atol: convergence threshold (max abs movement of any codepoint).
    Returns:
        (centers (k,) float32 SORTED ASCENDING, inertia float).
    """
    centers = np.sort(init.copy().astype(np.float32))
    for _ in range(max_iter):
        # Bisectors between consecutive centers form k Voronoi cells in 1-D.
        bisectors = (centers[:-1] + centers[1:]) / 2.0  # (k-1,)
        # Assignment by digitize (faster than (N, k) diff matrix).
        assignments = np.digitize(samples, bisectors)  # values in [0, k)
        new_centers = centers.copy()
        for j in range(len(centers)):
            mask = assignments == j
            if mask.any():
                new_centers[j] = samples[mask].mean()
        # Re-sort in case of cell shuffles.
        new_centers = np.sort(new_centers)
        max_move = np.max(np.abs(new_centers - centers))
        centers = new_centers
        if max_move < atol:
            break
    bisectors = (centers[:-1] + centers[1:]) / 2.0
    assignments = np.digitize(samples, bisectors)
    inertia = float(np.sum((samples - centers[assignments]) ** 2))
    return centers, inertia


def fit_per_layer_codebook(acts_layer_kv, n_clusters=8, warm_start=None):
    """Apply per-token RMS scale + WHT, flatten, fit Lloyd-Max. Returns (8,) fp32."""
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0).float()
    rms = stacked.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked / rms
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)
    flat = rotated.reshape(-1).numpy().astype(np.float32)

    init = warm_start.astype(np.float32) if warm_start is not None else np.linspace(
        flat.min(), flat.max(), n_clusters, dtype=np.float32
    )
    codebook, _ = lloyd_max_1d(flat, init)
    del stacked, rms, normalized, rotated, flat
    gc.collect()
    return codebook


def fit_per_head_codebook(acts_layer_kv, n_clusters=8, warm_start=None):
    """Per-head Lloyd-Max; returns (H_kv, 8) fp32."""
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0).float()
    H_kv = stacked.shape[1]
    rms = stacked.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked / rms
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)
    codebooks = np.zeros((H_kv, n_clusters), dtype=np.float32)
    init_arr = warm_start.astype(np.float32) if warm_start is not None else None
    for h in range(H_kv):
        flat = rotated[:, h, :].reshape(-1).numpy().astype(np.float32)
        init = init_arr if init_arr is not None else np.linspace(
            flat.min(), flat.max(), n_clusters, dtype=np.float32
        )
        codebooks[h], _ = lloyd_max_1d(flat, init)
    del stacked, rms, normalized, rotated
    gc.collect()
    return codebooks


def quality_simulator_delta(model, base_cache, tok, proxy_cb_k, proxy_cb_v,
                             n_samples, ctx_len, seed_base):
    """Reuse the already-patched (model, base_cache) chain from capture.
    Swap the module-level codebook, reset cache, generate; compare token streams.

    Proxy = per-layer mean — a conservative scalar approximation of the full
    per-layer codebook. If even the mean produces no token-level delta, per-layer
    is unlikely to either.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers

    original_K = kvq.K_TURBO_CODEBOOK.clone()
    original_V = kvq.V_TURBO_CODEBOOK.clone()

    proxy_cb_k_t = torch.tensor(proxy_cb_k, dtype=torch.float32, device="cuda")
    proxy_cb_v_t = torch.tensor(proxy_cb_v, dtype=torch.float32, device="cuda")

    @torch.no_grad()
    def gen(input_ids):
        base_cache._seen_tokens = [0] * num_layers
        out = model.generate(
            input_ids,
            max_new_tokens=32,
            do_sample=False,
            use_cache=True,
        )
        return out[0, input_ids.shape[-1]:].tolist()

    diff_count = 0
    per_sample = []

    try:
        for i in range(n_samples):
            prompt, expected = make_prompt(
                "multivalue", ctx_len=ctx_len, tokenizer=tok, seed=seed_base + i
            )
            input_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

            kvq.K_TURBO_CODEBOOK.copy_(original_K)
            kvq.V_TURBO_CODEBOOK.copy_(original_V)
            paper_gen = gen(input_ids)
            torch.cuda.empty_cache()

            kvq.K_TURBO_CODEBOOK.copy_(proxy_cb_k_t)
            kvq.V_TURBO_CODEBOOK.copy_(proxy_cb_v_t)
            calib_gen = gen(input_ids)
            torch.cuda.empty_cache()

            differs = paper_gen != calib_gen
            if differs:
                diff_count += 1
            per_sample.append({
                "sample_idx": i,
                "expected": list(expected),
                "paper_gen_len": len(paper_gen),
                "calib_gen_len": len(calib_gen),
                "differs": differs,
                "first_divergence_idx": (
                    next(
                        (k for k in range(min(len(paper_gen), len(calib_gen)))
                         if paper_gen[k] != calib_gen[k]),
                        None,
                    )
                ),
            })
            print(f"[probe.sim] sample {i}: paper_gen_len={len(paper_gen)} "
                  f"calib_gen_len={len(calib_gen)} differs={differs}")
    finally:
        kvq.K_TURBO_CODEBOOK.copy_(original_K)
        kvq.V_TURBO_CODEBOOK.copy_(original_V)

    return diff_count, per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--essays", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--sim-samples", type=int, default=3)
    ap.add_argument("--sim-ctx", type=int, default=4096)
    ap.add_argument("--out", default="benchmarks/phase11/probe.md")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[probe] loading model {args.model}")
    model, tok = load_awq_model(args.model)
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    paper_cb = kvq.K_TURBO_CODEBOOK.cpu().numpy().astype(np.float32)

    print(f"[probe] reading data/PaulGrahamEssays.json (single concatenated text)")
    raw = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    if not (isinstance(raw, dict) and "text" in raw):
        raise SystemExit(f"Unexpected PaulGrahamEssays.json schema: keys={list(raw)}")

    full_ids = tok(raw["text"], return_tensors="pt").input_ids[0]
    needed = args.essays * args.ctx
    if full_ids.numel() < needed:
        raise SystemExit(
            f"Corpus has {full_ids.numel()} tokens; need {needed} "
            f"({args.essays} × ctx={args.ctx}). Reduce --essays or --ctx."
        )
    prompts = []
    for i in range(args.essays):
        chunk = full_ids[i * args.ctx : (i + 1) * args.ctx].unsqueeze(0).to(model.device)
        prompts.append(chunk)
    print(f"[probe] split corpus into {args.essays} × {args.ctx}-tok chunks "
          f"({full_ids.numel()} tokens available)")

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    print(f"[probe] capturing POST-RoPE K + V via subclassed TurboQuant cache "
          f"({len(prompts)} prompts @ ctx={args.ctx})")
    acts, base_cache = capture_activations_via_cache(
        model, prompts, num_layers, head_dim, sim_max_ctx=args.sim_ctx,
    )

    print(f"[probe] fitting per-layer codebooks (K + V) for {num_layers} layers")
    per_layer_k = np.zeros((num_layers, 8), dtype=np.float32)
    per_layer_v = np.zeros((num_layers, 8), dtype=np.float32)
    for li in range(num_layers):
        per_layer_k[li] = fit_per_layer_codebook(acts[li]["K"], warm_start=paper_cb)
        per_layer_v[li] = fit_per_layer_codebook(acts[li]["V"], warm_start=paper_cb)
        print(f"  layer {li:2d} K={per_layer_k[li].round(3).tolist()}")

    print(f"[probe] fitting per-head codebooks (advisory granularity check)")
    per_head_k = []
    for li in range(num_layers):
        ph = fit_per_head_codebook(acts[li]["K"], warm_start=paper_cb)
        per_head_k.append(ph)
        # Free this layer's captured activations.
        acts[li]["K"].clear()
        acts[li]["V"].clear()
    per_head_k = np.stack(per_head_k)  # (L, H_kv, 8)

    # Codepoint divergence (per-layer vs paper).
    div_k = float(np.max(np.abs(per_layer_k - paper_cb[None, :]) / np.abs(paper_cb[None, :])))
    div_v = float(np.max(np.abs(per_layer_v - paper_cb[None, :]) / np.abs(paper_cb[None, :])))

    # Granularity ratio R.
    EPS = 1e-6
    per_layer_diff = float(np.max(np.abs(per_layer_k - paper_cb[None, :])))
    per_head_diff = float(np.max(np.abs(per_head_k - per_layer_k[:, None, :])))
    R = per_head_diff / max(per_layer_diff, EPS)

    print(f"[probe] divergence: K={div_k:.3%}  V={div_v:.3%}  R={R:.2f}")

    print(f"[probe] quality-simulator delta — {args.sim_samples} samples @ ctx={args.sim_ctx}")
    proxy_cb_k = per_layer_k.mean(axis=0)
    proxy_cb_v = per_layer_v.mean(axis=0)
    gc.collect()
    torch.cuda.empty_cache()
    sim_count, sim_per_sample = quality_simulator_delta(
        model, base_cache, tok, proxy_cb_k, proxy_cb_v,
        n_samples=args.sim_samples,
        ctx_len=args.sim_ctx,
        seed_base=20260514,
    )

    verdict_codepoint = "PASS" if (div_k >= 0.05 and div_v >= 0.05) else "KILL"
    verdict_simulator = "PASS" if sim_count >= 1 else "KILL"
    overall = "PROCEED" if (verdict_codepoint == "PASS" and verdict_simulator == "PASS") else "KILL"

    md = []
    md.append("# Phase 11 — Calibration Entry-Gate Probe")
    md.append("")
    md.append(f"**Verdict: {overall}**")
    md.append("")
    md.append(f"- Codepoint divergence (per-layer vs paper):")
    md.append(f"  K = {div_k:.3%}, V = {div_v:.3%} → **{verdict_codepoint}** (gate ≥5%).")
    md.append(f"- Quality-simulator delta (proxy = per-layer mean): "
              f"{sim_count}/{args.sim_samples} samples differ → "
              f"**{verdict_simulator}** (gate ≥1).")
    md.append(f"- Granularity pre-signal R (advisory): {R:.2f} "
              f"(R > 2 across multiple layers → consider Phase 11b per-head).")
    md.append("")
    md.append("## Paper codebook (reference)")
    md.append(f"`{paper_cb.round(4).tolist()}`")
    md.append("")
    md.append("## Per-layer calibrated K codebook (first 5 layers)")
    md.append("```")
    for li in range(min(5, num_layers)):
        md.append(f"  layer {li:2d}: {per_layer_k[li].round(4).tolist()}")
    md.append("```")
    md.append("")
    md.append("## Per-sample quality-simulator detail")
    md.append("| i | expected | paper_len | calib_len | differs | first_div |")
    md.append("|---|---|---|---|---|---|")
    for s in sim_per_sample:
        md.append(
            f"| {s['sample_idx']} | {s['expected']} | {s['paper_gen_len']} | "
            f"{s['calib_gen_len']} | {s['differs']} | {s['first_divergence_idx']} |"
        )
    out_path.write_text("\n".join(md) + "\n")

    json_path = out_path.parent / "probe_codepoints.json"
    json_path.write_text(json.dumps({
        "paper": paper_cb.tolist(),
        "per_layer_k": per_layer_k.tolist(),
        "per_layer_v": per_layer_v.tolist(),
        "per_head_k_shape": list(per_head_k.shape),
        "max_per_head_dev_from_per_layer": per_head_diff,
        "max_per_layer_dev_from_paper": per_layer_diff,
        "div_k": div_k,
        "div_v": div_v,
        "R": R,
        "sim_delta_count": sim_count,
        "sim_samples": sim_per_sample,
        "args": {
            "model": args.model,
            "essays": args.essays,
            "ctx": args.ctx,
            "sim_samples": args.sim_samples,
            "sim_ctx": args.sim_ctx,
        },
    }, indent=2))

    print(f"\n[probe] wrote {out_path}\n[probe] wrote {json_path}\n[probe] verdict: {overall}")
    return 0 if overall == "PROCEED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
