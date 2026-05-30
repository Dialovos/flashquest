"""Phase 11 — full per-layer codebook calibration for TurboQuant K3-V3.

Captures POST-RoPE K and unrotated V via a subclassed PersistentTurboKVCache
across N essays at ctx_len each, then fits per-layer Lloyd-Max codebooks
(k=8, warm-start from paper) per (layer, K|V). Writes:

  - src/flashquest/turbo/codebook_<model>.pt        (num_layers, 2, 8) fp32
  - src/flashquest/turbo/codebook_<model>.json      sidecar metadata

Usage:
  nice -n 19 .venv/bin/python scripts/phase11_calibrate_codebook.py
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from flashquest.kernel import kv_quant as kvq
from flashquest.kernel.wht import wht_along_head_dim
from flashquest.runtime.awq_load import load_awq_model

_MODEL_ID_TO_FILENAME = {
    "casperhansen/llama-3.2-3b-instruct-awq": "codebook_llama_3_2_3b",
}


def lloyd_max_1d(samples: np.ndarray, init: np.ndarray, max_iter: int = 500,
                 atol: float = 1e-6) -> tuple[np.ndarray, float]:
    """1-D Lloyd-Max k-means via bisector midpoints (fully vectorized)."""
    centers = np.sort(init.copy().astype(np.float32))
    for _ in range(max_iter):
        bisectors = (centers[:-1] + centers[1:]) / 2.0
        assignments = np.digitize(samples, bisectors)
        new_centers = centers.copy()
        for j in range(len(centers)):
            mask = assignments == j
            if mask.any():
                new_centers[j] = samples[mask].mean()
        new_centers = np.sort(new_centers)
        if np.max(np.abs(new_centers - centers)) < atol:
            centers = new_centers
            break
        centers = new_centers
    bisectors = (centers[:-1] + centers[1:]) / 2.0
    assignments = np.digitize(samples, bisectors)
    inertia = float(np.sum((samples - centers[assignments]) ** 2))
    return centers, inertia


def _layer_kv(pkv, li):
    """Post-RoPE K + V for layer ``li`` from a DynamicCache, version-robust.

    transformers 4.57 exposes ``pkv.layers[li].keys/.values``; older builds use
    ``pkv.key_cache[li]`` / ``pkv.value_cache[li]``. Both are (B, H_kv, S, D).
    """
    layers = getattr(pkv, "layers", None)
    if layers is not None and getattr(layers[li], "keys", None) is not None:
        return layers[li].keys, layers[li].values
    return pkv.key_cache[li], pkv.value_cache[li]


def capture_post_rope_kv(model, prompts, num_layers, tokens_per_chunk=4096, seed=0):
    """Plain forward per prompt; read post-RoPE K + V from the standard
    DynamicCache (``out.past_key_values``).

    No Quest patching: the Quest-patched prefill runs per-page INT4/turbo
    quantization + criticality + a dense reference path, which grew the process
    to ~8 GB during capture and was OOM-killed by the WSL watchdog. A plain
    forward uses the model's native memory-efficient attention, and the cache
    holds exactly the post-RoPE K (and RoPE-free V) the deployed quantizer sees,
    so the calibration target is unchanged.

    Host RAM is bounded two ways: only ``tokens_per_chunk`` random positions are
    kept per chunk (an 8-centroid codebook needs a representative sample, not
    every token), and the GPU cache is freed between prompts. At 4096 tok x 8
    chunks x 28 layers x (K+V) bf16 this is ~1.8 GB.
    """
    captured_K = {li: [] for li in range(num_layers)}
    captured_V = {li: [] for li in range(num_layers)}
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, prompt_ids in enumerate(prompts):
            out = model(prompt_ids, use_cache=True)
            pkv = out.past_key_values
            S = prompt_ids.shape[-1]
            keep = min(tokens_per_chunk, S)
            idx = torch.randperm(S, generator=gen)[:keep].to(prompt_ids.device)
            for li in range(num_layers):
                k, v = _layer_kv(pkv, li)  # (B, H_kv, S, D)
                assert k.dim() == 4 and k.shape[2] == S, \
                    f"unexpected K cache shape {tuple(k.shape)} for layer {li}"
                captured_K[li].append(
                    k[:, :, idx, :].transpose(1, 2).to(torch.bfloat16).cpu())
                captured_V[li].append(
                    v[:, :, idx, :].transpose(1, 2).to(torch.bfloat16).cpu())
            del out, pkv, idx
            torch.cuda.empty_cache()
            print(f"  captured chunk {n + 1}/{len(prompts)} ({keep} tok)", flush=True)
    return captured_K, captured_V


def fit_per_layer_codebook(acts_layer_kv, paper_cb, max_samples=1_000_000, seed=0):
    """Apply Phase 7 transform + Lloyd-Max. Returns (codebook (8,) fp32, residual_rms, n_tokens).

    Subsamples up to ``max_samples`` rotated codepoints before clustering. A full
    layer is ~33.5M scalars and full-resolution 1-D Lloyd-Max costs ~2.2 s/iter
    over ~96 iters = ~5 min/fit (measured), i.e. ~4.7 h for the 56-fit run --
    which is what stalled the earlier calibration (it was killed long before
    finishing, not OOM: peak RSS was only ~1.9 GB). A 1M uniform subsample yields
    centroids within ~4e-3 of the full-data fit (centroids are population means;
    1M points give <0.2 % relative error -- negligible for a 3-bit quantizer) at
    ~0.8 s/fit. The subsample is seeded so the shipped artifact is reproducible.
    """
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0).float()
    rms = stacked.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked / rms
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)
    flat = rotated.reshape(-1).numpy().astype(np.float32)
    n_tokens = stacked.shape[0] * stacked.shape[1]

    fit_samples = flat
    if flat.size > max_samples:
        rng = np.random.default_rng(seed)
        fit_samples = flat[rng.choice(flat.size, size=max_samples, replace=False)]

    cb, inertia = lloyd_max_1d(fit_samples, init=paper_cb.astype(np.float32))
    residual_rms = float(np.sqrt(inertia / fit_samples.size))
    del stacked, rms, normalized, rotated, flat, fit_samples
    gc.collect()
    return cb, residual_rms, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--essays", type=int, default=8,
                    help="N chunks of --ctx tokens from data/PaulGrahamEssays.json.")
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    if args.model not in _MODEL_ID_TO_FILENAME:
        raise SystemExit(
            f"Unknown model {args.model!r}; add to _MODEL_ID_TO_FILENAME."
        )
    stem = _MODEL_ID_TO_FILENAME[args.model]
    out_pt = Path(f"src/flashquest/turbo/{stem}.pt")
    out_json = out_pt.with_suffix(".json")
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    print(f"[calibrate] loading {args.model}")
    model, tok = load_awq_model(args.model)
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    paper_cb = kvq.K_TURBO_CODEBOOK.cpu().numpy().astype(np.float32)

    print(f"[calibrate] reading data/PaulGrahamEssays.json")
    raw = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    full_ids = tok(raw["text"], return_tensors="pt").input_ids[0]
    needed = args.essays * args.ctx
    if full_ids.numel() < needed:
        raise SystemExit(
            f"Corpus has {full_ids.numel()} tokens; need {needed} "
            f"(--essays={args.essays} × --ctx={args.ctx}). Reduce one."
        )
    prompts = [
        full_ids[i * args.ctx:(i + 1) * args.ctx].unsqueeze(0).to(model.device)
        for i in range(args.essays)
    ]
    print(f"[calibrate] capturing K + V from {args.essays} chunks @ ctx={args.ctx}")
    captured_K, captured_V = capture_post_rope_kv(model, prompts, num_layers)

    print(f"[calibrate] fitting per-layer codebooks (K + V)")
    cb = np.zeros((num_layers, 2, 8), dtype=np.float32)
    residuals = np.zeros((num_layers, 2), dtype=np.float32)
    n_tokens_each = np.zeros((num_layers, 2), dtype=np.int64)
    for li in range(num_layers):
        cb_k, r_k, n_k = fit_per_layer_codebook(captured_K[li], paper_cb)
        cb_v, r_v, n_v = fit_per_layer_codebook(captured_V[li], paper_cb)
        cb[li, 0] = cb_k
        cb[li, 1] = cb_v
        residuals[li, 0] = r_k
        residuals[li, 1] = r_v
        n_tokens_each[li] = (n_k, n_v)
        print(f"  layer {li:2d}: K_res={r_k:.4f}  V_res={r_v:.4f}  "
              f"K_cb={cb_k.round(3).tolist()}")
        captured_K[li].clear()
        captured_V[li].clear()

    torch.save(torch.from_numpy(cb).float(), out_pt)

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        commit = "unknown"

    meta = {
        "model_id": args.model,
        "corpus": "PaulGrahamEssays",
        "n_essays": args.essays,
        "ctx": args.ctx,
        "n_tokens_per_layer": int(n_tokens_each.min()),
        "calibration_commit": commit,
        "residual_rms_per_layer": residuals.tolist(),
    }
    out_json.write_text(json.dumps(meta, indent=2))

    print(f"\n[calibrate] wrote {out_pt} ({out_pt.stat().st_size} bytes)")
    print(f"[calibrate] wrote {out_json}")


if __name__ == "__main__":
    raise SystemExit(main())
