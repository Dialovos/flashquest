"""Phase 11 Task 9 — RULER NIAH 4k on TurboQuant K3-V3, calibrated vs paper.

THE quality gate: calibrated must hit >=95% on all three NIAH tasks (single,
multikey, multivalue) to ship K3-V3 calibrated as the default. Mirrors the
verified setup in scripts/phase7_run_ruler_4k_turbo.py: flashquest.eval.runner
.run_niah with a pre_sample cache reset. --codebook calibrated loads the
per-layer artifact via PersistentTurboKVCache(model_id=...); paper omits model_id.

Run one codebook per invocation (separate JSON each) for resilience:
  nice -n 19 .venv/bin/python scripts/phase11_run_ruler_4k_calibrated.py \
    --codebook calibrated --n-samples 20 --ctx-len 4096 --retention 0.20 \
    --out benchmarks/phase11/ruler_4k_calibrated.json
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.eval.runner import run_niah

TASKS = ["single", "multikey", "multivalue"]


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_codebook(codebook, model_name, ctx_len, n_samples, seed, max_new,
                 retention, num_sinks, window_pages, page_size, tasks=TASKS):
    from flashquest.cache.persistent_turbo import PersistentTurboKVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.runtime.awq_load import load_awq_model

    print(f"\n=== K3-V3 {codebook} (retention={retention}, all-retrieval) ===", flush=True)
    t_load = time.perf_counter()
    model, tok = load_awq_model(model_name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    cache_kwargs = dict(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx_len + max_new + 128, page_size=page_size, device="cuda",
    )
    if codebook == "calibrated":
        cache_kwargs["model_id"] = model_name
    cache = PersistentTurboKVCache(**cache_kwargs)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=retention,
        num_sinks=num_sinks, window_pages=window_pages, page_size=page_size,
    )
    print(f"[ruler] {codebook} setup done in {time.perf_counter() - t_load:.1f}s "
          f"(model load + patch + codebook)", flush=True)

    def reset_cache(_i):
        cache._seen_tokens = [0] * cache.num_layers

    out = {}
    for ti, task in enumerate(tasks):
        t0 = time.perf_counter()
        r = run_niah(model, tok, task=task, n_samples=n_samples, ctx_len=ctx_len,
                     seed=seed, max_new_tokens=max_new, pre_sample=reset_cache)
        wall = time.perf_counter() - t0
        rate = r["hits"] / r["total"] if r["total"] else 0.0
        note = " (incl. cold kernel compile)" if ti == 0 else ""
        print(f"  {task}: {r['hits']}/{r['total']} ({100 * rate:.0f}%) "
              f"wall={wall:.0f}s{note}", flush=True)
        out[task] = {"hits": r["hits"], "total": r["total"], "rate": rate,
                     "wall_s": round(wall, 1)}
    del model, tok, cache
    _free()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.20)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--codebook", choices=("calibrated", "paper"), default="calibrated")
    p.add_argument("--tasks", default="single,multikey,multivalue",
                   help="comma-separated subset of single,multikey,multivalue "
                        "(Phase 11b Task 1a uses 'multivalue' only to isolate the gap).")
    p.add_argument("--out", default="benchmarks/phase11/ruler_4k_calibrated.json")
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in TASKS]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; choose from {TASKS}")

    res = run_codebook(args.codebook, args.model, args.ctx_len, args.n_samples,
                       args.seed, args.max_new_tokens, args.retention,
                       args.num_sinks, args.window_pages, args.page_size, tasks=tasks)
    # Gate is only meaningful over the full task set; a subset run reports the
    # subset's own pass/fail for convenience but flags that it's partial.
    full = set(tasks) == set(TASKS)
    gate = all(res[t]["rate"] >= 0.95 for t in tasks)
    result = {
        "model": args.model, "ctx_len": args.ctx_len, "n_samples": args.n_samples,
        "retention": args.retention, "kv_bits": 3, "codebook": args.codebook,
        "seed": args.seed, "tasks": tasks, "results": res,
        "gate_all_ge_95": (gate if args.codebook == "calibrated" else None) if full else None,
        "subset_all_ge_95": None if full else gate,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\nWrote {args.out}", flush=True)
    line = "  ".join(f"{t}={res[t]['hits']}/{res[t]['total']}" for t in tasks)
    print(f"[{args.codebook} ret={args.retention}] {line}", flush=True)
    if full and args.codebook == "calibrated":
        print(f"GATE (calibrated all >=95%): {'PASS' if gate else 'FAIL'}", flush=True)
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
