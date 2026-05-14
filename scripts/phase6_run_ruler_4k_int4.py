"""Phase 6 task 5 — RULER NIAH 4k quality gate at INT4 KV.

Mirrors scripts/phase6_run_ruler_4k.py but the patched backend uses
PersistentInt4KVCache (--kv-bits 4 equivalent) instead of INT8. Compares
dense SDPA vs patched-INT4. Gate: patched_int4_hits / dense_hits >= 0.85
for every task.
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


def run_dense(model_name, ctx_len, n_samples, seed, max_new):
    from flashquest.runtime.awq_load import load_awq_model
    print(f"\n=== dense (vanilla SDPA) ===")
    model, tok = load_awq_model(model_name)
    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        r = run_niah(model, tok, task=task, n_samples=n_samples,
                     ctx_len=ctx_len, seed=seed, max_new_tokens=max_new)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok
    _free()
    return out


def run_patched_int4(model_name, ctx_len, n_samples, seed, max_new,
                     retention, num_sinks, window_pages, page_size):
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.runtime.awq_load import load_awq_model

    print(f"\n=== patched INT4 (retention={retention}, all-retrieval) ===")
    model, tok = load_awq_model(model_name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx_len + max_new + 128, page_size=page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=retention, num_sinks=num_sinks,
        window_pages=window_pages, page_size=page_size,
    )

    def reset_cache(_i: int) -> None:
        cache._seen_tokens = [0] * cache.num_layers

    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        r = run_niah(model, tok, task=task, n_samples=n_samples,
                     ctx_len=ctx_len, seed=seed, max_new_tokens=max_new,
                     pre_sample=reset_cache)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok, cache
    _free()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()

    dense = run_dense(args.model, args.ctx_len, args.n_samples, args.seed, args.max_new_tokens)
    patched = run_patched_int4(
        args.model, args.ctx_len, args.n_samples, args.seed, args.max_new_tokens,
        args.retention, args.num_sinks, args.window_pages, args.page_size,
    )

    tasks_out = {}
    all_pass = True
    for t in TASKS:
        k = f"niah_{t}"
        d, q = dense[k]["hits"], patched[k]["hits"]
        ratio = q / d if d > 0 else 0.0
        passed = ratio >= 0.85
        if not passed:
            all_pass = False
        tasks_out[k] = {
            "dense_hits": d, "patched_int4_hits": q,
            "total": dense[k]["total"], "ratio": ratio, "pass": passed,
            "dense_wall_s": dense[k]["wall_s"],
            "patched_int4_wall_s": patched[k]["wall_s"],
        }

    result = {
        "model": args.model, "ctx_len": args.ctx_len, "n_samples": args.n_samples,
        "retention": args.retention, "num_sinks": args.num_sinks,
        "window_pages": args.window_pages, "page_size": args.page_size,
        "seed": args.seed, "head_pattern": "all-retrieval", "kv_bits": 4,
        "tasks": tasks_out,
        "gate": "≥85% vs dense at retention=0.25 + all-retrieval head_pattern + INT4 KV",
        "all_pass": all_pass,
    }
    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_ruler_4k_int4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}\nAll-pass: {all_pass}")
    for t, info in tasks_out.items():
        print(f"  {t}: {info['patched_int4_hits']}/{info['dense_hits']} "
              f"= {info['ratio']:.2%} {'PASS' if info['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
