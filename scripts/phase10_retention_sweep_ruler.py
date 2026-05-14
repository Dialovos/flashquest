"""Phase 10 — retention sweep + RULER NIAH single 4k quality test.

Binary-search-style: run niah_single at given retention, gate on hits/20 == 20.
Skips the dense baseline re-run (Phase 6/7 already established 20/20 dense).

Usage:
  nice -n 19 .venv/bin/python scripts/phase10_retention_sweep_ruler.py --retention 0.10
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.runner import run_niah
from flashquest.runtime.awq_load import load_awq_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--retention", type=float, required=True)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--task", default="single", choices=["single", "multikey", "multivalue"])
    p.add_argument("--out-dir", default="benchmarks/phase10_retention")
    args = p.parse_args()

    print(f"=== retention={args.retention} task={args.task} n={args.n_samples} ctx={args.ctx_len} ===")
    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=args.ctx_len + args.max_new_tokens + 128,
        page_size=args.page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=args.retention, num_sinks=args.num_sinks,
        window_pages=args.window_pages, page_size=args.page_size,
    )

    def reset_cache(_i: int) -> None:
        cache._seen_tokens = [0] * cache.num_layers

    t0 = time.perf_counter()
    r = run_niah(
        model, tok, task=args.task, n_samples=args.n_samples,
        ctx_len=args.ctx_len, seed=args.seed,
        max_new_tokens=args.max_new_tokens, pre_sample=reset_cache,
    )
    wall = time.perf_counter() - t0
    hits = r["hits"]
    total = r["total"]
    print(f"\n  niah_{args.task}: hits={hits}/{total}  wall={wall:.0f}s")
    print(f"  GATE: {'PASS' if hits == total else 'FAIL'} (need {total}/{total})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"retention_{args.retention:.2f}_{args.task}.json"
    out_path.write_text(json.dumps({
        "retention": args.retention,
        "task": args.task,
        "ctx_len": args.ctx_len,
        "hits": hits, "total": total,
        "wall_s": wall,
        "gate_pass": hits == total,
        "samples": r["samples"],
    }, indent=2))
    print(f"\nResults: {out_path}")

    del model, tok, cache
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


if __name__ == "__main__":
    main()
