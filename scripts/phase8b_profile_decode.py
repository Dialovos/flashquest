"""Phase 8b — profile Phase 7 decode @ 32k to ground CUDA Graph ROI estimate.

Instruments the decode loop with torch.profiler. Reports:
  - Total CPU vs CUDA wall time
  - CPU/CUDA % of token wall time (CPU% upper-bounds graph savings)
  - Top-N kernels by CUDA time
  - Top-N CPU operators (Python+driver overhead candidates)

Usage:
  nice -n 19 python scripts/phase8b_profile_decode.py --ctx-len 32768 --n-decode 10
"""
from __future__ import annotations

import argparse
import gc
import time

import torch
from torch.profiler import ProfilerActivity, profile

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=32768)
    p.add_argument("--n-decode", type=int, default=10)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--top-k", type=int, default=15,
                   help="Number of top kernels/ops to print")
    args = p.parse_args()

    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    pattern = torch.ones(
        cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool,
    )
    cache = PersistentInt4KVCache(
        batch_size=1,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=args.ctx_len + args.n_decode + 128,
        page_size=args.page_size,
        device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=args.retention, num_sinks=args.num_sinks,
        window_pages=args.window_pages, page_size=args.page_size,
    )

    ids = torch.randint(0, cfg.vocab_size, (1, args.ctx_len), device="cuda")

    print(f"Prefilling {args.ctx_len} tokens ...")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    print(f"  prefill done in {time.perf_counter() - t0:.1f} s")

    next_ids = torch.tensor([[0]], device="cuda")

    # Warmup
    print(f"Warmup ({args.n_warmup} decode steps) ...")
    with torch.no_grad():
        for _ in range(args.n_warmup):
            out = model(input_ids=next_ids, use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()

    # Plain wall-time sanity check
    t_w0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.n_decode):
            out = model(input_ids=next_ids, use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    wall_per_token_unprofiled = (time.perf_counter() - t_w0) / args.n_decode
    tps = 1.0 / wall_per_token_unprofiled
    print(f"\n=== Plain wall-time (no profiler) ===")
    print(f"  {wall_per_token_unprofiled * 1000:.1f} ms/token  ({tps:.2f} tok/s)")

    # GPU-only timing via cuda.Event (bypasses CUPTI, works in WSL2)
    # cuda.Event measures GPU time between events on the default stream;
    # CPU+driver overhead does NOT count.
    print(f"\nMeasuring GPU-only time via cuda.Event ({args.n_decode} steps) ...")
    gpu_evt_per_step = []
    for _ in range(args.n_decode):
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_end = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        with torch.no_grad():
            out = model(input_ids=next_ids, use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1:].argmax(dim=-1)
        ev_end.record()
        torch.cuda.synchronize()
        gpu_evt_per_step.append(ev_start.elapsed_time(ev_end))
    gpu_per_token_ms = sum(gpu_evt_per_step) / len(gpu_evt_per_step)
    wall_ms = wall_per_token_unprofiled * 1000

    print(f"\n=== Per-token breakdown ===")
    print(f"  Wall:        {wall_ms:>7.2f} ms")
    print(f"  GPU (event): {gpu_per_token_ms:>7.2f} ms ({gpu_per_token_ms / wall_ms * 100:.1f}%)")
    cpu_overhead_ms = max(0, wall_ms - gpu_per_token_ms)
    print(f"  CPU+driver: {cpu_overhead_ms:>7.2f} ms ({cpu_overhead_ms / wall_ms * 100:.1f}%)")
    print()
    print("  CPU+driver fraction is the upper bound on Phase 8b graph savings:")
    print("  graphs eliminate Python+launch overhead but cannot reduce GPU compute.")

    # Profiled run for top-kernel breakdown (CPU side only on WSL2 — CUPTI missing)
    print(f"\nProfiling {args.n_decode} decode steps (CPU activity only) ...")
    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        with torch.no_grad():
            for _ in range(args.n_decode):
                out = model(input_ids=next_ids, use_cache=True, logits_to_keep=1)
                next_ids = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

    avg = prof.key_averages()
    total_self_cpu_us = sum(e.self_cpu_time_total for e in avg)

    print(f"\n=== Top {args.top_k} CPU ops (graph candidates) ===")
    cpu_ops = sorted(
        [e for e in avg if e.self_cpu_time_total > 0],
        key=lambda e: e.self_cpu_time_total, reverse=True,
    )[:args.top_k]
    print(f"{'OPERATOR':<60} {'CALLS':>8} {'CPU_ms':>10} {'%':>6}")
    for e in cpu_ops:
        pct = e.self_cpu_time_total / total_self_cpu_us * 100
        print(f"  {e.key[:58]:<58} {e.count:>8} {e.self_cpu_time_total / args.n_decode / 1e3:>10.2f} {pct:>5.1f}")

    print(f"\n=== Phase 8b graph ROI estimate ===")
    # CUDA Graphs eliminate CPU+driver overhead but not GPU compute.
    # Upper bound speedup = wall / gpu_only.
    if gpu_per_token_ms > 0:
        max_speedup = wall_ms / gpu_per_token_ms
        print(f"  Upper-bound Phase 8b speedup: {max_speedup:.2f}x")
        print(f"  (wall {wall_ms:.1f} ms / GPU compute {gpu_per_token_ms:.1f} ms)")
        # Realistic: graphs capture 60-90% of CPU overhead
        realistic_70 = wall_ms / (gpu_per_token_ms + cpu_overhead_ms * 0.3)
        realistic_85 = wall_ms / (gpu_per_token_ms + cpu_overhead_ms * 0.15)
        print(f"  Realistic (70% CPU overhead eliminated): {realistic_70:.2f}x")
        print(f"  Realistic (85% CPU overhead eliminated): {realistic_85:.2f}x")
    _free()


if __name__ == "__main__":
    main()
