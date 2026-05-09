"""Phase 10 entry probe — per-module GPU time breakdown at 32k decode.

Measures what fraction of decode-step wall time is spent in each MLP / attention
projection so we can ground the CATS speedup ceiling BEFORE writing a spec.

Wraps q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head,
and the patched LlamaAttention.forward with cuda.Event timing. Sums across 28
layers × N_DECODE steps, prints fractions of step wall time.

Usage:
  nice -n 19 .venv/bin/python scripts/phase10_profile_decode_breakdown.py \
      --ctx-len 32768 --n-decode 20 --n-warmup 5
"""
from __future__ import annotations

import argparse
import gc
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


_TIMINGS: dict[str, float] = defaultdict(float)
_CALL_COUNTS: dict[str, int] = defaultdict(int)


@contextmanager
def cuda_event_timer(label: str):
    """Block-level GPU timer using cuda.Event."""
    ev_start = torch.cuda.Event(enable_timing=True)
    ev_end = torch.cuda.Event(enable_timing=True)
    ev_start.record()
    try:
        yield
    finally:
        ev_end.record()
        torch.cuda.synchronize()
        elapsed_ms = ev_start.elapsed_time(ev_end)
        _TIMINGS[label] += elapsed_ms
        _CALL_COUNTS[label] += 1


def wrap_forward_with_timing(module: torch.nn.Module, label: str) -> None:
    """Replace module.forward with a timed version. Returns nothing; mutates in place."""
    orig_forward = module.forward

    def timed_forward(*args, **kwargs):
        with cuda_event_timer(label):
            return orig_forward(*args, **kwargs)

    module.forward = timed_forward


def free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=32768)
    p.add_argument("--n-decode", type=int, default=20)
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    args = p.parse_args()

    print(f"Loading model {args.model} ...")
    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=args.ctx_len + args.n_decode + args.n_warmup + 32,
        page_size=args.page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=args.retention, num_sinks=args.num_sinks,
        window_pages=args.window_pages, page_size=args.page_size,
    )

    # Wrap hot modules with cuda.Event timers AFTER the patched forward is set
    print("Wrapping forward methods with cuda.Event timers...")
    n_attn_wrapped = 0
    n_mlp_wrapped = 0
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if cls == "LlamaAttention":
            wrap_forward_with_timing(module, "attention.forward")
            wrap_forward_with_timing(module.q_proj, "q_proj")
            wrap_forward_with_timing(module.k_proj, "k_proj")
            wrap_forward_with_timing(module.v_proj, "v_proj")
            wrap_forward_with_timing(module.o_proj, "o_proj")
            n_attn_wrapped += 1
        elif cls == "LlamaMLP":
            wrap_forward_with_timing(module, "mlp.forward")
            wrap_forward_with_timing(module.gate_proj, "gate_proj")
            wrap_forward_with_timing(module.up_proj, "up_proj")
            wrap_forward_with_timing(module.down_proj, "down_proj")
            n_mlp_wrapped += 1
    if hasattr(model, "lm_head"):
        wrap_forward_with_timing(model.lm_head, "lm_head")
    print(f"  Wrapped {n_attn_wrapped} attention, {n_mlp_wrapped} MLP modules")

    # Prefill
    print(f"Prefilling {args.ctx_len} random tokens...")
    ids = torch.randint(0, cfg.vocab_size, (1, args.ctx_len), device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    print(f"  prefill done in {time.perf_counter() - t0:.1f} s")
    next_id = out.logits[:, -1].argmax(dim=-1)

    # Warmup
    print(f"Warmup ({args.n_warmup} decode steps)...")
    with torch.no_grad():
        for _ in range(args.n_warmup):
            out = model(input_ids=next_id.unsqueeze(0), use_cache=True, logits_to_keep=1)
            next_id = out.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()

    # Reset timers for measurement
    _TIMINGS.clear()
    _CALL_COUNTS.clear()

    # Measure
    print(f"Measuring {args.n_decode} decode steps...")
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.n_decode):
            out = model(input_ids=next_id.unsqueeze(0), use_cache=True, logits_to_keep=1)
            next_id = out.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()
    wall_total_ms = (time.perf_counter() - wall_start) * 1000
    wall_per_step_ms = wall_total_ms / args.n_decode
    tps = 1000.0 / wall_per_step_ms

    print(f"\n=== Phase 10 entry probe @ ctx={args.ctx_len} decode ===")
    print(f"  Wall time per step: {wall_per_step_ms:>7.2f} ms  ({tps:.2f} tok/s)")
    print(f"  Total decode wall:  {wall_total_ms:>7.2f} ms  ({args.n_decode} steps)\n")

    # Per-module summary (totals across 28 layers × n_decode steps)
    print(f"  {'MODULE':<24}{'TOTAL_ms':>12}{'PER_STEP_ms':>14}{'%STEP':>10}{'CALLS':>10}")
    rows = sorted(_TIMINGS.items(), key=lambda kv: -kv[1])
    for label, total_ms in rows:
        per_step_ms = total_ms / args.n_decode
        pct_step = per_step_ms / wall_per_step_ms * 100
        n_calls = _CALL_COUNTS[label]
        print(f"  {label:<24}{total_ms:>12.2f}{per_step_ms:>14.3f}{pct_step:>9.1f}%{n_calls:>10}")

    # Phase 10 ceiling estimates
    print(f"\n=== Phase 10 ceiling estimates ===")
    down_pct = _TIMINGS.get("down_proj", 0) / args.n_decode / wall_per_step_ms * 100
    mlp_pct = _TIMINGS.get("mlp.forward", 0) / args.n_decode / wall_per_step_ms * 100
    attn_pct = _TIMINGS.get("attention.forward", 0) / args.n_decode / wall_per_step_ms * 100
    cats_50pct_savings = down_pct * 0.5
    print(f"  down_proj % of step:        {down_pct:>5.1f}%   (target for CATS speedup)")
    print(f"  Whole MLP %:                {mlp_pct:>5.1f}%")
    print(f"  Whole attention %:          {attn_pct:>5.1f}%")
    print(f"  CATS@50%sparse ceiling:     1 / (1 - {cats_50pct_savings/100:.4f}) = {1/(1-cats_50pct_savings/100):.3f}x")
    print(f"  CATS@70%sparse ceiling:     1 / (1 - {down_pct*0.7/100:.4f}) = {1/(1-down_pct*0.7/100):.3f}x")
    free()


if __name__ == "__main__":
    main()
