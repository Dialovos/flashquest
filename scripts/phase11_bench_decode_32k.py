"""Phase 11 Task 10 — decode tok/s @ 32k context, K3-V3 calibrated vs paper.

Speed gate: per-layer constexpr-dispatch overhead must keep calibrated within
-5% of the paper baseline (28 distinct codebooks -> up to 28 compiled kernel
variants vs 1). Mirrors the verified cache/patch setup from
tests/test_niah_smoke.py. --codebook calibrated loads the artifact.

Usage:
  nice -n 19 .venv/bin/python scripts/phase11_bench_decode_32k.py \
    --codebook calibrated --ctx 32768 --retention 0.20
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flashquest.cache.persistent_turbo import PersistentTurboKVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--retention", type=float, default=0.20)
    ap.add_argument("--codebook", choices=("calibrated", "paper"), default="calibrated")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=25)
    ap.add_argument("--out", default="benchmarks/phase11/decode_32k_calibrated.json")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    cache_kwargs = dict(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=args.ctx + 128, page_size=64, device="cuda",
    )
    if args.codebook == "calibrated":
        cache_kwargs["model_id"] = args.model
    cache = PersistentTurboKVCache(**cache_kwargs)

    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=args.retention,
    )

    prompt_ids = torch.randint(0, cfg.vocab_size, (1, args.ctx), dtype=torch.long).cuda()
    t_pf = time.perf_counter()
    with torch.no_grad():
        model(prompt_ids, use_cache=True)
    torch.cuda.synchronize()
    print(f"[bench] prefill {args.ctx} done in {time.perf_counter() - t_pf:.1f}s "
          f"({args.codebook})", flush=True)

    last = prompt_ids[:, -1:].clone()
    for _ in range(args.warmup_steps):
        with torch.no_grad():
            model(last, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.measure_steps):
        with torch.no_grad():
            model(last, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tok_per_s = args.measure_steps / (t1 - t0)
    peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
    result = {
        "model": args.model, "ctx": args.ctx, "retention": args.retention,
        "kv_bits": 3, "codebook": args.codebook,
        "decode_tok_per_s": round(tok_per_s, 3),
        "peak_vram_mib": int(peak_mib), "measure_steps": args.measure_steps,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[bench] {args.codebook}: {tok_per_s:.2f} tok/s @ ctx={args.ctx}, "
          f"peak {peak_mib} MiB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
