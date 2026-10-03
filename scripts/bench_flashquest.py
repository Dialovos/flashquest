"""Warmed single-request FlashQuest bench with separate prefill/decode timing."""
from __future__ import annotations

import argparse
import time

from bench_common import (
    add_run_arguments,
    is_oom,
    new_record,
    provenance,
    run_config,
    summarize_samples,
    synthetic_token_ids,
    validate_run_arguments,
    write_benchmark_record,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(p)
    p.add_argument("--retention", type=float, default=0.20,
                   help="1.0 selects all pages for the dense INT4 ablation")
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--kv-bits", type=int, choices=[3, 4, 8], default=4)
    args = p.parse_args()
    validate_run_arguments(p, args)
    if not 0 < args.retention <= 1 or args.page_size < 1:
        p.error("0 < retention <= 1 and page-size >= 1 are required")
    config = run_config(args.ctx_len, args.n_decode, args.reps, args.seed,
                        model=args.model, kv_bits=args.kv_bits, retention=args.retention,
                        page_size=args.page_size, num_sinks=args.num_sinks,
                        window_pages=args.window_pages)
    mode = "TurboQuant K3-V3" if args.kv_bits == 3 else f"INT{args.kv_bits}"
    record = new_record("flashquest", f"AWQ-INT4 + {mode} KV, retention={args.retention:g}",
                        config)
    t_start = time.perf_counter()
    record["provenance"] = provenance()
    try:
        import torch

        from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
        from flashquest.runtime.awq_load import load_awq_model

        if args.kv_bits == 3:
            from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls
        elif args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls

        model, tok = load_awq_model(args.model)
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # Also warm the partial-page and page-flush paths before any timing.
        warmup_steps = max(args.n_decode - 1, args.page_size + 1)
        cache = CacheCls(
            batch_size=1, num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
            max_seq_len=args.ctx_len + warmup_steps + 1,
            page_size=args.page_size, device="cuda",
        )
        pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=args.retention, num_sinks=args.num_sinks,
            window_pages=args.window_pages, page_size=args.page_size,
        )
        ids = torch.tensor([synthetic_token_ids(len(tok), args.ctx_len, args.seed)],
                           device="cuda")
        positions = torch.arange(args.ctx_len + warmup_steps, device="cuda")

        @torch.inference_mode()
        def trial(decode_steps: int) -> dict:
            # Views expose only seen tokens; subsequent prefill overwrites them.
            # Reuse the allocation rather than temporarily allocating two caches.
            cache._seen_tokens = [0] * cache.num_layers
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=ids, cache_position=positions[:args.ctx_len],
                        use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1:].argmax(dim=-1)
            del out
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for step in range(decode_steps):
                # Attention owns a separate persistent cache, so HF's temporary
                # cache cannot infer the next RoPE position for manual forwards.
                position = args.ctx_len + step
                out = model(input_ids=next_ids, cache_position=positions[position:position + 1],
                            use_cache=True, logits_to_keep=1)
                next_ids = out.logits[:, -1:].argmax(dim=-1)
                del out
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            return {"input_tokens": args.ctx_len, "output_tokens": decode_steps + 1,
                    "prefill_s": t1 - t0, "decode_s": t2 - t1, "request_s": t2 - t0,
                    "prefill_tok_s": args.ctx_len / (t1 - t0),
                    "decode_tok_s": decode_steps / (t2 - t1),
                    "end_to_end_tok_s": (decode_steps + 1) / (t2 - t0),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}

        trial(warmup_steps)
        for _ in range(args.reps):
            torch.cuda.reset_peak_memory_stats()
            record["samples"].append(trial(args.n_decode - 1))
        summarize_samples(record)
        record["versions"] = {"torch": torch.__version__}
    except Exception as exc:  # noqa: BLE001 — record backend/import failures for the matrix
        record["oom"] = is_oom(str(exc))
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = time.perf_counter() - t_start
    exported = write_benchmark_record(args.out, record)
    print(exported)
    return int(record["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
