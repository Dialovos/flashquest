"""Warmed single-request FlashQuest bench with separate prefill/decode timing."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from bench_common import (
    REPO_ROOT,
    add_run_arguments,
    content_hash,
    is_oom,
    make_identity,
    model_identity,
    new_record,
    provenance,
    run_config,
    summarize_samples,
    synthetic_token_ids,
    validate_run_arguments,
    write_benchmark_record,
)

BENCH_PROTOCOL = {"version": 1, "timing": "cuda-synchronized-monotonic",
                  "warmup": "one-full-trial-including-page-flush",
                  "output_count": "first-from-prefill-plus-decode", "repetitions": "whole-arm"}


def write_markers(events: list[dict]) -> None:
    if path := os.environ.get("FLASHQUEST_MARKERS"):
        with Path(path).open("a", encoding="utf-8") as stream:
            stream.writelines(json.dumps(event) + "\n" for event in events)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(p)
    p.add_argument("--revision", help="immutable model revision; resolved before loading")
    p.add_argument("--retention", type=float, default=0.20,
                   help="1.0 selects all pages for the dense INT4 ablation")
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--kv-bits", type=int, choices=[3, 4, 8], default=4)
    args = p.parse_args()
    validate_run_arguments(p, args)
    if not 0 < args.retention <= 1 or args.page_size < 1 or min(args.num_sinks, args.window_pages) < 0:
        p.error("0 < retention <= 1 and page-size >= 1 are required")
    if args.out.exists():
        p.error("output already exists; use a fresh output or the schedule's strict resume")
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
        load_start = time.monotonic_ns()
        write_markers([{"event": "load_start", "monotonic_ns": load_start}])
        resolved_model = model_identity(args.model, args.revision)
        import torch

        from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
        from flashquest.runtime.awq_load import load_awq_model

        if args.kv_bits == 3:
            from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls
        elif args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls

        model, tok = load_awq_model(args.model, revision=resolved_model["revision"],
                                   cache_dir=str(REPO_ROOT / "artifacts" / "hf-cache"))
        cfg = model.config
        actual_revision = getattr(cfg, "_commit_hash", None)
        if actual_revision is not None and actual_revision != resolved_model["revision"]:
            raise ValueError("loaded model revision differs from its identity")
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # Also warm the partial-page and page-flush paths before any timing.
        warmup_steps = max(args.n_decode - 1, args.page_size + 1)
        if args.ctx_len + warmup_steps + 1 > cfg.max_position_embeddings:
            raise ValueError("input plus warmup exceeds model capacity")
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
        input_ids = synthetic_token_ids(len(tok), args.ctx_len, args.seed)
        ids = torch.tensor([input_ids], device="cuda")
        positions = torch.arange(args.ctx_len + warmup_steps, device="cuda")
        weights = [*model.parameters(), *model.buffers()]
        cache_tensors = [value for value in vars(cache).values() if isinstance(value, torch.Tensor)]
        devices = sorted({str(t.device) for t in [*weights, *cache_tensors]})
        if devices != [str(model.device)] or not devices[0].startswith("cuda"):
            raise ValueError("model/cache tensors are not on the selected CUDA device")
        def storage_bytes(tensors):
            storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in tensors}
            return sum(storages.values())
        gemm = sys.modules.get("awq.modules.linear.gemm")
        runtime_info = {
            "cuda": torch.version.cuda, "devices": devices, "dtype": str(model.dtype),
            "weight_kernel": "awq_ext" if getattr(gemm, "awq_ext", None) is not None else
                             "triton" if getattr(gemm, "TRITON_AVAILABLE", False) else "unknown",
            "weight_modules": sorted({type(m).__name__ for m in model.modules()
                                      if type(m).__module__.startswith("awq.")}),
            "weight_storage_bytes": storage_bytes(weights),
            "persistent_cache_storage_bytes": storage_bytes(cache_tensors),
        }
        config.update(model=resolved_model["model"], revision=resolved_model["revision"],
                      model_cache="project-hf-cache", warmup_steps=warmup_steps,
                      cache_capacity=args.ctx_len + warmup_steps + 1,
                      input_sha256=content_hash(input_ids), runtime=runtime_info,
                      generation="manual-greedy-fixed-output-count-v1",
                      memory_protocol=json.loads(os.environ["FLASHQUEST_MEMORY_PROTOCOL"])
                                      if os.environ.get("FLASHQUEST_MEMORY_PROTOCOL") else None)
        record.update(make_identity(config, resolved_model, BENCH_PROTOCOL, record["provenance"]))
        torch.cuda.synchronize()
        load_end = time.monotonic_ns()
        write_markers([{"event": "load_end", "monotonic_ns": load_end}])

        @torch.inference_mode()
        def trial(decode_steps: int, repetition: int | None = None) -> dict:
            # Views expose only seen tokens; subsequent prefill overwrites them.
            # Reuse the allocation rather than temporarily allocating two caches.
            cache._seen_tokens = [0] * cache.num_layers
            torch.cuda.synchronize()
            t0 = time.monotonic_ns()
            out = model(input_ids=ids, cache_position=positions[:args.ctx_len],
                        use_cache=True, logits_to_keep=1)
            next_ids = out.logits[:, -1:].argmax(dim=-1)
            del out
            torch.cuda.synchronize()
            t1 = time.monotonic_ns()
            for step in range(decode_steps):
                # Attention owns a separate persistent cache, so HF's temporary
                # cache cannot infer the next RoPE position for manual forwards.
                position = args.ctx_len + step
                out = model(input_ids=next_ids, cache_position=positions[position:position + 1],
                            use_cache=True, logits_to_keep=1)
                next_ids = out.logits[:, -1:].argmax(dim=-1)
                del out
            torch.cuda.synchronize()
            t2 = time.monotonic_ns()
            prefill_s, decode_s = (t1 - t0) / 1e9, (t2 - t1) / 1e9
            if repetition is not None:
                write_markers([{"event": event, "monotonic_ns": timestamp, "repetition": repetition}
                               for event, timestamp in (("prefill_start", t0), ("prefill_end", t1),
                                                        ("decode_start", t1), ("decode_end", t2))])
            return {"input_tokens": args.ctx_len, "output_tokens": decode_steps + 1,
                    "prefill_s": prefill_s, "decode_s": decode_s, "request_s": (t2 - t0) / 1e9,
                    "prefill_tok_s": args.ctx_len / prefill_s,
                    "decode_tok_s": decode_steps / decode_s,
                    "end_to_end_tok_s": (decode_steps + 1) / ((t2 - t0) / 1e9),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1)}

        write_markers([{"event": "warmup_start", "monotonic_ns": time.monotonic_ns()}])
        trial(warmup_steps)
        write_markers([{"event": "warmup_end", "monotonic_ns": time.monotonic_ns()}])
        for repetition in range(args.reps):
            torch.cuda.reset_peak_memory_stats()
            record["samples"].append(trial(args.n_decode - 1, repetition))
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
