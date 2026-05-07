"""Single-cell flashquest decode bench. Writes a per-cell JSON to --out.

Mirrors scripts/phase6_bench_decode_32k_v2.py but parametric over --ctx-len
and catches torch.cuda.OutOfMemoryError to record `oom: true` instead of
crashing the orchestrator.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, required=True)
    p.add_argument("--n-decode", type=int, default=32)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--kv-bits", type=int, choices=[4, 8, 3], default=4,
                   help="KV cache bit width. 4 = KIVI-INT4 (default, RULER 100/100/100). "
                        "3 = TurboQuant K3-V3 (Phase 7). 8 = KIVI-INT8.")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    quant_label = {
        3: "AWQ-INT4 + TurboQuant K3-V3 paged KV + Quest top-k retention=0.25",
        4: "AWQ-INT4 + INT4 paged KV + Quest top-k retention=0.25",
        8: "AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25",
    }[args.kv_bits]
    record = {
        "backend": "flashquest",
        "quant": quant_label,
        "ctx_len": args.ctx_len,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "peak_vram_mib": None,
        "wall_s": None,
        "oom": False,
        "error": None,
    }

    t_start = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        if args.kv_bits == 3:
            from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls
        elif args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls
        model, tok = load_awq_model(args.model)
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        pattern = torch.ones(
            cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool,
        )
        cache = CacheCls(
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

        torch.cuda.synchronize()
        t_pf0 = time.perf_counter()
        with torch.no_grad():
            # logits_to_keep=1 — bench discards prefill logits; without this,
            # lm_head materialises (1, ctx_len, vocab=128256) = 7.83 GiB at 32 k.
            _ = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        t_pf1 = time.perf_counter()
        record["prefill_tok_s"] = args.ctx_len / (t_pf1 - t_pf0)

        next_ids = torch.tensor([[0]], device="cuda")
        torch.cuda.synchronize()
        t_dec0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.n_decode):
                out = model(input_ids=next_ids, use_cache=True, logits_to_keep=1)
                next_ids = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_dec1 = time.perf_counter()
        record["decode_tok_s"] = args.n_decode / (t_dec1 - t_dec0)
        record["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    except torch.cuda.OutOfMemoryError as exc:
        record["oom"] = True
        record["error"] = f"torch.cuda.OutOfMemoryError: {exc}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_s"] = time.perf_counter() - t_start
        try:
            _free()
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
