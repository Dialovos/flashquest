"""Phase 5 control: Llama-3.1-8B-AWQ at the largest context that fits on 4 GB.

The SPEC §6 Phase 4 win was "8B at 32k, ≥4 tok/s, ≥80 % RULER" — but
Llama-3.1-8B AWQ-INT4 weights alone are ~4.5 GB, exceeding the 4 GB
envelope before any KV cache. This script demonstrates the 8B decode
path works (passkey at 4k context) and records OOM at 32k as the
expected hardware result.
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.duo import load_duo_pattern
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.passkey import make_example, score
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_TARGETS = [4096, 8192, 16384, 32768]
DEPTHS = [0.5]
N_TRIALS = 2
RETENTION = 0.25


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    name = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "phase5_8b_control.json"
    try:
        model, tok = load_awq_model(name)
    except (RuntimeError, torch.cuda.OutOfMemoryError, OSError) as e:
        out_path.write_text(json.dumps({
            "model": name, "result": "OOM_AT_LOAD", "error": str(e)[:200],
        }, indent=2))
        print(f"8B AWQ load failed (often OOM on 4 GB): {str(e)[:120]}")
        return

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    duo_path = Path("vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/"
                    "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv")
    if duo_path.exists():
        pattern = load_duo_pattern(str(duo_path))
        pattern_source = str(duo_path)
    else:
        pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)
        pattern_source = "synthetic 70/30"

    results = {"model": name, "pattern_source": pattern_source, "by_context": {}}

    for ctx_len in CONTEXT_TARGETS:
        print(f"\n--- 8B at context={ctx_len} ---")
        try:
            cache = PersistentInt8KVCache(
                batch_size=1, num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
                max_seq_len=ctx_len + 64, page_size=64, device="cuda",
            )
            patch_llama_for_quest_persistent(
                model, cache=cache, head_pattern=pattern,
                retention=RETENTION, num_sinks=4, window_pages=2, page_size=64,
            )

            rng = random.Random(ctx_len)
            torch.cuda.reset_peak_memory_stats()
            correct = 0
            total = 0
            for d in DEPTHS:
                for _ in range(N_TRIALS):
                    ex = make_example(
                        rng=rng, tokenizer=tok,
                        target_total_tokens=ctx_len, depth_pct=d,
                    )
                    cache._seen_tokens = [0] * cache.num_layers
                    ids = tok(ex.text, return_tensors="pt").to("cuda")
                    with torch.no_grad():
                        out = model.generate(
                            **ids, max_new_tokens=10, do_sample=False,
                            pad_token_id=tok.eos_token_id,
                        )
                    gen = tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
                    correct += int(score(gen, ex.passkey))
                    total += 1
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            results["by_context"][str(ctx_len)] = {
                "accuracy": correct / total, "peak_vram_mib": peak,
            }
            print(f"  accuracy: {correct}/{total}  peak VRAM: {peak:.0f} MiB")
            del cache
            _free()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            results["by_context"][str(ctx_len)] = {"result": "OOM", "error": str(e)[:200]}
            print(f"  OOM (expected at large ctx): {str(e)[:80]}")
            _free()
            break

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
