"""Phase 6 task 1c bench: re-run decode at 32k with the two-matmul
page_scores_int8_fast wired in. Target: ≥4 tok/s decode (SPEC §6 task 1
gate). Phase 6 task 1a measured 2.03 tok/s.
"""
from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


N_PREFILL_TOKENS = 32768
N_DECODE_TOKENS = 32
N_TRIALS = 1


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@torch.no_grad()
def measure_decode(model, tok, n_prefill: int, n_decode: int) -> dict:
    ids = synthetic_prompt_ids(tok, n_prefill)
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    out = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    for _ in range(n_decode):
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_dec

    return {
        "prefill_s": prefill_s,
        "prefill_tok_per_s": n_prefill / prefill_s,
        "decode_s": decode_s,
        "decode_tok_per_s": n_decode / decode_s,
    }


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    results = {
        "model": name, "n_prefill": N_PREFILL_TOKENS, "n_decode": N_DECODE_TOKENS,
        "n_trials": N_TRIALS, "trials": [],
    }

    print("=== Phase 6 task 1c (algebraic + two-matmul page_scores_int8_fast) ===")
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL_TOKENS + N_DECODE_TOKENS + 128,
        page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )
    for trial in range(N_TRIALS):
        cache._seen_tokens = [0] * cache.num_layers
        torch.cuda.reset_peak_memory_stats()
        m = measure_decode(model, tok, N_PREFILL_TOKENS, N_DECODE_TOKENS)
        m["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"  trial {trial}: prefill={m['prefill_tok_per_s']:.1f} tok/s, "
              f"decode={m['decode_tok_per_s']:.2f} tok/s, "
              f"peak VRAM={m['peak_vram_mib']:.0f} MiB")
        results["trials"].append(m)

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_decode_v2.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
