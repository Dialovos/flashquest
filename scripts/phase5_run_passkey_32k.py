"""Phase 5 passkey: Llama-3.2-3B-AWQ with persistent INT8 KV cache + fused
DuoAttention dispatch (synthetic 70/30 split). Runs at {8k, 16k, 32k}
contexts × {0.1, 0.5, 0.9} depths × 3 trials.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.passkey import make_example, score
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_LENS = [8192, 32768]
DEPTHS = [0.1, 0.5, 0.9]
N_TRIALS = 2
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2
RETRIEVAL_FRACTION = 0.7


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def synthetic_pattern(num_layers: int, num_kv: int, fraction: float, seed: int = 0):
    rng = torch.Generator().manual_seed(seed)
    return torch.rand((num_layers, num_kv), generator=rng) < fraction


@torch.no_grad()
def generate_passkey_answer(model, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def main() -> None:
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    pattern = synthetic_pattern(
        cfg.num_hidden_layers, cfg.num_key_value_heads, RETRIEVAL_FRACTION,
    )

    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    results = {
        "model": name,
        "context_lens": CONTEXT_LENS,
        "depths": DEPTHS,
        "n_trials": N_TRIALS,
        "retention": RETENTION,
        "retrieval_fraction": RETRIEVAL_FRACTION,
        "by_context": {},
    }

    for ctx_len in CONTEXT_LENS:
        print(f"\n=== context = {ctx_len} ===")
        cache = PersistentInt8KVCache(
            batch_size=1, num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
            max_seq_len=ctx_len + 64, page_size=64, device="cuda",
        )
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=RETENTION, num_sinks=NUM_SINKS,
            window_pages=WINDOW_PAGES, page_size=64,
        )

        rng = random.Random(ctx_len)
        examples = []
        for d in DEPTHS:
            for _ in range(N_TRIALS):
                examples.append(make_example(
                    rng=rng, tokenizer=tok,
                    target_total_tokens=ctx_len, depth_pct=d,
                ))

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        correct_by_depth = {d: 0 for d in DEPTHS}
        for ex in examples:
            cache._seen_tokens = [0] * cache.num_layers
            gen = generate_passkey_answer(model, tok, ex.text)
            if score(gen, ex.passkey):
                correct_by_depth[ex.depth_pct] += 1
        dt = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        accuracy = {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS}
        print(f"  accuracy: {accuracy}")
        print(f"  elapsed: {dt:.1f}s, peak VRAM: {peak_mb:.0f} MiB")

        results["by_context"][str(ctx_len)] = {
            "accuracy_by_depth": accuracy,
            "elapsed_s": dt,
            "peak_vram_mib": peak_mb,
        }

        del cache
        _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase5_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
