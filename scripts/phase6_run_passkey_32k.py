"""Phase 6 task 1 quality check: re-run Phase 5's 32k passkey eval with
the new algebraic + vectorized paths. Must still pass 6/6 across depths."""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


CONTEXT_LENS = [8192, 32768]
DEPTHS = [0.1, 0.5, 0.9]
N_TRIALS = 2
MAX_NEW = 8


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def make_passkey_prompt(tok, ctx_len: int, depth: float, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    passkey = "".join(str(rng.randint(0, 9)) for _ in range(5))
    filler = "The grass is green. The sky is blue. The sun is yellow. " * (ctx_len // 8)
    target_pos = int(depth * len(filler))
    prefix = filler[:target_pos]
    suffix = filler[target_pos:]
    prompt = (
        f"There is an important info hidden in the text. Find it.\n\n"
        f"{prefix} The pass key is {passkey}. Remember it. {passkey} is the pass key. {suffix}\n\n"
        f"What is the pass key? The pass key is "
    )
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > ctx_len:
        ids = ids[:, :ctx_len]
    return tok.decode(ids[0]), passkey


@torch.no_grad()
def run_one(model, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=max(CONTEXT_LENS) + MAX_NEW + 128,
        page_size=64, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    results = {"model": name, "tiers": []}
    for ctx in CONTEXT_LENS:
        tier = {"context": ctx, "depths": []}
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for depth in DEPTHS:
            hits = 0
            for trial in range(N_TRIALS):
                cache._seen_tokens = [0] * cache.num_layers
                _free()
                prompt, key = make_passkey_prompt(tok, ctx, depth, seed=ctx * 100 + trial)
                out = run_one(model, tok, prompt)
                if key in out:
                    hits += 1
                print(f"  ctx={ctx} depth={depth} trial={trial} key={key} hit={key in out} out={out!r}")
            tier["depths"].append({"depth": depth, "hits": hits, "trials": N_TRIALS})
        tier["wall_s"] = time.perf_counter() - t0
        tier["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"ctx={ctx}: wall={tier['wall_s']:.0f}s peak_vram={tier['peak_vram_mib']:.0f} MiB")
        results["tiers"].append(tier)

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
