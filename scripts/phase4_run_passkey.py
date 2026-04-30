"""Phase 4 passkey: Llama-3.2-1B with synthetic 70/30 DuoAttention split,
retention=0.25 sinks=4 window=2. Compare against Phase 1's all-retrieval patch.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo
from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.passkey import make_example, score


N_TRIALS = 5
DEPTHS = [0.1, 0.5, 0.9]
TARGET_TOTAL_TOKENS = 1024
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2
RETRIEVAL_FRACTION = 0.7


def fresh_model(name: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


@torch.no_grad()
def generate_passkey_answer(model: torch.nn.Module, tok, prompt: str) -> str:
    ids = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(
        **ids, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def synthetic_pattern(num_layers: int, num_kv: int, fraction_retrieval: float, seed: int = 0):
    """Random per-layer per-head pattern with fraction_retrieval True."""
    rng = torch.Generator().manual_seed(seed)
    return torch.rand((num_layers, num_kv), generator=rng) < fraction_retrieval


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)

    examples = []
    rng = random.Random(0)
    for d in DEPTHS:
        for _ in range(N_TRIALS):
            examples.append(
                make_example(
                    rng=rng,
                    tokenizer=tok,
                    target_total_tokens=TARGET_TOTAL_TOKENS,
                    depth_pct=d,
                )
            )

    results: dict = {
        "depths": DEPTHS,
        "n_trials": N_TRIALS,
        "retention": RETENTION,
        "num_sinks": NUM_SINKS,
        "window_pages": WINDOW_PAGES,
        "retrieval_fraction": RETRIEVAL_FRACTION,
    }

    print("phase 1 baseline (all retrieval) ...")
    m = fresh_model(name)
    patch_llama_for_quest_eager(
        m, retention=RETENTION, num_sinks=NUM_SINKS, window_pages=WINDOW_PAGES, page_size=64,
    )
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["phase1"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
    }
    print(f"  {results['phase1']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    _free()

    print("phase 4 (DuoAttention 70/30 split) ...")
    pattern = synthetic_pattern(cfg.num_hidden_layers, cfg.num_key_value_heads, RETRIEVAL_FRACTION)
    m = fresh_model(name)
    patch_llama_for_quest_duo(
        m, head_pattern=pattern, retention=RETENTION, num_sinks=NUM_SINKS,
        window_pages=WINDOW_PAGES, page_size=64,
    )
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["phase4"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
        "head_pattern_retrieval_fraction": pattern.float().mean().item(),
    }
    print(f"  {results['phase4']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase4_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
