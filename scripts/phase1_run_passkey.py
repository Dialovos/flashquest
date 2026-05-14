"""Run passkey at three depths × four retentions. Compute % correct."""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.passkey import make_example, score


N_TRIALS = 5
DEPTHS = [0.1, 0.5, 0.9]
RETENTIONS = [1.0, 0.5, 0.25, 0.1]
TARGET_TOTAL_TOKENS = 1024  # keeps prefill (B,H,S,P,D) broadcast within 4 GB


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


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

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

    # Sanity: how big are the prompts actually?
    prompt_lens = [len(tok.encode(ex.text, add_special_tokens=False)) for ex in examples]
    print(f"prompt token counts (target {TARGET_TOTAL_TOKENS}): min={min(prompt_lens)} max={max(prompt_lens)} median={sorted(prompt_lens)[len(prompt_lens)//2]}")

    results: dict = {"depths": DEPTHS, "n_trials": N_TRIALS, "configs": {}}

    print("dense baseline ...")
    m = fresh_model(name)
    t0 = time.perf_counter()
    correct_by_depth = {d: 0 for d in DEPTHS}
    for ex in examples:
        gen = generate_passkey_answer(m, tok, ex.text)
        if score(gen, ex.passkey):
            correct_by_depth[ex.depth_pct] += 1
    dt = time.perf_counter() - t0
    results["dense"] = {
        "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
        "elapsed_s": dt,
    }
    print(f"  {results['dense']['accuracy_by_depth']}  ({dt:.1f}s)")
    del m
    _free()

    for r in RETENTIONS:
        print(f"retention={r} ...")
        m = fresh_model(name)
        patch_llama_for_quest_eager(m, retention=r, num_sinks=4, window_pages=2, page_size=64)
        t0 = time.perf_counter()
        correct_by_depth = {d: 0 for d in DEPTHS}
        for ex in examples:
            gen = generate_passkey_answer(m, tok, ex.text)
            if score(gen, ex.passkey):
                correct_by_depth[ex.depth_pct] += 1
        dt = time.perf_counter() - t0
        results["configs"][f"retention_{r}"] = {
            "accuracy_by_depth": {str(d): correct_by_depth[d] / N_TRIALS for d in DEPTHS},
            "elapsed_s": dt,
        }
        print(f"  {results['configs'][f'retention_{r}']['accuracy_by_depth']}  ({dt:.1f}s)")
        del m
        _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase1_passkey.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
