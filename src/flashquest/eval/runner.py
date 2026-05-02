"""Run a NIAH task across n_samples; return hits/total."""
from __future__ import annotations

from typing import Callable, Optional

import torch

from .niah import make_prompt, score


@torch.no_grad()
def run_niah(
    model,
    tokenizer,
    task: str,
    n_samples: int,
    ctx_len: int,
    seed: int = 0,
    max_new_tokens: int = 128,
    pre_sample: Optional[Callable[[int], None]] = None,
) -> dict:
    """Run `task` on `model` for `n_samples` prompts. Returns:
        {"task": str, "hits": int, "total": int, "samples": [{prompt_tokens, expected, generated, hit}, ...]}

    `pre_sample(i)` is called before each sample's generate; use it to reset
    a persistent KV cache between prompts.
    """
    samples = []
    hits = 0
    for i in range(n_samples):
        if pre_sample is not None:
            pre_sample(i)
        prompt, expected = make_prompt(task, ctx_len, tokenizer, seed=seed * 10000 + i)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True
        )
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        hit = score(text, expected)
        if hit:
            hits += 1
        samples.append({
            "prompt_tokens": int(ids.shape[1]),
            "expected": list(expected),
            "generated": text,
            "hit": bool(hit),
        })
    return {"task": task, "hits": hits, "total": n_samples, "samples": samples}
