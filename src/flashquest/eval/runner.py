"""Run a NIAH task across n_samples; return hits/total."""
from __future__ import annotations

from collections.abc import Callable

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
    pre_sample: Callable[[int], None] | None = None,
    examples: list[dict] | None = None,
) -> dict:
    """Run `task` on `model` for `n_samples` prompts. Returns:
        {"task": str, "hits": int, "total": int, "samples": [{prompt_tokens, expected, generated, hit}, ...]}

    `pre_sample(i)` is called before each sample's generate; use it to reset
    a persistent KV cache between prompts.
    """
    if examples is not None and len(examples) != n_samples:
        raise ValueError("manifest length differs from n_samples")
    samples = []
    hits = 0
    for i in range(n_samples):
        if pre_sample is not None:
            pre_sample(i)
        if examples is None:
            prompt, expected = make_prompt(task, ctx_len, tokenizer, seed=seed * 10000 + i)
            ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            example_id = f"{task}:{seed}:{i}"
        else:
            example = examples[i]
            if example["task"] != task or example["seed"] != seed:
                raise ValueError("manifest task/seed differs from requested task/seed")
            expected = example["expected"]
            ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=model.device)
            example_id = example["example_id"]
        eos = model.generation_config.eos_token_id
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        pad = getattr(tokenizer, "pad_token_id", None)
        if pad is None:
            pad = eos if isinstance(eos, int) else (eos[0] if eos else None)
        generation = {"max_new_tokens": max_new_tokens, "do_sample": False, "use_cache": True}
        if pad is not None:
            generation["pad_token_id"] = pad
        # Every manifest is one unpadded prompt. Do not infer padding from EOS IDs.
        out = model.generate(ids, attention_mask=torch.ones_like(ids), **generation)
        generated_ids = out[0, ids.shape[1]:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        termination = "eos" if generated_ids.numel() and int(generated_ids[-1]) in eos_ids else "limit"
        hit = score(text, expected)
        if hit:
            hits += 1
        samples.append({
            "example_id": example_id,
            "prompt_tokens": int(ids.shape[1]),
            "expected": list(expected),
            "generated": text,
            "hit": bool(hit),
            "output_tokens": int(generated_ids.numel()),
            "termination": termination,
        })
    return {"task": task, "hits": hits, "total": n_samples, "samples": samples}
