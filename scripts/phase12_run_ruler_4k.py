"""Phase 12 Task B — RULER NIAH 4k quality confirmation for EAGLE-3 spec decode.

Mirrors scripts/phase7_run_ruler_4k_turbo.py, but the patched arm drives the
EAGLE-3 chain spec-decode dispatcher (``--speculative`` path) instead of
``model.generate``. Compares vanilla SDPA dense vs the spec path over the SAME
prompts/seeds. Gate: ``spec_hits / dense_hits >= 0.85`` for every task; we expect
the v1.0 default (single 100% / multikey 100% / multivalue >=95%).

This is a *confirmation*, not a re-derivation: Task 9 already proved end-to-end
losslessness (spec output == non-spec sparse greedy, bit-identical at n_draft 1
and 4), so the spec path's RULER numbers must match the non-spec sparse path's.
This run confirms it on the actual generative loop.

The patched arm uses ``PersistentInt4KVCache`` (retention 0.20, kv_bits 4, all-
retrieval head_pattern) + the EAGLE-3 draft head, exactly the shipped
``flashquest chat --speculative --n-draft 4 --kv-bits 4 --retention 0.20`` config.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.eval.niah import make_prompt, score
from flashquest.eval.runner import run_niah

TASKS = ["single", "multikey", "multivalue"]


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_dense(model_name, ctx_len, n_samples, seed, max_new):
    from flashquest.runtime.awq_load import load_awq_model

    print("\n=== dense (vanilla SDPA) ===")
    model, tok = load_awq_model(model_name)
    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        r = run_niah(model, tok, task=task, n_samples=n_samples,
                     ctx_len=ctx_len, seed=seed, max_new_tokens=max_new)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok
    _free()
    return out


@torch.no_grad()
def _run_niah_spec(model, tok, cache, draft, task, n_samples, ctx_len, seed,
                   max_new, n_draft, page_size):
    """NIAH runner that drives the EAGLE-3 spec-decode dispatcher.

    Reproduces ``run_niah`` (same ``make_prompt`` / ``score``, same seeds) but
    generates via ``make_quest_specdec(...).init/step`` instead of
    ``model.generate``. Resets the persistent cache before each sample and stops
    on EOS or when ``max_new`` tokens are emitted.
    """
    from flashquest.runtime.chat import _stop_token_ids
    from flashquest.specdec.dispatcher import make_quest_specdec

    stop_ids = _stop_token_ids(model, tok)
    hits = 0
    samples = []
    for i in range(n_samples):
        cache._seen_tokens = [0] * cache.num_layers
        prompt, expected = make_prompt(task, ctx_len, tok, seed=seed * 10000 + i)
        ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)

        init, step = make_quest_specdec(
            model, cache, draft, n_draft=n_draft, page_size=page_size,
        )
        init(ids)

        gen_ids: list[int] = []
        while len(gen_ids) < max_new:
            emitted = [int(x) for x in step().tolist()]
            stopped = False
            for tid in emitted:
                if tid in stop_ids:
                    stopped = True
                    break
                gen_ids.append(tid)
                if len(gen_ids) >= max_new:
                    break
            if stopped:
                break

        text = tok.decode(gen_ids, skip_special_tokens=True)
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


def run_patched_spec(model_name, ctx_len, n_samples, seed, max_new,
                     retention, num_sinks, window_pages, page_size,
                     n_draft, draft_model):
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import (
        patch_llama_for_quest_persistent,
    )
    from flashquest.runtime.awq_load import load_awq_model
    from flashquest.specdec import load_eagle3_draft

    print(f"\n=== patched INT4 + EAGLE-3 spec (retention={retention}, "
          f"n_draft={n_draft}, all-retrieval) ===")
    model, tok = load_awq_model(model_name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads,
                         dtype=torch.bool)
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx_len + max_new + 128, page_size=page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=retention, num_sinks=num_sinks,
        window_pages=window_pages, page_size=page_size,
    )
    draft = load_eagle3_draft(
        draft_model, device="cuda", dtype=torch.bfloat16,
        embed_weight=model.model.embed_tokens.weight,
    )

    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        r = _run_niah_spec(model, tok, cache, draft, task, n_samples, ctx_len,
                           seed, max_new, n_draft, page_size)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok, cache, draft
    _free()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--draft-model",
                   default="thoughtworks/Llama-3.2-3B-Instruct-Eagle3")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.20)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--n-draft", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    dense = run_dense(args.model, args.ctx_len, args.n_samples, args.seed,
                      args.max_new_tokens)
    patched = run_patched_spec(
        args.model, args.ctx_len, args.n_samples, args.seed, args.max_new_tokens,
        args.retention, args.num_sinks, args.window_pages, args.page_size,
        args.n_draft, args.draft_model,
    )

    tasks_out = {}
    all_pass = True
    for t in TASKS:
        k = f"niah_{t}"
        d, q = dense[k]["hits"], patched[k]["hits"]
        ratio = q / d if d > 0 else 0.0
        passed = ratio >= 0.85
        if not passed:
            all_pass = False
        tasks_out[k] = {
            "dense_hits": d, "spec_hits": q,
            "total": dense[k]["total"], "ratio": ratio, "pass": passed,
            "dense_wall_s": dense[k]["wall_s"],
            "spec_wall_s": patched[k]["wall_s"],
        }

    result = {
        "model": args.model, "draft_model": args.draft_model,
        "ctx_len": args.ctx_len, "n_samples": args.n_samples,
        "retention": args.retention, "num_sinks": args.num_sinks,
        "window_pages": args.window_pages, "page_size": args.page_size,
        "n_draft": args.n_draft, "seed": args.seed,
        "head_pattern": "all-retrieval", "kv_bits": 4, "speculative": True,
        "tasks": tasks_out,
        "gate": ("≥85% vs dense at retention=0.20 + all-retrieval head_pattern + "
                 "INT4 KV + EAGLE-3 spec decode (n_draft=4)"),
        "all_pass": all_pass,
        "note": ("Confirmation, not re-derivation: Task 9 proved spec output == "
                 "non-spec sparse greedy (bit-identical). Expect the v1.0 default "
                 "single 100% / multikey 100% / multivalue >=95%."),
    }
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (Path(__file__).resolve().parents[1] / "benchmarks" /
                    "phase12" / "ruler_4k_spec.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}\nAll-pass: {all_pass}")
    for t, info in tasks_out.items():
        print(f"  {t}: {info['spec_hits']}/{info['dense_hits']} "
              f"= {info['ratio']:.2%} {'PASS' if info['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
