"""Phase 9 Task 1 — PLD profile-first entry gate.

Measure on Llama-3.2-3B-AWQ at ctx=8k, greedy decoding:
  - mean accept count M_avg per PLD step
  - hit rate (% steps where PLD admissible)
  - S_q=5 vs S_q=1 dense verify cost ratio

Workloads:
  - PG-essay summarize 8k input
  - RULER NIAH single 4k

Gate (must clear ALL to proceed to Task 2+):
  - M_avg >= 2.0 on (PG + RULER) average
  - S_q=5 dense / S_q=1 dense <= 1.3
  - hit rate >= 30% on (PG + RULER)
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.eval.niah import make_prompt
from flashquest.runtime.awq_load import load_awq_model


def propose_draft_naive(
    prompt_ids: list[int],
    history_tail: list[int],
    K_match: int,
    N_draft: int,
) -> list[int] | None:
    if len(history_tail) < K_match:
        return None
    needle = tuple(history_tail[-K_match:])
    last_valid_start = len(prompt_ids) - K_match - N_draft
    for i in range(last_valid_start, -1, -1):
        if tuple(prompt_ids[i : i + K_match]) == needle:
            return prompt_ids[i + K_match : i + K_match + N_draft]
    return None


@torch.no_grad()
def run_workload(
    model, tok, prompts: list[str],
    *, n_decode: int, K_match: int, N_draft: int,
) -> dict:
    """Greedy decode each prompt for n_decode tokens, attempting PLD each step."""
    accept_counts = []
    pld_admissible = 0
    pld_steps_attempted = 0
    sq1_times_ms = []
    sq5_times_ms = []

    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        prompt_ids = inputs.input_ids[0].tolist()
        out = model(**inputs, use_cache=True)
        past = out.past_key_values
        next_argmax = out.logits[:, -1].argmax(dim=-1)
        next_input = next_argmax.clone()
        committed = list(prompt_ids)
        prev_argmax_valid = True
        emitted = 0
        while emitted < n_decode:
            history_tail = committed[-K_match:] if len(committed) >= K_match else []
            draft = (
                propose_draft_naive(prompt_ids, history_tail, K_match, N_draft)
                if history_tail and prev_argmax_valid else None
            )
            admissible = (
                draft is not None
                and prev_argmax_valid
                and draft[0] == int(next_argmax.item())
            )
            pld_steps_attempted += 1
            if not admissible:
                t0 = time.perf_counter()
                out = model(
                    input_ids=next_input.unsqueeze(0),
                    past_key_values=past, use_cache=True,
                )
                torch.cuda.synchronize()
                sq1_times_ms.append((time.perf_counter() - t0) * 1000)
                past = out.past_key_values
                new_argmax = out.logits[:, -1].argmax(dim=-1)
                committed.append(int(next_input.item()))
                next_input = new_argmax
                next_argmax = new_argmax
                prev_argmax_valid = True
                emitted += 1
                continue
            pld_admissible += 1
            verify_in = torch.tensor([draft], device="cuda")
            t0 = time.perf_counter()
            out = model(
                input_ids=verify_in,
                past_key_values=past, use_cache=True,
            )
            torch.cuda.synchronize()
            sq5_times_ms.append((time.perf_counter() - t0) * 1000)
            past = out.past_key_values
            argmax_seq = out.logits.argmax(dim=-1).squeeze(0)
            M = 0
            for i in range(N_draft - 1):
                if int(argmax_seq[i].item()) == int(draft[i + 1]):
                    M += 1
                else:
                    break
            accept_counts.append(M)
            free_token = argmax_seq[M].unsqueeze(0)
            for t in draft[: M + 1]:
                committed.append(int(t))
            emitted += M + 1
            next_input = free_token
            next_argmax = free_token
            prev_argmax_valid = False
        del past
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "accept_counts": accept_counts,
        "M_avg": sum(accept_counts) / max(1, len(accept_counts)),
        "pld_admissible": pld_admissible,
        "pld_steps_attempted": pld_steps_attempted,
        "hit_rate": pld_admissible / max(1, pld_steps_attempted),
        "sq1_avg_ms": sum(sq1_times_ms) / max(1, len(sq1_times_ms)),
        "sq5_avg_ms": sum(sq5_times_ms) / max(1, len(sq5_times_ms)),
        "n_pld_steps": len(accept_counts),
        "n_single_steps": len(sq1_times_ms),
    }


def load_pg_prompts(n: int, ctx_target: int, tok) -> list[str]:
    """Split data/PaulGrahamEssays.json (one big text blob) into N consecutive
    ~ctx_target-token chunks, append a summarize instruction to each.
    """
    pg_obj = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    text = pg_obj["text"] if isinstance(pg_obj, dict) else pg_obj
    ids = tok(text, return_tensors="pt").input_ids[0]
    instr_tail = "\n\nSummarize the above:\n\n"
    instr_tail_ids = tok(instr_tail, return_tensors="pt").input_ids[0]
    chunk_len = ctx_target - len(instr_tail_ids) - 8
    prompts = []
    for i in range(n):
        start = i * chunk_len
        end = start + chunk_len
        if end > len(ids):
            break
        clip = tok.decode(ids[start:end], skip_special_tokens=True)
        prompts.append(clip + instr_tail)
    return prompts


def load_ruler_prompts(n: int, tok, ctx_len: int = 4096) -> list[str]:
    prompts = []
    for i in range(n):
        prompt, _ = make_prompt(task="single", ctx_len=ctx_len, tokenizer=tok, seed=i)
        prompts.append(prompt)
    return prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--n-prompts", type=int, default=20)
    p.add_argument("--n-decode", type=int, default=64)
    p.add_argument("--K-match", type=int, default=3)
    p.add_argument("--N-draft", type=int, default=5)
    p.add_argument("--ctx-pg", type=int, default=8192)
    p.add_argument("--out", default="benchmarks/phase9_task1_results.json")
    args = p.parse_args()

    print(f"Loading model {args.model} ...")
    model, tok = load_awq_model(args.model)

    print(f"Loading {args.n_prompts} PG-summarize prompts (ctx~{args.ctx_pg})...")
    pg_prompts = load_pg_prompts(args.n_prompts, args.ctx_pg, tok)
    print(f"  loaded {len(pg_prompts)} PG prompts")

    print(f"Loading {args.n_prompts} RULER NIAH single 4k prompts...")
    ruler_prompts = load_ruler_prompts(args.n_prompts, tok, ctx_len=4096)
    print(f"  loaded {len(ruler_prompts)} RULER prompts")

    print(f"Running PG-summarize workload...")
    pg_res = run_workload(
        model, tok, pg_prompts,
        n_decode=args.n_decode, K_match=args.K_match, N_draft=args.N_draft,
    )
    print(f"  PG: M_avg={pg_res['M_avg']:.2f}, hit_rate={pg_res['hit_rate']:.2%}, "
          f"sq1={pg_res['sq1_avg_ms']:.1f}ms, sq5={pg_res['sq5_avg_ms']:.1f}ms")

    print(f"Running RULER workload...")
    ruler_res = run_workload(
        model, tok, ruler_prompts,
        n_decode=args.n_decode, K_match=args.K_match, N_draft=args.N_draft,
    )
    print(f"  RULER: M_avg={ruler_res['M_avg']:.2f}, hit_rate={ruler_res['hit_rate']:.2%}, "
          f"sq1={ruler_res['sq1_avg_ms']:.1f}ms, sq5={ruler_res['sq5_avg_ms']:.1f}ms")

    combined_M_avg = (pg_res["M_avg"] + ruler_res["M_avg"]) / 2
    combined_hit_rate = (pg_res["hit_rate"] + ruler_res["hit_rate"]) / 2
    sq1_avg = (pg_res["sq1_avg_ms"] + ruler_res["sq1_avg_ms"]) / 2
    sq5_avg = (pg_res["sq5_avg_ms"] + ruler_res["sq5_avg_ms"]) / 2
    sq5_to_sq1 = sq5_avg / max(0.001, sq1_avg)

    print(f"\n=== Phase 9 Task 1 Entry Gate ===")
    print(f"  M_avg (PG + RULER):       {combined_M_avg:.2f}     (gate: >= 2.0)")
    print(f"  hit_rate (PG + RULER):    {combined_hit_rate:.2%}    (gate: >= 30%)")
    print(f"  S_q=5 / S_q=1 wall:       {sq5_to_sq1:.2f}x    (gate: <= 1.3x)")
    gate_pass = (
        combined_M_avg >= 2.0
        and combined_hit_rate >= 0.30
        and sq5_to_sq1 <= 1.3
    )
    print(f"\n  ENTRY GATE: {'PASS' if gate_pass else 'FAIL'}")

    Path(args.out).write_text(json.dumps({
        "pg": pg_res,
        "ruler": ruler_res,
        "combined_M_avg": combined_M_avg,
        "combined_hit_rate": combined_hit_rate,
        "sq5_to_sq1_ratio": sq5_to_sq1,
        "gate_pass": gate_pass,
    }, indent=2))
    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
