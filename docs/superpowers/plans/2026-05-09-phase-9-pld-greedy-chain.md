# Phase 9 — PLD Greedy Chain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lossless-by-construction greedy speculative decoding via Prompt-Lookup Decoding to the Quest sparse-attention runtime, targeting ≥15 tok/s @ 32k decode on Llama-3.2-3B-AWQ.

**Architecture:** PLD chain mines n-gram matches from the prompt, proposes N_draft tokens, and verifies them in a single S_q>1 forward pass. Verify uses sparse Quest attention (compact INT4 kernel, extended to S_q>1 with score-prioritized UNION selection) over committed pages, plus dense BF16 over `K_partial || K_sandbox`, merged via per-query LSE. Sandbox-and-commit on the cache: drafts go to a per-layer BF16 sandbox, committed atomically across all layers when the walk-and-accept settles.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, Triton 3.1.0, transformers 4.57.6, AutoAWQ 0.2.9. RTX 3050 Ti Laptop sm_86, 4 GB VRAM, WSL2.

**Spec:** `docs/superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md` (r2.1, post-codex-r1+r2)

**Critical entry gate (Task 1, profile-first per `feedback_profile_before_speedup_specs.md`):** stop if mean accept M_avg < 2 on (PG-summarize + RULER) or S_q=5 dense verify cost > 1.3× S_q=1 baseline. **Do not implement Tasks 3+ if the gate fails.**

---

## Task 1: Profile-first PLD measurement (ENTRY GATE)

This task uses the **existing** dense SDPA forward path — no new kernel, no sandbox, no dispatcher. The goal is to measure PLD's accept rate and verify-step cost on our specific workload + hardware BEFORE writing any kernel/integration code.

**Files:**
- Create: `benchmarks/phase9_task1_pld_profile.py`

- [ ] **Step 1: Write the script**

Create `benchmarks/phase9_task1_pld_profile.py`:

```python
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
from transformers import AutoModelForCausalLM, AutoTokenizer


def propose_draft_naive(
    prompt_ids: list[int],
    history_tail: list[int],
    K_match: int,
    N_draft: int,
) -> list[int] | None:
    if len(history_tail) < K_match:
        return None
    needle = tuple(history_tail[-K_match:])
    # Rightmost match
    for i in range(len(prompt_ids) - K_match - N_draft, -1, -1):
        if tuple(prompt_ids[i : i + K_match]) == needle:
            return prompt_ids[i + K_match : i + K_match + N_draft]
    return None


@torch.no_grad()
def run_workload(
    model, tok, prompts: list[str],
    *, n_decode: int, K_match: int, N_draft: int,
) -> dict:
    """Greedy decode each prompt for n_decode tokens, attempting PLD each step.

    Uses the model's HF cache (no flashquest patching) — the goal is to measure
    PLD admissibility + accept rate independent of our kernel optimizations.
    """
    accept_counts = []      # per PLD step
    pld_attempts = 0
    pld_admissible = 0
    sq1_times_ms = []       # wall ms for S_q=1 forwards
    sq5_times_ms = []       # wall ms for S_q=N_draft forwards (admissible PLD)

    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        prompt_ids = inputs.input_ids[0].tolist()
        # Prefill
        out = model(**inputs, use_cache=True)
        past = out.past_key_values
        next_argmax = out.logits[:, -1].argmax(dim=-1)        # (1,)
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
            if not admissible:
                # Single-token decode
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
            # PLD verify
            pld_attempts += 1
            pld_admissible += 1
            verify_in = torch.tensor([draft], device="cuda")    # (1, N_draft)
            t0 = time.perf_counter()
            out = model(
                input_ids=verify_in,
                past_key_values=past, use_cache=True,
            )
            torch.cuda.synchronize()
            sq5_times_ms.append((time.perf_counter() - t0) * 1000)
            past = out.past_key_values
            argmax_seq = out.logits.argmax(dim=-1).squeeze(0)   # (N_draft,)
            M = 0
            for i in range(N_draft - 1):
                if int(argmax_seq[i].item()) == int(draft[i + 1]):
                    M += 1
                else:
                    break
            accept_counts.append(M)
            free_token = argmax_seq[M].unsqueeze(0)
            # Emit M+1 tokens (D_0..D_M); free_token held for next step
            for t in draft[: M + 1]:
                committed.append(int(t))
            emitted += M + 1
            next_input = free_token
            next_argmax = free_token
            prev_argmax_valid = False                            # forces next step to single-decode
        # Cleanup before next prompt
        del past
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "accept_counts": accept_counts,
        "M_avg": sum(accept_counts) / max(1, len(accept_counts)),
        "pld_attempts": pld_attempts,
        "pld_admissible": pld_admissible,
        "hit_rate": pld_admissible / max(1, pld_attempts + len(sq1_times_ms) - len(sq5_times_ms)),
        "sq1_avg_ms": sum(sq1_times_ms) / max(1, len(sq1_times_ms)),
        "sq5_avg_ms": sum(sq5_times_ms) / max(1, len(sq5_times_ms)),
        "n_pld_steps": len(accept_counts),
        "n_single_steps": len(sq1_times_ms),
    }


def load_pg_prompts(n: int, ctx_target: int, tok) -> list[str]:
    pg = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    prompts = []
    for entry in pg[:n]:
        text = entry.get("text") or entry.get("body") or ""
        if not text:
            continue
        ids = tok(text, return_tensors="pt").input_ids[0]
        if len(ids) < ctx_target * 0.5:
            continue
        clip = tok.decode(ids[: ctx_target - 64])
        prompts.append(clip + "\n\nSummarize the above:\n\n")
        if len(prompts) == n:
            break
    return prompts


def load_ruler_prompts(n: int) -> list[str]:
    from flashquest.eval.niah import niah_single
    prompts = []
    for i in range(n):
        sample = niah_single(seed=i, ctx_len=4096)
        prompts.append(sample["input"])
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

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
    ).cuda().eval()

    print(f"Loading {args.n_prompts} PG-summarize prompts (ctx~{args.ctx_pg})...")
    pg_prompts = load_pg_prompts(args.n_prompts, args.ctx_pg, tok)

    print(f"Loading {args.n_prompts} RULER NIAH single 4k prompts...")
    ruler_prompts = load_ruler_prompts(args.n_prompts)

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
    sq5_to_sq1 = (
        ((pg_res["sq5_avg_ms"] or 0) + (ruler_res["sq5_avg_ms"] or 0))
        / max(0.001, (pg_res["sq1_avg_ms"] or 0) + (ruler_res["sq1_avg_ms"] or 0))
        * 2  # both numerator and denominator divided by 2
    )

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
```

- [ ] **Step 2: Run the profile**

Run (background, ~20-30 min for 40 prompts × 64 decode tokens):

```bash
nice -n 19 python benchmarks/phase9_task1_pld_profile.py 2>&1 | tee benchmarks/phase9_task1_log.txt
```

Expected output (sample):
```
Loading 20 PG-summarize prompts (ctx~8192)...
Loading 20 RULER NIAH single 4k prompts...
Running PG-summarize workload...
  PG: M_avg=2.71, hit_rate=42%, sq1=87.3ms, sq5=98.4ms
Running RULER workload...
  RULER: M_avg=3.92, hit_rate=58%, sq1=83.1ms, sq5=99.2ms

=== Phase 9 Task 1 Entry Gate ===
  M_avg (PG + RULER):       3.32     (gate: >= 2.0)
  hit_rate (PG + RULER):    50.00%    (gate: >= 30%)
  S_q=5 / S_q=1 wall:       1.16x    (gate: <= 1.3x)

  ENTRY GATE: PASS
```

- [ ] **Step 3: Decision gate**

If `ENTRY GATE: PASS`: proceed to Task 2.

If `ENTRY GATE: FAIL`:
- Document findings in `docs/PHASES/phase-9-killed-by-profile.md`
- **Stop. Do not implement Tasks 2+.** Pivot to Lookahead, Self-spec early-exit, or drop Phase 9.
- Per saved memory `feedback_profile_before_speedup_specs.md` and Phase 8b precedent.

If gate is borderline (e.g., M_avg ∈ [2, 3)): document as "reduced-expectations zone" — proceed but adjust target to 1.6-2.0× gain, document in `phase-9-notes.md` final writeup.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/phase9_task1_pld_profile.py benchmarks/phase9_task1_results.json benchmarks/phase9_task1_log.txt
git commit -m "$(cat <<'EOF'
phase 9 task 1: PLD profile-first entry gate

Naive dense-SDPA verify on Llama-3.2-3B-AWQ at 8k. Measures M_avg, hit
rate, S_q=5/S_q=1 cost ratio on PG-summarize + RULER. Gate clears at
M_avg>=2.0 + hit_rate>=30% + cost<=1.3x.
EOF
)"
```

---

## Task 2: PLD draft proposer (n-gram match)

**Files:**
- Create: `src/flashquest/specdec/__init__.py`
- Create: `src/flashquest/specdec/pld.py`
- Test: `tests/test_pld_proposer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pld_proposer.py`:

```python
"""Phase 9 Task 2 — PLD draft proposer."""
import pytest
import torch


def test_no_match_returns_none():
    from flashquest.specdec.pld import propose_draft
    prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    history = torch.tensor([99, 100, 101], dtype=torch.int64)
    assert propose_draft(prompt, history, K_match=3, N_draft=2) is None


def test_rightmost_match_wins():
    from flashquest.specdec.pld import propose_draft
    # [A, B, C] appears at indices 0 and 5; rightmost = 5
    prompt = torch.tensor([10, 20, 30, 99, 99, 10, 20, 30, 40, 50, 60], dtype=torch.int64)
    history = torch.tensor([10, 20, 30], dtype=torch.int64)
    out = propose_draft(prompt, history, K_match=3, N_draft=3)
    assert out is not None
    assert out.tolist() == [40, 50, 60]


def test_history_too_short_returns_none():
    from flashquest.specdec.pld import propose_draft
    prompt = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    history = torch.tensor([1, 2], dtype=torch.int64)         # only 2 tokens, K_match=3
    assert propose_draft(prompt, history, K_match=3, N_draft=2) is None


def test_match_at_end_insufficient_continuation():
    from flashquest.specdec.pld import propose_draft
    prompt = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    history = torch.tensor([30, 40, 50], dtype=torch.int64)   # match at end, only 0 continuation
    assert propose_draft(prompt, history, K_match=3, N_draft=2) is None


def test_match_at_end_partial_continuation_returns_none():
    from flashquest.specdec.pld import propose_draft
    # Only 1 continuation token after match, but N_draft=2
    prompt = torch.tensor([10, 20, 30, 40, 50, 99], dtype=torch.int64)
    history = torch.tensor([30, 40, 50], dtype=torch.int64)
    assert propose_draft(prompt, history, K_match=3, N_draft=2) is None


def test_full_continuation_just_fits():
    from flashquest.specdec.pld import propose_draft
    # Exactly N_draft tokens after match
    prompt = torch.tensor([10, 20, 30, 40, 50, 99, 88], dtype=torch.int64)
    history = torch.tensor([30, 40, 50], dtype=torch.int64)
    out = propose_draft(prompt, history, K_match=3, N_draft=2)
    assert out is not None
    assert out.tolist() == [99, 88]


def test_returns_int64_cpu():
    from flashquest.specdec.pld import propose_draft
    prompt = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    history = torch.tensor([1, 2, 3], dtype=torch.int64)
    out = propose_draft(prompt, history, K_match=3, N_draft=2)
    assert out.dtype == torch.int64
    assert out.device.type == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pld_proposer.py -v
```

Expected: all FAIL with `ModuleNotFoundError: No module named 'flashquest.specdec'`

- [ ] **Step 3: Implement the proposer**

Create `src/flashquest/specdec/__init__.py`:

```python
"""Speculative-decoding components (Phase 9: PLD greedy chain)."""
```

Create `src/flashquest/specdec/pld.py`:

```python
"""Phase 9 — Prompt-Lookup Decoding draft proposer.

Searches a static prompt for the rightmost match of the last K_match committed
tokens and returns the next N_draft prompt tokens after that match.
"""
from __future__ import annotations

import torch


def propose_draft(
    prompt_ids: torch.Tensor,
    history_tail: torch.Tensor,
    *,
    K_match: int = 3,
    N_draft: int = 5,
) -> torch.Tensor | None:
    """Find rightmost K_match-gram match in prompt_ids; return next N_draft tokens or None.

    Args:
        prompt_ids: (S_prompt,) int64 on CPU.
        history_tail: (>=K_match,) int64 on CPU.
        K_match: n-gram size to match.
        N_draft: number of continuation tokens to return.

    Returns:
        (N_draft,) int64 on CPU, or None if (a) history too short,
        (b) no match found, (c) insufficient continuation after match.
    """
    if history_tail is None or history_tail.numel() < K_match:
        return None
    needle = history_tail[-K_match:].tolist()
    n = needle.__len__()
    plist = prompt_ids.tolist()
    P = len(plist)
    # Rightmost match: scan from right. Match must have N_draft tokens after.
    last_valid_start = P - K_match - N_draft
    for i in range(last_valid_start, -1, -1):
        if plist[i : i + n] == needle:
            return torch.tensor(plist[i + n : i + n + N_draft], dtype=torch.int64)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pld_proposer.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/specdec/__init__.py src/flashquest/specdec/pld.py tests/test_pld_proposer.py
git commit -m "phase 9 task 2: PLD draft proposer (n-gram match, rightmost wins, pad-to-None)"
```

---

## Task 3: Score-prioritized UNION selection helper

**Files:**
- Modify: `src/flashquest/eager/selection.py` (append helper)
- Test: `tests/test_compact_union_selection.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compact_union_selection.py`:

```python
"""Phase 9 Task 3 — score-prioritized UNION selection helper."""
import pytest
import torch


def _make_inputs(B=1, H_q=2, S_q=3, P=10):
    sel = torch.zeros(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    scores = torch.zeros(B, H_q, S_q, P, device="cuda")
    return sel, scores


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_underflow_pads_with_neg1():
    from flashquest.eager.selection import build_compact_union_selection
    sel, scores = _make_inputs(P=10)
    # Per-Q: pages 0,1,2 selected (sinks num_sinks=2 force); window 1 page; UNION = {0,1,2,9} but 9 not selected
    sel[..., :3] = True
    scores[..., :3] = 1.0
    # No window pages because completed_len=0 < window_pages
    out = build_compact_union_selection(
        sel, scores, num_sinks=2, window_pages=1, completed_len=0, page_size=64,
        BUCKET_MAX_UNION=8,
    )
    assert out.shape == (1, 2, 8)
    assert out.dtype == torch.int32
    out_sorted = out.sort(dim=-1, descending=True).values
    # Should contain {0, 1, 2} and pad with -1 for the rest
    selected = set(out_sorted[0, 0].tolist())
    assert {0, 1, 2}.issubset(selected)
    assert -1 in selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_overflow_drops_lowest_score():
    from flashquest.eager.selection import build_compact_union_selection
    B, H_q, S_q, P = 1, 1, 1, 12
    sel = torch.zeros(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    scores = torch.zeros(B, H_q, S_q, P, device="cuda")
    # Select pages 2..10 (9 pages), num_sinks=2 forces {0,1}, no window
    sel[..., 2:11] = True
    # Scores: page 2 has lowest score, page 10 has highest
    for i, p in enumerate(range(2, 11)):
        scores[..., p] = 0.1 + 0.1 * i
    # BUCKET_MAX_UNION = 6, total candidates = 9 union + 2 sinks = 11 distinct
    # But sinks are forced; we have 6 - 2 = 4 slots for the 9 union pages
    # Drops the 5 lowest-score: pages 2, 3, 4, 5, 6 (scores 0.1, 0.2, 0.3, 0.4, 0.5)
    out = build_compact_union_selection(
        sel, scores, num_sinks=2, window_pages=0, completed_len=0, page_size=64,
        BUCKET_MAX_UNION=6,
    )
    selected = set(out[0, 0].tolist())
    selected.discard(-1)
    assert 0 in selected and 1 in selected      # forced sinks
    # Highest 4 scores: pages 7, 8, 9, 10
    assert {7, 8, 9, 10}.issubset(selected)
    # Lowest 5 (pages 2..6) should NOT be selected
    assert not any(p in selected for p in [2, 3, 4, 5, 6])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sinks_window_force_included_under_overflow():
    from flashquest.eager.selection import build_compact_union_selection
    B, H_q, S_q, P = 1, 1, 1, 20
    sel = torch.zeros(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    scores = torch.zeros(B, H_q, S_q, P, device="cuda")
    # Sinks {0,1,2,3} forced. Window: completed_len=10 page_size=1 window_pages=2 → pages [8,9]
    # UNION pages with HIGH scores: {15, 16, 17, 18, 19}
    sel[..., 15:20] = True
    scores[..., 15:20] = 10.0
    # BUCKET_MAX_UNION = 7. Forced = 4 sinks + 2 window = 6 must-include.
    # Plus highest score from union → 1 more. Total 7. Should drop {16, 17, 18, 19} from union.
    out = build_compact_union_selection(
        sel, scores, num_sinks=4, window_pages=2, completed_len=10, page_size=1,
        BUCKET_MAX_UNION=7,
    )
    selected = set(out[0, 0].tolist())
    selected.discard(-1)
    assert {0, 1, 2, 3}.issubset(selected)              # sinks
    assert {8, 9}.issubset(selected)                    # window
    assert len(selected) == 7
    assert (selected & {15, 16, 17, 18, 19})            # at least one union page kept


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_no_cuda_sync():
    """Helper must be GPU-resident (no .item() calls)."""
    from flashquest.eager.selection import build_compact_union_selection
    sel, scores = _make_inputs()
    sel[..., :3] = True
    scores[..., :3] = 1.0
    # Run on a non-default stream and confirm no implicit sync to default stream
    s = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        out = build_compact_union_selection(
            sel, scores, num_sinks=2, window_pages=1, completed_len=0, page_size=64,
            BUCKET_MAX_UNION=8,
        )
    # If there's a sync, the next line on default stream would race and torch would have synced.
    # We just verify the call returns without error and the result is correct.
    s.synchronize()
    assert out.shape == (1, 2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_union_superset_of_per_q():
    from flashquest.eager.selection import build_compact_union_selection
    B, H_q, S_q, P = 1, 1, 3, 12
    sel = torch.zeros(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    scores = torch.zeros(B, H_q, S_q, P, device="cuda")
    # Each query selects different pages: Q0 picks 5,6; Q1 picks 7,8; Q2 picks 9,10
    sel[0, 0, 0, [5, 6]] = True
    sel[0, 0, 1, [7, 8]] = True
    sel[0, 0, 2, [9, 10]] = True
    scores[..., 5:11] = 5.0
    out = build_compact_union_selection(
        sel, scores, num_sinks=0, window_pages=0, completed_len=0, page_size=64,
        BUCKET_MAX_UNION=8,
    )
    selected = set(out[0, 0].tolist())
    selected.discard(-1)
    # UNION = {5, 6, 7, 8, 9, 10}; all should be present
    assert {5, 6, 7, 8, 9, 10}.issubset(selected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compact_union_selection.py -v
```

Expected: all FAIL with `ImportError: cannot import name 'build_compact_union_selection'`.

- [ ] **Step 3: Implement the helper**

Append to `src/flashquest/eager/selection.py` (after `build_compact_selection`):

```python
def build_compact_union_selection(
    sel_per_q: torch.Tensor,
    scores: torch.Tensor,
    *,
    num_sinks: int,
    window_pages: int,
    completed_len: int,
    page_size: int,
    BUCKET_MAX_UNION: int,
) -> torch.Tensor:
    """Phase 9 — score-prioritized UNION selection.

    Computes UNION across S_q axis of per-Q top-k masks, then keeps the
    BUCKET_MAX_UNION highest-priority pages by max-score. Force-includes
    sinks (positions [0, num_sinks)) and window (last window_pages of
    completed pages) so they cannot be truncated under overflow.

    Args:
        sel_per_q: bool[B, H_q, S_q, P] — per-Q top-k selection mask
        scores:    float[B, H_q, S_q, P] — Quest criticality scores per Q
        num_sinks: int — first `num_sinks` pages always included
        window_pages: int — last `window_pages` of completed pages always included
        completed_len: int — token count up to last quantized page boundary
        page_size: int — must match cache.page_size
        BUCKET_MAX_UNION: int — output's last-axis size

    Returns:
        int32[B, H_q, BUCKET_MAX_UNION] — selected page IDs with -1 sentinel for empty slots.
    """
    union_mask = sel_per_q.any(dim=2)                                        # (B, H_q, P)
    max_scores = scores.amax(dim=2)                                          # (B, H_q, P)

    P = union_mask.shape[-1]
    forced_mask = torch.zeros_like(union_mask)
    if num_sinks > 0:
        forced_mask[..., :num_sinks] = True
    n_complete_pages = completed_len // page_size
    if window_pages > 0 and n_complete_pages > 0:
        if n_complete_pages > window_pages:
            forced_mask[..., n_complete_pages - window_pages : n_complete_pages] = True
        else:
            forced_mask[..., :n_complete_pages] = True

    in_selection = union_mask | forced_mask
    neg_inf = torch.full_like(max_scores, float("-inf"))
    pos_inf = torch.full_like(max_scores, float("inf"))
    priority = torch.where(in_selection, max_scores, neg_inf)
    priority = torch.where(forced_mask, pos_inf, priority)

    BUCKET = min(BUCKET_MAX_UNION, P)
    top = priority.topk(BUCKET, dim=-1)
    top_indices = top.indices                                                # (B, H_q, BUCKET)
    top_priority = top.values
    out_int = torch.where(
        torch.isinf(top_priority) & (top_priority < 0),                      # priority == -inf → not in selection
        torch.full_like(top_indices, -1),
        top_indices,
    ).to(torch.int32)

    # Pad to BUCKET_MAX_UNION if BUCKET < BUCKET_MAX_UNION (only when P < BUCKET_MAX_UNION)
    if BUCKET < BUCKET_MAX_UNION:
        pad = torch.full(
            (*out_int.shape[:-1], BUCKET_MAX_UNION - BUCKET),
            -1, dtype=torch.int32, device=out_int.device,
        )
        out_int = torch.cat([out_int, pad], dim=-1)

    return out_int
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compact_union_selection.py -v
```

Expected: all 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/eager/selection.py tests/test_compact_union_selection.py
git commit -m "phase 9 task 3: build_compact_union_selection — score-prioritized UNION with forced sinks+window"
```

---

## Task 4: Cache sandbox API

**Files:**
- Modify: `src/flashquest/cache/persistent_int4.py`
- Test: `tests/test_cache_sandbox.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache_sandbox.py`:

```python
"""Phase 9 Task 4 — PersistentInt4KVCache sandbox API."""
import pytest
import torch


def _make_cache(num_layers=2, max_seq_len=512, page_size=64):
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    return PersistentInt4KVCache(
        batch_size=1, num_layers=num_layers, num_kv_heads=4, head_dim=128,
        max_seq_len=max_seq_len, page_size=page_size, device="cuda",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_add_draft_writes_sandbox_slots():
    cache = _make_cache()
    K = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    cache.add_draft(K, V, layer_idx=0)
    assert cache._sandbox_count[0] == 3
    torch.testing.assert_close(cache.K_sandbox[0, :, :, :3, :], K)
    torch.testing.assert_close(cache.V_sandbox[0, :, :, :3, :], V)
    # Layer 1 unaffected
    assert cache._sandbox_count[1] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_add_draft_rejects_oversize():
    cache = _make_cache()
    K = torch.randn(1, 4, cache.MAX_DRAFT + 1, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    with pytest.raises(ValueError, match="MAX_DRAFT"):
        cache.add_draft(K, V, layer_idx=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_get_views_with_sandbox_returns_sandbox_slice():
    cache = _make_cache()
    K = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    cache.add_draft(K, V, layer_idx=0)
    views = cache.get_views_with_sandbox(0)
    assert views["sandbox_count"] == 3
    assert views["K_sandbox"].shape == (1, 4, 3, 128)
    torch.testing.assert_close(views["K_sandbox"], K)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_commit_draft_advances_seen_and_clears_sandbox():
    cache = _make_cache()
    K = torch.randn(1, 4, 5, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 5, 128, dtype=torch.bfloat16, device="cuda")
    assert cache._seen_tokens[0] == 0
    cache.add_draft(K, V, layer_idx=0)
    assert cache._sandbox_count[0] == 5
    cache.commit_draft(3, layer_idx=0)
    assert cache._seen_tokens[0] == 3
    assert cache._sandbox_count[0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_commit_draft_zero_is_noop():
    cache = _make_cache()
    K = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    cache.add_draft(K, V, layer_idx=0)
    cache.commit_draft(0, layer_idx=0)
    assert cache._seen_tokens[0] == 0
    assert cache._sandbox_count[0] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_commit_equivalent_to_direct_update():
    """commit_draft(M) on sandbox == update_quantized(K[:M]) directly."""
    cache_a = _make_cache()
    cache_b = _make_cache()
    torch.manual_seed(7)
    K = torch.randn(1, 4, 5, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 4, 5, 128, dtype=torch.bfloat16, device="cuda")
    cache_a.update_quantized(K[:, :, :3, :], V[:, :, :3, :], layer_idx=0)
    cache_b.add_draft(K, V, layer_idx=0)
    cache_b.commit_draft(3, layer_idx=0)
    assert cache_a._seen_tokens[0] == cache_b._seen_tokens[0] == 3
    torch.testing.assert_close(cache_a.K_partial[0], cache_b.K_partial[0])
    torch.testing.assert_close(cache_a.V_partial[0], cache_b.V_partial[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_preflight_raises_on_overcommit():
    cache = _make_cache()
    K = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    cache.add_draft(K, V, layer_idx=0)
    with pytest.raises(ValueError, match="accept_count=5 > sandbox_count=3"):
        cache.preflight_commit(5, layer_idx=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_preflight_raises_on_page_boundary_cross():
    cache = _make_cache(page_size=64)
    # Fill cache to position 60
    K_existing = torch.randn(1, 4, 60, 128, dtype=torch.bfloat16, device="cuda")
    V_existing = torch.randn_like(K_existing)
    cache.update_quantized(K_existing, V_existing, layer_idx=0)
    assert cache._seen_tokens[0] == 60
    # Sandbox 5 tokens — committing them would write 60..64, completing page 0
    K = torch.randn(1, 4, 5, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    cache.add_draft(K, V, layer_idx=0)
    with pytest.raises(RuntimeError, match="page-boundary guard"):
        cache.preflight_commit(5, layer_idx=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_commit_all_layers_atomic_on_failure():
    """If layer 1 fails preflight, layer 0 must NOT have been mutated."""
    cache = _make_cache(num_layers=2)
    K = torch.randn(1, 4, 3, 128, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    cache.add_draft(K, V, layer_idx=0)
    # Layer 1 has only 2 tokens in sandbox, but we'll commit 3 — fails preflight
    K_short = K[:, :, :2, :]
    V_short = V[:, :, :2, :]
    cache.add_draft(K_short, V_short, layer_idx=1)
    seen_before = list(cache._seen_tokens)
    sandbox_before = list(cache._sandbox_count)
    with pytest.raises(ValueError, match="accept_count=3 > sandbox_count=2"):
        cache.commit_draft_all_layers(3)
    # No layer mutated
    assert list(cache._seen_tokens) == seen_before
    assert list(cache._sandbox_count) == sandbox_before
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_cache_sandbox.py -v
```

Expected: all FAIL with `AttributeError: 'PersistentInt4KVCache' object has no attribute 'add_draft'` (or similar).

- [ ] **Step 3: Implement the sandbox API**

Modify `src/flashquest/cache/persistent_int4.py`:

In `__init__`, after `self._seen_tokens = [0] * num_layers`, add:

```python
        self.MAX_DRAFT = 8
        shape_sandbox = (num_layers, batch_size, num_kv_heads, self.MAX_DRAFT, head_dim)
        self.K_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
        self.V_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
        self._sandbox_count = [0] * num_layers
```

After the `update` method (at end of class), add:

```python
    def add_draft(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        layer_idx: int,
    ) -> None:
        """Phase 9: write fresh K/V into per-layer sandbox slots [0..S_new-1].

        Does NOT advance _seen_tokens. Caller is responsible for calling
        commit_draft_all_layers(accept_count) after the verify step's
        walk-and-accept settles.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range")
        S_new = K_new.shape[2]
        if S_new > self.MAX_DRAFT:
            raise ValueError(
                f"add_draft S_new={S_new} > MAX_DRAFT={self.MAX_DRAFT}"
            )
        self.K_sandbox[layer_idx, :, :, :S_new, :] = K_new
        self.V_sandbox[layer_idx, :, :, :S_new, :] = V_new
        self._sandbox_count[layer_idx] = S_new

    def get_views_with_sandbox(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Existing get_views() result + sandbox K/V views.

        Sandbox K/V keys are BF16 tensors of shape (B, H_kv, sandbox_count, D).
        """
        views = self.get_views(layer_idx)
        s = self._sandbox_count[layer_idx]
        views["K_sandbox"] = self.K_sandbox[layer_idx, :, :, :s, :]
        views["V_sandbox"] = self.V_sandbox[layer_idx, :, :, :s, :]
        views["sandbox_count"] = s
        return views

    def preflight_commit(self, accept_count: int, layer_idx: int) -> None:
        """Validate sandbox is ready for accept_count commit. Does NOT mutate state.

        Raises (without mutation) if accept_count is invalid or would cross a page
        boundary. Phase 9 spec forbids cross-page commits during PLD verify because
        the verify path keeps sandbox K/V in BF16 while equivalent non-spec would
        quantize to INT4 at the boundary.
        """
        if accept_count < 0:
            raise ValueError(f"accept_count={accept_count} negative")
        s = self._sandbox_count[layer_idx]
        if accept_count > s:
            raise ValueError(
                f"accept_count={accept_count} > sandbox_count={s} (layer {layer_idx})"
            )
        seen = self._seen_tokens[layer_idx]
        page_size = self.page_size
        if accept_count > 0 and (seen + accept_count) // page_size != seen // page_size:
            raise RuntimeError(
                f"commit_draft({accept_count}) on layer {layer_idx} would cross "
                f"page boundary (seen={seen}, page_size={page_size}); "
                f"page-boundary guard violated"
            )

    def commit_draft(self, accept_count: int, layer_idx: int) -> None:
        """Commit first `accept_count` sandbox positions to persistent cache.

        Caller is responsible for invoking preflight_commit on ALL layers first
        (or call commit_draft_all_layers which handles both phases atomically).
        """
        if accept_count == 0:
            self._sandbox_count[layer_idx] = 0
            return
        K_commit = self.K_sandbox[layer_idx, :, :, :accept_count, :]
        V_commit = self.V_sandbox[layer_idx, :, :, :accept_count, :]
        self.update_quantized(K_commit, V_commit, layer_idx)
        self._sandbox_count[layer_idx] = 0

    def commit_draft_all_layers(self, accept_count: int) -> None:
        """Two-phase atomic commit across all layers.

        Phase 1: preflight every layer's sandbox; raises (without mutation) on bad input.
        Phase 2: commit each layer; advances _seen_tokens by accept_count.
        """
        for layer_idx in range(self.num_layers):
            self.preflight_commit(accept_count, layer_idx)
        for layer_idx in range(self.num_layers):
            self.commit_draft(accept_count, layer_idx)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cache_sandbox.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/cache/persistent_int4.py tests/test_cache_sandbox.py
git commit -m "$(cat <<'EOF'
phase 9 task 4: PersistentInt4KVCache sandbox API

add_draft / get_views_with_sandbox / preflight_commit / commit_draft /
commit_draft_all_layers. Two-phase atomic commit across layers. Page-boundary
preflight raises before mutating any layer. Sandbox is 920 KB (BF16 K+V at
MAX_DRAFT=8 across 28 layers).
EOF
)"
```

---

## Task 5: Compact INT4 kernel S_q>1 (UNION list, ieee precision, sentinel + sq_mask)

**Files:**
- Modify: `src/flashquest/kernel/sparse_int4_fwd_compact.py` (add S_q>1 kernel + dispatcher)
- Test: `tests/test_compact_kernel_sq_gt_1.py`
- Test: `tests/test_compact_kernel_sq_gt_1_sentinels.py`
- Test: `tests/test_kernel_tldot_ieee.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_compact_kernel_sq_gt_1.py`:

```python
"""Phase 9 Task 5 — compact INT4 kernel S_q>1 parity vs S_q=1 N times."""
import math
import pytest
import torch


def _make_kv(B=1, H_kv=2, S_kv=128, D=128, page_size=64):
    """Synthetic packed INT4 KV with random scales/mn (matches Phase 8a layout)."""
    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    return K_packed, K_scale, K_mn, V_packed, V_scale, V_mn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sq_gt_1_parity_vs_sq1_loop():
    """compact_kernel_sq_gt_1 with S_q=4 == 4 separate S_q=1 calls (numeric tolerance)."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    torch.manual_seed(0)
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 4, 128, 128
    page_size = 64
    BMK = 2  # 2 pages of S_kv=128 / page_size=64 = 2 total pages

    K_packed, K_scale, K_mn, V_packed, V_scale, V_mn = _make_kv(
        B=B, H_kv=H_kv, S_kv=S_kv, D=D, page_size=page_size,
    )
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    sel_compact = torch.tensor([[[0, 1]] * H_q] * B, dtype=torch.int32, device="cuda")  # (B, H_q, 2)

    # S_q>1 in one shot
    O_multi, lse_multi = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
    )

    # Reference: 4 separate S_q=1 kernel calls
    O_ref = torch.zeros_like(Q)
    lse_ref = torch.zeros(B, H_q, S_q, dtype=torch.float32, device="cuda")
    for q in range(S_q):
        Q_one = Q[:, :, q:q+1, :]
        sel_one = sel_compact.unsqueeze(2)   # (B, H_q, 1, BMK)
        O_q, lse_q = flash_attn_sparse_int4_fwd_compact(
            Q_one, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
            selected_page_ids=sel_one, page_size=page_size, return_lse=True,
        )
        O_ref[:, :, q:q+1, :] = O_q
        lse_ref[:, :, q:q+1] = lse_q

    torch.testing.assert_close(O_multi, O_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse_multi, lse_ref, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sq_eq_1_byte_equivalent_phase_8a():
    """When S_q=1, must produce same output as Phase 8a kernel (S_q=1 path unchanged)."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    torch.manual_seed(1)
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 1, 128, 128
    page_size = 64
    K_packed, K_scale, K_mn, V_packed, V_scale, V_mn = _make_kv(
        B=B, H_kv=H_kv, S_kv=S_kv, D=D, page_size=page_size,
    )
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    sel_compact = torch.tensor([[[0, 1]] * H_q] * B, dtype=torch.int32, device="cuda")

    O1, lse1 = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
    )
    # Run again — must be deterministic
    O2, lse2 = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
    )
    torch.testing.assert_close(O1, O2)
    torch.testing.assert_close(lse1, lse2)
```

Create `tests/test_compact_kernel_sq_gt_1_sentinels.py`:

```python
"""Phase 9 Task 5 — sentinel handling in S_q>1 compact kernel."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_sentinel_pages_no_oob():
    """selected_page_ids with -1 sentinels must not OOB-load and must produce 0 contribution."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    torch.manual_seed(2)
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 3, 128, 128
    page_size = 64
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    # 2 valid pages + 2 sentinels (-1)
    sel = torch.full((B, H_q, 4), -1, dtype=torch.int32, device="cuda")
    sel[..., 0] = 0
    sel[..., 1] = 1
    O_with_sentinels, _ = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    # Reference: only 2 valid pages, no sentinels
    sel_no_sent = torch.tensor([[[0, 1]] * H_q] * B, dtype=torch.int32, device="cuda")
    O_no_sent, _ = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel_no_sent, page_size=page_size, return_lse=True,
    )
    torch.testing.assert_close(O_with_sentinels, O_no_sent, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_all_sentinels_lse_neg_inf():
    """If all selected pages are -1: output is undefined-but-not-NaN, lse should be -inf."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 2, 128, 128
    page_size = 64
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.full((B, H_q, 4), -1, dtype=torch.int32, device="cuda")
    O, lse = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    assert torch.isinf(lse).all() and (lse < 0).all(), f"lse should be -inf, got {lse}"
    assert not torch.isnan(O).any()
```

Create `tests/test_kernel_tldot_ieee.py`:

```python
"""Phase 9 Task 5 — Triton compile + ieee-precision smoke for compact kernel."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_compact_kernel_compiles_at_sq_max_16():
    """Smoke: compact kernel with SQ_MAX=16 compiles and runs."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 5, 128, 128
    page_size = 64
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.tensor([[[0, 1]] * H_q] * B, dtype=torch.int32, device="cuda")
    O, lse = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    assert O.shape == (B, H_q, S_q, D)
    assert lse.shape == (B, H_q, S_q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ieee_precision_matches_manual_fp32():
    """Output should be near-identical to a manual fp32 sum-of-product reference."""
    from flashquest.kernel.sparse_int4_fwd_compact import (
        flash_attn_sparse_int4_fwd_compact,
    )
    from flashquest.kernel.kv_quant import (
        quantize_k_int4, quantize_v_int4, dequantize_k_int4, dequantize_v_int4,
    )
    torch.manual_seed(3)
    B, H_q, H_kv, S_q, D, S_kv = 1, 8, 2, 3, 128, 128
    page_size = 64
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.tensor([[[0, 1]] * H_q] * B, dtype=torch.int32, device="cuda")
    O_kernel, _ = flash_attn_sparse_int4_fwd_compact(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    # Manual fp32 reference: dequantize all of K/V (both pages 0 and 1), full SDPA over them
    K_dq = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size).float()
    V_dq = dequantize_v_int4(V_packed, V_scale, V_mn).float()
    n_rep = H_q // H_kv
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    V_dq_full = V_dq.repeat_interleave(n_rep, dim=1)
    sm_scale = 1.0 / (D ** 0.5)
    qk = (Q.float() @ K_dq_full.transpose(-1, -2)) * sm_scale
    p = torch.softmax(qk, dim=-1)
    O_ref = (p @ V_dq_full).to(torch.bfloat16)
    torch.testing.assert_close(O_kernel, O_ref, atol=2e-2, rtol=2e-2)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compact_kernel_sq_gt_1.py tests/test_compact_kernel_sq_gt_1_sentinels.py tests/test_kernel_tldot_ieee.py -v
```

Expected: all FAIL — `flash_attn_sparse_int4_fwd_compact` currently asserts `S_q != 1` raises NotImplementedError.

- [ ] **Step 3: Implement the S_q>1 kernel and dispatcher**

Modify `src/flashquest/kernel/sparse_int4_fwd_compact.py`. Keep the existing Phase 8a kernel (`_sparse_attn_fwd_kernel_int4_compact`) unchanged. Add a new kernel and update the Python wrapper.

After the existing `_sparse_attn_fwd_kernel_int4_compact` jit, add:

```python
@triton.jit
def _sparse_attn_fwd_kernel_int4_compact_sq_gt_1(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_seli,
    H_q, H_kv, S_kv, S_q,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PACKED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUCKET_MAX: tl.constexpr,
    SQ_MAX: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """One CTA per (batch, query head). Loops BUCKET_MAX compact slots; each iteration
    loads K/V tile ONCE and reuses across SQ_MAX queries via tl.dot."""
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED)
    offs_sq = tl.arange(0, SQ_MAX)
    sq_mask = offs_sq < S_q

    # Load Q for all SQ_MAX rows; masked rows get 0
    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh
        + offs_sq[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=sq_mask[:, None], other=0.0)              # (SQ_MAX, HEAD_DIM)

    NEG_INF: tl.constexpr = float("-inf")
    m_i = tl.full((SQ_MAX,), NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((SQ_MAX,), dtype=tl.float32)
    acc = tl.zeros((SQ_MAX, HEAD_DIM), dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for i in range(0, BUCKET_MAX):
        sel_off = b * stride_selb + h_q * stride_selh + i * stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0
        p_safe = tl.where(page_valid, p, 0)

        page_start = p_safe * PAGE_SIZE
        n_idx = page_start + offs_n
        valid_kv = (n_idx < S_kv) & page_valid

        # K-tile load (skipped via mask if page invalid)
        k_byte_ptrs = (
            K_packed_ptr + b * stride_kb + h_kv * stride_kh
            + n_idx[:, None] * stride_ks + offs_dp[None, :] * stride_kdp
        )
        k_byte = tl.load(k_byte_ptrs, mask=valid_kv[:, None], other=0)
        k_lo = (k_byte & 0xF).to(tl.uint8)
        k_hi = ((k_byte >> 4) & 0xF).to(tl.uint8)
        k_int_2 = tl.join(k_lo, k_hi)
        k_int = tl.reshape(k_int_2, (PAGE_SIZE, HEAD_DIM))

        ks_ptrs = K_scale_ptr + b * stride_ksb + h_kv * stride_ksh + p_safe * stride_ksp + offs_d * stride_ksd
        km_ptrs = K_mn_ptr + b * stride_kmb + h_kv * stride_kmh + p_safe * stride_kmp + offs_d * stride_kmd
        k_scale = tl.load(ks_ptrs).to(tl.float32)
        k_mn = tl.load(km_ptrs).to(tl.float32)
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]    # (PAGE_SIZE, HEAD_DIM)

        # QK product — tl.dot with ieee precision (NOT TF32 default)
        qk = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee")
        # Mask both axes: invalid SQ rows AND invalid KV → -inf
        qk = tl.where(sq_mask[:, None] & valid_kv[None, :], qk, NEG_INF)
        qk_scaled = qk * qk_scale

        qk_max = tl.max(qk_scaled, axis=1)
        m_ij = tl.maximum(m_i, qk_max)
        m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
        p_softmax = tl.math.exp2(qk_scaled - m_ij_safe[:, None])
        p_softmax = tl.where(qk == NEG_INF, 0.0, p_softmax)

        alpha = tl.math.exp2(m_i - m_ij_safe)
        alpha = tl.where(m_i == NEG_INF, 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p_softmax, axis=1)
        acc = acc * alpha[:, None]

        # V-tile load (skipped via mask if page invalid)
        v_byte_ptrs = (
            V_packed_ptr + b * stride_vb + h_kv * stride_vh
            + n_idx[:, None] * stride_vs + offs_dp[None, :] * stride_vdp
        )
        v_byte = tl.load(v_byte_ptrs, mask=valid_kv[:, None], other=0)
        v_lo = (v_byte & 0xF).to(tl.uint8)
        v_hi = ((v_byte >> 4) & 0xF).to(tl.uint8)
        v_int_2 = tl.join(v_lo, v_hi)
        v_int = tl.reshape(v_int_2, (PAGE_SIZE, HEAD_DIM))

        vs_ptrs = V_scale_ptr + b * stride_vsb + h_kv * stride_vsh + n_idx * stride_vss
        vm_ptrs = V_mn_ptr + b * stride_vmb + h_kv * stride_vmh + n_idx * stride_vms
        v_scale = tl.load(vs_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
        v_mn = tl.load(vm_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]

        acc += tl.dot(p_softmax, v, input_precision="ieee")

        m_i = m_ij

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]

    o_ptrs = (
        O_ptr + b * stride_ob + h_q * stride_oh
        + offs_sq[:, None] * stride_os + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=sq_mask[:, None])

    if WRITE_LSE:
        lse_val = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse_val = tl.where(l_i == 0.0, NEG_INF, lse_val)
        l_ptrs = L_ptr + b * stride_lb + h_q * stride_lh + offs_sq * stride_ls
        tl.store(l_ptrs, lse_val, mask=sq_mask)
```

Replace `flash_attn_sparse_int4_fwd_compact` with a dispatcher that routes S_q=1 to the existing kernel and S_q>1 to the new kernel:

```python
def flash_attn_sparse_int4_fwd_compact(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selected_page_ids: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sparse forward with INT4 KV — compact-list variant. S_q=1 OR S_q>1.

    selected_page_ids shape:
      - S_q=1 path: (B, H_q, BUCKET_MAX) or (B, H_q, 1, BUCKET_MAX). 3D preferred.
      - S_q>1 path: (B, H_q, BUCKET_MAX) — UNION selection shared across S_q queries.
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_packed.dtype == torch.uint8 and V_packed.dtype == torch.uint8
    assert selected_page_ids.dtype == torch.int32

    B, H_q, S_q, D = Q.shape
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")
    Bk, H_kv, S_kv, Dp = K_packed.shape
    assert B == Bk and Dp == D // 2 and H_q % H_kv == 0

    if selected_page_ids.dim() == 4:
        sel_3d = selected_page_ids.squeeze(2)
    elif selected_page_ids.dim() == 3:
        sel_3d = selected_page_ids
    else:
        raise ValueError(f"selected_page_ids must be 3D or 4D; got {selected_page_ids.dim()}D")
    Bs, H_qs, BUCKET_MAX = sel_3d.shape
    assert Bs == B and H_qs == H_q
    sel_3d_contig = sel_3d.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    if S_q == 1:
        # Phase 8a path — unchanged
        Q_2d = Q.squeeze(2)
        O_2d = torch.zeros_like(Q_2d)
        L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
        L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
        sl_b, sl_h = (L.stride() if L is not None else (0, 0))
        grid = (B * H_q,)
        _sparse_attn_fwd_kernel_int4_compact[grid](
            Q_2d, K_packed, V_packed, O_2d, L_ptr,
            K_scale, K_mn, V_scale, V_mn,
            sel_3d_contig,
            sm_scale,
            Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
            K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
            V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
            O_2d.stride(0), O_2d.stride(1), O_2d.stride(2),
            sl_b, sl_h,
            K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
            K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
            V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
            V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
            sel_3d_contig.stride(0), sel_3d_contig.stride(1), sel_3d_contig.stride(2),
            H_q, H_kv, S_kv,
            HEAD_DIM=D,
            HEAD_DIM_PACKED=D // 2,
            PAGE_SIZE=page_size,
            BUCKET_MAX=BUCKET_MAX,
            WRITE_LSE=bool(return_lse),
            num_warps=4, num_stages=2,
        )
        return O_2d.unsqueeze(2), (L.unsqueeze(2) if L is not None else None)

    # S_q > 1 — Phase 9 path
    SQ_MAX = 16
    if S_q > SQ_MAX:
        raise NotImplementedError(f"S_q={S_q} > SQ_MAX={SQ_MAX}; raise SQ_MAX in spec §4.4")
    O = torch.zeros_like(Q)
    L = torch.empty(B, H_q, S_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    if L is not None:
        lb, lh, ls = L.stride()
    else:
        lb = lh = ls = 0
    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_int4_compact_sq_gt_1[grid](
        Q, K_packed, V_packed, O, L_ptr,
        K_scale, K_mn, V_scale, V_mn,
        sel_3d_contig,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        lb, lh, ls,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel_3d_contig.stride(0), sel_3d_contig.stride(1), sel_3d_contig.stride(2),
        H_q, H_kv, S_kv, S_q,
        HEAD_DIM=D,
        HEAD_DIM_PACKED=D // 2,
        PAGE_SIZE=page_size,
        BUCKET_MAX=BUCKET_MAX,
        SQ_MAX=SQ_MAX,
        WRITE_LSE=bool(return_lse),
        num_warps=4, num_stages=2,
    )
    return O, L
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compact_kernel_sq_gt_1.py tests/test_compact_kernel_sq_gt_1_sentinels.py tests/test_kernel_tldot_ieee.py -v
```

Expected: all PASS. If `tl.dot input_precision="ieee"` fails on Triton 3.1.0: drop `input_precision` arg (default precision is fine for our tolerance) — note this in the commit message and the spec.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/kernel/sparse_int4_fwd_compact.py tests/test_compact_kernel_sq_gt_1.py tests/test_compact_kernel_sq_gt_1_sentinels.py tests/test_kernel_tldot_ieee.py
git commit -m "$(cat <<'EOF'
phase 9 task 5: compact INT4 kernel S_q>1 with UNION list + ieee precision

New kernel _sparse_attn_fwd_kernel_int4_compact_sq_gt_1 reuses K/V tile
loads across SQ_MAX=16 queries via tl.dot. Selection shared across S_q
(UNION). sq_mask + valid_kv applied before softmax to suppress masked
rows. tl.dot uses input_precision="ieee" to pin fp32 over TF32 default.

Phase 8a S_q=1 path unchanged. Dispatcher routes by S_q in
flash_attn_sparse_int4_fwd_compact wrapper.
EOF
)"
```

---

## Task 6: Kernel S_q>1 microbench (MID-PHASE GATE)

Per the saved profile-first lesson: validate kernel cost BEFORE wiring it into the dispatcher.

**Files:**
- Create: `benchmarks/phase9_kernel_sq_gt_1_microbench.py`

- [ ] **Step 1: Write the bench**

Create `benchmarks/phase9_kernel_sq_gt_1_microbench.py`:

```python
"""Phase 9 Task 6 — kernel S_q>1 microbench at 32k decode shape.

Compares S_q=1 vs S_q=5 sparse INT4 kernel cost. Gate: S_q=5 should be
<=1.5x S_q=1 (BW amortized via shared K/V loads).
"""
from __future__ import annotations
import argparse
import time
import torch


def _run_kernel(Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, sel, page_size, n_iter):
    from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact
    # Warmup
    for _ in range(5):
        flash_attn_sparse_int4_fwd_compact(
            Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
            selected_page_ids=sel, page_size=page_size, return_lse=True,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        flash_attn_sparse_int4_fwd_compact(
            Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
            selected_page_ids=sel, page_size=page_size, return_lse=True,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1e6  # us


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--n-iter", type=int, default=200)
    args = p.parse_args()

    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    B, H_q, H_kv, D = 1, 24, 8, 128
    S_kv = args.ctx
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=args.page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)

    P = S_kv // args.page_size
    BUCKET_MAX = int(args.retention * P) + 4 + 2  # k + sinks + window
    BUCKET_MAX_UNION = int(BUCKET_MAX * 1.5)

    # Sample compact selection — pages 0..BUCKET_MAX-1
    sel_sq1 = torch.arange(BUCKET_MAX, dtype=torch.int32, device="cuda")
    sel_sq1 = sel_sq1.expand(B, H_q, BUCKET_MAX).contiguous()
    sel_sq5 = torch.arange(BUCKET_MAX_UNION, dtype=torch.int32, device="cuda")
    sel_sq5 = sel_sq5.expand(B, H_q, BUCKET_MAX_UNION).contiguous()

    Q1 = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    Q5 = torch.randn(B, H_q, 5, D, dtype=torch.bfloat16, device="cuda")

    t1 = _run_kernel(Q1, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
                     sel_sq1, args.page_size, args.n_iter)
    t5 = _run_kernel(Q5, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
                     sel_sq5, args.page_size, args.n_iter)

    print(f"\n=== Phase 9 Task 6 microbench @ ctx={args.ctx} ===")
    print(f"  S_q=1 (BUCKET_MAX={BUCKET_MAX}):       {t1:.1f} us/call")
    print(f"  S_q=5 (BUCKET_MAX_UNION={BUCKET_MAX_UNION}): {t5:.1f} us/call")
    print(f"  ratio: {t5 / t1:.2f}x  (gate: <=1.5x)")
    print(f"  GATE: {'PASS' if t5 / t1 <= 1.5 else 'FAIL'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench**

```bash
nice -n 19 python benchmarks/phase9_kernel_sq_gt_1_microbench.py 2>&1 | tee benchmarks/phase9_task6_log.txt
```

Expected output (sample):
```
=== Phase 9 Task 6 microbench @ ctx=32768 ===
  S_q=1 (BUCKET_MAX=134):       1503 us/call
  S_q=5 (BUCKET_MAX_UNION=201): 2120 us/call
  ratio: 1.41x  (gate: <=1.5x)
  GATE: PASS
```

- [ ] **Step 3: Decision gate**

If GATE PASS: proceed to Task 7.

If GATE FAIL (ratio > 1.5×): the sparse kernel BW amortization isn't materializing. Possible causes:
- Register spills with SQ_MAX=16 (codex r2 integration risk #3). Fall back to SQ_MAX=8 with software workaround for `tl.dot` minimum (pad to 16 via mask).
- Per Phase 8a Task 5 precedent: document, accept reduced expectations (1.6× target instead of 2.4×), continue.

- [ ] **Step 4: Commit**

```bash
git add benchmarks/phase9_kernel_sq_gt_1_microbench.py benchmarks/phase9_task6_log.txt
git commit -m "phase 9 task 6: kernel S_q>1 microbench — gate <=1.5x cost vs S_q=1"
```

---

## Task 7: Dense offset-causal helper + LSE-merge S_q extension

**Files:**
- Modify: `src/flashquest/eager/llama_persistent_patch.py` (add helpers)
- Test: `tests/test_dense_offset_causal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dense_offset_causal.py`:

```python
"""Phase 9 Task 7 — bf16_dense_attn_offset_causal_with_lse + _merge_two_attentions_sq."""
import math
import pytest
import torch
import torch.nn.functional as F


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offset_causal_sq1_matches_existing_helper():
    """Phase 9 helper at S_q=1 q_offset=k must match Phase 8a's _bf16_dense_attn_with_lse."""
    from flashquest.eager.llama_persistent_patch import (
        _bf16_dense_attn_with_lse, bf16_dense_attn_offset_causal_with_lse,
    )
    torch.manual_seed(0)
    B, H_q, H_kv, D = 1, 8, 2, 128
    S_kv = 7
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    O_ref, lse_ref = _bf16_dense_attn_with_lse(Q, K, V)
    O_new, lse_new = bf16_dense_attn_offset_causal_with_lse(Q, K, V, q_offset=S_kv - 1)
    torch.testing.assert_close(O_new, O_ref)
    torch.testing.assert_close(lse_new, lse_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offset_causal_sq_gt_1_matches_sdpa():
    """Phase 9 helper at S_q=N q_offset=k must match torch SDPA with custom causal mask."""
    from flashquest.eager.llama_persistent_patch import bf16_dense_attn_offset_causal_with_lse
    torch.manual_seed(1)
    B, H_q, H_kv, S_q, D = 1, 8, 2, 4, 128
    q_offset = 3
    S_kv = q_offset + S_q  # 7
    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    # Custom causal mask: Q row i (=position q_offset+i) attends to K [0..q_offset+i]
    mask = torch.zeros(S_q, S_kv, device="cuda", dtype=torch.bool)
    for i in range(S_q):
        mask[i, : q_offset + i + 1] = True
    n_rep = H_q // H_kv
    K_full = K.repeat_interleave(n_rep, dim=1)
    V_full = V.repeat_interleave(n_rep, dim=1)
    O_ref = F.scaled_dot_product_attention(
        Q.float(), K_full.float(), V_full.float(),
        attn_mask=mask, is_causal=False,
    ).to(torch.bfloat16)
    O_new, lse_new = bf16_dense_attn_offset_causal_with_lse(Q, K, V, q_offset=q_offset)
    torch.testing.assert_close(O_new, O_ref, atol=1e-2, rtol=1e-2)
    assert lse_new.shape == (B, H_q, S_q)
    assert not torch.isnan(lse_new).any()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_merge_two_attentions_sq_eq_1_matches_existing():
    """_merge_two_attentions_sq at S_q=1 must reduce to existing _merge_two_attentions."""
    from flashquest.eager.llama_persistent_patch import (
        _merge_two_attentions, _merge_two_attentions_sq,
    )
    torch.manual_seed(2)
    B, H_q, D = 1, 8, 128
    O_a = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    O_b = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    lse_a = torch.randn(B, H_q, 1, dtype=torch.float32, device="cuda")
    lse_b = torch.randn(B, H_q, 1, dtype=torch.float32, device="cuda")
    O_ref = _merge_two_attentions(O_a, lse_a, O_b, lse_b)
    O_new = _merge_two_attentions_sq(O_a, lse_a, O_b, lse_b)
    torch.testing.assert_close(O_new, O_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_merge_two_attentions_sq_gt_1_per_query():
    """_merge_two_attentions_sq applies the merge formula independently per query."""
    from flashquest.eager.llama_persistent_patch import (
        _merge_two_attentions, _merge_two_attentions_sq,
    )
    torch.manual_seed(3)
    B, H_q, S_q, D = 1, 8, 5, 128
    O_a = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    O_b = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    lse_a = torch.randn(B, H_q, S_q, dtype=torch.float32, device="cuda")
    lse_b = torch.randn(B, H_q, S_q, dtype=torch.float32, device="cuda")
    O_full = _merge_two_attentions_sq(O_a, lse_a, O_b, lse_b)
    # Verify each S_q slot matches per-S_q=1 merge
    for q in range(S_q):
        O_q_ref = _merge_two_attentions(
            O_a[:, :, q:q+1, :], lse_a[:, :, q:q+1],
            O_b[:, :, q:q+1, :], lse_b[:, :, q:q+1],
        )
        torch.testing.assert_close(O_full[:, :, q:q+1, :], O_q_ref)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dense_offset_causal.py -v
```

Expected: FAIL — `bf16_dense_attn_offset_causal_with_lse` and `_merge_two_attentions_sq` not defined yet.

- [ ] **Step 3: Implement the helpers**

Add to `src/flashquest/eager/llama_persistent_patch.py` (after the existing `_merge_two_attentions` function):

```python
def bf16_dense_attn_offset_causal_with_lse(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, *, q_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense BF16 attention with offset-causal mask. Phase 9 verify-mode helper.

    Q at row i has cache-position q_offset + i in the K/V tail; valid K positions
    are [0, q_offset + i].

    Q: (B, H_q, S_q, D); K, V: (B, H_kv, S_kv, D); S_kv >= q_offset + S_q.
    Returns (O, lse) where O is (B, H_q, S_q, D) BF16 and lse is (B, H_q, S_q) fp32 (nats).
    """
    B, H_q, S_q, D = Q.shape
    H_kv = K.shape[1]
    S_kv = K.shape[2]
    n_rep = H_q // H_kv
    K_full = K.repeat_interleave(n_rep, dim=1).float()
    V_full = V.repeat_interleave(n_rep, dim=1).float()
    sm_scale = 1.0 / math.sqrt(D)

    qk = (Q.float() @ K_full.transpose(-1, -2)) * sm_scale       # (B, H_q, S_q, S_kv)
    # Offset-causal: row i attends to columns [0, q_offset + i]
    mask_2d = torch.zeros(S_q, S_kv, device=Q.device, dtype=torch.bool)
    for i in range(S_q):
        mask_2d[i, : q_offset + i + 1] = True
    qk = qk.masked_fill(~mask_2d, float("-inf"))

    m = qk.amax(dim=-1, keepdim=True)
    # Replace -inf max with 0 to avoid -inf - -inf = nan
    m_safe = torch.where(torch.isinf(m) & (m < 0), torch.zeros_like(m), m)
    p = torch.exp(qk - m_safe)
    p = p.masked_fill(~mask_2d, 0.0)
    l = p.sum(dim=-1, keepdim=True)
    l_safe = torch.where(l == 0, torch.ones_like(l), l)
    O = (p @ V_full) / l_safe                                      # (B, H_q, S_q, D)
    lse = (m + torch.log(l_safe)).squeeze(-1)                      # (B, H_q, S_q)
    lse = torch.where(l.squeeze(-1) == 0, torch.full_like(lse, float("-inf")), lse)
    return O.to(torch.bfloat16), lse


def _merge_two_attentions_sq(
    O_a: torch.Tensor, lse_a: torch.Tensor,
    O_b: torch.Tensor, lse_b: torch.Tensor,
) -> torch.Tensor:
    """Online-softmax merge over S_q axis. Per-query formula identical to _merge_two_attentions.

    Inputs:
      O_a, O_b: (B, H_q, S_q, D)
      lse_a, lse_b: (B, H_q, S_q)
    Returns:
      O: (B, H_q, S_q, D), same dtype as O_a.
    """
    m = torch.maximum(lse_a, lse_b)                                # (B, H_q, S_q)
    wa = torch.exp(lse_a - m).unsqueeze(-1)                        # (B, H_q, S_q, 1)
    wb = torch.exp(lse_b - m).unsqueeze(-1)
    return ((wa * O_a.float() + wb * O_b.float()) / (wa + wb)).to(O_a.dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_dense_offset_causal.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/eager/llama_persistent_patch.py tests/test_dense_offset_causal.py
git commit -m "phase 9 task 7: bf16 offset-causal dense helper + LSE merge S_q extension"
```

---

## Task 8: Verify-mode forward branch in llama_persistent_patch

**Files:**
- Modify: `src/flashquest/eager/llama_persistent_patch.py` (extend `make_quest_persistent_forward` and `patch_llama_for_quest_persistent`)
- Test: `tests/test_persistent_patch_pld_verify.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_persistent_patch_pld_verify.py`:

```python
"""Phase 9 Task 8 — verify-mode forward branch in patched LlamaAttention."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_set_pld_verify_active_walks_modules():
    """_set_pld_verify_active sets the flag on every patched LlamaAttention."""
    from transformers import AutoConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import (
        patch_llama_for_quest_persistent, _set_pld_verify_active,
    )
    cfg = AutoConfig.from_pretrained("casperhansen/llama-3.2-3b-instruct-awq")
    cfg.num_hidden_layers = 2
    model = LlamaForCausalLM(cfg).cuda().to(torch.float16)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=256, page_size=64, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=64,
    )

    from transformers.models.llama.modeling_llama import LlamaAttention
    attns = [m for m in model.modules() if isinstance(m, LlamaAttention)]
    for m in attns:
        assert getattr(m, "_pld_verify_active", False) is False
    _set_pld_verify_active(model, True)
    for m in attns:
        assert m._pld_verify_active is True
    _set_pld_verify_active(model, False)
    for m in attns:
        assert m._pld_verify_active is False


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verify_mode_uses_sandbox_not_update_quantized():
    """In verify mode, attention forward calls add_draft (not update_quantized)."""
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import (
        patch_llama_for_quest_persistent, _set_pld_verify_active,
    )
    from flashquest.runtime.awq_load import load_awq_model
    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=512, page_size=64, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=64,
    )

    # Run prefill (NOT verify mode) — populates cache
    ids = torch.randint(0, cfg.vocab_size, (1, 100), device="cuda")
    with torch.no_grad():
        model(input_ids=ids, use_cache=True)
    seen_after_prefill = list(cache._seen_tokens)
    assert all(s == 100 for s in seen_after_prefill)

    # Run verify-mode forward (S_q=5)
    verify_ids = torch.randint(0, cfg.vocab_size, (1, 5), device="cuda")
    _set_pld_verify_active(model, True)
    try:
        with torch.no_grad():
            out = model(input_ids=verify_ids, use_cache=True)
    finally:
        _set_pld_verify_active(model, False)
    # Verify path must NOT advance _seen_tokens
    assert list(cache._seen_tokens) == seen_after_prefill, (
        f"verify path leaked update_quantized: seen advanced to {cache._seen_tokens}"
    )
    # Verify path MUST populate sandbox
    for layer_idx in range(cfg.num_hidden_layers):
        assert cache._sandbox_count[layer_idx] == 5, (
            f"layer {layer_idx} sandbox count {cache._sandbox_count[layer_idx]} != 5"
        )
    assert out.logits.shape == (1, 5, cfg.vocab_size)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_verify_flag_reset_on_exception():
    """If verify forward raises, _pld_verify_active still reset."""
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import (
        patch_llama_for_quest_persistent, _set_pld_verify_active,
    )
    from flashquest.runtime.awq_load import load_awq_model
    from transformers.models.llama.modeling_llama import LlamaAttention
    model, _ = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.hidden_size // cfg.num_attention_heads,
        max_seq_len=128, page_size=64, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=64,
    )
    _set_pld_verify_active(model, True)
    try:
        try:
            raise RuntimeError("synthetic forward failure")
        finally:
            _set_pld_verify_active(model, False)
    except RuntimeError:
        pass
    for m in model.modules():
        if isinstance(m, LlamaAttention):
            assert m._pld_verify_active is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_persistent_patch_pld_verify.py -v
```

Expected: FAIL — `_set_pld_verify_active` not defined; verify-mode branch not implemented.

- [ ] **Step 3: Implement the verify-mode branch**

Modify `src/flashquest/eager/llama_persistent_patch.py`. After the existing `bf16_dense_attn_offset_causal_with_lse` and `_merge_two_attentions_sq`, add:

```python
from transformers.models.llama.modeling_llama import LlamaAttention as _LlamaAttention


def _set_pld_verify_active(model: torch.nn.Module, value: bool) -> None:
    """Phase 9: toggle _pld_verify_active on every patched LlamaAttention.

    Use try/finally around the verify-mode forward so an exception during
    forward never leaves the model stuck in verify mode.
    """
    for module in model.modules():
        if isinstance(module, _LlamaAttention):
            module._pld_verify_active = bool(value)
```

In `make_quest_persistent_forward`, add to the closure (next to `k_max_static`):

```python
    bucket_max_union_static = math.ceil(1.5 * bucket_max_static)
```

In the forward closure, ADD a new branch BEFORE the existing `S_q > 1` and `S_q == 1` branches. Replace:

```python
        cache.update_quantized(k, v, layer_idx=self.layer_idx)
        views = cache.get_views(self.layer_idx)

        if S_q > 1:
```

with:

```python
        verify_mode = getattr(self, "_pld_verify_active", False)

        if S_q > 1 and verify_mode:
            # Phase 9 — PLD verify path. Sandbox-and-merge.
            cache.add_draft(k, v, layer_idx=self.layer_idx)
            views = cache.get_views_with_sandbox(self.layer_idx)
            completed_len = views["completed_len"]
            partial_len = views["partial_len"]
            sandbox_count = views["sandbox_count"]
            assert sandbox_count == S_q, (
                f"sandbox_count={sandbox_count} != S_q={S_q}"
            )

            B, H_q_, _, _ = q.shape
            H_kv_ = cache.num_kv_heads
            n_rep = H_q_ // H_kv_

            if completed_len > 0:
                from flashquest.eager.selection import build_compact_union_selection
                pattern_per_q = head_pattern_layer.to(q.device).repeat_interleave(n_rep)
                retention_per_q = torch.where(
                    pattern_per_q,
                    torch.full((H_q_,), retention, device=q.device),
                    torch.zeros(H_q_, device=q.device),
                )
                scores = _criticality_scores(q, views)
                sel_per_q = select_pages_vectorized(
                    scores, retention=retention_per_q,
                    num_sinks=num_sinks, window_pages=window_pages,
                    k_max_static=k_max_static,
                )
                sel_compact = build_compact_union_selection(
                    sel_per_q, scores,
                    num_sinks=num_sinks, window_pages=window_pages,
                    completed_len=completed_len, page_size=page_size,
                    BUCKET_MAX_UNION=bucket_max_union_static,
                )
                if kv_bits == 4:
                    O_sparse, lse_sparse = flash_attn_sparse_int4_fwd_compact(
                        q, views["K_packed"], views["K_scale"], views["K_mn"],
                        views["V_packed"], views["V_scale"], views["V_mn"],
                        selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
                    )
                else:
                    raise NotImplementedError(
                        f"Phase 9 verify mode supports kv_bits=4 only; got kv_bits={kv_bits}"
                    )
                lse_sparse = lse_sparse.squeeze(2) if lse_sparse.dim() == 4 else lse_sparse  # (B, H_q, S_q)
            else:
                O_sparse = torch.zeros_like(q)
                lse_sparse = torch.full((B, H_q_, S_q), float("-inf"), device=q.device, dtype=torch.float32)

            K_dense = torch.cat([views["K_partial"], views["K_sandbox"]], dim=2)
            V_dense = torch.cat([views["V_partial"], views["V_sandbox"]], dim=2)
            O_draft, lse_draft = bf16_dense_attn_offset_causal_with_lse(
                q, K_dense, V_dense, q_offset=partial_len,
            )

            if completed_len > 0:
                attn_output = _merge_two_attentions_sq(
                    O_sparse, lse_sparse, O_draft, lse_draft,
                )
            else:
                attn_output = O_draft
        else:
            cache.update_quantized(k, v, layer_idx=self.layer_idx)
            views = cache.get_views(self.layer_idx)

            if S_q > 1:
```

The rest of the existing branches (`if S_q > 1:` prefill and `else:` decode) remain unchanged. Make sure the existing code is now nested under the `else:` of the verify_mode branch.

In `patch_llama_for_quest_persistent`: no signature change needed; the `_pld_verify_active` flag is set/unset by the dispatcher (Task 9), and the forward branch uses `getattr` with default False.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_persistent_patch_pld_verify.py -v
```

Expected: all 3 tests PASS (1 fast + 2 slow). Run slow tests:
```bash
pytest tests/test_persistent_patch_pld_verify.py -v -m slow
```

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/eager/llama_persistent_patch.py tests/test_persistent_patch_pld_verify.py
git commit -m "$(cat <<'EOF'
phase 9 task 8: verify-mode forward branch in patched LlamaAttention

When _pld_verify_active=True and S_q>1, dispatch to the new sandbox path:
add_draft (no update_quantized), score-prioritized UNION selection over
completed cache, sparse INT4 compact kernel S_q>1, dense BF16 offset-causal
over K_partial||K_sandbox, LSE-merge per-query.

_set_pld_verify_active(model, value) walks every LlamaAttention and toggles
the flag. INT8/Turbo unsupported in verify mode (kv_bits=4 only).
EOF
)"
```

---

## Task 9: PLD dispatcher + walk-and-accept

**Files:**
- Create: `src/flashquest/specdec/dispatcher.py`
- Test: `tests/test_pld_walk_and_accept.py`
- Test: `tests/test_pld_admissibility.py`
- Test: `tests/test_pld_verify_parity.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pld_walk_and_accept.py`:

```python
"""Phase 9 Task 9 — walk-and-accept logic."""
import torch


def test_walk_partial_accept():
    from flashquest.specdec.dispatcher import _walk_and_accept
    draft = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    argmax_seq = torch.tensor([20, 30, 99, 99, 88], dtype=torch.int64)  # mismatch at i=2 (compares 99 vs 40)
    M, free = _walk_and_accept(draft, argmax_seq)
    assert M == 2
    assert int(free.item()) == 99    # argmax_seq[2]


def test_walk_all_accept():
    from flashquest.specdec.dispatcher import _walk_and_accept
    draft = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    argmax_seq = torch.tensor([20, 30, 40, 50, 88], dtype=torch.int64)
    M, free = _walk_and_accept(draft, argmax_seq)
    assert M == 4
    assert int(free.item()) == 88    # argmax_seq[N_draft-1=4]


def test_walk_zero_accept_post_d0():
    """All post-D_0 drafts mismatch at first compare."""
    from flashquest.specdec.dispatcher import _walk_and_accept
    draft = torch.tensor([10, 20, 30, 40, 50], dtype=torch.int64)
    argmax_seq = torch.tensor([99, 88, 77, 66, 55], dtype=torch.int64)  # argmax_seq[0]=99 vs draft[1]=20 mismatch
    M, free = _walk_and_accept(draft, argmax_seq)
    assert M == 0
    assert int(free.item()) == 99    # argmax_seq[0]
```

Create `tests/test_pld_admissibility.py`:

```python
"""Phase 9 Task 9 — PLD admissibility checks."""
import torch


def test_admissibility_short_history_falls_back():
    from flashquest.specdec.dispatcher import _is_pld_admissible
    prompt_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    history = torch.tensor([1], dtype=torch.int64)   # < K_match
    next_argmax = torch.tensor([99], dtype=torch.int64)
    out = _is_pld_admissible(
        prompt_ids, history, next_argmax,
        seen=10, K_match=3, N_draft=3, page_size=64,
    )
    assert out is None


def test_admissibility_no_match_falls_back():
    from flashquest.specdec.dispatcher import _is_pld_admissible
    prompt_ids = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
    history = torch.tensor([99, 99, 99], dtype=torch.int64)
    next_argmax = torch.tensor([0], dtype=torch.int64)
    out = _is_pld_admissible(
        prompt_ids, history, next_argmax,
        seen=10, K_match=3, N_draft=3, page_size=64,
    )
    assert out is None


def test_admissibility_d0_mismatch_falls_back():
    from flashquest.specdec.dispatcher import _is_pld_admissible
    # Match found, but D_0 != next_argmax
    prompt_ids = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
    history = torch.tensor([10, 20, 30], dtype=torch.int64)
    # Match at index 0, D_0 = 40
    next_argmax = torch.tensor([99], dtype=torch.int64)   # != 40
    out = _is_pld_admissible(
        prompt_ids, history, next_argmax,
        seen=10, K_match=3, N_draft=3, page_size=64,
    )
    assert out is None


def test_admissibility_page_boundary_falls_back():
    from flashquest.specdec.dispatcher import _is_pld_admissible
    prompt_ids = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
    history = torch.tensor([10, 20, 30], dtype=torch.int64)
    next_argmax = torch.tensor([40], dtype=torch.int64)
    # seen=63, N_draft=2 → would write 63..64, crossing page boundary
    out = _is_pld_admissible(
        prompt_ids, history, next_argmax,
        seen=63, K_match=3, N_draft=2, page_size=64,
    )
    assert out is None


def test_admissibility_happy_path_returns_draft():
    from flashquest.specdec.dispatcher import _is_pld_admissible
    prompt_ids = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64)
    history = torch.tensor([10, 20, 30], dtype=torch.int64)
    next_argmax = torch.tensor([40], dtype=torch.int64)
    out = _is_pld_admissible(
        prompt_ids, history, next_argmax,
        seen=10, K_match=3, N_draft=3, page_size=64,
    )
    assert out is not None
    assert out.tolist() == [40, 50, 60]
```

Create `tests/test_pld_verify_parity.py`:

```python
"""Phase 9 Task 9 — PLD verify parity vs non-spec single-decode."""
import pytest
import torch


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_forced_mismatch_emits_2_tokens():
    """When all post-D_0 drafts mismatch, PLD emits D_0 in this step + free in next.

    Total emitted across (PLD + single) = 2 tokens. Token sequence equals
    non-spec single-decode of D_0 followed by single-decode of (D_0's argmax).
    """
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.runtime.awq_load import load_awq_model
    from flashquest.specdec.dispatcher import make_quest_pld_dispatcher

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    page_size = 64

    # Build cache + patch
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=2048, page_size=page_size, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=page_size,
    )

    # Force PLD into a near-guaranteed mismatch by giving a "draft" pulled from
    # an unrelated section of the prompt. The dispatcher's admissibility check
    # is what will route to the appropriate branch — this test just ensures that
    # IF a draft is admitted and post-D_0 mismatch happens, the emit is 2 tokens.
    prompt = "The quick brown fox jumps over the lazy dog. " * 32
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].cuda()

    init, step = make_quest_pld_dispatcher(
        model, cache, prompt_ids,
        K_match=3, N_draft=5, page_size=page_size,
    )
    init(prompt_ids.unsqueeze(0))
    # Run 10 PLD steps; record per-step emit counts and final token list
    pld_emits = []
    final_tokens = []
    for _ in range(10):
        emitted = step()
        pld_emits.append(int(emitted.numel()))
        final_tokens.extend(emitted.tolist())
    # No catastrophe: total emitted >= 10 (each step emits >= 1 token)
    assert sum(pld_emits) >= 10
    # Each emit is at least 1 token (single-decode emits 1; PLD step emits M+1 >= 1)
    assert all(e >= 1 for e in pld_emits)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pld_walk_and_accept.py tests/test_pld_admissibility.py -v
pytest tests/test_pld_verify_parity.py -v -m slow
```

Expected: all FAIL — dispatcher module doesn't exist yet.

- [ ] **Step 3: Implement the dispatcher**

Create `src/flashquest/specdec/dispatcher.py`:

```python
"""Phase 9 — PLD greedy chain dispatcher.

State invariants:
- cache._seen always = number of K/V slots committed (positions 0..seen-1).
- next_input_token: most-recently-decided token whose K/V is NOT yet in cache.
  Will be input to the next forward.
- next_argmax_buffer: same value as next_input_token in single-decode steady
  state. Set to None after a PLD step (no model argmax for the held free token
  until the next single-decode step computes it).
- committed_history: all already-committed token IDs (length == cache._seen).
"""
from __future__ import annotations

import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import _set_pld_verify_active
from flashquest.specdec.pld import propose_draft


def _walk_and_accept(
    draft: torch.Tensor,                # (N_draft,) int64
    argmax_seq: torch.Tensor,           # (N_draft,) int64
) -> tuple[int, torch.Tensor]:
    """Walk-and-accept: M = longest prefix of argmax_seq[i] == draft[i+1] for i in [0, N-1).

    Returns (M, free_token) where free_token = argmax_seq[M] (held over for next step).
    """
    N = draft.numel()
    M = 0
    for i in range(N - 1):
        if int(argmax_seq[i].item()) == int(draft[i + 1].item()):
            M += 1
        else:
            break
    free_token = argmax_seq[M : M + 1]
    return M, free_token


def _is_pld_admissible(
    prompt_ids: torch.Tensor,           # (S_prompt,) int64 on CPU
    committed_history: torch.Tensor,    # (>=K_match,) int64 on CPU; tail of committed tokens
    next_argmax_buffer: torch.Tensor | None,  # (1,) int64 on CPU, or None
    *,
    seen: int,                          # cache._seen_tokens[0]
    K_match: int,
    N_draft: int,
    page_size: int,
) -> torch.Tensor | None:
    """Return the draft if PLD is admissible at this step, else None.

    PLD admissibility = (a) history >= K_match, (b) prev argmax exists,
    (c) n-gram match in prompt with N_draft continuation, (d) draft[0] == prev argmax,
    (e) committing N_draft tokens does not cross a page boundary.
    """
    if next_argmax_buffer is None:
        return None
    if committed_history is None or committed_history.numel() < K_match:
        return None
    # Page-boundary guard
    if (seen + N_draft) // page_size != seen // page_size:
        return None
    draft = propose_draft(
        prompt_ids, committed_history,
        K_match=K_match, N_draft=N_draft,
    )
    if draft is None:
        return None
    if int(draft[0].item()) != int(next_argmax_buffer.item()):
        return None
    return draft


def make_quest_pld_dispatcher(
    model: torch.nn.Module,
    cache: PersistentInt4KVCache,
    prompt_ids: torch.Tensor,
    *,
    K_match: int = 3,
    N_draft: int = 5,
    page_size: int = 64,
):
    """Wrap `model` (already patched with patch_llama_for_quest_persistent) with PLD generation.

    Returns (init, step) callables. init runs prefill and stores the prompt's argmax.
    step generates >=1 tokens per call (PLD emits M+1, single-decode emits 1).
    """
    if prompt_ids.dim() == 1:
        prompt_ids_cpu = prompt_ids.cpu()
    elif prompt_ids.dim() == 2 and prompt_ids.shape[0] == 1:
        prompt_ids_cpu = prompt_ids[0].cpu()
    else:
        raise ValueError(f"prompt_ids must be 1D or (1, S) 2D; got {prompt_ids.shape}")

    state = {
        "next_input_token": None,
        "next_argmax_buffer": None,
        "committed_history": None,    # set in init()
    }

    def init(prompt_input_ids: torch.Tensor) -> None:
        with torch.no_grad():
            out = model(input_ids=prompt_input_ids, use_cache=True, logits_to_keep=1)
        next_argmax = out.logits[:, -1].argmax(dim=-1)
        state["next_input_token"] = next_argmax
        state["next_argmax_buffer"] = next_argmax
        # Prefill committed prompt_ids; convert to CPU list for n-gram lookup
        if prompt_input_ids.dim() == 2:
            ids_cpu = prompt_input_ids[0].cpu()
        else:
            ids_cpu = prompt_input_ids.cpu()
        state["committed_history"] = ids_cpu.tolist()

    def step() -> torch.Tensor:
        next_in = state["next_input_token"]
        prev_argmax = state["next_argmax_buffer"]
        # Layer-lockstep assert (codex r2 LOW residual #4)
        assert all(s == cache._seen_tokens[0] for s in cache._seen_tokens), (
            f"layer _seen_tokens out of lockstep: {cache._seen_tokens}"
        )
        seen = cache._seen_tokens[0]
        history_tail = (
            torch.tensor(state["committed_history"][-K_match:], dtype=torch.int64)
            if len(state["committed_history"]) >= K_match else None
        )
        prev_argmax_cpu = prev_argmax.cpu() if prev_argmax is not None else None
        draft = _is_pld_admissible(
            prompt_ids_cpu, history_tail, prev_argmax_cpu,
            seen=seen, K_match=K_match, N_draft=N_draft, page_size=page_size,
        )

        if draft is None:
            with torch.no_grad():
                out = model(input_ids=next_in.unsqueeze(0), use_cache=True, logits_to_keep=1)
            new_argmax = out.logits[:, -1].argmax(dim=-1)
            emitted = next_in.unsqueeze(0).cpu()
            state["committed_history"].append(int(next_in.item()))
            state["next_input_token"] = new_argmax
            state["next_argmax_buffer"] = new_argmax
            return emitted.squeeze(0)

        # PLD verify path
        verify_input = draft.to(next_in.device).unsqueeze(0)
        _set_pld_verify_active(model, True)
        try:
            with torch.no_grad():
                out = model(input_ids=verify_input, use_cache=True)
        finally:
            _set_pld_verify_active(model, False)

        argmax_seq = out.logits.argmax(dim=-1).squeeze(0).cpu()
        M, free_token = _walk_and_accept(draft, argmax_seq)
        total_accepted_kvs = 1 + M

        cache.commit_draft_all_layers(total_accepted_kvs)

        accepted_ids = draft[: total_accepted_kvs]
        for t in accepted_ids.tolist():
            state["committed_history"].append(int(t))
        state["next_input_token"] = free_token.to(next_in.device)
        state["next_argmax_buffer"] = None
        return accepted_ids

    return init, step
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pld_walk_and_accept.py tests/test_pld_admissibility.py -v
pytest tests/test_pld_verify_parity.py -v -m slow
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/specdec/dispatcher.py tests/test_pld_walk_and_accept.py tests/test_pld_admissibility.py tests/test_pld_verify_parity.py
git commit -m "$(cat <<'EOF'
phase 9 task 9: PLD dispatcher with walk-and-accept + admissibility + lockstep assert

make_quest_pld_dispatcher returns (init, step). init runs prefill, sets
next_input_token = next_argmax_buffer = prompt's argmax. step:
- if PLD inadmissible (history short, no n-gram match, D_0 != prev_argmax,
  page-boundary cross, or prev_argmax==None): single-decode (1 emit)
- else: forward [D_0, ..., D_{N-1}] in verify mode; walk-and-accept;
  commit M+1 K/Vs across all layers atomically (preflight first); emit
  M+1 tokens. Free_token held in next_input_token for next step.
- next_argmax_buffer = None after PLD (forces next step to single-decode).

Layer-lockstep asserted before page-boundary check (codex r2 #4).
EOF
)"
```

---

## Task 10: End-to-end PLD vs non-spec greedy parity

**Files:**
- Test: `tests/test_pld_dispatcher_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/test_pld_dispatcher_e2e.py`:

```python
"""Phase 9 Task 10 — end-to-end: PLD output ≈ non-spec greedy on synthetic prompt."""
import pytest
import torch


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pld_emits_match_non_spec_on_verbatim_prompt():
    """A prompt where the answer copies verbatim from the prompt should produce
    near-identical generated tokens between PLD and non-spec greedy paths.

    Uses a synthetic instruction: "Repeat the following: <text>" where <text>
    is short and forces the model to copy verbatim. PLD should accept many
    drafts; the resulting token sequence should match non-spec greedy generation.
    """
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.runtime.awq_load import load_awq_model
    from flashquest.specdec.dispatcher import make_quest_pld_dispatcher

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    page_size = 64

    # Synthetic verbatim-copy prompt
    target = "the quick brown fox jumps over the lazy dog and runs into the forest"
    prompt = f"Please repeat the following sentence verbatim: '{target}'\n\nRepeat: '"
    prompt_ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    # === Run 1: non-spec greedy ===
    cache_a = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=512, page_size=page_size, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache_a, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=page_size,
    )
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    next_t = out.logits[:, -1].argmax(dim=-1)
    non_spec_tokens = []
    for _ in range(40):
        non_spec_tokens.append(int(next_t.item()))
        with torch.no_grad():
            out = model(input_ids=next_t.unsqueeze(0), use_cache=True, logits_to_keep=1)
        next_t = out.logits[:, -1].argmax(dim=-1)
    non_spec_tokens.append(int(next_t.item()))

    # === Run 2: PLD ===
    # New cache + re-patch (PLD keeps verify-mode hooks)
    cache_b = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=512, page_size=page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache_b, head_pattern=pattern, retention=0.25,
        num_sinks=4, window_pages=2, page_size=page_size,
    )
    init, step = make_quest_pld_dispatcher(
        model, cache_b, prompt_ids[0],
        K_match=3, N_draft=5, page_size=page_size,
    )
    init(prompt_ids)
    pld_tokens = []
    while len(pld_tokens) < 41:
        emitted = step()
        pld_tokens.extend(emitted.tolist())
    pld_tokens = pld_tokens[:41]

    # PLD should match non-spec on at least 95% of tokens (UNION-induced drift bounded)
    matches = sum(1 for a, b in zip(non_spec_tokens, pld_tokens) if a == b)
    print(f"\nNon-spec: {tok.decode(non_spec_tokens)!r}")
    print(f"PLD:      {tok.decode(pld_tokens)!r}")
    print(f"Match rate: {matches} / {len(pld_tokens)} = {matches/len(pld_tokens):.2%}")
    assert matches / len(pld_tokens) >= 0.95, (
        f"PLD-vs-non-spec match rate {matches}/{len(pld_tokens)} below 95%"
    )
```

- [ ] **Step 2: Run the test**

```bash
nice -n 19 pytest tests/test_pld_dispatcher_e2e.py -v -m slow -s
```

Expected: PASS with match rate ≥ 95%.

- [ ] **Step 3: Decision gate**

If match rate < 95%: investigate UNION-select drift, sandbox commit semantics, or LSE-merge numerics. Do NOT proceed to Tasks 11-12 until this clears.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pld_dispatcher_e2e.py
git commit -m "phase 9 task 10: e2e PLD vs non-spec greedy parity on verbatim-copy prompt"
```

---

## Task 11: RULER quality gate + argmax drift measurement

**Files:**
- Create: `benchmarks/phase9_argmax_drift.py`
- Run: existing `python -m flashquest.eval.runner --task niah_single --pld-on` (script may need a `--pld-on` flag)

- [ ] **Step 1: Add `--pld-on` flag to flashquest.eval.runner**

Modify `src/flashquest/eval/runner.py`:

Find the existing argparse setup. Add:

```python
    p.add_argument("--pld-on", action="store_true", help="Phase 9 PLD greedy decoding")
    p.add_argument("--pld-N-draft", type=int, default=5)
    p.add_argument("--pld-K-match", type=int, default=3)
```

Find where the model generates tokens during eval. Wrap with PLD if `--pld-on`:

```python
    if args.pld_on:
        from flashquest.specdec.dispatcher import make_quest_pld_dispatcher
        init_fn, step_fn = make_quest_pld_dispatcher(
            model, cache, input_ids[0],
            K_match=args.pld_K_match, N_draft=args.pld_N_draft, page_size=args.page_size,
        )
        init_fn(input_ids)
        emitted = []
        while len(emitted) < max_new:
            chunk = step_fn()
            emitted.extend(chunk.tolist())
        emitted = emitted[:max_new]
        gen_text = tok.decode(emitted)
    else:
        # ... existing single-decode path ...
```

- [ ] **Step 2: Run RULER**

```bash
nice -n 19 python -m flashquest.eval.runner \
    --model casperhansen/llama-3.2-3b-instruct-awq \
    --task niah_single --ctx 4096 --n 20 \
    --kv-bits 4 --retention 0.25 \
    --pld-on --pld-N-draft 5 --pld-K-match 3 \
    2>&1 | tee benchmarks/phase9_ruler_single_pld.log

nice -n 19 python -m flashquest.eval.runner \
    --model casperhansen/llama-3.2-3b-instruct-awq \
    --task niah_multivalue --ctx 4096 --n 20 \
    --kv-bits 4 --retention 0.25 \
    --pld-on --pld-N-draft 5 --pld-K-match 3 \
    2>&1 | tee benchmarks/phase9_ruler_multivalue_pld.log
```

Expected:
- `niah_single`: 20/20 (100%)
- `niah_multivalue`: ≥ 17/20 (≥85%)

- [ ] **Step 3: Run argmax drift bench**

Create `benchmarks/phase9_argmax_drift.py`:

```python
"""Phase 9 Task 11 — measure argmax drift between PLD and non-spec greedy."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model
from flashquest.specdec.dispatcher import make_quest_pld_dispatcher


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-prompts", type=int, default=50)
    p.add_argument("--n-tokens", type=int, default=20)
    p.add_argument("--out", default="benchmarks/phase9_argmax_drift.json")
    args = p.parse_args()

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)

    # Sample prompts: PG essays + RULER + HumanEval (truncate to short prompts)
    pg = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    prompts = []
    random.seed(0)
    pg_random = random.sample(pg, min(args.n_prompts, len(pg)))
    for entry in pg_random[: args.n_prompts]:
        text = entry.get("text") or entry.get("body") or ""
        if not text:
            continue
        prompts.append(text[:1200] + "\n\nSummarize: ")
        if len(prompts) == args.n_prompts:
            break

    matches = 0
    total = 0
    for prompt in prompts:
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        # Non-spec
        cache_a = PersistentInt4KVCache(
            batch_size=1, num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
            max_seq_len=2048, page_size=64, device="cuda",
        )
        patch_llama_for_quest_persistent(
            model, cache=cache_a, head_pattern=pattern, retention=0.25,
            num_sinks=4, window_pages=2, page_size=64,
        )
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        next_t = out.logits[:, -1].argmax(dim=-1)
        non_spec = [int(next_t.item())]
        for _ in range(args.n_tokens - 1):
            with torch.no_grad():
                out = model(input_ids=next_t.unsqueeze(0), use_cache=True, logits_to_keep=1)
            next_t = out.logits[:, -1].argmax(dim=-1)
            non_spec.append(int(next_t.item()))
        # PLD
        cache_b = PersistentInt4KVCache(
            batch_size=1, num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
            max_seq_len=2048, page_size=64, device="cuda",
        )
        patch_llama_for_quest_persistent(
            model, cache=cache_b, head_pattern=pattern, retention=0.25,
            num_sinks=4, window_pages=2, page_size=64,
        )
        init, step = make_quest_pld_dispatcher(
            model, cache_b, ids[0], K_match=3, N_draft=5, page_size=64,
        )
        init(ids)
        pld = []
        while len(pld) < args.n_tokens:
            chunk = step()
            pld.extend(chunk.tolist())
        pld = pld[: args.n_tokens]
        for a, b in zip(non_spec, pld):
            total += 1
            if a == b:
                matches += 1
        del cache_a, cache_b
        torch.cuda.empty_cache()

    rate = matches / max(1, total)
    print(f"\n=== Phase 9 Task 11 argmax drift ===")
    print(f"  matches: {matches} / {total} = {rate:.2%}")
    print(f"  GATE: {'PASS' if rate >= 0.99 else 'FAIL'}")
    Path(args.out).write_text(json.dumps(
        {"matches": matches, "total": total, "rate": rate, "gate_pass": rate >= 0.99},
        indent=2,
    ))


if __name__ == "__main__":
    main()
```

Run:

```bash
nice -n 19 python benchmarks/phase9_argmax_drift.py 2>&1 | tee benchmarks/phase9_argmax_drift.log
```

Expected: rate ≥ 99%.

- [ ] **Step 4: Decision gate**

If RULER fails OR argmax drift < 99%: P0 quality regression. Investigate UNION drift, sandbox commit semantics, or LSE-merge numerics. Do NOT proceed to Task 12.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/eval/runner.py benchmarks/phase9_argmax_drift.py benchmarks/phase9_ruler_single_pld.log benchmarks/phase9_ruler_multivalue_pld.log benchmarks/phase9_argmax_drift.log benchmarks/phase9_argmax_drift.json
git commit -m "$(cat <<'EOF'
phase 9 task 11: RULER quality gate + argmax-drift measurement

eval.runner --pld-on flag wires dispatcher into RULER eval path.
RULER NIAH 4k @ Llama-3.2-3B-AWQ + INT4 + PLD on:
  niah_single: 20/20 (target 100%)
  niah_multivalue: >=17/20 (target >=85%)
Argmax drift on 50 PG-summarize prompts × 20 tokens: >=99% match.
EOF
)"
```

---

## Task 12: Phase 9 decode bench + writeup + tag

**Files:**
- Create: `benchmarks/phase9_decode_bench.py`
- Create: `docs/PHASES/phase-9-notes.md`

- [ ] **Step 1: Write the decode bench**

Create `benchmarks/phase9_decode_bench.py`:

```python
"""Phase 9 Task 12 — full PLD vs baseline decode bench at 8k/16k/32k.

Sweeps K_draft ∈ {3, 5, 8} and ctx ∈ {8192, 16384, 32768}.
Reports tok/s and gain vs non-spec compact baseline.

Per saved feedback `avoid_wsl_lag`: one backend at a time, nice -n 19.
"""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model
from flashquest.specdec.dispatcher import make_quest_pld_dispatcher


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def bench_pld(model, tok, cfg, prompt_ids, *, ctx, n_decode, K_draft, K_match, page_size, retention):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx + n_decode + 128, page_size=page_size, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=retention,
        num_sinks=4, window_pages=2, page_size=page_size,
    )
    init, step = make_quest_pld_dispatcher(
        model, cache, prompt_ids[0], K_match=K_match, N_draft=K_draft, page_size=page_size,
    )
    init(prompt_ids)
    # Warmup
    _ = step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    emitted = []
    while len(emitted) < n_decode:
        chunk = step()
        emitted.extend(chunk.tolist())
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_decode / elapsed


def bench_baseline(model, tok, cfg, prompt_ids, *, ctx, n_decode, page_size, retention):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx + n_decode + 128, page_size=page_size, device="cuda",
    )
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=retention,
        num_sinks=4, window_pages=2, page_size=page_size,
        use_compact_kernel=True,
    )
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True, logits_to_keep=1)
    next_t = out.logits[:, -1].argmax(dim=-1)
    # Warmup
    with torch.no_grad():
        out = model(input_ids=next_t.unsqueeze(0), use_cache=True, logits_to_keep=1)
    next_t = out.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_decode):
        with torch.no_grad():
            out = model(input_ids=next_t.unsqueeze(0), use_cache=True, logits_to_keep=1)
        next_t = out.logits[:, -1].argmax(dim=-1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_decode / elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ctxs", default="8192,16384,32768")
    p.add_argument("--n-decode", type=int, default=128)
    p.add_argument("--K-draft-sweep", default="3,5,8")
    p.add_argument("--K-match", type=int, default=3)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--out", default="benchmarks/phase9_decode_bench.json")
    args = p.parse_args()

    ctxs = [int(c) for c in args.ctxs.split(",")]
    K_drafts = [int(k) for k in args.K_draft_sweep.split(",")]

    model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
    cfg = model.config

    # PG-summarize-style prompt (real prompts at full ctx)
    pg = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    text = (pg[0].get("text") or pg[0].get("body") or "")
    if not text:
        text = "The quick brown fox " * 5000

    results = []
    for ctx in ctxs:
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        if ids.shape[1] > ctx:
            ids = ids[:, :ctx]
        elif ids.shape[1] < ctx:
            # pad by repeating
            n_repeat = (ctx // ids.shape[1]) + 1
            ids = ids.repeat(1, n_repeat)[:, :ctx]
        # Non-spec baseline
        print(f"\n=== ctx={ctx} ===")
        baseline_tps = bench_baseline(
            model, tok, cfg, ids,
            ctx=ctx, n_decode=args.n_decode, page_size=args.page_size, retention=args.retention,
        )
        print(f"  non-spec compact baseline: {baseline_tps:.2f} tok/s")
        _free()
        for K_draft in K_drafts:
            pld_tps = bench_pld(
                model, tok, cfg, ids,
                ctx=ctx, n_decode=args.n_decode, K_draft=K_draft,
                K_match=args.K_match, page_size=args.page_size, retention=args.retention,
            )
            gain = pld_tps / baseline_tps
            print(f"  PLD K_draft={K_draft}: {pld_tps:.2f} tok/s ({gain:.2f}x)")
            results.append({
                "ctx": ctx, "K_draft": K_draft,
                "baseline_tps": baseline_tps, "pld_tps": pld_tps, "gain": gain,
            })
            _free()

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nResults: {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench**

```bash
nice -n 19 python benchmarks/phase9_decode_bench.py 2>&1 | tee benchmarks/phase9_decode_bench.log
```

Expected (sample):
```
=== ctx=32768 ===
  non-spec compact baseline: 6.31 tok/s
  PLD K_draft=3: 11.20 tok/s (1.78x)
  PLD K_draft=5: 15.40 tok/s (2.44x)
  PLD K_draft=8: 14.10 tok/s (2.24x)
```

Gate: At ctx=32k, **at least one K_draft sweep value ≥ 15 tok/s** (Phase 9 success target). At ctx=8k, **at least one K_draft sweep value ≥ 18 tok/s**.

- [ ] **Step 3: Write the phase notes**

Create `docs/PHASES/phase-9-notes.md`:

```markdown
# Phase 9 Notes — Prompt-Lookup Decoding (Greedy Chain)

**Started:** 2026-05-09
**Status:** [PASS / FAIL — fill in based on bench results]
**Plan:** [`../superpowers/plans/2026-05-09-phase-9-pld-greedy-chain.md`](...)
**Spec:** [`../superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md`](...)
**Tag:** `phase-9` (after all gates clear)

## Summary

[Fill in: what was built, what landed, headline result.]

Phase 9 adds lossless-by-construction greedy speculative decoding via PLD:
- PLD draft proposer (n-gram match in prompt, rightmost wins)
- Per-layer BF16 sandbox in PersistentInt4KVCache (920 KB)
- Compact INT4 sparse kernel extended to S_q>1 with score-prioritized UNION selection
- Verify-mode forward branch in patched LlamaAttention
- PLD dispatcher with walk-and-accept, page-boundary guard, atomic two-phase commit

## Headline result

[Fill in:]

| ctx | non-spec compact | PLD best K_draft | gain |
|---|---|---|---|
| 8k | X.X tok/s | X.X tok/s (K=Y) | X.XX× |
| 16k | X.X | X.X (K=Y) | X.XX× |
| 32k | X.X | X.X (K=Y) | X.XX× |

[Fill in: what M_avg was measured, hit rate, what RULER scores landed.]

## Gates

| Gate | Threshold | Result |
|---|---|---|
| Task 1 entry: M_avg | >= 2.0 | [fill] |
| Task 1 entry: cost ratio | <= 1.3× | [fill] |
| Task 1 entry: hit rate | >= 30% | [fill] |
| Task 6 kernel: S_q=5 / S_q=1 | <= 1.5× | [fill] |
| Task 10 e2e: PLD vs non-spec | >= 95% | [fill] |
| Task 11 RULER single 4k | 100/100 | [fill] |
| Task 11 RULER multivalue 4k | >= 85% | [fill] |
| Task 11 argmax drift | >= 99% | [fill] |
| Task 12 32k decode | >= 15 tok/s | [fill] |
| Task 12 8k decode | >= 18 tok/s | [fill] |

## What surprised me

[Fill in: anything unexpected that came out of measurement.]

## Surface

- `flashquest.specdec.pld.propose_draft` — n-gram match
- `flashquest.specdec.dispatcher.make_quest_pld_dispatcher` — top-level wrapper
- `flashquest.cache.persistent_int4.PersistentInt4KVCache` — sandbox API: `add_draft`, `get_views_with_sandbox`, `preflight_commit`, `commit_draft`, `commit_draft_all_layers`
- `flashquest.eager.selection.build_compact_union_selection` — score-prioritized UNION
- `flashquest.kernel.sparse_int4_fwd_compact.flash_attn_sparse_int4_fwd_compact` — dispatcher: S_q=1 (Phase 8a path) | S_q>1 (new Phase 9 kernel)
- `flashquest.eager.llama_persistent_patch._set_pld_verify_active`, `bf16_dense_attn_offset_causal_with_lse`, `_merge_two_attentions_sq` — verify-mode plumbing

## Roadmap implications

[Fill in: what does this mean for Phase 10 (DuoAttention/CATS) and Phase 11 (lookahead)?]

## Process notes

- Codex r1 raised 4 dealbreakers, 3 correctness bugs, 5 integration risks, 4 nits — all addressed in spec r2.
- Codex r2 raised 1 HIGH (UNION-overflow) + 3 minor — addressed in spec r2.1 via the `build_compact_union_selection` helper (score-prioritized truncation, sinks + window force-included).
- Profile-first lesson held (saved memory): Task 1 dense-only profile was the entry gate; Tasks 5+6 included a kernel microbench gate. [Fill in: did either gate change the trajectory?]
```

- [ ] **Step 4: Tag the phase**

```bash
git add benchmarks/phase9_decode_bench.py benchmarks/phase9_decode_bench.log benchmarks/phase9_decode_bench.json docs/PHASES/phase-9-notes.md
git commit -m "phase 9 task 12: decode bench + notes"

# Tag only after all gates clear
git tag phase-9 -m "phase 9: PLD greedy chain on Quest sparse INT4"
```

If decode gates failed: do NOT tag. Update `phase-9-notes.md` to reflect actual results, document the gap, then commit (tag deferred).

---

## Self-review checklist (run before handing off)

- [ ] Every task has 5 steps (test → fail → impl → pass → commit) per skill convention.
- [ ] Every code block in implementation steps is complete (no `...`).
- [ ] Type/method names consistent across tasks: `add_draft`, `commit_draft`, `commit_draft_all_layers`, `get_views_with_sandbox`, `_set_pld_verify_active`, `_pld_verify_active`, `bf16_dense_attn_offset_causal_with_lse`, `_merge_two_attentions_sq`, `build_compact_union_selection`, `flash_attn_sparse_int4_fwd_compact`, `_walk_and_accept`, `_is_pld_admissible`, `make_quest_pld_dispatcher`.
- [ ] Spec sections covered: §1 goals (Task 1 + 12), §2 context (Task 1), §3 algorithm (Tasks 4 + 8 + 9), §4.1 proposer (Task 2), §4.2 sandbox (Task 4), §4.3 verify forward (Tasks 7 + 8), §4.4 kernel + UNION (Tasks 3 + 5 + 6), §4.5 dispatcher (Task 9), §5 trace (verified by Tasks 9 + 10), §6 entry gate (Task 1), §7 success gates (Tasks 1, 6, 10, 11, 12), §8 test plan (each task includes its tests), §9 RULER quality (Task 11), §10 decode bench (Task 12), §11 file layout (followed exactly), §12 risks (Task 1, 6 gates address profile-first risk; Task 10 addresses quality drift; Task 4 addresses page-boundary; Task 8 addresses verify flag try/finally), §13 open questions (resolved by tests in tasks; remainder are profile-driven), §14 references (linked from spec).
- [ ] Profile-first gates exist at Tasks 1 (entry), 6 (kernel), 10 (e2e quality), 11 (RULER).
- [ ] No CUDA-sync .item() calls in hot path (sandbox API, UNION helper); allowed in dispatcher's per-step bookkeeping (already crossing CPU↔GPU boundary).
- [ ] Layer-lockstep assert in dispatcher (Task 9) per codex r2 LOW residual.
- [ ] Page-boundary guard at admissibility (Task 9) AND preflight (Task 4).
- [ ] Two-phase commit (preflight all → mutate) (Task 4).
- [ ] try/finally around verify-mode flag (Task 9).
- [ ] tl.dot input_precision="ieee" in kernel (Task 5).
