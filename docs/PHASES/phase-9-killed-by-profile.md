# Phase 9 PLD — Killed by Profile-First Entry Gate

**Date:** 2026-05-09
**Status:** dropped — empirical Task 1 entry gate FAILED on (M_avg, hit_rate)
**Spec:** `docs/superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md` (r2.1, kept for reference)
**Plan:** `docs/superpowers/plans/2026-05-09-phase-9-pld-greedy-chain.md` (Tasks 2-12 not executed)
**Tag:** none (no kernel/dispatcher work landed)

## Why

Phase 9 was scoped to deliver 2.4× decode throughput via Prompt-Lookup Decoding (PLD greedy chain). Per the profile-first lesson saved after Phase 8a Task 5 + Phase 8b (`feedback_profile_before_speedup_specs.md`), the FIRST task was an empirical measurement gate on dense (no-kernel) PLD across the targeted workloads (PG-essay summarize 8k + RULER NIAH single 4k). Entry gate definition (from spec §6):

- **M_avg ≥ 2.0** averaged across (PG + RULER) — minimum 1.6× per-cycle gain
- **hit_rate ≥ 30%** averaged — fraction of decode steps where PLD is admissible
- **S_q=5 dense / S_q=1 dense ≤ 1.3×** — verify cost ratio bound

## Result @ Llama-3.2-3B-AWQ, 18 PG-summarize prompts × 64 decode + 20 RULER prompts × 64 decode

| Metric | Combined | PG | RULER | Gate | Verdict |
|---|---|---|---|---|---|
| M_avg | **1.64** | 1.67 | 1.61 | ≥ 2.0 | **FAIL** |
| hit_rate | **3.87%** | 5.05% | 2.69% | ≥ 30% | **FAIL hard** |
| S_q=5 / S_q=1 ratio | **0.63×** | 0.57× | 0.87× | ≤ 1.3× | PASS (S_q=5 cheaper than S_q=1) |

Script: `benchmarks/phase9_task1_pld_profile.py`. Results JSON: `benchmarks/phase9_task1_results.json`. Log: `benchmarks/phase9_task1_log.txt`.

## Root causes

### 1. Pre-admissibility check is the dominant filter

The spec's PLD design uses an admissibility check: PLD verify is invoked only when `draft[0] == next_argmax_buffer` (= the model's natural next-token argmax already matches the prompt-lookup proposal). This is the lossless-by-construction criterion — D_0 gets verified by the previous step's already-computed argmax.

In practice: among "potentially-PLD-eligible" steps (every other step, since each PLD step is followed by a forced single-decode that re-establishes `next_argmax_buffer`), only ~10% pass the admissibility gate. The rest fall back to single-decode. This caps the practical hit rate well below 30%.

The alternative is HF-style "always propose, verify decides" — drops the pre-admissibility check and runs verify on every step where an n-gram match exists. Per-cycle math (with our measured 0.63× S_q=5/S_q=1 cost ratio + 1.64 M_avg + 10% D_0-accept rate inside admissibility): best-case ~1.32× speedup. Below the 1.6× minimum gain for Phase 9 to deliver any meaningful win.

### 2. Naive rightmost K_match=3 picks wrong-context matches

Our `propose_draft` uses fixed K_match=3 and rightmost match. RULER NIAH's prompt repeats "Some special magic numbers are hidden..." many times across the haystack, and the rightmost occurrence usually isn't adjacent to the answer needle. HF's reference impl uses longest-prefix-first matching (try K=K_max..K_min, pick the longest match) which biases toward recent context. Even with longest-prefix, the answer-needle phase is short (~10-20 tokens) within a 64-decode window, limiting upside.

### 3. PG-summarize doesn't actually verbatim-copy

Generated summarize text paraphrases the prompt; PLD's prompt-mining advantage applies only when the answer copies verbatim. Our targeted "long-context retrieval / RAG / summarize" framing was correct in spirit but wrong in instance — we picked summarize (generative) instead of true RAG (extractive).

## Diagnostic numbers (from `benchmarks/phase9_task1_results.json`)

```
PG (1069 steps, 18 prompts × ~60 decode):
  pld_admissible:   54     (5.05%)
  pld_steps:        54
  M_avg:            1.67   (range 0..4)
  sq1_avg_ms:       670.7
  sq5_avg_ms:       384.3

RULER (1227 steps, 20 prompts × ~61 decode):
  pld_admissible:   33     (2.69%)
  pld_steps:        33
  M_avg:            1.61
  sq1_avg_ms:       169.6
  sq5_avg_ms:       147.0
```

Per-cycle gain math under the as-designed (admissibility-gated) algorithm:
```
hit_rate × cycle_gain + (1 - hit_rate) × 1.0×
= 0.05 × ((M+2) / 2.5) + 0.95 × 1.0×
= 0.05 × 1.46 + 0.95
= 1.02×
```
Net: **basically no speedup**. Phase 9 as designed delivers ~1× on these workloads.

## What's preserved

- `benchmarks/phase9_task1_pld_profile.py` — profile script. Reusable for re-measuring PLD if a future phase revisits with different workload mix or proposer strategy.
- `docs/superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md` — design spec stays in the repo for reference. The architectural decisions (sandbox-and-commit cache, score-prioritized UNION selection, S_q>1 kernel) may apply to future specdec phases if we revisit.
- `docs/superpowers/plans/2026-05-09-phase-9-pld-greedy-chain.md` — implementation plan stays. Tasks 2-12 are unbuilt; spec dependencies on the kernel S_q>1 + sandbox API are still useful design assets.

## Lesson logged

**Profile gate caught the algorithmic premise mismatch BEFORE we wrote any kernel code.** This is the third time the profile-first rule has paid off:

| Phase | Predicted | Measured | Saved |
|---|---|---|---|
| 8a Task 5 | 1.3-1.6× | 1.01× | 5 days of integration work |
| 8b graphs | 1.3-1.4× | 1.02× | 1 week of CUDA Graph plumbing |
| 9 PLD admissibility | 2.4× | ~1.0× | ~2 weeks of kernel + dispatcher work |

The spec's r1 → r2 codex review cycles caught **correctness bugs** (4 dealbreakers + 3 correctness in r1; 1 HIGH + 3 minor in r2). Codex review does NOT catch algorithmic-premise failures — it sees code/specs, not measurements.

**The profile-first rule is now load-bearing on three phases.** It belongs in CLAUDE.md (project-level), not just in memory.

## Roadmap implications

Phase 9 PLD greedy chain is dropped. The OOM roadmap math is now:

| Phase | Speedup | Cumulative tok/s @ 32k |
|---|---|---|
| baseline today | — | 6.29 |
| ~~Phase 9 PLD~~ | ~~2.4×~~ | ~~15.1~~ (dropped) |
| Phase 10 DuoAttention/CATS | 1.5× | 9.4 |
| Phase 11 lookahead (gen-side) | 1.3× | 12.3 |

To recover Phase 9's 2.4× contribution, candidate replacements:
1. **Lookahead / Jacobi decoding** — n-gram mining from the *generated* tail (not the prompt). Different accept-rate distribution; less workload-dependent.
2. **Self-speculative early-exit** — first ~14 layers as draft, full 28 as verifier. Workload-agnostic (~1.4-1.7× expected without training; lower than 2.4× target).
3. **HF-style PLD without pre-admission** — recovers some hit-rate via "always-propose, verify-decides" + longest-prefix matching, but bounded by our 0.63× cost ratio + 1.6 M_avg → ~1.3× ceiling. May be a quick win if redone correctly.

User's choice (post-kill): pivot to Lookahead or self-spec brainstorm.

## Process notes

- The Task 1 script had two pre-launch bugs (wrong python interpreter; wrong PG corpus format) caught and fixed at runtime, costing ~5 minutes. Future plans should include a "smoke test" step (load model, load 1 prompt, run 1 step) before the full bench.
- The plan called `niah_single` from `flashquest.eval.niah` — actual API is `make_prompt(task="single", ctx_len, tokenizer, seed)`. Plan should be patched to fix this or future replays will hit the same wall.
- Codex r2 verdict was clean ("no new dealbreakers"). Codex's review can't catch algorithmic-premise mismatches — only empirical measurement does.
