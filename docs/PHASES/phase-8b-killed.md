# Phase 8b Killed by Profile

**Date:** 2026-05-08
**Status:** dropped — empirical profile showed CUDA Graph upper bound is 1.02×
**Spec deferred indefinitely.** Cache redesign piece folds into Phase 9 (specdec needs it for tree-verify cache management anyway).

## Why

Phase 8b was scoped (in Phase 8a's notes) as the phase that delivers wall-time savings via CUDA Graph capture: estimated 1.3-1.4× from launch-overhead elimination. That estimate was back-of-envelope.

User insisted on profile-first per the lesson from Phase 8a Task 5 (where my "compact kernel: 1.3-1.6×" claim was measured at 1.01×). Profile script: `scripts/phase8b_profile_decode.py` using `cuda.Event`-based GPU timing (CUPTI is missing perms in WSL2; cuda.Event bypasses CUPTI).

## Result @ 8k decode (Llama-3.2-3B-AWQ INT4)

```
Wall:          99.42 ms (10.06 tok/s)
GPU (event):   97.58 ms (98.1% of wall)
CPU+driver:    1.84 ms (1.9% of wall)

Upper-bound Phase 8b graph speedup: 1.02×
Realistic (85% CPU overhead eliminated): 1.02×
```

CPU+driver overhead is the upper bound on CUDA-Graph savings. At 1.9%, graphs cannot deliver more than 1.02× even with perfect capture.

**At 32k the ratio is even worse** — GPU compute scales with context (more pages selected, more K/V tile loads), CPU dispatch overhead is roughly constant per step. Expected CPU% at 32k: ≤1%.

## Why launch overhead is fully hidden

Kernel launches are async on CUDA. While the GPU executes kernel N, the CPU enqueues kernel N+1. As long as the GPU is the bottleneck (which it is at our decode shape — sparse INT4 attention is ~1.5 ms × 28 layers + AWQ GEMMs), CPU dispatch overhead overlaps with GPU compute and never shows up in wall time.

Codex r2 review on the original Phase 8 spec hinted at this with finding #9 ("graph capture boundary inconsistent") but didn't reject the launch-overhead premise outright. Static analysis can't see "GPU is already saturated"; only profiling can.

## Side-finding — clean baselines

The profiler also showed today's clean (no host contention) baselines are higher than Phase 7's reported numbers:

| ctx | Phase 7 reported | Today's clean | factor |
|---|---|---|---|
| 8k | 4.94 tok/s | 10.06 tok/s | 2.04× |
| 32k | 3.88 tok/s | 6.29 tok/s | 1.62× |

Phase 7's measurements were taken during a multi-backend head-to-head with vLLM and llama.cpp running concurrently (Phase 7 notes line 94 already documented this). Single-cell clean baselines today are much higher.

**Updated OOM roadmap math** (using 32k clean baseline 6.29 tok/s):

| Phase | Speedup | Cumulative tok/s @ 32k |
|---|---|---|
| baseline (today) | — | 6.29 |
| Phase 9 specdec | 2.5-3.5× | 15.7-22.0 |
| Phase 10 DuoAttention/CATS | 1.5× | 23.6-33.0 |
| Phase 11 lookahead | 1.3× | 30.7-42.9 |

User's "OOM gain to ~30-50 tok/s" target is achievable through Phases 9+10+11 alone. Phase 8b graphs were never going to materially contribute.

## What's preserved

- `scripts/phase8b_profile_decode.py` — profiling tool. Reusable for future phases.
- Phase 8a's compact-list kernel + sync-free selection are still valid foundation work for Phase 9 (specdec's tree-verify forward pass benefits from a kernel that takes a compact list of selected pages, even if standalone speedup is ~0×).
- The cache view redesign Phase 8b would have done becomes a Phase 9 task: tree verification needs snapshot/rollback semantics on the KV cache.

## Lesson logged

**Profile before predicting throughput.** This is the second back-of-envelope claim that didn't survive measurement (Phase 8a Task 5 was the first). Phase 9's spec MUST start with a "what does the workload actually look like" measurement task before designing speedup mechanisms around assumed bottlenecks.
