# Phase 11 — Quality-gate FAIL (TurboQuant calibrated codebook)

**Date:** 2026-05-30
**Outcome:** KILLED at the Task 9 quality gate. Per-layer calibrated codebooks do
**not** recover RULER NIAH multivalue, so K3-V3 calibrated does **not** become the
v1.2 default. It ships as an **opt-in** (`--kv-bits 3 --codebook calibrated`);
`--kv-bits 4` (KIVI-INT4) remains the v1 default.
**Plan/Spec:** `docs/superpowers/plans/2026-05-14-phase-11-turboquant-calibrated-codebook.md`

## The gate (Task 9): RULER NIAH 4k, n=20, ctx=4096, retention=0.20

| Task | Calibrated (K3-V3) | Gate ≥95% |
|---|---|---|
| single | 19/20 (95%) | pass (just) |
| multikey | 20/20 (100%) | pass |
| **multivalue** | **17/20 (85%)** | **FAIL** |

`gate_all_ge_95 = False`. Multivalue lands at exactly the Phase 7 paper-codebook
baseline (17/20). Calibration moved nothing on the task it was built to fix, and
single is 95% vs Phase 7's 100% — i.e. not an improvement.
(Apples-to-apples paper-at-0.20 rerun: `benchmarks/phase11/ruler_4k_paper.json`.)

## Why it failed — the entry probe predicted it

The Task 1 entry probe (`benchmarks/phase11/probe_tighter.md`) cleared the
codepoint-divergence gate but flagged the risk:

- Codepoint divergence (per-layer vs paper): K = 5.75%, V = 13.37% (≥5% gate → PROCEED)
- Quality-simulator delta: 1 of 3 samples differ
- **Granularity pre-signal R (per-head vs per-layer) = 5.70**

R = 5.70 means head-level codepoint divergence is ~5.7× the layer-level
divergence. Per-**layer** calibration therefore captures only ~1/5.7 of the
structure that actually varies. Multivalue NIAH stresses simultaneous retrieval
across heads, exactly where per-layer averaging washes out the signal. The gate
result confirms the probe's advisory: **per-layer is the wrong granularity.**

## Decision

- **Do not** promote calibrated to default. Keep `--kv-bits 4` as the v1 default.
- Ship Phase 11 as an **opt-in** capability: `--codebook calibrated` works, is
  tested, and is correct — it just doesn't beat INT4 on quality, so it isn't the
  default. No `phase-11` promotion tag.
- **Task 10 (decode speed) skipped** — moot once the quality gate failed.

## What still shipped (committed, tested)

- `flashquest.turbo.codebook.load_codebook(model_id)` loader + paper fallback.
- `PersistentTurboKVCache(model_id=...)` autoloads the per-layer artifact.
- Parameterized `quantize/dequantize_{k,v}_turbo(codebook=...)` and
  `flash_attn_sparse_turbo_fwd(codebook_k=, codebook_v=)` (28-variant constexpr
  dispatch).
- CLI `--codebook {calibrated,paper}`; calibration script
  `scripts/phase11_calibrate_codebook.py`; artifact
  `src/flashquest/turbo/codebook_llama_3_2_3b.pt` (28,2,8).
- Tests: loader, monotonicity, subsample-hot-path, kernel parity, cache smoke.

## Next: Phase 11b candidate (per-head)

R = 5.70 points squarely at **per-head calibrated codebooks** as the escalation
that could actually move multivalue. The kernel already takes codepoints as
constexpr; the lift is (a) fit per-(layer,head) codebooks, (b) widen the
artifact to (num_layers, num_kv_heads, 2, 8), (c) thread a per-head codebook
through the dispatch. Profile-first as always: re-run this same Task 9 gate
before investing in speed work.
