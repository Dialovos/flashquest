# Phase 11 — Quality-gate FAIL (TurboQuant calibrated codebook)

**Date:** 2026-05-30
**Outcome:** Calibrated K3-V3 **fails the ≥95% quality gate** (multivalue 85%), so
it does **not** become the v1.2 default. It ships as an **opt-in**
(`--kv-bits 3 --codebook calibrated`); `--kv-bits 4` (KIVI-INT4) remains the v1
default. Nuance below: at matched retention, calibration *does* improve K3-V3
multivalue over the paper codebook (70% → 85%) — it just doesn't reach INT4's
95% or clear the gate.
**Plan/Spec:** `docs/superpowers/plans/2026-05-14-phase-11-turboquant-calibrated-codebook.md`

## The gate (Task 9): RULER NIAH 4k, n=20, ctx=4096, retention=0.20

Apples-to-apples, all at retention 0.20, same harness/seed
(`benchmarks/phase11/ruler_4k_{calibrated,paper}.json`); INT4 row from Phase 10 /
v1.0 (the current default):

| mode | single | multikey | multivalue | gate ≥95% all |
|---|---|---|---|---|
| INT4 (`--kv-bits 4`, default) | 100% | 100% | **95%** | pass |
| K3-V3 paper codebook | 100% | 100% | **70%** | fail |
| **K3-V3 calibrated** | 95% | 100% | **85%** | **fail (multivalue)** |

**Correction to an earlier draft of this doc.** A first draft claimed calibration
"moved nothing — multivalue 17/20, identical to paper." That was wrong: the
"17/20 paper" was Phase 7's number at retention **0.25** (its default then),
mis-cited as a matched comparison. The Task-9 matched paper rerun at retention
0.20 is **14/20 (70%)**. So calibration actually **improves** multivalue by
+3 samples (70% → 85%) at matched retention — it just plateaus below the 95% gate,
and dips single by one sample (100% → 95%, within n=20 noise). The matched
baseline the plan insisted on is what caught the error.

## Why it still fails the gate

- Calibration helps multivalue but **stops at 85%**, short of the ≥95% bar.
- INT4 (the incumbent default) already does **95%** multivalue at the same
  retention with no calibration — so on quality there is no reason to switch the
  default to calibrated K3-V3. K3-V3's edge is the **25% smaller KV cache**, which
  makes it a memory opt-in, not a quality default.

The Task 1 entry probe (`benchmarks/phase11/probe_tighter.md`) flagged the
granularity risk that bounds per-layer:

- Codepoint divergence (per-layer vs paper): K = 5.75%, V = 13.37% (≥5% gate → PROCEED)
- Quality-simulator delta: 1 of 3 samples differ
- **Granularity pre-signal R (per-head vs per-layer) = 5.70**

R = 5.70 means head-level codepoint divergence is ~5.7× the layer-level
divergence. Per-**layer** calibration captures the layer-mean structure (enough to
move multivalue 70 → 85) but averages out the per-head structure — a plausible
reason it plateaus. Whether closing that per-head gap actually buys the last 10
points, or whether the residual is page-selection, is exactly what Phase 11b's
Task 1 disambiguator tests before any per-head build.

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

## Next: Phase 11b (profile-first; not assumed to be per-head)

Now that we know calibration *partially* works (multivalue 70 → 85), the open
question is what closes the last 10 points: more granular fidelity (per-head
codebooks) or page selection (the remaining misses are pages top-k dropped). R =
5.70 makes per-head plausible, but per-layer already helped, so the lever is no
longer obvious. Phase 11b therefore leads with a cheap disambiguator — vary only
page selection at fixed 3-bit and see if multivalue recovers — and only builds
per-head if fidelity is shown to be the binding constraint. Full design (codex-
reviewed): `docs/superpowers/specs/2026-05-30-phase-11b-multivalue-gap-design.md`.
