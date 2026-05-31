# Phase 11b — Closing the RULER multivalue gap (profile-first design)

**Status:** spec / design (not yet planned or executed)
**Author:** autopilot continuation of Phase 11
**Predecessor:** Phase 11 (per-layer calibrated codebook) — **KILLED at quality gate**, see `docs/PHASES/phase-11-quality-gate-fail.md`

## Goal

Close the RULER NIAH **multivalue** gap at K3-V3 (TurboQuant 3-bit KV) from
17/20 (85%) to ≥95%, so the smaller-cache mode can match INT4 quality and become
the default. single (19/20) and multikey (20/20) already clear; multivalue is the
sole blocker.

## Why Phase 11 failed, and the central question

Per-**layer** calibration improved the codepoint fit (codepoint divergence vs
paper: K 5.75%, V 13.37%) yet moved multivalue by **exactly zero** (17/20 →
17/20, identical to the paper codebook). That is strong evidence that
**codebook fidelity is not the binding constraint for multivalue.**

The proposed Phase 11b (per-**head** codebooks) is motivated only by the Task 1
granularity signal R=5.70 (per-head codepoint divergence ≈ 5.7× per-layer). But
"divergence exists at head level" ≠ "that divergence causes the multivalue
miss." Before building anything, we must disambiguate two hypotheses:

- **H-quant:** multivalue fails because 3-bit quantization loses information that
  matters per-head. → per-head codebooks could help.
- **H-select:** multivalue fails because Quest top-k page selection at
  retention=0.20 drops pages holding *some* of the multiple needle values (a
  recall problem, not a fidelity problem). Supporting evidence: Phase 10 showed
  multivalue **collapses to 65% at retention=0.10** — multivalue is acutely
  selection-sensitive — while single stays 100%. And per-layer fidelity gains
  did nothing, which is exactly what H-select predicts.

Per the project's load-bearing **profile-first discipline** (4 prior
profile-kills; `feedback_profile_before_speedup_specs`), Phase 11b's first task
is a cheap experiment that decides which hypothesis is real — and may kill the
per-head idea outright.

## Task 1 — entry-gate experiment (the disambiguator)

> **Codex review (2026-05-30, read-only advisor):** agreed with the premise —
> "per-layer calibration cutting codepoint error to ~zero effect on multivalue
> is strong evidence the bottleneck is upstream of dequant fidelity; R=5.70 only
> says per-head distributions differ, not that the difference costs recall.
> Don't build per-head codebooks yet." It sharpened the experiment: the single
> cleanest discriminator changes **one** variable (the selection set) at fixed
> 3-bit precision. It noted my original 2×2's `dense × bf16` cell changes *two*
> things at once (selection breadth AND the math path), so it's a weaker
> isolator. Folded in below: Task 1a is the one-variable selection test; the
> bf16 cells are demoted to a conditional follow-up (Task 1c) that only runs if
> selection is exonerated.

The hypotheses, restated as a question about a single needle-bearing page: is the
miss **"can't see it"** (selection dropped the page) or **"saw it, lost it"**
(quant/attention mangled it)? Hold KV = 3-bit paper fixed throughout; same
harness/seed as the Phase 11 gate (`flashquest.eval.runner.run_niah`,
`scripts/phase7_run_ruler_4k_turbo.py` template), RULER **multivalue**, n=20,
ctx=4096, all-retrieval head pattern.

**Task 1a — retention sweep at fixed 3-bit (zero instrumentation, do this first).**
Run multivalue at retention ∈ {0.20 (known: 17/20), 0.50, 1.00 (dense)}, all at
3-bit. Only the selection breadth changes; precision is constant.
- **Dense (1.00) ≈ 100% at 3-bit:** the gap is **selection** (H-select). Per-head
  codebooks will NOT help — **kill the per-head angle**; pivot to multivalue-aware
  selection (Task 2'). The 0.50 point shows how steep the recall curve is.
- **Dense (1.00) still ≈ 85% at 3-bit:** selection is exonerated (even seeing all
  pages doesn't fix it) → the loss is downstream → proceed to Task 1c.

This needs no new code: retention is a CLI/arg knob, KV stays 3-bit. ~3 runs.

**Task 1b — oracle-selection toggle (codex's targeted version; only if 1a is
ambiguous, e.g. dense lands ~90–94%).** At retention=0.20, log which pages the
needle values land in, then re-run forcing exactly those needle-bearing pages
into the selection set (keep top-k for the rest). This is a tighter probe than
full dense: it tests whether *the needle pages specifically* are being dropped,
and doubles as the upper bound a real multivalue-aware policy would chase. Needs
a small selection-override hook; no kernel/calibration work. **Caveat (codex):**
an oracle that force-includes needle pages is a diagnostic upper bound, not a
shippable policy.

**Task 1c — bf16-exact ceiling (only if selection is exonerated by 1a/1b).**
Run multivalue dense at bf16 (existing dense reference path). This separates
"quant is binding under full attention" (H-quant, bf16 ≈ 100% but 3-bit ≈ 85%) →
per-head probe justified, from "3B/RULER model ceiling" (bf16 also ≈ 85%) →
**kill Phase 11b entirely**.

Total cost: ~3 runs for 1a (~30–60 min); 1b/1c are conditional. **Stop after 1a
unless it says selection is exonerated.**

## Task 2+ (only if H-quant confirmed) — per-head codebooks

Design notes (deferred until Task 1 justifies them):

- **Artifact:** widen to `(num_layers, num_kv_heads, 2, 8)` fp32 (Llama-3.2-3B:
  28×8×2×8×4 B ≈ 14 KB — still trivial). Calibration reuses the Phase 11
  pipeline verbatim: plain-forward DynamicCache capture (NOT the Quest-patched
  cache — it OOMs; see `project_phase11_calibration`), token-cap per chunk,
  subsample ≤1M points/fit before Lloyd-Max. Just fit per-(layer,head) instead
  of per-(layer).
- **Kernel — the real change:** per-head codepoints **cannot stay `constexpr`**
  (constexpr is per-launch, uniform across the grid; different KV heads in one
  launch need different codepoints). Per-head therefore *requires* moving the
  codebook to a **runtime tensor** indexed per KV head inside the kernel
  (`tl.load(codebook_ptr + kv_head*8 + ...)`). This trades the ~10% constexpr-
  inline win for a single compiled variant (vs the 224-variant constexpr
  explosion 28×8 would otherwise cause) — likely a net positive on compile
  budget, a small steady-state decode cost. Re-run the Task 9 quality gate AND a
  decode-speed check before shipping.
- **Cache/dispatch:** `PersistentTurboKVCache` holds `(num_layers, num_kv_heads,
  8)` K/V codebooks; `make_quest_persistent_forward` passes the layer's
  per-head codebook tensor to the kernel.

## Task 2' (if H-select confirmed) — multivalue-aware selection (sketch)

Not specced in detail here; candidate levers: per-needle/union page selection,
multivalue-tuned `retention`, or criticality scoring that doesn't collapse when
relevant mass is spread across many pages. This would be its own phase.

## Risks / notes

- Biggest risk is building per-head on the R=5.70 signal alone; Task 1 exists to
  prevent that. Treat Task 1 as a hard gate. (Codex concurred this is the trap.)
- bf16-exact dense at ctx=4096 fits the 4 GB GPU (it's the existing reference
  path used in `scripts/phase7_run_ruler_4k_turbo.py`'s dense arm).
- Keep `--kv-bits 4` (INT4) as the default throughout; nothing here changes the
  default until a gate passes.

## Out of scope

- Per-channel codebooks; other models; per-token RMS-scale recalibration.
- Shipping anything as default before the Task 9 quality gate + a speed check.

## Next step

Write the implementation plan for Task 1 only (the disambiguator), execute it in
a **clean session** (this one had tool-output corruption), and let the 2×2 table
decide whether Phase 11b is per-head, selection, or a kill.
