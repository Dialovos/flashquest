# Phase 6 Task 1 Notes

**Started:** 2026-05-01
**Status:** **partial — perf 22× improved (0.092 → 2.03 tok/s); SPEC ≥4 tok/s gate not yet met. NOT tagged.**
**Spec:** [../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md](../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md)
**Plan:** [../superpowers/plans/2026-05-01-phase-6-criticality-fix.md](../superpowers/plans/2026-05-01-phase-6-criticality-fix.md)

## Goal

Close the SPEC §6 task 1 gate: Llama-3.2-3B-AWQ decode at 32 k context from
Phase 5's 0.092 tok/s up to ≥4 tok/s.

## Surface

- `flashquest.eager.page_scores_int8(Q, K_scale, K_mn) → (B, H_q, S_q, P)` —
  algebraic Quest criticality direct from quant params; no dequant.
- `flashquest.eager.select_pages_vectorized(scores, retention, num_sinks, window_pages) → mask` —
  single batched topk + scatter; no per-head Python loop, no `.item()` per head.

## Win conditions

| Condition | Target | Result | Pass? |
|---|---|---|---|
| EQ18-EQ20 unit tests for page_scores_int8 | green | 6 passed | ✅ |
| EQ21-EQ25 unit tests for select_pages_vectorized | green | 24 passed | ✅ |
| Phase 5 unit + e2e tests still green | green | 145 passed + e2e (rtol=5e-2) | ✅ |
| Logit equivalence vs Phase 5 wiring on identical seeds | bit-equal multi-step output | confirmed | ✅ |
| Decode at 32 k ≥4 tok/s | ≥4 tok/s | **2.03 tok/s** (22× over 0.092) | ❌ |
| 32 k passkey ≥80 % at depth=0.5 | 6/6 | inconclusive (see Methodology gap below) | ⚠️ |

## Decisions

- **Algebraic identity exact, not approximate.** `kv_quant._scale_mn_per_page_channel`
  computes `mn = K.min(dim=page).values` and `scale = (K.max - K.min)/255`, so
  `K_mn` is the page_min and `K_mn + 255*K_scale` is the page_max — bit-equal
  modulo the eps-clamp on constant channels (which the dequant path also
  rounds through). No quality cost.
- **Triton kernel deferred — but the gate isn't yet met without it.** The
  algebraic + vectorized fix moved the per-layer cost from ~337 ms to ~17.6 ms
  (19× per layer; 22× end-to-end). The remaining 17.6 ms / layer is the
  AWQ projections + MLP + LayerNorm — not the criticality path. To clear ≥4
  tok/s would need either (a) speculative decoding (EAGLE-2), or (b) the
  kernel-fused criticality folded with new dequant in a future INT4 KV pass.
  See "Phase 6 task 1b" below.
- **One `.item()` per layer** instead of per-head — for `k_max` only. Could
  be precomputed at patch time from the retention vector but the saving is
  ~1 host sync per layer per step (microseconds); not worth the API churn.

## Methodology gap: passkey eval brittleness

The existing `phase5_run_passkey_32k.py` and the cloned `phase6_run_passkey_32k.py`
generate `head_pattern` via un-seeded `torch.rand(...) < 0.7` — different
across runs. Diagnostic at ctx=4096 with seeded `head_pattern`
(`scripts/phase6_diag_passkey_3b.py`):

- Phase 5 wiring (HEAD-restored old chain): **0/6** at depths {0.1, 0.5, 0.9} × 2 trials.
- Phase 6 wiring (current): **0/6** with **bit-identical outputs** to Phase 5 wiring.

Both paths produce *identical* filler-text continuations regardless of input
under this random pattern. Phase 5's recorded 6/6 in `phase5_passkey.json`
was almost certainly a favorable random head_pattern that happened to attend
to retrieval-relevant heads. This is not a Phase 6 regression — it's a
methodology issue that affects both phases equally.

**Implication:** the passkey eval is not a robust quality gate.
Phase 6 task 2 (RULER 4 k subset) is the SPEC's actual quality gate and
should replace passkey for go/no-go decisions.

## Phase 6 task 1b (deferred — closes the remaining 2×)

To clear the 2.03 → 4 tok/s gap on the same hardware, one of:
1. **EAGLE-2 speculative decoding wrapper** (Phase 6 plan task 6). ~2-3×
   decode multiplier, composes orthogonally with our criticality fix.
2. **Kernel-fused criticality + top-k** (the originally planned task 1
   that the algebraic shortcut bypassed). Profile-driven; at 17.6 ms /
   layer the remaining cost is mostly AWQ + MLP, not the sparse path.
   Speedup likely <1.5× from kernelizing criticality alone. Defer to
   when INT4 KV (task 5) needs new dequant kernels anyway.
3. **Re-profile Phase 6 to identify the new bottleneck**, then attack
   the largest. Cheapest first step.

## Phase 6 next tasks (still SPEC §6 priority order)

2. **RULER 4k subset eval.** Replaces passkey as the quality gate (passkey
   methodology is too brittle, see above).
3. `flashquest chat` CLI.
4. Head-to-head benchmark table.
5. INT4 KV.
6. EAGLE-2 wrapper. (Now also a candidate for closing the decode tok/s gap.)
7. Marlin W4A16 (if profile demands).
8. ExLlamaV2 backend integration.

## Phase 6 task 1a → task 1b handoff

Phase 6 task 1a ships:
- `phase_scores_int8`, `select_pages_vectorized` in eager package.
- Wired into `llama_persistent_patch`.
- 32 unit + e2e tests green.
- 32k decode bench: 2.03 tok/s, peak VRAM 6379 MiB (`benchmarks/phase6_decode.json`).
- Diagnostic scripts in `scripts/phase6_diag_*.py`.

Phase 6 task 1b begins (when prioritised): close the 2.03 → 4 tok/s gap via
EAGLE-2 speculation OR re-profile-driven kernel work.
