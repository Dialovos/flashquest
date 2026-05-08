# Phase 8a Notes — Compact-List Kernel + Sync Cleanup + Fused-Proj Scaffolding

**Started:** 2026-05-07
**Completed:** 2026-05-08
**Status:** **complete (tag `phase-8a`); landed as graph-prep foundation for Phase 8b. Standalone throughput delta is ~0× — see "Honest finding" below. Quality unchanged (compact path is mathematically identical to bool-mask path, validated by parity tests).**
**Plan:** [`../superpowers/plans/2026-05-07-phase-8a-compact-kernel-fusion.md`](../superpowers/plans/2026-05-07-phase-8a-compact-kernel-fusion.md)
**Spec:** [`../superpowers/specs/2026-05-07-phase-8a-compact-kernel-fusion-design.md`](../superpowers/specs/2026-05-07-phase-8a-compact-kernel-fusion-design.md)

## Summary

Phase 8a was scoped (after a r1→r4 spec cycle with three codex review rounds) as the kernel + dispatch foundation for an OOM cumulative push: 8a → 8b graphs → 9 specdec → 10 DuoAttention → 11 lookahead. Predicted standalone ceiling: 1.29-1.50× at 32k decode.

**Empirical finding: the predicted standalone speedups did not materialize.** The compact-list sparse kernel runs at the same wall-time as the bool-mask kernel (~1.01× microbench at 32k decode shape), and the fused-proj wrapper, as scoped (Python wrapper around AutoAWQ Linear forwards), gives no measurable launch reduction either.

**What did land** is the **architectural foundation that Phase 8b graph capture will consume**:

- A graph-friendly compact-list sparse INT4 kernel with `selected_page_ids: int32[B, H_q, BUCKET_MAX]` + `-1` sentinel + load-time masking + `qk = NEG_INF` for invalid slots. Static-shape, ready for `torch.cuda.CUDAGraph` capture.
- An `.item()`-free selection path: `select_pages_vectorized(..., k_max_static=...)` removes the per-step CPU sync that would otherwise break graph capture.
- A GPU-resident `build_compact_selection` (sort-based, no `.item()`) that bridges the existing bool-mask selection output to the new compact-list kernel input.
- Verified AWQ layout (`qweight (in, out//8)` packed along OUT axis, not IN as I initially guessed — codex r3 #11 caught the assumption mismatch via the audit step).
- Flag-gated `--use_compact_kernel` integration in `patch_llama_for_quest_persistent` (default OFF; INT8/Turbo paths raise NotImplementedError pending Phase 8b).

## Honest finding — why the predicted speedup did not materialize

### Compact kernel @ 32k decode shape (Task 5 microbench)

`benchmarks/phase8a_microbench_kernel.py`:

| kernel | µs/call | speedup |
|---|---|---|
| bool-mask `flash_attn_sparse_int4_fwd` (NUM_PAGES=512) | 1503 | 1.00× (baseline) |
| compact-list `flash_attn_sparse_int4_fwd_compact` (BUCKET_MAX=134) | 1491 | **1.01×** |

The bool-mask kernel iterates 512 pages with `if is_sel:` branch but **already skips non-selected pages cheaply**: the branch is uniform across all CTA threads (every thread sees the same `is_sel` value for a given page), so there's no thread-divergence penalty — it's just one bool load + skip per non-selected page.

The compact-list kernel iterates 134 pages and skips the bool-mask branch. Loop control savings are **~1% of kernel time**, not the 1.3-1.6× I'd assumed. The expensive work — K/V tile loads (memory bandwidth) + dequant + softmax + V loads — happens 128 times in BOTH versions because both versions only do real work for the 128 selected pages.

`num_warps=8 num_stages=3` regresses to 0.75× (Ampere SM_86 with PAGE_SIZE=64 has only 64 elements/load; 256 threads/CTA leaves most idle). Reverted to 4/2.

### Fused-proj wrapper

`fused_qkv_proj` and `fused_gate_up_proj` literally call `q_proj(...)`, `k_proj(...)`, `v_proj(...)` in sequence inside a Python function. No GEMM fusion happens — each AWQ Linear forward still launches its own INT4 kernel. The wrapper is mathematically identical to the inline calls; **no launch reduction, no win**.

The original spec described a "Triton fused QKV kernel" reading 3 source weight tensors. Writing such a kernel from scratch to beat AutoAWQ's hand-tuned INT4 GEMM at decode batch=1 is a large undertaking — the AutoAWQ GEMM is already memory-bandwidth bound at batch=1, so a custom Triton kernel would not deliver meaningful gains. (Phase 6 notes line 148 already documented this for Marlin: "Marlin at M=1 ≈ AWQ. Marlin's design point is M=16-32." The same bandwidth-bound argument applies to a custom Triton fusion at our workload.)

The plan scope-reduced this to a Python wrapper, accepting that the standalone speedup would be 0×, on the rationale that the wrapper would be a useful integration point for Phase 8b graph capture. That's still true; it's just not a Phase 8a speedup.

### Sync removal in selection

`select_pages_vectorized` previously did `int(k_per_h.max().item())` per call — a CPU sync. Phase 8a precomputes `k_max_static` at module init from `retention × P_max` (where `P_max = cache.max_seq_len // page_size`) and passes it in, eliminating the per-step `.item()`.

The win here is also small standalone: a `.item()` is ~30 µs, but it overlaps with subsequent CUDA work on the default stream because it's the only sync in the hot path. Real benefit emerges in Phase 8b when **all** syncs in the hot path must be eliminated for `torch.cuda.CUDAGraph` capture to work.

## Why the spec was wrong

The r1 spec assumed bucket-padded bool masks would reduce kernel work — codex r2 caught this and led to the compact-list rewrite. The r2/r3/r4 spec then assumed the compact-list rewrite itself would deliver ~1.3-1.6× from "fewer loop iterations." This was wrong because:

1. The bool-mask kernel already does ~no work on non-selected pages (uniform branch, no divergence).
2. Memory bandwidth is the bottleneck (loading K/V tiles for 128 selected pages × 64 elements × INT4); both kernel variants pay the same bandwidth cost.
3. Loop control overhead (the actual delta between the kernels) is a small fraction of total kernel time on Ampere.

The fix would have been: actually run the microbench BEFORE writing the spec, not after. Codex's r3 review was suspicious of the speedup claim ("Marlin at M=1 ≈ AWQ" generalization should have flagged this) but didn't reject the compact-kernel claim outright. Reaffirms: **microbench before predicting throughput**.

## Result — quality

Compact path is mathematically equivalent to bool-mask path (with sentinel padding contributing 0 to softmax via `qk = -inf`). All parity tests pass:

- `tests/test_sparse_int4_fwd_compact_parity.py` — 3 random-seed parity tests, max abs err < 1e-2 BF16.
- `tests/test_sparse_int4_fwd_compact_padding.py` — sentinel padding equivalence, all-sentinel zero output + lse=-inf.
- `tests/test_sparse_compact_kernel_address_safety.py` — no OOB on `p=-1` sentinel.
- `tests/test_persistent_patch_compact.py` — full-forward parity on toy 2-layer Llama, kv_bits=4.

RULER NIAH 4k @ Llama-3.2-3B-AWQ NOT re-run because the compact path produces mathematically identical output to the bool-mask path — Phase 7's RULER scores carry over unchanged. (RULER would only be needed if the compact path numerics deviated, which the parity tests prove they don't.)

## Surface

- `flashquest.kernel.sparse_int4_fwd_compact` — compact-list INT4 sparse Triton kernel + Python wrapper
- `flashquest.kernel.fused_proj.{fused_qkv_proj, fused_gate_up_proj}` — launch-reduction wrappers (currently no-op standalone; integration point for Phase 8b)
- `flashquest.eager.selection.build_compact_selection(mask, BUCKET_MAX)` — bool mask → int32 compact list (sort-based, GPU-resident, dedup-naturally)
- `flashquest.eager.selection.select_pages_vectorized(..., k_max_static=...)` — `.item()`-free path
- `flashquest.quant.awq_layout.{AWQLayout, assert_awq_layout, AWQ_GROUP_SIZE, AWQ_PACK_FACTOR}` — AWQ tensor layout constants + load-time assertion (verified against Llama-3.2-3B-AWQ on 2026-05-08)
- `patch_llama_for_quest_persistent(..., use_compact_kernel=False)` — flag-gated dispatch (default OFF; kv_bits=4 only)

## Tasks landed vs. deferred

| Task | Status | Note |
|---|---|---|
| 1 — AWQ layout audit | ✅ landed | caught codex r3 #11 layout assumption mismatch |
| 2 — k_max_static precompute | ✅ landed | sync removed; standalone gain small |
| 3 — build_compact_selection | ✅ landed | GPU-resident, no .item() |
| 4 — Compact INT4 kernel | ✅ landed | parity verified; standalone ~1× per Task 5 |
| 5 — Compact INT4 microbench | ✅ landed | gate FAIL documented honestly |
| 6 — Compact INT8 + Turbo kernels | ⏸️ deferred to Phase 8b | same standalone ~0× expected; will land when graphs need them |
| 7 — Wire compact kernel into Llama patch | ✅ landed | flag-gated, kv_bits=4 only |
| 8 — Ablation 1 (compact-only bench) | ⏸️ deferred | result predicted to be ~1× from microbench; not worth wallclock |
| 9 — Fused QKV wrapper | ✅ landed | as scaffolding (no real fusion happens) |
| 10 — Fused gate+up wrapper test | ✅ landed | parity confirmed |
| 11 — Wire fused proj into Llama patch | ⏸️ deferred to Phase 8b | wrapper is currently no-op; integrating a no-op flag is anti-YAGNI |
| 12 — RULER + ablation 2 + writeup | ✅ writeup landed; benches deferred | RULER unchanged via parity proof |

## Roadmap implications

Phase 8a delivered foundation work but no throughput. Phase 9-11 cumulative ceiling re-estimation:

- **Phase 8b — cache view redesign + bucketed CUDA Graphs.** This is now THE phase that delivers wall-time savings. Estimated ~1.3-1.4× from Python+launch overhead elimination. Brainstorm starts after this phase tag.
- **Phase 9 — speculative decoding (EAGLE-3 / Medusa-2).** Biggest single win on the roadmap (~2.5-3.5×). Phase 6 notes flagged 2-3 weeks of cache-bridge work; Phase 8b's cache redesign should overlap cleanly with that.
- **Phase 10 — DuoAttention head split + CATS.** ~1.5× expected. Per Phase 6 notes, Llama-3.2-3B head pattern needs training (no upstream pattern exists).
- **Phase 11 — lookahead/Jacobi + prompt-lookup drafting.** ~1.3-1.5× general, up to 5× for retrieval/code workloads.

Cumulative honest re-estimate (with Phase 8a delivering ~1.0×, not the planned 1.5×): ~6-9× cumulative target on top of Phase 6's 3.88 tok/s @ 32k → ~25-35 tok/s @ 32k cumulative. Lower than the 50 tok/s target from the original brainstorm, but honest given Phase 8a's gate-fail finding.

## Process notes

- The audit step (Task 1) caught a real bug (codex r3 #11 was right about AWQ packing axis). **Always audit external library tensor layouts before writing kernels against them.**
- Microbenching should happen BEFORE writing the spec, not after. The compact-kernel claim was an architectural pretty-print that didn't survive contact with `triton.jit` reality. Codex's review process didn't catch this — codex only sees code/specs, not measurements. **Empirical gates need empirical evidence, not just static analysis.**
- The spec went r1 → r4 with three codex review rounds because the early specs piled assumptions on top of assumptions. Splitting Phase 8 into 8a + 8b (after codex r2) was the right call; even with that split, Phase 8a's standalone-speedup story didn't survive measurement. Future specs should start with a microbench task to validate the underlying speedup model BEFORE building integration scaffolding around it.
- "Continue with reduced expectations" (user's call after Task 5 microbench failed) was the right strategic move: keep the foundation work that's necessary for 8b graphs, drop the pieces that don't compose to a real win standalone. Phase 8a as committed is structurally sound; the ceiling claim was wrong, not the architecture.

## v2 follow-ups

- **Phase 8b brainstorm** — cache view redesign (fixed-max-size + length-mask tensors, no Python-int slicing) + bucketed CUDA Graph dispatcher. The compact-list kernel + sync removal landed in 8a are the prerequisite. **This is the next session's focus.**
- **True Triton fused INT4 GEMM** — only worth pursuing if Phase 9 specdec raises effective M to 8-16, where Marlin and custom fused INT4 kernels start beating AutoAWQ.
- **Compact INT8 + Turbo kernels** — write when 8b graphs need them per kv_bits.
- **`partial_len`/`completed_len` GPU-resident scalar tensors + predicated partial-merge** — required for 8b graph capture (current Python-int branches break graph capture).
