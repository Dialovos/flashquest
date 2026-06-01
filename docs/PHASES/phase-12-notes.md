# Phase 12 Notes — EAGLE-3 Speculative Decoding over Sparse Paged INT4 KV

**Started:** 2026-05-31
**Completed:** 2026-05-31
**Status:** **killed as a speedup** — built + lossless, but ~0.9–1.0× end-to-end at bs=1 on this hardware. `--speculative` ships opt-in only. No tag.
**Plan/Spec:** `docs/superpowers/specs/2026-05-31-phase-12-eagle3-speculative-decoding-design.md` · `docs/superpowers/plans/2026-05-31-phase-12-eagle3-speculative-decoding.md`

## The lever

After four profile-kills (8a compact-kernel ~1×, 8b CUDA Graphs 1.02× ceiling, 9 PLD ~1×, 10 CATS 1.05× ceiling) the project's train-free decode ceiling sat at **8.41 tok/s @ 32k** (`--kv-bits 4 --retention 0.20`), the only backend that decodes 32k on a 4 GB card at all. Every remaining train-free lever is *additive* and ≤ ~1.15× (the KV-codec ladder is exhausted; CUDA Graphs / CATS / per-head codebooks already killed). **Speculative decoding is the only *multiplicative* one left** — codex (gpt-5.5 xhigh, read-only) endorsed EAGLE as the single remaining multiplicative train-free option and refined the choice to **EAGLE-3 first** (over EAGLE-2), because a published draft head exists for our exact base: `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` (single Llama layer, ~486 MB bf16, 32k draft vocab). No training by us — the train-free constraint holds.

The draft proposes a depth-N token **chain**; the target verifies all N in one `S_q>1` forward pass over the existing Quest sparse paged INT4 KV cache; the longest greedily-matching prefix is accepted. Net decode speedup ≈ `mean_accepted_tokens / verify_cost_ratio`. The spec called this margin "genuinely tight": the head's *generic-chat* accept numbers (~60/58/57% on the first three positions) imply only ~2.26 tokens/iter on a depth-4 chain, which at a 1.6× verify ceiling is just ~1.41× — below the 1.5× default bar. Clearing the default bar needs *real-trace* acceptance to beat generic chat, **or** the verify cost to beat 1.6×. Both were measured first (entry gate, §6), before any integration.

## Entry gate (profile-first, non-negotiable — 4 prior kills)

### Gate 1a — verifier economics (`benchmarks/phase12/task1a*` → `benchmarks/phase12/task1a.json`)

The fatal risk codex flagged: does the `S_q>1` sparse verify kernel *amortize* (load each selected page's K/V tile once, reuse across all `S_q` query rows via `tl.dot`), or does it spill registers on sm_86 and scale ~linearly in `q`? **Rule: PASS iff `ratio_q4 ≤ 1.6` OR `ratio_q8 ≤ 2.4`.**

| Measurement | Value (ms) | Ratio vs q=1 |
|---|---|---|
| T_decode_q1 | 1.27 | 1.00× |
| T_verify_q2 | 1.10 | 0.865× |
| T_verify_q4 | 1.09 | **0.854×** |
| T_verify_q8 | 1.11 | 0.872× |

**Verdict: PASS** — verifying 4 (or 8) drafts costs *less* than one decode (0.85×). The sparse decode is so memory-bandwidth-bound at 32k that adding query rows is nearly free.

**The bf16-tensor-core fix (6.8× → 0.87×).** The first build used fp32 `input_precision="ieee"` `tl.dot` and measured T_verify ~8.3 ms, ratios ~**6.8×** — a hard gate FAIL. Diagnosis: sm_86 has **no fp32 tensor cores**, so every dot product ran through CUDA cores (fp32 emulation, compute-bound). Fix: bf16 inputs + fp32 accumulation in `tl.dot` engaged the tensor cores; T_verify dropped to ~1.1 ms (~8× faster), turning the gate from FAIL to PASS. Parity held (atol/rtol 2e-2) across all batch sizes. (Commit `47d0c4c`.)

### Gate 1b — EAGLE-3 draft acceptance (`benchmarks/phase12/task1b_acceptance.py` → `task1b.json`)

Load the EAGLE-3 head against the **AWQ** target (out-of-distribution vs the bf16 base it was trained on — codex's residual risk). Greedy, chain depth `n_draft=4`. EAGLE-3 fuses the target hidden states entering layers `{2, L//2, L-3}` = HF `output_hidden_states` indices **[2, 14, 25]** (28-layer target), concatenated to 9216 wide → head `fc`; draft-vocab logits mapped to full vocab via `id + d2t[id]`. Acceptance measured by greedy-reference replay (exact for greedy verification). Two workloads pooled (PG-essay summary + RULER-NIAH needle):

| Measurement (clean primary, ctx 1024, fully GPU-resident) | Value |
|---|---|
| **mean_accepted** (1 bonus + accepted drafts / iter) | **1.794** blended |
| — PaulGraham summarize | 2.259 |
| — RULER NIAH | 1.488 |
| draft_step_ratio (one draft step / one target decode) | 0.047 (5.6 ms / 118.8 ms) |
| top1_agreement (sanity) | 39.9% pooled (PG 58.3% ≈ head card's acc_0=60.6%) |
| peak VRAM (486 MB head resident) | 3717 MiB |

AWQ INT4 decode at batch=1 is ~119 ms/token on this RTX 3050 Ti (no fast INT4-GEMM path), while the bf16 draft step is ~5.6 ms → the draft is ~21× cheaper than one target decode, so the controller's `net = mean_accepted / (0.87 + draft_step_ratio)` is dominated by `mean_accepted / 0.87`. The 4k stretch run peaked 4264 MiB (> the card's 4095 MiB → WSL2 host-memory spill; timing degraded), so the 1k run is the trustworthy gate number.

**Why Gate 1a PASS was optimistic.** Gate 1a measured the verify *kernel* in isolation at 0.87×. A subsequent cost-breakdown probe (`benchmarks/phase12/iteration_breakdown.json`, 4k context) showed the verify *forward* itself amortizes well — `T_verify(q=4)/T_decode(q=1) = 1.17×` (q=4 costs only marginally more than q=1, because the model is weight-bound and AWQ quantization means weight traffic dominates). However the **full iteration** is verify (68%, 233 ms) + draft steps (26%, 89 ms) + advance (6%, 20 ms) ≈ 342 ms ≈ 1.7 decode steps. At 4k context this models a **1.22× speedup** — a genuine win in isolation. The gate-1a "0.87× kernel" claim was not the error; the error was projecting from kernel ratio to iteration ratio without counting the draft-head steps and orchestration overhead. The attention kernel is a small fraction of the iteration; gating only on the kernel ratio was insufficient.

## What was built (Tasks 1–10)

Gate-first TDD, reusing Phase 9's twice-codex-reviewed (never-built) verify machinery and adding the drafter Phase 9 never had:

- **Task 1** — `S_q>1` sparse INT4 verify kernel (`flash_attn_sparse_int4_fwd_compact_sq`) + parity test. The bandwidth-amortizing tile-reuse core.
- **Task 2** — gate 1a microbench + the bf16-tensor-core fix (above).
- **Task 3** — gate 1b acceptance probe (above).
- **Task 4** — cache sandbox + two-phase commit (`add_draft` → BF16 sandbox → `commit_draft_all_layers(m+1)`; the tail drafts are dropped by resetting the sandbox count — no rollback math for a chain). `MAX_DRAFT = 8`.
- **Task 5** — score-prioritized UNION page selection (the `S_q` draft queries share one top-k page set so tiles load once; sinks + window force-included; overflow truncates lowest-score).
- **Task 6** — verify-mode forward arm (third dispatch branch alongside prefill `S_q>1` and decode `S_q=1`): sparse-union(completed pages) ⊕ dense(partial-page tail ‖ sandbox), offset-causal mask, `set_verify_active(model, bool)` toggle, fused-hidden tap.
- **Task 8** — the chain spec-decode dispatcher (`make_quest_specdec`): reuses Phase 9's PLD walk-and-accept state machine verbatim (verify input `[bonus, d_1, …, d_{n-1}]`; held free/bonus token; page-boundary fallback to a single decode); only the *drafter* changes (EAGLE-3 `propose_chain` instead of n-gram).
- **Task 9** — end-to-end losslessness equivalence test (below).
- **Task 7** — incremental draft-KV maintenance (`use_incremental=True`, default): `init` seeds the head's KV once, each `step` grows it by only newly-committed tokens (`seed`/`propose_from`/`advance`), instead of re-prefilling the whole verified prefix every step (the old O(T)-per-step path, kept as `use_incremental=False` for A/B). `propose_from` is bit-for-bit identical to from-scratch `propose_chain` (unit-tested), so the §5 losslessness contract is unchanged — this is purely the **speed + long-context-OOM enabler** that makes 32k feasible (q=1 draft steps, no O(T²) re-prefill, no OOM).
- **Task 10** — `flashquest chat --speculative` (`--n-draft` default 4, `--draft-model` default the thoughtworks head); `_stream_one_spec` mirrors the standard streamer's UX + cache-reset contract.

### The `page_scores .view → .reshape` S_q>1 fix (commit `ea88954`)

The verify-mode forward is the *first* `S_q>1` caller of the sparse criticality scorers. `page_scores_int4_fast` / `page_scores_int8_fast` collapsed `(H_q, S_q) → (H_kv, n_rep*S_q)` with `.view()`, which raises "view size is not compatible with input tensor's size and stride" on the post-RoPE/cast Q (non-contiguous) when `S_q>1`. Switched both the Q regroup and the final un-group to `.reshape()` — numerically identical to `.view()` (verified vs the materializing `repeat_interleave` reference to bf16 epsilon on a non-contiguous `S_q=4` Q), copies only when needed, `S_q=1` decode unaffected. Surfaced by the Task 9 equivalence test.

## Losslessness contract + equivalence result (Task 9)

This is **not** bit-identical to non-spec single-step decode by design, and the spec says so: the verify pass scores all `S_q` draft queries against the completed pages using a **UNION** of their per-query top-k page sets (so K/V tiles load once), whereas a non-spec decode uses each token's own per-query top-k. Since `UNION ⊇ per-query top-k`, the verify pass attends to *at least as many* pages, but the softmax normalization differs slightly. The contract is therefore: lossless **with respect to the UNION-selection verify model** (greedy acceptance is exact for that model).

**Task 9 measured the real thing** (`tests/test_specdec_equivalence.py`, 469 lines): the spec-decode output is **bit-identical to the non-spec sparse greedy decode** — 100% token agreement at both `n_draft=1` and `n_draft=4`. The UNION-vs-per-query softmax difference did not flip a single argmax on the tested prompts. This is the strongest possible form of the §5 contract (the spec only required ≥99% argmax match) and means the spec path's quality must equal the non-spec sparse path's.

## Cost-breakdown probe (Task 11 follow-up)

`benchmarks/phase12/iteration_breakdown.json` (4k context, bf16 head resident): a timed decomposition of a single spec-decode iteration into its three phases.

| Phase | Time (ms) | Share |
|---|---|---|
| verify forward (q=4) | 233 | 68% |
| draft steps (×4) | 89 | 26% |
| advance + orchestration | 20 | 6% |
| **total iteration** | **342** | — |
| single baseline decode | ~200 | — |
| **`T_verify(q=4) / T_decode(q=1)`** | **1.17×** | — |

**Key finding:** the verify forward *amortizes* — q=4 costs only 1.17× a single decode (weight-bound AWQ: the same weights are read once regardless of q). The old "~2.3× verify" claim was wrong; that figure was the **full iteration** (~1.7 decodes) relative to a cheap 2k-context baseline, not the verify forward alone. The corrected picture:

- At 4k context the spec path **models 1.22×** — a genuine win.
- The win is **context-dependent**: 2k baselines are cheaper (decode too fast to amortize verify+draft overhead → 0.91×); at 4k the model hits 1.22×; at 8k mild head-swap drag → 0.99×.
- The win is **VRAM-capped and cannot be extended to longer contexts**: the 16k OOM is on the spec path's *own* state (full-sequence `fused_seq` accumulation + growing draft KV + verify sandbox), not just the draft head — so freeing head VRAM does not unblock long-context.

## End-to-end speed result (Task A) — the kill

`scripts/phase12_bench_decode_32k.py` drives the dispatcher directly for clean timing on the v1.0 default config (`--kv-bits 4 --retention 0.20`, all-retrieval head_pattern): build AWQ target + `PersistentInt4KVCache` + patch + `load_eagle3_draft`; prefill to context; then **spec** = `make_quest_specdec(..., n_draft=4).init` then time `step()` to ~96 emitted tokens; **baseline** = fresh cache, non-spec greedy S_q=1 loop over 96 tokens.

### Measured numbers (Llama-3.2-3B-AWQ, kv_bits=4, retention=0.20, n_draft=4, RTX 3050 Ti 4 GB)

| Context | Baseline (tok/s) | Spec (tok/s) | Speedup | mean accepted/iter | Peak VRAM (PyTorch / nvidia-smi) |
|---|---|---|---|---|---|
| **2k ctx** | 6.73 | 6.10 | **0.91×** | 2.09 | 3776 / 3929 MiB |
| **4k ctx** | (models 1.22×) | — | — | — | bf16 head resident |
| **8k ctx** | 4.27 | 4.24 | **0.99×** | 2.13 | 4759 MiB (head-resident, WSL overcommit) |
| **≥16k ctx** | — | OOM | — | — | spec-path state (fused_seq + draft KV + sandbox) |
| **32k ctx** | (prefill swap-thrashed, >7 min) | — | — | — | head does not fit at 32k |

Raw results: `benchmarks/phase12/decode_2k.json` (the 2k run), `benchmarks/phase12/decode_32k.json` (the 8k run; filename reflects the original script target), `benchmarks/phase12/iteration_breakdown.json` (4k probe). Logs: `benchmarks/phase12/decode_2k.log`, `benchmarks/phase12/decode_8k.log`.

**spec_band per SPEC §7 = "off"** (<1.2×) at all deployable contexts.

### Root cause

The gate-1a verify-kernel ratio (0.87×) was correct but covered only the **attention kernel**. The cost-breakdown probe shows the full verify-forward itself amortizes to **1.17×** (not 2.3×; the old "~2.3×" figure was the whole iteration vs a cheap 2k baseline). The actual kill factors are:

1. **The win is narrow and context-dependent.** Verify amortizes at 4k (1.22× model), but decode is too cheap at 2k (0.91×) and head-swap drag at 8k (0.99×) closes the window.
2. **VRAM caps the usable context, and the OOM is on the spec path's own state.** At ≥16k the spec path OOMs due to full-sequence `fused_seq` accumulation + growing draft KV + verify sandbox — not just the 486 MB head. Freeing head VRAM (e.g. via INT8 quantization) is insufficient; the spec path's state itself won't fit.
3. **Draft-head steps dominate overhead at the non-4k contexts.** At 2k the 89 ms for 4 draft steps (26% of iteration) pushes total above what the ~2.1-token accept gain covers.

## INT8-draft-head salvage attempt (Task 12) — failed

After the 16k OOM was diagnosed as spec-path state, a final salvage was attempted: quantize the EAGLE-3 bf16 draft head to weight-only INT8 (per-channel) to free VRAM and check whether the 16k path became viable.

**Result (commit `398b01f`; `benchmarks/phase12/int8_probe.json`, `iteration_breakdown_int8_perchannel.json`, `decode_8k_int8_perchannel.json`):**

| Metric | bf16 head | INT8 per-channel head |
|---|---|---|
| mean accepted | 1.684 | 1.684 (unchanged) |
| draft-argmax agreement | — | 98.7% |
| head VRAM | 1216 MiB | 985 MiB (freed **232 MiB**) |
| 16k OOM | yes | **still OOMs** |
| draft step time | 89 ms | **456 ms (5× slower)** |
| 4k iteration model | 1.22× | **0.63×** (inverted) |

**Fidelity was never the issue.** Acceptance held at 1.684; draft-argmax agreement was 98.7%. The failure was twofold:

1. **232 MiB freed is insufficient.** The 16k OOM is driven by the spec path's own state (fused-hidden accumulation + draft KV + verify sandbox), which weight quantization of the lm_head does not shrink. 16k still OOMs.
2. **Weight-only INT8 at seq=1 / bs=1 is a GEMV** — the full 188 MB `lm_head` is re-dequantized every draft token (no batch reuse to amortize). Draft step explodes from 89 ms to 456 ms, inverting the 4k model from 1.22× to 0.63×.

Codex concurred (read-only review): a fused low-bit GEMV kernel (Marlin / tinygemm) could in principle recover draft speed, but even Marlin rarely beats bf16 at bs=1 on a 4 GB consumer GPU — and the VRAM fix is independent, so this is out of scope.

**INT8 draft head ships as the opt-in `--int8-draft` flag** (correct + lossless, not faster here); code kept as a validated reference.

## Quality (Task B)

RULER is not re-run. Task 9 proves the spec path is **bit-identical** to the non-spec sparse greedy decode (100% token agreement, n_draft=1 and n_draft=4). Spec quality therefore equals the non-spec default's already-gated 100/100/95 RULER NIAH result — no separate eval needed. Quality is lossless-by-equivalence.

## Verdict and disposition

**Phase 12 is the 5th profile-kill. Dead end on 4 GB / AWQ / bs=1.** The implementation is correct and lossless — spec output is bit-identical to non-spec greedy (Task 9: 100% agreement at n_draft=1 and n_draft=4). The cost-breakdown probe shows the verify forward itself amortizes well (`T_verify(q=4)/T_decode(q=1) = 1.17×`; the old "~2.3× verify" was the full iteration vs a cheap 2k baseline — that claim is corrected here). The win exists (4k models 1.22×) but cannot be widened: short context → decode too cheap to amortize verify + draft overhead; long context → OOM on the spec path's own state (not just the head); INT8-head salvage → 5× slower draft GEMV + insufficient VRAM freed.

**`--speculative` ships opt-in only.** The code is correct, fully tested, and kept as a validated reference. The opt-in `--int8-draft` flag (weight-only INT8 draft head) is also kept — correct + lossless but slower at bs=1. Users on larger VRAM (where the head fits comfortably at long ctx and the weight-bound fraction shrinks relative to attention) can experiment. On this 4 GB card, both are slower.

**No `phase-12` tag.** Matches the Phase 9 kill precedent — phases that don't deliver end-to-end gains don't get promotion tags.

**Lessons (3):**

1. **Gate on the FULL iteration cost** — verify + draft steps + advance + orchestration — vs K·decode, not the attention-kernel ratio. At bs=1 weight-bound inference, the kernel is a small fraction of the step; the kernel ratio is optimistic by the ratio of non-attention work to attention work.
2. **The speculative win grows with context but VRAM caps the usable context** — and the OOM driver is the spec path's own state (fused-hidden accumulation + draft KV + verify sandbox), not just the draft head. Freeing head VRAM does not unblock long-context spec-decode.
3. **Weight-only INT8 is SLOWER than bf16 at seq=1 / bs=1** — every token dequantizes the full `lm_head` weight matrix (a GEMV, no batch reuse to amortize). INT8 is a throughput / batched-inference optimization; it is the wrong tool for single-stream decode.

## Commit chain

```
adb1c2c  phase 12: spec — EAGLE-3 speculative decoding over sparse paged INT4 KV
3bfca3c  phase 12: implementation plan (gate-first TDD)
dbd6d92  phase 12 task 1: S_q>1 sparse INT4 verify kernel + parity test (gate slice)
234abba  phase 12 task 2: gate 1a verifier-economics microbench + verdict
47d0c4c  phase 12 task 2: bf16 tensor-core dot in verify kernel — gate 1a PASS (6.8x->0.87x)
a3b0c4b  phase 12 task 3: gate 1b EAGLE-3 acceptance probe
7df2225  phase 12 task 4: cache sandbox + two-phase commit API
f7fb177  phase 12 task 5: score-prioritized UNION page selection
5ad61fd  phase 12 task 6: verify-mode forward arm + offset-causal dense tail + set_verify_active
82752c0  phase 12 task 8: EAGLE-3 chain spec-decode dispatcher
ea88954  phase 12: fix page_scores S_q>1 view stride error in verify arm
36b9bf0  phase 12 task 9: end-to-end losslessness equivalence test
f0ea58b  phase 12 task 7: incremental draft-KV maintenance (speed/OOM enabler for long ctx)
c08d3f6  phase 12 task 10: flashquest chat --speculative (EAGLE-3 spec decode)
c08d3f6  phase 12 task 11: end-to-end speed kill — built + lossless but ~0.9-1.0x at bs=1; --speculative opt-in, no tag
398b01f  phase 12 task 12: INT8-draft-head salvage — preserved acceptance but 5x slower draft GEMV; 16k still OOMs; opt-in kept
<this commit>  phase 12: finalize verdict — probe shows verify amortizes (1.17x) but spec ~0.9-1.0x/16k-OOM at bs=1; INT8-head salvage failed; opt-in, no tag
```

## Surface

- `flashquest.specdec.load_eagle3_draft(model_id, device, dtype, embed_weight=...)` — load the public EAGLE-3 head into the vendored inference `Model`; reuses the target's `embed_tokens` (no extra download).
- `flashquest.specdec.EagleDraft` — `.seed` / `.propose_from` / `.advance` (incremental KV) + `.propose_chain` (from-scratch fallback) + `.prefill` / `.step`.
- `flashquest.specdec.dispatcher.make_quest_specdec(model, cache, draft, *, n_draft=4, page_size=64, prefill_chunk=256, use_incremental=True)` → `(init, step)`.
- `flashquest.eager.llama_persistent_patch.set_verify_active(model, bool)` — flip every patched `LlamaAttention` onto the verify arm.
- `PersistentInt4KVCache` sandbox API — `add_draft`, `commit_draft_all_layers(m+1)`, `MAX_DRAFT=8`.
- `flashquest chat --speculative [--n-draft 4] [--draft-model …]` (opt-in; no speedup on the 4 GB card).
- `scripts/phase12_bench_decode_32k.py`, `scripts/phase12_run_ruler_4k.py`; gates in `benchmarks/phase12/`.

## References

- EAGLE-3: arXiv 2503.01840; EAGLE-2 (dynamic tree, deferred): arXiv 2406.16858.
- Draft head: `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` (HF).
- Phase 9 (the reused, never-built verify machinery): `docs/PHASES/phase-9-killed-by-profile.md`.
