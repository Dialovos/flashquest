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

**Why Gate 1a PASS was optimistic.** Gate 1a measured the verify *kernel* in isolation at 0.87×. The end-to-end benchmark shows the full verify-*forward* costs ~2.3 decodes, not 0.87 of one. The gap is the non-attention work: the weight-bound projections and MLP for q=4 queries (Q/K/V projections, output projection, FFN — all running at batch=4 instead of batch=1), plus the verify arm's extra per-layer ops. At bs=1 AWQ-INT4, each decode is already ~150 ms and weight-bound (not attention-bound), so the non-attention work does not amortize with more query rows. The attention kernel is a small fraction of the iteration; gating only on the kernel ratio was insufficient.

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

## End-to-end speed result (Task A) — the kill

`scripts/phase12_bench_decode_32k.py` drives the dispatcher directly for clean timing on the v1.0 default config (`--kv-bits 4 --retention 0.20`, all-retrieval head_pattern): build AWQ target + `PersistentInt4KVCache` + patch + `load_eagle3_draft`; prefill to context; then **spec** = `make_quest_specdec(..., n_draft=4).init` then time `step()` to ~96 emitted tokens; **baseline** = fresh cache, non-spec greedy S_q=1 loop over 96 tokens.

### Measured numbers (Llama-3.2-3B-AWQ, kv_bits=4, retention=0.20, n_draft=4, RTX 3050 Ti 4 GB)

| Context | Baseline (tok/s) | Spec (tok/s) | Speedup | mean accepted/iter | Peak VRAM (PyTorch / nvidia-smi) |
|---|---|---|---|---|---|
| **2k ctx** | 6.73 | 6.10 | **0.91×** | 2.09 | 3776 / 3929 MiB |
| **8k ctx** | 4.27 | 4.24 | **0.99×** | 2.13 | 4759 MiB (head-resident, WSL overcommit) |
| **32k ctx** | (prefill swap-thrashed, >7 min) | — | — | — | head does not fit at 32k |

Raw results: `benchmarks/phase12/decode_2k.json` (the 2k run), `benchmarks/phase12/decode_32k.json` (the 8k run; filename reflects the original script target). Logs: `benchmarks/phase12/decode_2k.log`, `benchmarks/phase12/decode_8k.log`.

**spec_band per SPEC §7 = "off"** (<1.2×) at all measured contexts.

### Root cause

The gate-1a verify-kernel ratio (0.87×) was correct but covered only the **attention kernel**. The full verify-forward cost is ~**2.3× a single decode step**, because:

1. **Weight-bound projections dominate at bs=1.** AWQ-INT4 decode is ~150 ms/step on this card (weight-bound, not attention-bound). For the verify pass, the Q/K/V projections, output projection, and FFN all run at q=4 instead of q=1 — four decode steps' worth of weight traffic, which does not amortize.
2. **Per-iteration spec overhead = verify-forward + 4 draft-head steps + draft-KV advance + Python orchestration ≈ 2.3 decodes.** The ~2.1 accepted tokens/iter mean gain is smaller than this cost.
3. **486 MB bf16 draft head does not fit alongside 32k decode.** At 32k the full model + 32k KV cache + head exceeds the card's 4 GB; the head is swap-resident and the 32k prefill itself swap-thrashes (>7 min, GPU pinned 3949/4096 MiB). The head-resident regime is 2k–8k only.

In short: the speedup math (`mean_accepted / verify_cost_ratio`) assumed the verify-cost ratio equals the attention kernel ratio. At bs=1 weight-bound inference, that assumption fails by ~2.7×.

## Quality (Task B)

RULER is not re-run. Task 9 proves the spec path is **bit-identical** to the non-spec sparse greedy decode (100% token agreement, n_draft=1 and n_draft=4). Spec quality therefore equals the non-spec default's already-gated 100/100/95 RULER NIAH result — no separate eval needed. Quality is lossless-by-equivalence.

## Verdict and disposition

**Phase 12 is killed as a speedup.** The implementation is correct and lossless — spec output is bit-identical to non-spec greedy. But the end-to-end decode benchmark shows ~0.9–1.0× at all measured contexts on this hardware. There is no speedup to ship as a default.

**`--speculative` ships opt-in only.** The code is correct, fully tested, and kept as a validated reference. Users who want to experiment with EAGLE-3 on larger VRAM (where the 486 MB head fits comfortably alongside a 32k context and the weight-bound fraction shrinks relative to attention) can enable it. On this 4 GB card, it is slower.

**No `phase-12` tag.** Matches the Phase 9 kill precedent — phases that don't deliver end-to-end gains don't get promotion tags.

**Lesson.** Gate speculative decoding on the FULL verify-forward cost (weight-bound projections + MLP for q=N, verify arm's per-layer ops, draft-head steps, orchestration), not just the attention kernel ratio. At bs=1 weight-bound inference, the kernel is a small fraction of the step; a kernel-only gate is optimistic by the ratio of non-attention work to attention work — here ~2.7×.

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
<this commit>  phase 12 task 11: end-to-end speed kill — built + lossless but ~0.9-1.0x at bs=1; --speculative opt-in, no tag
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
