# Phase 12 — EAGLE-3 speculative decoding over sparse paged INT4 KV — Design

**Status:** drafted 2026-05-31; pending user review.
**Plan:** to be written by `superpowers:writing-plans` after spec approval.
**Phase:** 12 (after Phase 11/11c calibrated codebook; realizes the long-deferred SPEC §5.9 / §6 item 8 "EAGLE-2 wrapper").
**Target:** ≥1.5× decode @ 32 k on Llama-3.2-3B-AWQ (8.41 → ≥ ~12.6 tok/s), lossless-by-contract, RTX 3050 Ti Laptop sm_86 4 GB WSL2.

**Refs:**
- EAGLE repo (draft model code, authoritative): https://github.com/SafeAILab/EAGLE — `eagle/model/{ea_model,cnets}.py`. EAGLE-3: arXiv 2503.01840; EAGLE-2 (dynamic tree): arXiv 2406.16858. Already in `docs/REFERENCES.md` (EAGLE-2, EMNLP 2024).
- **Pretrained draft head for our exact base:** `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` (HF) — ~250 M params, single-layer, 486 MB safetensors, 32 k draft vocab, reported ~60/58/57 % accept on the first three draft positions. Trained against `meta-llama/Llama-3.2-3B-Instruct` (our target is `casperhansen/llama-3.2-3b-instruct-awq` = the AWQ of that base).
- **Reusable prior design — the spine of this phase:** Phase 9 PLD spec `docs/superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md` (§4.2 sandbox/commit API, §4.3 verify-mode forward branch, §4.4 `S_q>1` compact kernel + UNION selection + LSE-merge) and plan `docs/superpowers/plans/2026-05-09-phase-9-pld-greedy-chain.md`. Kill record: `docs/PHASES/phase-9-killed-by-profile.md` (killed at the n-gram-accept gate, not the verify machinery).
- Codex consult (this session, gpt-5.5 xhigh, read-only): endorsed EAGLE as the only remaining multiplicative train-free lever; refined EAGLE-2→**EAGLE-3 first**; identified `q>1` verify economics as the true fatal risk; supplied the verifier-cost gate thresholds adopted in §6.
- Discipline: `feedback_profile_before_speedup_specs.md` (4 prior profile-kills), `project_v1_ceiling.md` (train-free ceiling; EAGLE-class speculation is the one structural lever not yet pulled).

---

## §1 Goal

Break the v1.0 train-free decode ceiling (8.41 tok/s @ 32 k, the only backend that decodes 32 k on 4 GB) with **lossless-by-contract EAGLE-3 speculative decoding**, using the *published* draft head — no training by us. The draft proposes a depth-N token **chain**; the target verifies all N in one `q>1` forward pass over the existing Quest sparse paged INT4 KV cache; the longest greedily-matching prefix is accepted.

**Why this is worth a phase despite the ceiling memo's "5–15 % per phase."** Every other remaining train-free lever is *additive* and ≤ ~1.15× (KV-codec ladder is exhausted; CUDA Graphs, CATS, per-head codebooks already killed). Speculation is the only *multiplicative* one. Passing both entry-gate thresholds (§6) implies a net decode speedup of `mean_accepted_tokens / verify_cost_ratio` ≈ **1.4–1.7×** for a chain — 3–5× larger than anything else on the table — and the EAGLE-2 dynamic tree (deferred, §8) is the path to 2–3×. The entry gate makes a non-result cheap to discover.

**Throughput target (chain, gate-conditional):** ≥1.5× over 8.41 → **≥ ~12.6 tok/s @ 32 k** clean earns the default; ≥1.3× clears the opt-in floor (§6/§7). (Phase 9's "clean" 32 k baseline was extrapolated at 6.29 tok/s under no host contention; the published v1.0 headline is 8.41. We benchmark against 8.41 and report both.)

## §2 Context — what this reuses and what is new

**Reused from Phase 9 (designed + twice-codex-reviewed, never built — `specdec/` does not exist, the sandbox API is absent from `persistent_int4.py`):**
- `S_q>1` compact INT4 verify kernel (`_sparse_attn_fwd_kernel_int4_compact_sq_gt_1`, Phase 9 §4.4): loads each selected page's K/V tile **once** and reuses it across all `S_q` query rows via `tl.dot` — the memory-bandwidth amortization that makes verifying N tokens cost far less than N decodes.
- Cache sandbox + two-phase atomic commit (Phase 9 §4.2): `add_draft` → BF16 sandbox → `commit_draft_all_layers(accept_count)` (commit accepted prefix, drop the rest by resetting sandbox count — **no rollback math needed for a chain**).
- Verify-mode forward branch (Phase 9 §4.3): third dispatch arm in `make_quest_persistent_forward` alongside prefill (`S_q>1`) and decode (`S_q=1`).
- Score-prioritized UNION page selection + per-query LSE-merge (Phase 9 §4.4), including the page-boundary commit guard.

**New in Phase 12 (the part Phase 9 never had — a drafter that actually accepts):**
- EAGLE-3 draft head: load the public single-layer checkpoint, run it over fused target hidden states, emit a depth-N chain.
- Wire the target's fused low/mid/high hidden states out to the drafter (EAGLE-3 consumes a 3-layer feature fusion, unlike EAGLE-1/2's last-hidden+embedding).
- A small dense BF16 KV cache for the one draft layer.
- `flashquest chat --speculative`.

## §3 Approach overview

Steady-state cycle (chain, depth `N_draft`, e.g. 4):
1. **Draft.** From the last accepted token + fused target hidden states, the EAGLE-3 head autoregressively proposes `[D_1, …, D_{N_draft}]` (its own tiny dense KV; cheap).
2. **Verify.** Feed `[D_0, D_1, …, D_{N_draft-1}]` (`S_q = N_draft`, `D_0` = the already-decided last token) through the target in one forward pass. Per layer: write K/V to the BF16 sandbox; sparse-attend over completed pages with UNION selection; dense-attend over (committed partial-page tail ‖ sandbox); LSE-merge per query.
3. **Accept.** `argmax(logits[i])` predicts position `i+1`. Walk `i = 0…N_draft-2`; accept while `argmax(logits[i]) == D_{i+1}`; stop at first mismatch. Accept the matched prefix + one "free" correction token.
4. **Commit.** `commit_draft_all_layers(M+1)` writes only accepted K/V to the cache; the sandbox tail is discarded. Continue.

This is Phase 9 §3's cadence verbatim, with `propose_draft` (n-gram) replaced by the EAGLE-3 head. The free-token hold + forced-single-decode bookkeeping and the page-boundary guard carry over unchanged.

## §4 Architecture — components touched

| Component | Change |
|---|---|
| `src/flashquest/specdec/__init__.py` (new) | Package init (Phase 9's intended namespace; never created). |
| `src/flashquest/specdec/eagle_draft.py` (new) | Load `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` safetensors; single draft-layer forward + draft-vocab↔full-vocab map (`d2t`/`t2d`); `propose_chain(hidden_states, last_token, n_draft) -> LongTensor[N]`. Draft code cribbed from `SafeAILab/EAGLE` `cnets.py`, not SGLang. |
| `src/flashquest/specdec/dispatcher.py` (new) | `init/step` generation loop = Phase 9 §4.5 with `propose_chain` swapping `propose_draft`; walk-and-accept; `_set_verify_active` try/finally guard. |
| `src/flashquest/cache/persistent_int4.py` | Add Phase 9 §4.2 sandbox + commit API (`MAX_DRAFT`, `K/V_sandbox`, `add_draft`, `get_views_with_sandbox`, `preflight_commit`, `commit_draft`, `commit_draft_all_layers`). ~1 MB sandbox footprint. |
| `src/flashquest/kernel/sparse_int4_fwd_compact.py` | Add `_sparse_attn_fwd_kernel_int4_compact_sq_gt_1` (Phase 9 §4.4) + `S_q>1` wrapper. The existing `S_q=1` compact kernel stays byte-identical (decode path unchanged). |
| `src/flashquest/eager/selection.py` | Add `build_compact_union_selection` (Phase 9 §4.4): score-prioritized UNION, sinks + window force-included, sentinel-padded, GPU-resident (no CUDA sync). |
| `src/flashquest/eager/llama_persistent_patch.py` | Add the verify-mode forward arm (Phase 9 §4.3) and expose the fused hidden states EAGLE-3 consumes (it already receives `hidden_states` per layer). |
| `src/flashquest/runtime/chat.py` | `--speculative` (opt-in), `--n-draft` (default 4), `--draft-model` (default the thoughtworks head). Plumbing only. |

**Draft KV.** The single draft layer needs its own KV across the 32 k context. At 1 layer × 8 KV heads × 64 dim × 32 k × (K+V) × 2 B ≈ **67 MiB** BF16 — small. Reuse an HF `DynamicCache` for the draft or a thin dense buffer; INT8 it only if §6 VRAM is tight.

## §5 Losslessness contract (honest framing — inherited from Phase 9 §1)

This is **not** bit-identical to non-spec single-step decode, and the spec says so. The verify pass scores all `S_q` draft queries against the completed pages using a **UNION** of their per-query top-k page sets (so K/V tiles load once). A non-spec decode would instead use each token's own per-query top-k. Since `UNION ⊇ per-query top-k`, the verify pass attends to *at least as many* pages (≥ per-query coverage), but the softmax normalization differs slightly. Therefore:

- **Contract:** lossless **with respect to the UNION-selection verify model** (greedy acceptance is exact for that model), and empirically **argmax matches non-spec sparse greedy ≥ 99 %** on representative prompts (verified by the §9 end-to-end equivalence test; reported per-workload in `phase-12-notes.md`).
- **Quality is re-gated, not assumed:** RULER NIAH 4 k must hold at the v1.0 default (single 100 %, multikey 100 %, multivalue ≥ 95 %). If it regresses, the phase ships opt-in or is killed.
- **Rejected alternative:** force exact per-query selection in verify (loop the kernel per draft position). Recovers bit-identity but reloads tiles per query → kills the bandwidth amortization that is the entire point. Revisit only if §6 Task 1a shows large headroom.

## §6 Entry gate — profile-first (non-negotiable; 4 prior kills)

Per `feedback_profile_before_speedup_specs.md`, **no kernel/draft integration is built until both sub-gates pass.** Codex flagged `q>1` verify economics as the true fatal risk (memory and "no head exists" are largely retired); both numbers below are measured first.

### Task 1a — verifier economics (`benchmarks/phase12/task1a_verify_economics.py`)
Build the *minimal* `_sparse_attn_fwd_kernel_int4_compact_sq_gt_1` (this is a real first slice of the eventual kernel, not throwaway). On a 32 k INT4 cache with synthetic known-greedy candidates (no draft head yet), measure `T_verify(q ∈ {2,4,8})` vs `T_decode(q=1)`, plus peak VRAM and Triton occupancy/spill report.
- **PROCEED iff** `T_verify(q=4) ≤ ~1.6× T_decode(q=1)` **or** `T_verify(q=8) ≤ ~2.4×`.
- **Watch:** Phase 9 §4.4 flagged `SQ_MAX=16` register pressure (~80–100 KiB live/CTA) may spill on sm_86. If it spills and verify scales ~linearly in `q`, **kill** and pivot to Lookahead decoding (~1.8×, draft-free, exact — codex's ranked-second option).

### Task 1b — acceptance (`benchmarks/phase12/task1b_acceptance.py`)
Load the EAGLE-3 head against the **AWQ** target (out-of-distribution vs the bf16 base it was trained on — codex's residual risk #3). Measure mean accepted tokens/iteration on **real 32 k traces** (PG-essay summarize + RULER NIAH single & multivalue), greedy, plus peak VRAM with the 486 MB head resident.
- **PROCEED iff** mean accepted ≥ **2.2** tokens/iter (chain) **and** peak VRAM fits (≤ ~6.0 GiB incl. WSL overcommit; else fall back to INT8/INT4 draft-head quant and re-measure).
- Report the acceptance histogram + the AWQ-vs-bf16 acceptance delta.

**Combined go/no-go.** Project net decode speedup ≈ `mean_accepted / verify_cost_ratio(chosen q)`. The margin is genuinely tight: the head's *generic-chat* numbers (~60/58/57 %) imply only ~2.26 accepted tokens/iter on a depth-4 chain, which at the verify-cost ceiling (1.6×) is just **~1.41× — below the default bar.** Clearing it needs the *real-trace* acceptance to beat generic chat (plausible — long-context summarize/retrieval echoes the context verbatim) **or** the measured verify cost to beat the 1.6× ceiling. That uncertainty is exactly why both numbers are measured on real traces before any build, and why a deeper chain / the EAGLE-2 tree (§8) may be needed for a decisive win. **Build iff projected ≥ ~1.3×** (the §7 opt-in floor); below that the added complexity isn't worth it — document the kill in `docs/PHASES/phase-12-killed-by-gate.md` and pivot (Lookahead ~1.8×, or stop).

## §7 Success criteria

| Axis | Bar | Kill / downgrade if |
|---|---|---|
| Decode tok/s @ 32 k (clean) | **≥ 1.5×** over 8.41 → ≥ ~12.6 tok/s earns the default | 1.2–1.5× → ship `--speculative` opt-in only; < 1.2× → under-delivered vs the gate projection, keep the kernel but disable by default (or revert). |
| Losslessness | argmax matches non-spec sparse greedy **≥ 99 %** on PG-summarize + RULER traces | < 99 %, or any catastrophic >1-token drift on retrieval → kill. |
| Quality (RULER NIAH 4 k, n=20, ret 0.20) | single 100 %, multikey 100 %, multivalue ≥ 95 % (= v1.0 default) | multivalue < 95 % → opt-in only. |
| Peak VRAM @ 32 k | ≤ ~6.0 GiB incl. draft head (vs current 5478 MiB) | over budget even with draft-head quant → kill. |
| Draft head | loads from the public checkpoint; **no training by us** | no head loads/maps cleanly against the AWQ target → descope (training the head breaks the train-free constraint → out of scope). |

If decode + losslessness + quality all clear, **`--speculative` becomes the default** at 32 k; `--no-speculative` stays as the deterministic-baseline escape hatch.

## §8 Non-goals / follow-ups

- **EAGLE-2 dynamic tree.** Chain only in Phase 12 (simpler verify; reuses Phase 9's chain machinery). The tree (tree-attention mask in the verify kernel, dynamic draft expansion) is the path to 2–3× — Phase 12b/13 if the chain clears the gate.
- **Sampling.** Greedy only. Speculative sampling (Leviathan/Chen rejection) is deferred (same stance as Phase 9).
- **Training our own draft head** (EAGLE-3 fine-tune). Breaks the train-free constraint; only revisited if no public head works against the AWQ target.
- **VocabTrim / draft-latency optimization** (arXiv 2506.22694) — a follow-on *on top of* speculation if the draft step turns out memory-bound; not a Phase 12 dependency.
- **TurboQuant K3-V3 / INT8 verify paths.** INT4 only, matching Phase 8a's compact-kernel constraint.
- **Other models / 8 B / 128 k.** Out of scope; orthogonal to this lever.

## §9 Testing strategy

| Test | Checks | Where | Marker |
|---|---|---|---|
| `S_q>1` kernel vs dense reference parity | Verify kernel ≡ a BF16 dense reference over the same pages (within tol); tile-reuse + UNION mask correct | `tests/test_sparse_int4_compact_sq_gt_1.py` | (not slow) |
| UNION selection helper | Sinks + window force-included; overflow truncates lowest-score; sentinel padding; no CUDA sync | `tests/test_union_selection.py` | (not slow) |
| Sandbox + commit semantics | `add_draft`/`commit_draft(M+1)` advances `_seen_tokens` by exactly M+1; discard tail; two-phase atomicity; page-boundary guard raises | `tests/test_persistent_int4_sandbox.py` | (not slow) |
| EAGLE draft load + forward | Checkpoint loads; draft-layer output shape; `d2t`/`t2d` vocab map round-trips | `tests/test_eagle_draft.py` | (not slow) |
| Dispatcher walk-and-accept | Synthetic logits → correct accept count, free-token hold, committed-history bookkeeping | `tests/test_specdec_dispatcher.py` | (not slow) |
| End-to-end losslessness | `--speculative` argmax stream matches `--no-speculative` greedy ≥ 99 % on fixed seeds/prompts | `tests/test_specdec_equivalence.py` | `slow` |
| `chat --speculative` smoke | Streams coherent tokens end-to-end on Llama-3.2-3B-AWQ | `tests/test_chat_speculative_smoke.py` | `slow` |
| Entry-gate probes (1a, 1b) | Documented kill-or-proceed verdict; gates spec→plan→build | `benchmarks/phase12/task1{a,b}_*.py` + `benchmarks/phase12/gate.md` | manual |
| RULER + decode bench @ 32 k | Quality + speed gates (§7) | `scripts/phase12_run_ruler_4k.py`, `scripts/phase12_bench_decode_32k.py` | manual |

CI runs `pytest -m "not slow"` only (existing pattern). Slow + manual gates run during the verification pass.

## §10 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| `q>1` sparse verify doesn't amortize on sm_86 (register spill → ~linear in q) | **High** (the fatal one) | Task 1a measures directly before any integration; cheap kill → Lookahead pivot. Phase 9 §4.4 already analyzed `SQ_MAX={8,16}` + `tl.dot` ≥16 constraint. |
| AWQ-INT4 hidden states OOD vs the head's bf16 training → acceptance sags | Medium | Task 1b measures on the real target; verification stays exact (low accept only lowers speedup, never corrupts output). |
| 486 MB draft head + draft KV doesn't fit 4 GB | Medium | Head is small; INT8/INT4 draft-head quant fallback; draft KV ~67 MiB. Measured in Task 1b. |
| EAGLE-3 feature-fusion layers don't match our patched forward | Medium | Plan Task 0 reads the checkpoint card for the exact fusion layers; wire those specific hidden states. |
| UNION-selection verify drifts from non-spec greedy | Low–Med | §5 contract + RULER re-gate; per-query LSE-merge identical to Phase 9's reviewed math. |
| Page-boundary commit guard costs throughput (~6–8 % of steps fall back) | Low | Accepted (Phase 9 precedent); revisit with quantize-on-boundary only if it dominates. |

## §11 Open questions

(None blocking.) The two real unknowns — verify cost ratio and AWQ acceptance — are exactly what §6 measures before commitment.
