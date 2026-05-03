# Phase 6 Task 1 Notes

**Started:** 2026-05-01
**Status:** **complete (tag `phase-6-task-1`); SPEC ≥4 tok/s gate cleared at 5.14 tok/s.**
**Spec:** [../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md](../superpowers/specs/2026-05-01-phase-6-criticality-fix-design.md)
**Plan:** [../superpowers/plans/2026-05-01-phase-6-criticality-fix.md](../superpowers/plans/2026-05-01-phase-6-criticality-fix.md)

## Summary

Three sub-tasks shipped:
- **Task 1a** (2026-05-01): algebraic page_scores_int8 + select_pages_vectorized. Decode 0.092 → 2.03 tok/s (22×). Skipped the dequant chain by exploiting `page_min ≡ K_mn`, `page_max ≡ K_mn + 255*K_scale`. Below ≥4 tok/s gate.
- **Task 1b** (2026-05-02): re-profile + research. `page_scores_int8` was 49% of per-layer time. EAGLE-2 deferred (cache-incompatible per vendor/eagle inspection); Marlin deferred (M=1 design point ≈ AWQ).
- **Task 1c** (2026-05-02): two-matmul `page_scores_int8_fast`. Decode 2.03 → **5.14 tok/s** (2.5× over 1a, 56× total over Phase 5). Peak VRAM unchanged at 6379 MiB.

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
| Decode at 32 k ≥4 tok/s | ≥4 tok/s | **5.14 tok/s** (56× over 0.092) | ✅ |
| 32 k passkey ≥80 % at depth=0.5 | 6/6 | inconclusive — methodology issue (see below) | ⚠️ |

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

**Re-profile after task 1a** (`benchmarks/phase6_profile_after.json`,
`scripts/phase6_profile_after.py`) overturns the earlier "AWQ + MLP is
the bottleneck" hypothesis. Per-layer cost at 32 k:

| op | ms | % |
|---|---|---|
| **`page_scores_int8`** | **5.687** | **49.2** |
| `flash_attn_sparse_fwd` | 1.581 | 13.7 |
| `select_pages_vectorized` | 0.938 | 8.1 |
| MLP (gate + up + down + act_mul) | 1.893 | 16.4 |
| AWQ attn projections (q + k + v + o) | 1.416 | 12.3 |
| reshape | 0.046 | 0.4 |
| **total / layer** | **11.56** | |

× 28 layers ≈ 324 ms / step. End-to-end 295 ms (variance vs the bench's
493 ms / step at 2.03 tok/s — likely first-decode warmup difference;
this profile run measured 3.39 tok/s end-to-end).

**`page_scores_int8` is 49 % of per-layer time.** AWQ projections + MLP
combined are only ~29 %. This means EAGLE-2 (complex, cache-bridge work)
and Marlin (small gain at M=1, see research notes below) are NOT the right
next steps.

### Task 1c — algebraic-fast page_scores via two GEMMs (proposed, not yet implemented)

Math: `K_scale ≥ 0` everywhere, so:

```
max(Q[d]·K_mn[p,d], Q[d]·(K_mn[p,d] + 255·K_scale[p,d]))
  = Q[d]·K_mn[p,d] + relu(255·Q[d]·K_scale[p,d])
  = Q[d]·K_mn[p,d] + 255·relu(Q[d])·K_scale[p,d]   (since K_scale ≥ 0)
```

Sum over D:

```
score(p) = Q · K_mn[p]  +  255 · relu(Q) · K_scale[p]
```

Two batched matmuls + one `clamp(min=0)`. Memory traffic drops from
~150 MB (the current `(B, H_q, S_q, P, D)` fp32 intermediates) to
~12 MB (just K_mn and K_scale loaded once each). Expected runtime:
< 0.5 ms vs current 5.69 ms / layer. New per-layer total ≈ 6.4 ms
→ end-to-end ≈ **5.5 tok/s** — clears SPEC ≥4 tok/s gate without
any kernel work.

Implementation sketch (PyTorch only):

```python
def page_scores_int8_fast(Q, K_scale, K_mn):
    B, H_q, S_q, D = Q.shape
    H_kv = K_scale.shape[1]
    n_rep = H_q // H_kv

    # Group GQA: (B, H_kv, n_rep * S_q, D) — broadcast in matmul, not memory.
    Q_g = Q.view(B, H_kv, n_rep, S_q, D).reshape(B, H_kv, n_rep * S_q, D)
    Q_pos_g = Q_g.clamp(min=0)

    # Two batched matmuls — torch picks the fastest path (cuBLAS / cuDNN).
    term1 = torch.matmul(Q_g.float(), K_mn.float().transpose(-1, -2))
    term2 = 255.0 * torch.matmul(Q_pos_g.float(), K_scale.float().transpose(-1, -2))
    scores = term1 + term2
    return scores.reshape(B, H_kv, n_rep, S_q, -1).reshape(B, H_q, S_q, -1)
```

Quality validation: identity is exact (sign-of-Q proof above) — same EQ18-EQ20
unit tests apply unchanged. Expected to pass at rtol=0 (modulo fp32 add reorder).

### Research findings (2026-05-02 — see scripts/phase6_diag_*.py + research agent)

- **Marlin at M=1 ≈ AWQ.** Marlin's design point is M=16-32 (vendor/marlin/README.md
  L7-8). Only worth migrating once speculation raises effective M.
- **EAGLE-2 not orthogonal to our cache.** vendor/eagle/eagle/model/kv_cache.py
  L103-111 hard-codes dense FP16 contiguous KV; verify step does dense
  tree-causal attention over full cache. Bridge work to dequant on demand
  for verify is 2-3 weeks. Not the right next step given the page_scores_int8
  finding.
- **No upstream DuoAttention pattern for Llama-3.2-3B.** Only 7B/8B variants
  ship in vendor/duo-attention/attn_patterns/. Synthetic random head_pattern
  is not a sound substitute. Train via DuoAttention's run_train.sh OR fall
  back to all-retrieval baseline.
- **Passkey methodology is broken** for our setup (un-seeded torch.rand);
  RULER 4k subset (Phase 6 task 2) is the SPEC's actual quality gate and
  should replace passkey for go/no-go.

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

---

# Phase 6 Task 2 Notes

**Started:** 2026-05-02
**Status:** **complete (tag `phase-6-task-2`); SPEC §6 task 2 gate cleared.**
**Spec:** [../superpowers/specs/2026-05-02-phase-6-ruler-niah-4k-design.md](../superpowers/specs/2026-05-02-phase-6-ruler-niah-4k-design.md)
**Plan:** [../superpowers/plans/2026-05-02-phase-6-ruler-niah-4k.md](../superpowers/plans/2026-05-02-phase-6-ruler-niah-4k.md)

## Summary

Replaced the broken passkey methodology (Phase 5/6 task 1's un-seeded
`torch.rand` head_pattern variance) with the SPEC's actual quality gate —
RULER NIAH (single, multikey, multivalue) at ctx=4 k on
Llama-3.2-3B-AWQ, retention=0.25 vs dense SDPA, ≥85 % pass per task.

## Decisions

- **Hybrid harness.** Own pure-Python generator (no `nltk`/`wonderwords`/`tqdm`
  deps) but RULER's prompt template + scoring vendored verbatim from
  `vendor/RULER/scripts/data/synthetic/niah.py`. UUIDs for keys, 7-digit
  numbers for values (RULER defaults `type_needle_k=uuids`,
  `type_needle_v=numbers`).
- **All-retrieval head_pattern** (`head_pattern = ones(L, H_kv)`). Eliminates
  the un-seeded `torch.rand` variance that broke the passkey eval. Every
  head runs Quest top-k at retention=0.25 + sinks + window. Harder test
  (no streaming-head escape); DuoAttention pattern training deferred.
- **Dense baseline = vanilla SDPA**, same model, no flashquest patches.
  SPEC gate is exactly "≥85 % vs dense"; this is dense.
- **n=20/task** by default (CI ~±10 %, ~45 min wall on 3050 Ti). CLI takes
  `--n-samples 64` for release-grade.
- **Corpus committed** (`data/PaulGrahamEssays.json`, ~660 KB).
  RULER ships URLs and a download script that needs html2text + bs4 + tqdm
  to scrape paulgraham.com; we fetch the gkamradt-pre-extracted .txt subset
  via `scripts/fetch_ruler_corpus.sh` to avoid that dep tree, and commit
  the resulting JSON for reproducibility.

## Result (2026-05-02)

`benchmarks/phase6_ruler_4k.json`:

| task | dense | patched | ratio | pass |
|---|---|---|---|---|
| niah_single | 20/20 | 20/20 | 100 % | ✅ |
| niah_multikey | 20/20 | 20/20 | 100 % | ✅ |
| niah_multivalue | 20/20 | 19/20 | 95 % | ✅ |

**all_pass: True.** Wall: dense 15.2 min, patched 25.2 min, total 40.4 min.

The single multivalue miss at retention=0.25 is below the 85 % gate margin;
the patched backend recovers all 4 distractor values across 19 of 20 prompts.

## Surface

- `flashquest.eval.niah` — `make_prompt(task, ctx_len, tokenizer, seed)`,
  `score(generated, expected_keys)`, `random_uuid`, `random_number`,
  `NEEDLE_TEMPLATE`, `PROMPT_TEMPLATE`.
- `flashquest.eval.runner.run_niah(model, tokenizer, task, n_samples, ctx_len, ..., pre_sample=None)`.
- `scripts/phase6_run_ruler_4k.py` — CLI loads dense + patched, runs all 3 tasks under both, writes JSON.
- `scripts/fetch_ruler_corpus.sh` — re-build `data/PaulGrahamEssays.json` from upstream.

## Tests

- `tests/test_eval_niah.py` — ER1 (prompt fits ctx), ER2 (seed determinism),
  ER3 (multikey 4 distinct keys), multivalue 4-values check, ER4 (score
  substring rule), ER6 (n=0), ER5 (slow-marked smoke on Llama-3.2-1B).
- 10 fast tests + 1 slow smoke; all green; full fast suite 164 passed
  (no regressions in Phases 1–6).

## v2 follow-ups

- Variable context lengths (8 k, 16 k, 32 k).
- Other RULER tasks (variable_tracking, qa_*, common_words, freq_words).
- LongBench.
- DuoAttention pattern training for Llama-3.2-3B (so streaming heads are real, not all-retrieval).
- vLLM / llama.cpp head-to-head numbers (Phase 6 task 4 charter).
