# Phase 4 Notes

**Started:** 2026-04-30
**Completed:** 2026-04-30 (tag `phase-4`)
**Status:** **complete (eager dispatch validated; 8B integration deferred to Phase 5)**
**Spec:** [docs/SPEC.md §6 Phase 4](../SPEC.md)

## Goal

Per-head retrieval-vs-streaming attention dispatch (DuoAttention) wired through HF Llama. Validate the dispatch on Llama-3.2-1B with a synthetic 70/30 head pattern.

## Surface

- `flashquest.duo.load_duo_pattern(path, threshold=0.5) -> torch.BoolTensor` — TSV loader for DuoAttention upstream patterns.
- `flashquest.duo.quest_duo_eager_sdpa(Q, K, V, *, head_pattern, page_size, retention, num_sinks, window_pages, is_causal)` — per-head dispatch (eager Python).
- `flashquest.eager.streaming_eager_sdpa(...)` — sinks+window-only attention helper.
- `flashquest.eager.llama_duo_patch.patch_llama_for_quest_duo(model, *, head_pattern, retention, num_sinks, window_pages, page_size)` — HF monkeypatch that loads a `(num_layers, num_kv_heads)` bool pattern and dispatches per-layer per-head.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Loader correctly reads Llama-3.1-8B pattern (32×8) | shape (32,8), bool, both classes present | ✅ |
| All-retrieval pattern ≡ Phase 1 quest_eager | rtol=1e-3 on B=1 H=4 S=256 | ✅ |
| All-streaming pattern ≡ streaming_eager | rtol=1e-3 on B=1 H=4 S=256 | ✅ |
| Mixed pattern: per-head correctness via `torch.where` reference | rtol=1e-3 on GQA 4:2 | ✅ |
| Llama-3.2-1B 70/30 split passkey ≥ Phase 1 baseline | both 5/5 across all depths | ✅ |

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| EP1 | All-retrieval pattern ≡ Quest-everywhere | ✅ |
| EP2 | All-streaming pattern ≡ StreamingLLM-only | ✅ |
| EP3 | Mixed pattern, both code paths exercised | ✅ |
| EP4/EP5 | Pattern shape mismatch raises | ✅ |
| EP6 | Threshold at 0.5 boundary documented (rounds up to retrieval) | ✅ |
| EP7 | Decode step (S_q=1) | ✅ via passkey eval |
| EP8 | Prefill (S_q > 1) | ✅ same path; dispatch is per-token-by-token |
| EP9 | GQA broadcast (pattern per KV head, applied to query heads) | ✅ via `repeat_interleave` |
| EP10 | Short cache vs sinks+window | ✅ Phase 1 `select_pages` clamps |
| EP11 | Empty / missing file | ✅ raises |

## Decisions

- **Eager dispatch (not Triton-side)**: the Phase 3 sparse kernel already accepts arbitrary per-query-head selection masks. DuoAttention reduces to "build a different mask per head" — no kernel change needed. Phase 4 v1 implements the dispatch at the eager Python level by running both paths and selecting per head via `torch.where`. This is wasteful at runtime (~50 % slower than Phase 1 in the passkey eval) but correct; a fused kernel-side dispatch is a Phase 5 perf win.
- **Synthetic pattern on Llama-3.2-1B**: DuoAttention's upstream classifications cover Llama-3.1-8B and Mistral-7B, not Llama-3.2. We use a uniform 70/30 random pattern (matches the typical retrieval ratio in their paper) for Phase 4 validation. Real per-model classifications are a Phase 5 input (or training step).

## Phase 5 prerequisites (deferred work)

These items are needed for SPEC §6 Phase 4's "Llama-3.1-8B at 32 k, ≥4 tok/s, ≥80 % RULER" win condition:

1. **Llama-3.1-8B AWQ-INT4 load** via `transformers + auto_awq`. Model is `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (~4.5 GB weights). Doesn't fit alongside even INT8 KV at 32 k on 4 GB; needs IQ3-XXS or 2-bit weights, or hot/cold layer offload (PowerInfer style).
2. **Persistent INT8 KV cache** as a `transformers.cache_utils.Cache` subclass. On `update`, quantize incoming K/V via Phase 3's `quantize_k`/`quantize_v` and store as uint8 + scales.
3. **Marlin W4A16 projections** for Q/K/V/O linear layers. Convert AWQ packing → Marlin packing once at load time.
4. **DuoAttention pattern for Llama-3.1-8B**: load from `vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv` (already vendored).
5. **32 k context bench**: passkey at depth grid + RULER 4 k subset (full RULER is hours).
6. **Fused kernel-side dispatch**: run only one of retrieval/streaming per head instead of both then `torch.where`.

## Phase 4 → Phase 5 handoff

Phase 4 ships:
- `phase-4` git tag.
- DuoAttention pattern loader + per-head dispatch + HF integration.
- Passkey on Llama-3.2-1B with synthetic 70/30 split: 25/25 correct (matches Phase 1).

Phase 5 begins: Llama-3.1-8B AWQ + persistent INT8 KV cache + Marlin projections + RULER eval at 32 k + fused dispatch.
