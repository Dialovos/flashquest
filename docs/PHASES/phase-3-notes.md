# Phase 3 Notes

**Started:** 2026-04-30
**Completed:** 2026-04-30 (tag `phase-3`)
**Status:** **complete (sparse decode 2.48× faster than dense; all edge cases pass)**
**Spec:** [docs/SPEC.md §6 Phase 3](../SPEC.md)

## Goal

Layer Quest-style sparse selection on top of Phase 2's dense kernel, with KIVI-style INT8 KV. Decode-only sparse forward (`S_q == 1`).

## Surface

- `flashquest.kernel.kv_quant.{quantize_k, dequantize_k, quantize_v, dequantize_v}` — KIVI convention asymmetric uint8 (per-page channel-wise K, per-token V).
- `flashquest.kernel.flash_attn_sparse_fwd(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, selection_mask, page_size, sm_scale, return_lse) -> (O, lse)` — decode-only Triton kernel that dequantizes inside.
- `flashquest.eager.quest_eager_sparse_int8(...)` — pure-PyTorch correctness oracle.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Sparse ≡ Phase 2 dense at full mask within INT8 tolerance | rtol=5e-2 at S_kv=256 GQA 4:1 | ✅ |
| Sparse ≡ eager INT8 reference at retention=0.25 + sinks + window | rtol=5e-2 across hypothesis-fuzzed grid | ✅ |
| Decode speedup vs Phase 2 dense at 8 k context, retention=0.25 | **2.48× (0.181 ms vs 0.449 ms/step)** | ✅ (target ≥ 1.5×) |

## Perf headline

Llama-3.2-3B geometry (B=1, H_q=24, H_kv=8, S_kv=8192, D=64), retention=0.25, sinks=4, window=128:

| Backend | ms / decode step | speedup |
|---|---|---|
| Phase 2 dense (BF16 KV) | 0.449 | 1.0× |
| **Phase 3 sparse (INT8 KV)** | **0.181** | **2.48×** |

The sparsity gain (~25 % page retention plus sinks/window) dominates the per-page scale-load overhead even at this small per-step budget.

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| ES1 | selection_mask all True | ✅ matches dense within INT8 tolerance |
| ES2 | selection_mask all False | ✅ output is zero |
| ES3 | S_kv not multiple of page_size | ✅ across {64, 96, 100, 128, 192, 1023} |
| ES4 | S_q == 1 (decode hot path) | ✅ primary use case |
| ES5 | GQA per-head selection | ✅ n_rep ∈ {2, 4, 8} |
| ES6 | head_dim ∈ {64, 128} | ✅ |
| ES7 | S_kv == page_size (single page) | ✅ |
| ES8 | causal trivializes for decode | ✅ (kernel has no causal flag — selection mask is the only constraint) |
| ES9 | zero-range channel (constant K) | ✅ no NaN in output |
| ES10 | INT8 quant round-trip drift | ✅ within (max-min)/255 per element |
| ES11 | multi-query (S_q > 1) | ✅ rejected at the wrapper |

## Decisions

- **Decode-only sparse for v1**. Multi-query / chunked-prefill rejected at wrapper. Prefill uses Phase 2 dense kernel.
- **uint8 storage** (no bit-packing). INT4/INT2 deferred — start simple, halve the cache vs BF16 today, push lower in v2.
- **Per-page channel-wise K + per-token V scales** (KIVI convention, page_size aligned to Phase 2 BLOCK_N=64).
- **Selection mask computed outside the kernel** (Phase 1 page_summary + page_scores + select_pages) and passed in as a `(B, H_q, 1, num_pages) bool` tensor. Keeps the kernel simple; fusing top-k inside is a v2 optimization.
- **Scalar `m_i` / `l_i`** (not `[1]`-shape tensors) — required for Triton compile correctness on the LSE store path.

## Phase 3 → Phase 4 handoff

Algorithm validated; INT8 KV works; decode sparsity yields a real perf win. Phase 4:
- DuoAttention head split (per-head retrieval-vs-streaming pattern; load pre-trained classifications).
- HF Llama integration (patch LlamaAttention to use the sparse kernel + INT8 cache).
- Larger model (Llama-3.1-8B IQ3_XXS) with Marlin W4A16 weight projections.
- Real end-to-end at 32 k context with RULER quality eval.

Open items deferred:
- Sparse prefill (chunked) — track when it becomes load-bearing.
- INT4 / INT2 KV — Phase 5+.
- vLLM-style page_table for non-contiguous KV — Phase 5+.
- Fused top-k inside the kernel — v2 optimization.
