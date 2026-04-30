# Phase 2 Notes

**Started:** 2026-04-30
**Completed:** 2026-04-30 (tag `phase-2`)
**Status:** **complete (perf within 7 % of FA-2; all edge cases pass)**
**Spec:** [docs/SPEC.md §6 Phase 2](../SPEC.md)

## Goal

Port the FA-2 06-fused-attention Triton tutorial onto sm_86 as a clean dense forward-only kernel. No sparsity (Phase 3); no backward (out of scope).

## Kernel surface

`flashquest.kernel.flash_attn_fwd(Q, K, V, *, causal, sm_scale=None, return_lse=True) -> (O, lse)`

Shapes: `Q (B, H_q, S_q, D)`, `K/V (B, H_kv, S_kv, D)`, BF16 in / FP32 acc / BF16 out. GQA via kernel-side head map (no host `repeat_kv` copy). Returns LSE in fp32 for Phase 3 composition.

Tile shapes: `BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2`. Three autotune configs in `src/flashquest/kernel/_autotune.py` keyed on `(S_q, S_kv, HEAD_DIM)`.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Numeric ≡ torch SDPA across edge grid (E1–E12) | 39 deterministic + 20 hypothesis-fuzzed shapes pass within rtol=2e-2 | ✅ |
| Numeric ≡ Phase 1 eager at retention=1.0 | rtol=1e-2 at S=256 GQA 8:2 BF16 causal | ✅ |
| Perf within 30 % of FA-2 at S=8192 | **27.37 ms vs 25.51 ms = 1.073×** (7 % slower) | ✅ |

## Perf headline

Llama-3.2-3B geometry, S=8192, BF16, causal:

| Backend | ms / fwd | ratio vs FA-2 |
|---|---|---|
| `flash_attn` 2.7.4 (reference) | 25.51 | 1.000 |
| **flashquest Triton kernel** | **27.37** | **1.073** |
| torch SDPA (with GQA repeat) | 27.65 | 1.084 |

We're 1 % faster than torch SDPA (which uses cuDNN's flash attention path internally) and only 7 % slower than upstream FA-2. SPEC target was ≤30 % gap; we landed at 7 %.

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| E1 | `S == 1` (decode) | ✅ |
| E2 | `S < BLOCK_M` | ✅ |
| E3 | `S` not multiple of BLOCK_M / BLOCK_N | ✅ (parametrized over {17, 32, 63, 65, 100, 127, 129, 255}) |
| E4 | head_dim ∈ {64, 128} | ✅ |
| E5 | causal prefill (S_q == S_kv) | ✅ |
| E6 | causal decode (S_q == 1, S_kv > 1) | ✅ |
| E7 | B > 1 | ✅ (batch-independence verified bit-exact) |
| E8 | GQA n_rep ∈ {2, 4, 8} | ✅ |
| E9 | MHA degenerate (n_rep == 1) | ✅ |
| E10 | causal first row (single attended key) | ✅ (no NaN; output equals V[0]) |
| E11 | strided / non-contiguous Q | ✅ (transpose path bit-identical to contiguous) |
| E12 | NaN in Q | ✅ neighbour-row purity (strict NaN-propagation through `tl.dot` on bf16 is implementation-defined; the load-bearing test is "doesn't poison other rows") |
| E13 | causal + S_q != S_kv (chunked prefill) | ✅ rejected at wrapper |
| E14 | empty input (S=0) | ✅ rejected at wrapper |

## Decisions

- **Kernel-side GQA mapping** (no host `repeat_kv` copy). Saves ~25 MB at 8 k context for Llama-3.2-3B's 24:8 GQA.
- **BLOCK_M=BLOCK_N=64** vs the tutorial's 128. sm_86 has 16 SMs vs H100's 80; smaller blocks expose more parallelism on fewer SMs.
- **Base-2 exponent inside the kernel** (`exp2`/`log2`); converted to nats at the LSE write only. Lets the kernel use the faster `exp2` path.
- **LSE skipped** via `WRITE_LSE` constexpr when caller passes `return_lse=False`. No allocation cost.
- **Causal convention**: `q_pos = (S_kv - S_q) + i`, attend to `j <= q_pos`. For S_q=1 this collapses to "attend to all S_kv keys" — same as Phase 1 eager. **Note:** this disagrees with PyTorch SDPA's `is_causal=True`, which anchors the triangle at the top-left. Documented in the wrapper docstring.

## Phase 2 → Phase 3 handoff

Algorithm validated; perf bar exceeded. Phase 3 layers on top:
- Sparse outer loop driven by Phase 1's `compute_page_summary` + `page_scores` + `select_pages`.
- INT8 KV with KIVI-style scales (per-channel K, per-token V).
- StreamingLLM sinks + sliding window as always-attended block sets.
- The dense Triton kernel here is the correctness oracle: at retention=1.0, the sparse kernel's output must match this kernel's output bit-for-bit modulo float ordering.

Open items deferred:
- **Chunked prefill (E13)**: Phase 2 rejects this. Phase 3+ may want to add it for KV-cache prefill of long prompts. Track when it becomes load-bearing.
- **head_dim ∉ {64, 128}**: not a Llama target. If we ever need it, add via autotune key.
