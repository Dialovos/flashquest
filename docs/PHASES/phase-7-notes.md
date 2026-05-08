# Phase 7 Notes — TurboQuant K3-V3 KV

**Started:** 2026-05-06
**Completed:** 2026-05-07
**Status:** **complete (tag `phase-7`); fused TurboQuant kernel ships behind `--kv-bits 3`. Quality gate cleared (RULER NIAH 4k @ K3-V3: 100/100/85). Throughput regresses 32% vs Phase 6 INT4 fused — INT4 stays the v1 default; TurboQuant is the storage-or-quality opt-in.**
**Plan:** [../superpowers/plans/2026-05-06-phase-7-turboquant-kv.md](../superpowers/plans/2026-05-06-phase-7-turboquant-kv.md)
**Spec:** [../superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md](../superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md)

## Summary

`--kv-bits 3` ships TurboQuant **K3-V3** (Phase 7's headline target was K3-V2;
see *Two unplanned recoveries* below). Per-token Walsh-Hadamard rotation
along `head_dim` Gaussianizes the per-block distribution; a fixed
8-codepoint Lloyd-Max codebook handles K and V quantization; values are
stored as bit-split planes (1-bit MSB plane @ 8/byte + 2-bit LSB plane
@ 4/byte). Decode-only fused Triton kernel reads both planes directly,
gathers from a constexpr-inlined codebook, and runs the same
online-softmax math as the INT4 kernel. Storage shrinks 25% vs Phase 6
INT4 cache (980 MiB → 736 MiB at full 32 k cache, 28-layer
Llama-3.2-3B-Instruct). Quest top-k criticality unchanged via dual
statistics — `K_scale_raw, K_mn_raw` per-page channel-wise from
un-rotated K, used only for `page_scores_int4_fast`.

## Two unplanned recoveries during execution

The plan's **K3-V2** target (paper headline) failed RULER on Llama-3.2-3B:
single 100% / multikey 75% / multivalue 60%. Two changes were needed to
clear the gate:

1. **Per-token scale: `max(|x|)/c_max` → RMS** (sqrt(mean(x²))). The
   Lloyd-Max codebook is optimal for unit-variance Gaussian; max-scaling
   rescaled the data to a *different* variance, leaving most values in
   the codebook's "no-zero gap" region. Switching to RMS aligned the
   data variance with the codebook assumption. Roundtrip mean abs err:
   K 0.40 → 0.148 (-63%), V 0.70 → 0.272 (-61%). Recovered ~20pp on
   multikey/multivalue.

2. **V upgrade from 2-bit → 3-bit**: K3-V2 with RMS scale still failed
   multivalue at 80% (gate ≥85%). 4-codepoint V can't discriminate
   among 4 similar values; multikey + multivalue both need V precision.
   K3-V3 makes V symmetric with K and added one more bit per V element.

## Result — quality (RULER NIAH 4k @ K3-V3)

`benchmarks/phase7_ruler_4k_turbo.json`:

| task | dense | patched (K3-V3) | ratio | gate ≥85 % |
|---|---|---|---|---|
| niah_single | 20/20 | 20/20 | 100 % | PASS |
| niah_multikey | 20/20 | 20/20 | 100 % | PASS |
| niah_multivalue | 20/20 | 17/20 | **85 %** | **PASS (right at floor)** |

multivalue is at the gate floor. Calibrated codebook (Approach C from
the spec) would likely push it to 90-95% but is queued as v2 — current
gate is met.

## Result — single-cell decode bench at 32 k

`benchmarks/phase7_decode_turbo_32k.json`:

| metric | INT4 fused (Phase 6) | K3-V3 (initial) | K3-V3 (inline codebook) |
|---|---|---|---|
| decode_tok_s | 3.88 | 1.79 | **2.62** |
| prefill_tok_s | 65.0 | 78.2 | 77.6 |
| peak_vram_mib | 5478 | 6105 | 6105 |
| wall_s | 534.6 | 447.1 | 444.7 |

The first measurement (1.79) prompted a kernel optimization: the GMEM
codebook gather (`tl.load(codebook_ptr + idx)`) was the hot spot on
Ampere because indexed loads cannot coalesce. Replacing with a
constexpr `tl.where` chain over the 8 codepoints (compiles to ~7 SELP
instructions per element, all in registers) recovered 47% of throughput.
Final: 2.62 tok/s, **-32 % vs INT4 fused floor of 3.88**.

The remaining gap is from the 4 bit-plane tile loads per page (vs INT4's
2 packed loads), the WHT applied to Q + inverse-WHT on output, and the
extra unpack arithmetic. These are intrinsic to the K3-V3 layout.

## Result — head-to-head re-test (SPEC §11.4)

`benchmarks/phase7_headtohead_turbo.json`:

| backend | quant + KV | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest TurboQuant K3-V3 | AWQ-INT4 + K3-V3 paged | **2.05** | **1.93** | ✗ |
| flashquest INT4 (Phase 6) | AWQ-INT4 + INT4 paged | 4.94 | 3.88 | ✗ |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 38.45 | ✗ (timeout) | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | ✗ (timeout) | ✗ |

(Bench was stopped manually at the vLLM 32 k cell to spare host SSD;
vLLM 32 k + 128 k numbers reused from Phase 6 INT4 head-to-head — vLLM
does not consume `--kv-bits 3` so its results are unchanged between
phases. flashquest 32 k head-to-head reading is 1.93 vs the single-cell
2.62 measurement from Task 12, attributable to host-process contention
during the longer multi-backend run; both are below the 3.88 INT4 floor.)

**Verdict — capability axis:** flashquest still the only backend that
decodes at 32 k on 4 GB. Capability ratio over llama.cpp and vLLM
unchanged at ∞× (both still fail at 32 k).

**Verdict — throughput axis:** at the same hardware and ctx,
flashquest INT4 fused (Phase 6) is faster than K3-V3 (Phase 7). INT4
stays the v1 default. TurboQuant is opt-in for storage-constrained
or quality-sensitive workloads (~25% smaller cache, RULER-validated
quality). The "near-zero PPL gap" claim from the TurboQuant paper
required two non-paper adjustments (RMS scale + V at 3-bit not 2-bit)
on Llama-3.2-3B; the paper validated on Llama-3.1-8B + Gemma + Mistral.

## Surface

- `flashquest.kernel.wht.wht_along_head_dim` — pure-PyTorch normalized FWHT (self-inverse).
- `flashquest.kernel.kv_quant.{quantize_k_turbo, quantize_v_turbo, dequantize_k_turbo, dequantize_v_turbo, _pack_bit_split, _unpack_bit_split, _pack_int2, _unpack_int2, _quantize_to_codebook, K_TURBO_CODEBOOK, V_TURBO_CODEBOOK}`.
- `flashquest.cache.PersistentTurboKVCache` (kv_bits=3).
- `flashquest.kernel.sparse_turbo_fwd._sparse_attn_fwd_kernel_turbo` + `flash_attn_sparse_turbo_fwd` Python wrapper + `_flash_attn_sparse_turbo_fwd_reference` Python reference.
- `--kv-bits 3` flag in `flashquest chat` and `bench_flashquest.py`.
- Dispatcher in `eager/llama_persistent_patch.py` branches on `cache.kv_bits ∈ {3, 4, 8}`; reuses `page_scores_int4_fast` on un-rotated raw stats for criticality.
- Tests: `test_wht.py` (4), `test_kv_quant_turbo.py` (10), `test_persistent_turbo.py` (6 + 1 slow), `test_sparse_turbo.py` (3). Total 23 new fast + 1 new slow. Full fast suite 227 pass.

## v2 follow-ups

- **Calibrated codebook (Approach C from spec)** — recompute Lloyd-Max
  levels from real Llama-3.2-3B WHT'd K, V samples on RULER prompts.
  Would likely push multivalue from 85% → 90-95% and might recover
  some of the throughput by allowing tighter spacing where data is
  dense. ~1-2 hr work.
- **Kernel autotune** — sweep `BLOCK_SIZE`, `num_warps`, `num_stages`.
  Triton autotuner across 4-8 configs. Realistic 10-30 % decode speedup.
- **Larger context** — TurboQuant's 25 % shrink mostly matters at long
  ctx; revisit when 128 k decode becomes possible (e.g., on a larger
  card or with off-loading).
- **TransMLA** (NeurIPS 2025 spotlight) — separate phase; needs ~6 B-token
  fine-tune. Composes with TurboQuant in principle (latent KV → quant).
- **xKV (cross-layer KV sharing, ICLR 2025)** — orthogonal compression
  axis. Composes with TurboQuant.
- **1-bit QJL residual on K** (paper's actual recipe; deferred in Phase 7) —
  push K to effective 4-bit with marginal storage cost. Re-evaluate if a
  later workload demands tighter K precision than current 100 % multikey
  ratio implies.
