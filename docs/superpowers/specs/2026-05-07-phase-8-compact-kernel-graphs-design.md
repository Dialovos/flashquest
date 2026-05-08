# Phase 8 — Compact-List Sparse Kernel + GPU-Resident Dispatch + Bucketed CUDA Graphs — Design (revision 2)

**Author:** flashquest core
**Date:** 2026-05-07
**Revision:** 2 (codex review on r1 found 4 dealbreakers + 8 corrections; full rewrite. See Appendix A for diff.)
**Status:** Approved post-rewrite
**Phase 7 baseline:** [`docs/PHASES/phase-7-notes.md`](../../PHASES/phase-7-notes.md) — 3.88 tok/s @ 32k INT4 fused, 2.62 tok/s @ 32k TurboQuant K3-V3.
**Roadmap position:** First of a 4-phase OOM-cumulative push. Phase 8 lands the **kernel ABI + dispatch foundation**; Phase 9 (specdec) raises effective batch and unlocks Marlin; Phase 10 (DuoAttention/CATS) needs head-pattern training first; Phase 11 (lookahead/MTP) composes on top.

---

## 1. Goal

Take Llama-3.2-3B AWQ-INT4 decode throughput on RTX 3050 Ti Laptop (sm_86, 4 GB VRAM, WSL2) from **3.88 tok/s @ 32k → ≥6 tok/s (1.55× floor) / ≥7.5 tok/s stretch (1.94×)**, without changing the model, KV bit-width, or RULER NIAH 4k quality.

Phase 8 is the foundation. Its job is to **rebuild the kernel ABI and decode dispatch** so that Phase 9-11 can compose on top:
- The compact-list sparse kernel replaces the current full-mask iteration that wastes ~75% of loop overhead at retention=0.25.
- Eliminating host-side `.item()` calls in the hot path unblocks CUDA Graph capture (without sync removal, graphs buy ~nothing).
- Bucketed CUDA Graphs make the rebuilt path replayable, removing per-step launch overhead.
- Fused QKV + gate+up via custom Triton GEMM (NOT Marlin — see §2.4) reduces 4 GEMMs/layer to 2.

## 2. Background — corrected from r1

### 2.1 Why r1 was wrong (codex review summary)

r1 proposed Marlin INT4 GEMM + bucketed CUDA Graphs + async two-stream layer pipeline. Codex flagged 4 dealbreakers:

1. **Bucketed bool-mask doesn't reduce kernel work.** Current `sparse_int4_fwd` in `src/flashquest/kernel/sparse_int4_fwd.py:73` iterates `for p in range(0, NUM_PAGES)` with an `is_sel` branch inside. Padding the top-k count to a bucket size doesn't reduce the loop; the kernel runs all NUM_PAGES iterations regardless.

2. **r1 bucket sizes `[8, 16, 24, 32]` were calibrated for 4k context, not 32k.** At 32k context with `page_size=64`, `num_pages=512`. Selection in `selection.py:96` is `k = ceil(retention × P)` = `ceil(0.25 × 512) = 128`. Real 32k buckets need to cover ~128, not 32.

3. **Marlin doesn't help at M=1.** Phase 6 notes (`docs/PHASES/phase-6-notes.md:148`) already document: "Marlin at M=1 ≈ AWQ. Marlin's design point is M=16-32. Only worth migrating once speculation raises effective M." Marlin standalone gain at decode batch=1 is ≈0×. r1's 1.5-2× Marlin claim was unsupported.

4. **Async two-stream layer pipeline can't hide writeback.** `eager/llama_persistent_patch.py:167-168` writes K/V then immediately reads `cache.get_views(...)` in the same layer for attention. Same-layer write→read dependency means there's nothing to hide behind the next layer.

Plus 8 corrections (Marlin dtype is FP16 not BF16, AWQ→Marlin conversion was naive, partial-page dynamics ignored, host syncs in hot path, fallback claim weak, VRAM gate optimistic, missing gates, conversion test methodology). All are addressed in this revision.

### 2.2 Where the real wins live (corrected)

#### Compact-list sparse kernel rewrite — primary win
The current kernel iterates ALL 512 pages with an `is_sel` branch each iteration. At retention=0.25 only 128 pages contribute, but the loop runs full. The compact-list rewrite takes `selected_page_ids: int32[BUCKET]` and iterates `for i in range(0, BUCKET)` with `p = selected_page_ids[i]`. Loop count goes from 512 → 128 (or BUCKET-padded). Memory traffic per iteration is unchanged (already gated by `mask=valid_kv` in the load), but loop control overhead drops 4×. Estimated ~1.3-1.6× kernel-wall.

#### Host-sync elimination — unblocks graphs
Three sync points in the decode hot path:
- `selection.py:100` — `int(k_per_h.max().item())` forces a CPU sync to compute static k_max for `topk(k_max)`.
- `llama_persistent_patch.py:177-178, 180, 201` — `partial_len`, `completed_len` accessed as Python ints, used in `if completed_len == 0` and `if partial_len == 0` branches.

Without removing these, CUDA Graphs cannot wrap the dispatch logic — every `.item()` and Python branch breaks graph capture. Phase 8 makes selection produce a fixed-shape `selected_page_ids` tensor (k_max precomputed at warmup from retention × max P), and replaces partial/completed branches with **predicated execution** (always run both partial and sparse paths, predicate the merge by tensor-resident flags).

#### Bucketed CUDA Graphs over compact-list — secondary win
Once selection returns a static-shape `selected_page_ids` tensor and the kernel takes a compact list, the entire decode step becomes a static-shape computation graph. Capture once per bucket size, replay per step.

At decode shapes (batch=1, single token), per-step launch overhead from ~250 kernel launches per token (28 layers × ~9 kernels each) is on the order of 30-50% of decode wall time. Graph replay collapses this to one CUDA graph launch + parameter copies. Estimated ~1.2-1.4× from launch overhead elimination.

#### Fused QKV + gate+up via custom Triton GEMM — tertiary win
4 separate AWQ-INT4 GEMMs per layer (Q, K, V, gate, up — actually 5; O and down are post-attention) → 2 fused Triton GEMMs (qkv-fused, gate-up-fused). The fusion is implemented as a custom Triton kernel that operates directly on AWQ's existing INT4 packed weights (no Marlin layout reorder). Wins from larger tile sizes + fewer launches. Estimated ~1.05-1.1×.

#### What's NOT in Phase 8 (deferred)
- **Marlin INT4 GEMM** — defers to Phase 9 where speculative decoding raises effective M to 8-16 and Marlin starts winning. Currently Marlin standalone at M=1 ≈ AWQ.
- **Async two-stream layer pipeline** — broken dependency model; dropped permanently.
- **AWQ → Marlin weight conversion** — only relevant when Marlin lands.
- **EAGLE-3 / Medusa specdec** — Phase 9. Phase 6 notes flag that EAGLE-2's vendored cache layout (`vendor/eagle/eagle/model/kv_cache.py:103-111`) hardcodes dense FP16 contiguous KV, requires 2-3 weeks of bridge work to integrate with paged sparse KV. Phase 9 brainstorm will address.

### 2.3 Realistic ceiling
Compounded with interaction discounts:
- Compact-list kernel: 1.45× (mid)
- Sync elimination: 1× direct (enables graphs)
- Bucketed CUDA Graphs: 1.3× (mid)
- Fused QKV/gate+up Triton: 1.07× (mid)

Multiplicative: 1.45 × 1.3 × 1.07 = **2.0×** mid estimate. Floor 1.55× / stretch 1.94×.

The OOM cumulative roadmap (Phase 8 × 9 × 10 × 11) is unchanged in spirit — Phase 9 remains the heavy lifter (2.5-3.5× from specdec), Phase 10 (DuoAttention/CATS) and Phase 11 (lookahead/MTP) compose. Cumulative target ~50 tok/s @ 32k stays plausible. Phase 8's specific contribution is the foundation that makes the others composable.

### 2.4 Why Triton fusion (not Marlin)
Marlin requires (a) FP16 activations (not BF16), (b) a specific weight permutation, (c) batch ≥ 16 to amortize tile setup. None hold at our current decode regime. Custom Triton fusion of AWQ-INT4 weights:
- Operates on AWQ's existing packed INT4 layout (no repack)
- BF16-clean (matches current dispatcher contract)
- Wins from launch reduction + larger tile (vs. 5 separate small GEMMs)
- Simpler to maintain (one Triton file vs. vendored CUDA + repack pipeline)

When Phase 9 ships specdec and effective M rises to 8-16, the Marlin migration becomes worth it. Phase 8 doesn't do that work prematurely.

## 3. Design overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 8 decode step (per layer × 28 layers)                        │
│                                                                     │
│  input ─┐                                                           │
│         ├── Triton fused QKV GEMM (AWQ-INT4 direct, BF16 in/out)    │
│         │   slice → Q, K, V                                         │
│         │   RoPE on Q, K                                            │
│         │                                                           │
│         ├── cache.update_quantized(K, V)  [no async, same stream]   │
│         │                                                           │
│         ├── Quest criticality (existing) → top-k indices            │
│         │   ┌──────────────────────────────────────────────┐        │
│         │   │  GPU-RESIDENT DISPATCH                       │        │
│         │   │  k_max = static (precomputed at warmup)      │        │
│         │   │  topk → selected_page_ids: int32[BUCKET]     │        │
│         │   │  pad with first-page (no-op)                 │        │
│         │   │  bucket = lookup_bucket(actual_count)        │        │
│         │   │  (no .item() — bucket is a tensor index)     │        │
│         │   └──────────────────────────────────────────────┘        │
│         │                                                           │
│         ├── graph_cache[bucket].replay()                            │
│         │   [contains: sparse_int4_fwd_compact + LSE merge          │
│         │              + Triton fused gate+up + SwiGLU + down       │
│         │              + residual + RMSNorm                  ]      │
│         │   no .item(), no Python branches inside graph             │
│         │                                                           │
│         └── output (next-token hidden state)                        │
└─────────────────────────────────────────────────────────────────────┘
```

Differences from r1 (post-codex):
- **No Marlin** anywhere — Triton fusion only.
- **No async streams** — single stream, predicated execution.
- **No bool selection_mask** — compact list of page IDs.
- **No `.item()` in hot path** — bucket dispatch is GPU-resident.
- **Buckets calibrated at REAL 32k** — boundaries reflect ~128-page top-k.

## 4. Components

### 4.1 Compact-list sparse kernel rewrite

**Surface:**
```python
# new — kernel/sparse_int4_fwd_compact.py
flash_attn_sparse_int4_fwd_compact(
    q,                              # (B, H_q, 1, D) bf16
    K_packed, K_scale, K_mn,
    V_packed, V_scale, V_mn,
    selected_page_ids: torch.Tensor,  # int32 (B, H_kv, BUCKET)
    bucket_size: int,                 # constexpr at graph-capture time
    page_size: int,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor]   # (O, lse)
```

**Kernel inner loop (replaces lines 73-145 of current `sparse_int4_fwd.py`):**
```python
for i in range(0, BUCKET_SIZE):
    p = tl.load(selected_page_ids_ptr + i)  # int32
    page_start = p * PAGE_SIZE
    n_idx = page_start + offs_n
    valid_kv = n_idx < S_kv
    # ... rest of body unchanged: load K/V tile, dequant, qk, softmax merge, accumulate V
```

Compared to current kernel:
- Loop runs `BUCKET_SIZE` (e.g., 128 or 160) iterations, not `NUM_PAGES` (e.g., 512).
- No `is_sel` branch — every iteration contributes.
- Padding pages: caller pads `selected_page_ids` with sentinel `-1` for unused slots. Kernel skips them via `tl.where(p >= 0, body_contribution, 0.0)` for both QK and V accumulators, and `tl.where(p >= 0, qk, NEG_INF)` for the running max so the padded slot contributes zero to softmax. Initial design considered "pad with first page repeated" but verified algebraically: duplicating a page when other pages are present skews the weighted average — `(2·num_X + num_Y)/(2·den_X + den_Y) ≠ (num_X + num_Y)/(den_X + den_Y)` in general. Sentinel `-1` is correct.

**Three kernel variants (one per kv_bits):**
- `sparse_int4_fwd_compact` (kv_bits=4)
- `sparse_int8_fwd_compact` (kv_bits=8) — currently `sparse_fwd.py`
- `sparse_turbo_fwd_compact` (kv_bits=3) — currently `sparse_turbo_fwd.py`

All three follow the same pattern: replace bool-mask iteration with compact-list iteration. Quantization-specific dequant logic unchanged.

**Compatibility:** the existing bool-mask kernel paths stay as fallbacks for non-graph-captured operation (initial validation, debugging). Both paths must produce identical output (parity test).

### 4.2 Host-sync elimination

**Three sync points to remove:**

1. **`selection.py:100` — `k_max.item()`**
   - **Fix:** precompute `k_max` at warmup from `(retention × P_max).ceil()` where `P_max` is the maximum `num_pages` the cache will ever have (= `max_context_len // page_size`). Store as a Python int known statically.
   - All `topk(k_max)` calls use this precomputed value.
   - Side effect: must re-compute when context grows past `P_max`; rare and triggered explicitly.

2. **`llama_persistent_patch.py:177-180, 201` — `partial_len`, `completed_len` Python ints**
   - **Fix:** convert to GPU-resident scalar tensors. Use predicated execution: always run both `_bf16_dense_attn_with_lse` (partial) and `_sparse_fwd_call` (sparse), then merge with online-softmax. If `partial_len == 0` or `completed_len == 0`, the corresponding output's LSE is `-inf`, which the merge handles correctly (the other branch dominates).
   - Cost: small extra compute when one path is empty, but no sync, no Python branch. Replicates Phase 6 task 6's "always merge" pattern in a graph-friendly way.

3. **Replace bucket dispatch sync** (introduced naturally by §4.3)
   - **Fix:** bucket lookup is implemented as a GPU-resident `torch.searchsorted` on bucket boundaries against the actual top-k count tensor. Replay path uses bucket index directly (no `.item()`).
   - At graph-capture time, the bucket is known statically (one graph per bucket); replay path indexes a Python dict of pre-captured graphs by the bucket integer. The integer comes from `searchsorted(...).item()` — but this is the ONLY remaining `.item()` and it's outside the captured graph itself, just selecting which graph to replay.

### 4.3 Bucketed CUDA Graph dispatcher

**Surface:** `flashquest.runtime.graph_dispatcher.GraphDispatcher`.

**Bucket sizes — calibrated from real 32k histogram (see §6):**
Initial estimate: `[64, 96, 128, 160, 192]` covering p25 → p99 of real 32k selection-count distribution. Final values come from §6 calibration.

**State:**
- `bucket_sizes: list[int]` — calibrated at warmup.
- `graphs: dict[int, torch.cuda.CUDAGraph]` — per bucket size.
- `static_inputs: dict[int, dict[str, torch.Tensor]]` — pre-allocated input buffers per bucket. `selected_page_ids` shape `(B, H_kv, bucket_size)`.
- `static_outputs: dict[int, dict[str, torch.Tensor]]` — pre-allocated output buffers.

**Capture (warmup):**
```python
def capture(self, bucket_size, model_step_fn):
    si = self.static_inputs[bucket_size]
    so = self.static_outputs[bucket_size]
    # warmup: 3 forward passes to settle Triton autotune
    for _ in range(3):
        model_step_fn(si)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = model_step_fn(si)
        so["hidden_out"].copy_(out["hidden_out"])
        # ... copy any other outputs
    self.graphs[bucket_size] = g
```

**Replay (per decode step):**
```python
def step(self, real_inputs, selected_page_ids_compact, actual_count):
    # GPU-resident bucket lookup
    bucket_idx = torch.searchsorted(self.bucket_boundaries, actual_count)
    bucket_size = int(bucket_idx.item())  # only .item() — outside graph
    if bucket_size > self.bucket_sizes[-1]:
        # rare: actual count exceeds max bucket. fall back to eager.
        return self._step_eager(real_inputs, selected_page_ids_compact, actual_count)
    si = self.static_inputs[bucket_size]
    # copy real inputs into static buffers
    si["hidden_in"].copy_(real_inputs["hidden_in"])
    si["selected_page_ids"][:, :, :actual_count].copy_(selected_page_ids_compact)
    si["selected_page_ids"][:, :, actual_count:].fill_(-1)  # sentinel — kernel skips via tl.where(p >= 0, ...)
    # replay
    self.graphs[bucket_size].replay()
    return self.static_outputs[bucket_size]["hidden_out"].clone()
```

**Re-capture trigger:** if context length crosses page-flush boundary (i.e., `partial_len` rolls over from 63 → 64), the decode-step kernel sees a different completed-page count. Mitigation: capture two graphs per bucket — one for "mid-page" steps, one for "page-flush" steps. Or: predicate the page-flush logic inside the kernel and avoid graph re-capture.

**Decision:** Phase 8 ships **predicated page-flush** (no re-capture). The kernel handles both states via tensor-resident flags. This is implementation detail in the plan.

### 4.4 Fused QKV + gate+up via custom Triton GEMM

**Surface:** `flashquest.kernel.fused_proj.fused_qkv_proj`, `fused_gate_up_proj`.

**Logic:** stack `q_proj.weight`, `k_proj.weight`, `v_proj.weight` (AWQ-INT4 packed) along output dimension into one weight tensor. Custom Triton GEMM operates on the stacked weight, output is sliced into Q, K, V views.

```python
@triton.jit
def fused_int4_gemm_kernel(
    A_ptr,           # (M, K) bf16 activation
    W_packed_ptr,    # (K // 8, N_total) int32 packed AWQ weight, N_total = N_q + N_k + N_v
    W_scale_ptr,     # (K // group_size, N_total) bf16 scales
    W_zero_ptr,      # (K // group_size, N_total // 8) int32 packed zeros
    O_ptr,           # (M, N_total) bf16 output
    M, K, N_total,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    # Standard tiled GEMM with AWQ INT4 dequant inside the K-loop.
    # ... (similar to AutoAWQ's GEMM kernel, just larger N tile)
```

**Why this beats AWQ's per-Linear GEMM:**
- 1 launch instead of 3 for QKV (same for 1 vs 2 for gate+up).
- Larger N tile fits Ampere tensor-core shape better (currently AWQ uses N=3072 for Q and N=1024 for K/V; fused gets N=5120, single launch).
- BF16-clean (AWQ already supports BF16).

**Compatibility:** doesn't touch AWQ weight format. Existing AWQ checkpoints work unchanged.

**Estimated standalone gain:** 1.05-1.1× (decode batch=1, launch reduction). Smaller than Marlin would be at M≥16, but real and composable.

### 4.5 RoPE-post cache verification

**Audit task:** confirm that K is stored *post-RoPE* in `PersistentInt8KVCache`, `PersistentInt4KVCache`, `PersistentTurboKVCache`. Reading `llama_persistent_patch.py:155` (`q, k = apply_rotary_pos_emb(q, k, cos, sin)`) followed by `:167` (`cache.update_quantized(k, v, ...)`), it appears K is post-RoPE on write. Verify this and document.

If pre-RoPE somewhere, fix so decode reads cached K directly without re-rotation.

Small task — likely already correct, but worth confirming as part of foundation.

### 4.6 Real-histogram calibration utility

**Surface:** `flashquest.runtime.calibrate_buckets.collect_selection_histogram`.

**Logic:** run model in current Phase 7 mode at **REAL 32k context** on RULER NIAH 4k prompts (extended to 32k context) or representative 32k prompts. At each decode step, log the actual top-k count per (layer, head-group). Build histogram. Choose 4-5 bucket boundaries.

**At retention=0.25, page_size=64, 32k context:**
- `num_pages = 32768 / 64 = 512`
- `k_per_h = ceil(0.25 × 512) = 128`
- All heads request 128 pages by default.

But `num_sinks` (default 4) and `window_pages` (default 2) add to `k`. So actual unique selected pages can be 128 + sinks + window − overlap. Practical range: 128-134.

**However:** when retention is per-head (DuoAttention pattern, Phase 10), some heads get retention=0 (streaming heads use only sinks + window = 6 pages). Phase 8 still uses uniform retention=0.25 for all heads (DuoAttention is Phase 10), so the histogram is tight around 128.

**Calibration heuristic:** since the distribution at uniform retention is tight, Phase 8 may only need 1-2 buckets (e.g., `[128, 192]`). Calibration confirms.

**Output:**
```json
{
  "model": "unsloth/Llama-3.2-3B-Instruct",
  "context_length": 32768,
  "page_size": 64,
  "retention": 0.25,
  "histogram": {"128": 542, "129": 35, "132": 12, "...": "..."},
  "p99_count": 132,
  "max_count": 134,
  "bucket_sizes": [128, 144],
  "expected_pad_overhead": 0.04
}
```

If Phase 10 (DuoAttention) lands later and per-head retention varies, recalibrate at that point.

## 5. Compatibility matrix

| `--kv-bits` | Compact kernel | CUDA Graphs | Status after Phase 8 |
|---|---|---|---|
| 4 (Phase 6 INT4) | ✅ rewritten | ✅ | Primary target |
| 8 (Phase 3 INT8) | ✅ rewritten | ✅ | Maintained |
| 3 (Phase 7 Turbo) | ✅ rewritten | ✅ | Maintained |
| any | ✅ | OFF (`--no-cuda-graphs`) | Fallback for debugging |
| any | OFF (`--no-compact-kernel`) | OFF | Phase 7 path (regression-test reference) |

Both flags default ON post-validation. Defaults OFF during Phase 8 task progression until each is validated independently.

## 6. Calibration step (Task 1 of plan)

```bash
python -m flashquest.runtime.calibrate_buckets \
    --model unsloth/Llama-3.2-3B-Instruct \
    --kv-bits 4 \
    --retention 0.25 \
    --context-length 32768 \
    --num-prompts 5 \
    --output flashquest/runtime/_bucket_calibration_32k.json
```

Run on 5 long-context prompts (RULER 32k variants, or synthetic 32k contexts). At each decode step, log actual top-k count. Build histogram, pick bucket boundaries.

**Gate:** expected_pad_overhead ≤ 10% (very tight at uniform retention). If higher, increase bucket count.

## 7. Tests

### 7.1 Parity tests (fast suite)

`tests/test_sparse_int4_fwd_compact_parity.py`:
- Random K/V cache, random selection of 32 pages out of 64. Compare:
  - Current `flash_attn_sparse_int4_fwd` with bool mask
  - New `flash_attn_sparse_int4_fwd_compact` with compact list
- Output max abs err < 1e-2 BF16. Must hold for all `bucket_size ∈ {16, 32, 64, 128}`.

`tests/test_sparse_int4_fwd_compact_padding.py`:
- Compact list with padding (real_count < bucket_size, padded with first page). Output identical to compact list at exact real_count.

`tests/test_sparse_int8_fwd_compact_parity.py`, `tests/test_sparse_turbo_fwd_compact_parity.py`:
- Same pattern, INT8 and Turbo K3-V3 variants.

`tests/test_select_pages_no_sync.py`:
- Run `select_pages_vectorized` with k_max precomputed; verify no `.item()` in trace via `torch.cuda.synchronize` timing or PyTorch profiler.

`tests/test_predicated_partial_merge.py`:
- Verify that always-merge logic gives correct output when:
  - `partial_len == 0` (no partial-page tail) — sparse path dominates
  - `completed_len == 0` (no completed pages, only tail) — partial path dominates
  - both non-zero — both contribute

`tests/test_graph_dispatcher.py`:
- Capture graphs for buckets `[128, 144]` on a 2-layer toy decoder with compact-list kernel. Replay with various selection counts. Output identical to eager-mode (max abs err < 1e-3).
- Bucket overflow: actual_count > max_bucket → falls back to eager, returns correct output.
- Page-flush boundary: capture with `partial_len = 63`, replay with `partial_len = 0` (just rolled over). Predicated path handles both correctly.

`tests/test_fused_qkv_triton.py`:
- Random AWQ-INT4 weights for q/k/v. Compare AWQ per-Linear output to fused Triton output. Max abs err < 5e-2 BF16.

`tests/test_fused_gate_up_triton.py`:
- Same pattern for gate+up + SwiGLU.

`tests/test_calibrate_buckets.py`:
- Synthetic histogram input. Verify bucket selection logic + p99 coverage + pad-overhead constraint.

### 7.2 Quality gate (slow suite)

`tests/test_phase8_ruler_4k.py` (slow):
- RULER NIAH 4k @ Llama-3.2-3B AWQ-INT4 with `--compact-kernel --cuda-graphs --kv-bits 4`.
- All 3 categories ≥85% (Phase 7 baseline).
- Stretch: 100/100/100 (Phase 6 INT4 baseline).

`tests/test_phase8_logit_parity_32k.py` (slow):
- Run dense (no Phase 8) and Phase 8 on a 32k prompt. Compare next-token logit top-k overlap. Top-1 must match; top-5 overlap ≥4/5. (Codex finding #12: 32k logit parity beyond RULER 4k.)

### 7.3 Performance benches (slow)

`benchmarks/phase8_decode_8k.py`:
- 8k decode tok/s @ AWQ-INT4 + INT4 KV + compact + graphs. Expect ≥7 tok/s (vs Phase 6 4.94).

`benchmarks/phase8_decode_32k.py`:
- 32k decode tok/s @ same config. Expect ≥6 tok/s floor (vs Phase 6 3.88), stretch ≥7.5.

`benchmarks/phase8_ablation.py`:
- Per-feature on/off matrix. 4 cells:
  - Phase 7 baseline (no Phase 8)
  - Compact kernel only (no graphs, no fused proj)
  - Compact + graphs (no fused proj)
  - Compact + graphs + fused proj (full Phase 8)
- Confirms each component's contribution. Codex finding #12.

`benchmarks/phase8_headtohead.py`:
- Same orchestrator as Phase 7. Backends: flashquest Phase 8 (this), flashquest Phase 6 INT4 (control), llama.cpp Q4_K_M, vLLM AWQ. Cells: 8k decode, 32k decode. Reuse Phase 7's vLLM/llama.cpp 32k+128k cells (they don't change between phases).

## 8. Acceptance gates

| Gate | Floor | Stretch | Test/Bench |
|---|---|---|---|
| Compact kernel parity (vs current) | max abs err < 1e-2 BF16, all kv_bits | 0 errors | `test_sparse_*_compact_parity.py` |
| RULER NIAH 4k all 3 cats | ≥85% (no regression vs Phase 7) | 100/100/100 | `test_phase8_ruler_4k.py` |
| 32k logit parity (top-1) | match dense | top-5 ≥4/5 | `test_phase8_logit_parity_32k.py` |
| 32k decode tok/s (kv-bits 4) | ≥6 (1.55× current 3.88) | ≥7.5 (1.94×) | `phase8_decode_32k.py` |
| 8k decode tok/s | ≥7 (1.42× current 4.94) | ≥9 (1.82×) | `phase8_decode_8k.py` |
| Per-feature ablation | each component non-negative | each ≥1.05× | `phase8_ablation.py` |
| Peak VRAM @ 32k | ≤ Phase 7 + 400 MiB | ≤ Phase 7 + 200 MiB | bench output |
| Fast suite | 227+ pass, 0 regressions | — | `pytest tests/ -m "not slow"` |
| Compatibility | `--kv-bits {3,4,8}` all still pass | — | bench at each kv-bits |
| Pad overhead (calib) | ≤10% wasted compute | ≤5% | `_bucket_calibration_32k.json` |
| Graph capture sanity (WSL2) | toy 2-layer captures + replays | — | sanity test in Task 2 |

VRAM gate raised to +400 MiB (Phase 7 already at 6105 MiB; graph pools + static buffers add ~150-200 MiB per bucket × 2 buckets + static input/output buffers). Fits within 4 GB total VRAM with margin.

## 9. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Compact kernel doesn't beat full-mask kernel | Low | Phase 8 ceiling drops to graphs+fusion only (~1.3×) | Microbench in Task 3 (compact vs current at 32k shapes); if no win, kernel unchanged but graphs may still help. |
| WSL2 CUDA Graph capture broken | Low (CUDA 12 supports) | Phase 8 ceiling drops to compact+fusion only (~1.5×) | Sanity test on toy 2-layer in Task 2 BEFORE full integration. |
| Predicated partial-merge has measurable overhead | Medium | ~5-10% extra compute when one path is empty | Acceptable; eliminates sync. Measure in Task 8 ablation. |
| Bucket calibration shows tight distribution (1 bucket) | Low | Simpler dispatcher, may not need bucketing | Good outcome; reduces graph-capture surface. |
| Bucket overflow rate > 1% | Low | Eager fallback eats graph win | Calibrate with margin (largest bucket = p99 + 10%). |
| `searchsorted` GPU-resident bucket lookup slow | Low | Negligible latency add | Single bucket case skips searchsorted entirely. |
| Page-flush predicate wrong | Medium | Wrong output near page boundary | Parity test (`test_graph_dispatcher.py` page-flush case) catches it. |
| Sentinel `-1` padding kernel-side skip has bugs | Low | Wrong output near bucket boundary | Parity test verifies sentinel padding equals exact-count compact-list output. (Initial "pad with first page repeated" idea was rejected — duplicate skews softmax weighted average when other pages are present.) |
| Triton fused QKV/gate+up doesn't beat AWQ | Medium | Component drops, ceiling −5-10% | Microbench in Task 11; if no win, drop. Don't ship a regression. |
| AWQ-INT4 BF16 dequant in Triton has accuracy issues | Low | Quality gate fail | RULER 4k catches it. |

## 10. Non-goals (explicitly deferred)

- **Marlin INT4 GEMM** — Phase 9 (post-specdec, when M ≥ 8). Phase 6 notes already document Marlin at M=1 ≈ AWQ.
- **Async two-stream layer pipeline** — DROPPED (broken dependency model — same layer writes K/V then reads its own cache for attention).
- **AWQ → Marlin weight conversion** — Phase 9.
- **Speculative decoding (any flavor)** — Phase 9. EAGLE-2 cache integration (2-3 weeks per Phase 6 notes) tackled there.
- **DuoAttention head split** — Phase 10. Phase 6 notes flag: no Llama-3.2-3B head pattern exists; either train via DuoAttention's `run_train.sh` or use all-retrieval baseline.
- **Activation sparsity (CATS)** — Phase 10.
- **Lookahead / Jacobi / prompt-lookup** — Phase 11.
- **LayerSkip / early exit** — bonus Phase 12.
- **Token-level Quest (LSH-based)** — bonus Phase 13.
- **Calibrated TurboQuant codebook** — bonus Phase 14.
- **Persistent fused decoder kernel** — bonus Phase 15.
- **Prefill optimization** — Phase 8 leaves prefill as-is (BF16 SDPA).
- **Multi-batch / continuous batching** — single-user laptop runtime; not in scope.

## 11. Novel-angle callout

The novel piece of Phase 8 is the **GPU-resident sparse-attention dispatch under bucketed CUDA Graph capture**. Three sub-pieces, none in any public LLM runtime as of 2026-05:
- Compact-list sparse paged attention kernel taking `selected_page_ids: int32[BUCKET]` (vs vLLM's full-page-table indexing or current bool-mask iteration).
- Bucketed graph capture keyed by quantized top-k count, with sentinel `-1` padding and kernel-side `tl.where(p >= 0, ...)` skip for sub-bucket selections.
- Predicated partial-page tail merge that eliminates `partial_len == 0` / `completed_len == 0` Python branches.

Combined, this is the technique that lets sparse paged attention compose with CUDA Graphs at decode batch=1. If it works at the gate, the technique generalizes to any sparse paged attention runtime and is independently publishable.

## 12. Surface — files to create / modify

### New files
- `src/flashquest/kernel/sparse_int4_fwd_compact.py` — compact-list INT4 kernel
- `src/flashquest/kernel/sparse_int8_fwd_compact.py` — compact-list INT8 kernel
- `src/flashquest/kernel/sparse_turbo_fwd_compact.py` — compact-list TurboQuant kernel
- `src/flashquest/kernel/fused_proj.py` — fused QKV + gate+up Triton GEMM
- `src/flashquest/runtime/__init__.py`
- `src/flashquest/runtime/graph_dispatcher.py` — `GraphDispatcher`
- `src/flashquest/runtime/calibrate_buckets.py` — calibration utility
- `src/flashquest/runtime/_bucket_calibration_32k.json` — calibration output (committed)
- `tests/test_sparse_int4_fwd_compact_parity.py`
- `tests/test_sparse_int8_fwd_compact_parity.py`
- `tests/test_sparse_turbo_fwd_compact_parity.py`
- `tests/test_sparse_int4_fwd_compact_padding.py`
- `tests/test_select_pages_no_sync.py`
- `tests/test_predicated_partial_merge.py`
- `tests/test_graph_dispatcher.py`
- `tests/test_fused_qkv_triton.py`
- `tests/test_fused_gate_up_triton.py`
- `tests/test_calibrate_buckets.py`
- `tests/test_phase8_ruler_4k.py` (slow)
- `tests/test_phase8_logit_parity_32k.py` (slow)
- `benchmarks/phase8_decode_8k.py`
- `benchmarks/phase8_decode_32k.py`
- `benchmarks/phase8_ablation.py`
- `benchmarks/phase8_headtohead.py`
- `docs/PHASES/phase-8-notes.md` (post-execution writeup)

### Modified files
- `src/flashquest/eager/selection.py` — eliminate `.item()` (precompute k_max from retention × max_P)
- `src/flashquest/eager/llama_persistent_patch.py` — wire compact kernel + graph dispatcher; replace partial/completed Python branches with predicated execution; integrate fused proj
- `src/flashquest/cache/persistent_int4.py`, `persistent_int8.py`, `persistent_turbo.py` — confirm post-RoPE storage; expose `partial_len`, `completed_len` as GPU-resident scalar tensors (not Python ints) where graphs need them
- `scripts/bench_flashquest.py` — add `--compact-kernel / --no-compact-kernel`, `--cuda-graphs / --no-cuda-graphs`, `--graph-buckets` flags
- `src/flashquest/cli.py` — same flags
- `README.md` — Phase 8 section under benchmarks (post-execution)
- `DOC.md` — Phase 8 section (post-execution)

### Deleted (none — keep current bool-mask kernels as fallback; they remain importable)

## 13. Implementation order (preview — full plan in writing-plans)

Rough sequencing (15 tasks):
1. **Calibration** — run §6 to get real 32k bucket sizes. Update spec values if surprising.
2. **WSL2 CUDA Graph sanity** — toy 2-layer model captures + replays cleanly. De-risk environment.
3. **Compact INT4 kernel** — write `sparse_int4_fwd_compact`, parity test against bool-mask version.
4. **Compact INT8 + Turbo kernels** — same pattern.
5. **Eliminate `.item()` in selection** — precompute k_max from retention × max_P.
6. **Predicated partial-merge** — replace `if partial_len == 0` / `if completed_len == 0` with always-merge.
7. **Page-flush predicate** — handle the partial_len rollover boundary inside the graph (no re-capture).
8. **Wire compact kernel into Llama patch** (no graphs yet, eager mode). Quality gate: RULER 4k holds.
9. **Per-feature ablation 1**: compact-only tok/s vs Phase 7 baseline. Validate kernel wins.
10. **Graph dispatcher implementation** — capture + replay logic.
11. **Wire dispatcher into Llama patch**. Capture at warmup; replay per step.
12. **Per-feature ablation 2**: compact + graphs tok/s. Validate graph win.
13. **Fused QKV / gate+up Triton GEMM** — write kernel, parity test against AWQ.
14. **Wire fused proj into Llama patch**.
15. **Final RULER + 32k logit parity + bench + writeup**.

## 14. References

- **Quest paper:** "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference" — Tang et al., ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774).
- **CUDA Graphs guide:** NVIDIA CUDA C Programming Guide §3.2.6.
- **PyTorch CUDA Graphs:** [pytorch.org/docs/stable/notes/cuda.html#cuda-graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
- **Triton tutorials — Fused matrix multiplication:** [triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
- **AutoAWQ GEMM kernel reference:** [github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py](https://github.com/casper-hansen/AutoAWQ).
- **Phase 6 notes (Marlin batch=1 finding):** `docs/PHASES/phase-6-notes.md:148`.
- **Phase 7 spec:** `docs/superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md`.
- **Phase 7 notes:** `docs/PHASES/phase-7-notes.md`.
- **Codex review of r1:** internal — see commit log. 12 ranked findings; r2 addresses all.

---

## Appendix A — Revision history

### r1 → r2 diff (codex review 2026-05-07)

**Removed (dealbreakers):**
- §4.1 Marlin INT4×BF16 GEMM (Phase 6 notes already show Marlin ≈ AWQ at M=1)
- §4.2 AWQ → Marlin weight repack (only matters when Marlin lands)
- §4.3 r1's Fused QKV (rewritten to Triton-direct, not Marlin-based)
- §4.6 Async two-stream layer pipeline (broken dependency: same layer write→read)
- r1's bucket sizes `[8, 16, 24, 32]` (calibrated for 4k, not 32k)

**Added (corrections):**
- §4.1 Compact-list sparse kernel rewrite (real fix to bucketing not reducing kernel work)
- §4.2 Host-sync elimination (`.item()` calls in selection + partial/completed Python branches)
- §4.3 Bucketed graph dispatcher rebuilt over compact kernel + GPU-resident bucket lookup
- §4.4 Triton fused QKV/gate+up (no Marlin)
- §6 Real-32k calibration (k≈128, not 8-32)
- Per-feature ablation as a gate
- 32k logit parity test
- WSL2 graph-capture sanity test in Task 2

**Updated:**
- Ceiling claims: r1 said 1.8-2.6×; r2 says 1.55-1.94× (honest given Marlin defer).
- VRAM gate: r1 +200 MiB; r2 +400 MiB (CUDA Graph pools).
- Risks: r2 includes WSL2 graph compat, predicated partial overhead, padding-page softmax math.

**Process note:** I should have read Phase 6 notes (Marlin M=1 finding) and Phase 6's actual kernel structure before writing r1. Codex caught what I missed by skipping that ground-up read. Adding "re-read prior phase findings before drafting" as a personal pre-spec habit.
