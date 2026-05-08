# Phase 8 — Marlin INT4 GEMM + Bucketed CUDA Graphs + Fusions — Design

**Author:** flashquest core
**Date:** 2026-05-07
**Status:** Approved (brainstorming complete; spec ready for plan)
**Phase 7 baseline:** [`docs/PHASES/phase-7-notes.md`](../../PHASES/phase-7-notes.md) — 3.88 tok/s @ 32k INT4 fused, 2.62 tok/s @ 32k TurboQuant K3-V3.
**Roadmap position:** First of a 4-phase order-of-magnitude push: **Phase 8 Foundation** → 9 Speculative decoding → 10 DuoAttention head split + CATS activation sparsity → 11 Lookahead/Jacobi + prompt-lookup. Bonus phases queued (LayerSkip, token-level Quest, calibrated TurboQuant codebook, persistent fused decoder).

---

## 1. Goal

Take Llama-3.2-3B AWQ-INT4 decode throughput on RTX 3050 Ti Laptop (sm_86, 4 GB VRAM, WSL2) from **3.88 tok/s @ 32k → ≥7 tok/s (1.8× floor) / ≥10 tok/s stretch (2.6×)**, without changing the model, KV bit-width, or RULER NIAH 4k quality.

All speedup comes from the decode-side path. Prefill stays through dense BF16 SDPA — Marlin and the graph dispatcher are decode-only. The phase explicitly establishes the kernel + runtime foundation that Phase 9-11 compose on top of.

## 2. Background — why these wins exist

Three classes of inefficiency in the current decode loop:

### 2.1 AWQ INT4 weights, BF16 GEMM execution
Current AutoAWQ kernel path runs `dequant INT4 → BF16 → matmul against BF16 activations`. The dequant intermediate is materialized in registers/SMEM, but the matmul itself runs at BF16 tensor-core throughput, not INT4. On Ampere (sm_86) the tensor cores natively support INT4×INT4 → INT32 accumulation at **8× the BF16 throughput per cycle**. Marlin (IST-DASLab, NeurIPS 2024) is a fused INT4-weight × FP16-activation GEMM that keeps weights in INT4 in registers, dequantizes per-tile inside the MMA loop, and skips the BF16 intermediate. Real Ampere measurements show 1.5-2× over dequant-then-matmul at batch=1 decoding.

### 2.2 Per-step Python + Triton launch overhead
The decode step runs ~7 GEMMs (Q, K, V, O, gate, up, down) + RoPE + attention + RMSNorm × 28 layers = ~250 kernel launches per token. Each kernel launch costs ~10-30 µs of CPU + driver overhead. At batch=1 small-shape kernels, launch overhead is on the order of 30-50% of decode wall time. CUDA Graphs collapse the launch sequence into a single replayable graph, dropping Python/driver time to near-zero.

### 2.3 GPU underutilization from sequential layer dependence
Each layer's KV writeback to the persistent paged cache is serial with the next layer's projection start. Writeback is INT4-pack + scale-compute (the `_pack_int4` and quantize_v paths); projection is the next layer's QKV GEMM. They use disjoint resources (memory store vs. tensor-core compute) and can overlap on a single GPU via two CUDA streams.

### 2.4 What blocks naïve CUDA Graphs under Quest
Quest selects a **dynamic top-k page count per step**. CUDA Graphs require static shapes. Public CUDA-Graph-using runtimes (vLLM, TensorRT-LLM) all use it with **dense** or static-shape attention. No public repo captures graphs across dynamic sparse-page selection. Phase 8's bucketed dispatcher is the genuinely novel piece: it discretizes the page-count axis into fixed buckets, pads selection masks within a bucket to a static count, and routes each step to the matching pre-captured graph.

## 3. Design overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 8 decode step (per layer × 28 layers)                        │
│                                                                     │
│  input ─┐                                                           │
│         ├── Marlin fused QKV GEMM (INT4×BF16, 1 GEMM not 3)         │
│         │   slice → Q, K, V                                         │
│         │   RoPE on Q, K                                            │
│         │                                                           │
│         │ [stream_kv async] ─── write K,V to persistent paged cache │
│         │                       (post-RoPE storage; no recompute)   │
│         │                                                           │
│         ├── Quest criticality (already exists) → top-k pages        │
│         │   round top-k count UP to nearest bucket                  │
│         │                                                           │
│         │   ┌───────────────────────────────────────────────────┐   │
│         │   │  graph_cache[bucket].replay()                     │   │
│         │   │  (pre-captured CUDA Graph)                        │   │
│         │   │     sparse_int4_fwd kernel                        │   │
│         │   │       (existing fused kernel from Phase 6 task 6) │   │
│         │   │     LSE merge with partial-page tail              │   │
│         │   │     Marlin O projection                           │   │
│         │   │     Marlin fused gate+up GEMM                     │   │
│         │   │     SwiGLU                                        │   │
│         │   │     Marlin down projection                        │   │
│         │   │     residual + RMSNorm                            │   │
│         │   └───────────────────────────────────────────────────┘   │
│         │                                                           │
│         └── output (next-token hidden state)                        │
└─────────────────────────────────────────────────────────────────────┘
```

The two CUDA streams (`stream_compute`, `stream_kv`) interleave: layer N+1's QKV GEMM begins on `stream_compute` as soon as layer N's residual+norm completes, while layer N's KV writeback continues on `stream_kv`. Stream events at the layer boundary serialize only the data dependency (next layer's hidden state).

## 4. Components

### 4.1 Marlin INT4 × BF16 GEMM kernel

**Source:** vendored from [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) (MIT licensed). Direct C++/CUDA kernel.

**Surface:** `flashquest.quant.marlin_linear.MarlinLinear(in_features, out_features, group_size=128, bias=False)` — drop-in replacement for `nn.Linear`. Forward signature: `(B, ..., in_features) BF16 → (B, ..., out_features) BF16`. Internally calls Marlin's `mul()` C extension.

**Constraints:**
- Group size 128 (matches AWQ default; Marlin also supports 64).
- `out_features` must be divisible by 256 (Marlin's tile constraint).
- Input shape must be 2D-flattenable; nothing else.

**Vendoring:** copy `marlin/marlin_cuda_kernel.cu`, `marlin/marlin_cuda.cpp`, `marlin/__init__.py` into `flashquest/quant/_marlin/`. Build with `setup.py` (extension already part of pyproject.toml's torch build deps). Pinned commit, vendored as source — do not depend on `pip install marlin` (unmaintained PyPI package).

### 4.2 AWQ → Marlin weight repack utility

**Surface:** `flashquest.quant.awq_to_marlin.repack(awq_state_dict) → marlin_state_dict`.

**Logic:** AWQ stores INT4 weights as `(out//8, in)` packed-int32 with `(in_groups, out)` BF16 scales and `(in_groups, out//8)` packed-int32 zero-points. Marlin expects `(out, in//pack)` permuted INT4 with `(in_groups, out)` FP16 scales and *no* zero-points (Marlin uses symmetric quantization).

The conversion:
1. Unpack AWQ INT4 → INT8 dense weight tensor.
2. Apply AWQ zero-points to convert to symmetric: `w_sym = w_int4 - zero_point`. Result is signed INT4 in `[-8, 7]`.
3. Re-permute to Marlin's interleaved layout (every 8th column adjacent for 4-bit packing along the K-dim of the GEMM tile).
4. Cast scales BF16 → FP16 (Marlin uses FP16 scales). Loss is negligible since AWQ scales are already coarse-grained.

**Reference implementations:**
- [vLLM `awq_marlin.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_marlin.py) — production-grade conversion path.
- [AutoGPTQ `marlin_utils.py`](https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/utils/marlin_utils.py) — simpler reference.

We do NOT depend on either at runtime — we vendor the conversion logic only.

**Output:** new state dict with keys `{layer}.{q,k,v,o,gate,up,down}_proj.weight_marlin`, `weight_scale_marlin`. AWQ keys are dropped after conversion. Conversion runs once at model load (cached on disk in `~/.cache/flashquest/marlin-{model_id}/` to avoid re-running every session).

### 4.3 Fused QKV + fused gate+up projections

Llama-3.2-3B has separate `q_proj`, `k_proj`, `v_proj` (3 GEMMs). With GQA (24 Q heads, 8 KV heads) and head_dim=128:
- `q_proj`: 3072 × 3072
- `k_proj`: 3072 × 1024
- `v_proj`: 3072 × 1024

Stacked vertically into one Marlin weight `[5120, 3072]`, one GEMM, then sliced into Q/K/V slices.

Same for `gate_proj` (3072 × 8192) + `up_proj` (3072 × 8192) → stacked `[16384, 3072]`. One GEMM, slice into gate/up, apply SwiGLU.

Result per layer: 4 projection GEMMs (qkv-fused, o, gate-up-fused, down) instead of 7. ~10-15% standalone gain at decode batch=1 because fewer launch overhead + larger GEMM tiles fit Marlin's tile constraint better.

**Surface:** `flashquest.quant.marlin_linear.FusedMarlinLinear(in_features, out_features_list, group_size=128)` — internally one Marlin weight, returns a tuple of sliced output views.

### 4.4 RoPE-post cache verification

**Audit task:** confirm that K is stored *post-RoPE* in `PersistentInt8KVCache`, `PersistentInt4KVCache`, `PersistentTurboKVCache`. If pre-RoPE, decode reads must apply RoPE on every read — wasted work.

Look at `kv_quant.quantize_k`'s caller path in `eager/llama_persistent_patch.py` (it should already write post-RoPE because that's standard HF practice). Audit and document.

If pre-RoPE: fix by applying RoPE before quantize_k call, and verify decode reads use cached K directly without re-rotation.

This is small but important — RoPE is ~3% of decode time.

### 4.5 Bucketed CUDA Graph dispatcher

**Surface:** `flashquest.runtime.cuda_graph_dispatcher.GraphDispatcher`.

**State:**
- `bucket_sizes: list[int]` — e.g., `[8, 16, 24, 32]` (calibrated; see §6).
- `graphs: dict[int, torch.cuda.CUDAGraph]` — one captured graph per bucket size.
- `static_inputs: dict[int, dict[str, torch.Tensor]]` — pre-allocated input tensor stubs per bucket. The graph captures with these as inputs; replay copies actual data into them.
- `static_outputs: dict[int, torch.Tensor]` — captured output buffer per bucket.

**Capture** (one-time, during warmup):
```python
for bucket in bucket_sizes:
    # allocate static buffers
    static_inputs[bucket] = {
        "hidden_states": torch.empty((1, 1, hidden), dtype=bf16, device="cuda"),
        "selection_mask": torch.zeros((num_layers, 1, num_heads, num_kv_heads, max_pages), dtype=bool, device="cuda"),
        # ... + cache pointers, position_ids
    }
    # warmup: run forward 3× to settle autotune
    for _ in range(3):
        _decode_step_padded(static_inputs[bucket], pad_to=bucket)
    torch.cuda.synchronize()
    # capture
    graphs[bucket] = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graphs[bucket]):
        static_outputs[bucket] = _decode_step_padded(static_inputs[bucket], pad_to=bucket)
```

**Replay** (per decode step):
```python
def step(real_inputs):
    # 1. compute Quest selection (still eager — graphs don't capture this)
    selection = quest_select(...)
    actual_count = selection.sum(dim=-1).max().item()
    # 2. round up to bucket
    bucket = next(b for b in bucket_sizes if b >= actual_count)
    if bucket is None:
        # rare: top-k > max bucket. fall back to eager mode for this step.
        return _decode_step_eager(real_inputs)
    # 3. pad selection mask to bucket size
    padded = pad_selection_to_bucket(selection, bucket)
    # 4. copy real inputs into static buffers
    static_inputs[bucket]["hidden_states"].copy_(real_inputs["hidden_states"])
    static_inputs[bucket]["selection_mask"].copy_(padded)
    # ...
    # 5. replay
    graphs[bucket].replay()
    # 6. return output
    return static_outputs[bucket].clone()
```

**Padding rule:** the `selection_mask` is bool over pages. Padded slots are False (kernel masks them out). The kernel iterates over all `bucket` slots regardless — wasted compute on padding slots, but they zero-out via the mask and contribute nothing.

**Re-capture trigger:** if the real `actual_count` exceeds `max(bucket_sizes)` 2 steps in a row, log a warning + fall back to eager. If sustained, recapture with a larger bucket (one-time cost, but rare).

### 4.6 Async layer pipeline (two-stream)

**Surface:** built into the patched Llama forward. Two streams:
- `stream_compute = torch.cuda.current_stream()` — default stream for tensor-core compute.
- `stream_kv = torch.cuda.Stream(priority=-1)` — high-priority stream for KV writeback.

**Per layer:**
```python
# layer N
hidden_n = layer_n_compute(hidden_in)  # on stream_compute

# kick KV writeback for layer N onto stream_kv
with torch.cuda.stream(stream_kv):
    cache.write_layer_n(K_n, V_n)

event_kv = stream_kv.record_event()

# layer N+1 begins on stream_compute IMMEDIATELY using hidden_n
# (no dependency on KV writeback completing)
hidden_np1 = layer_np1_compute(hidden_n)
# ...

# at end of layer N+1's attention, before reading layer N's KV:
# (we don't actually need this — layer N+1 reads its own KV after it writes it)
# the only sync is at the END of the layer stack, before logits
stream_compute.wait_event(stream_kv.record_event())
```

Note: each layer reads its OWN KV cache (which it just wrote) within the same step. So the only cross-layer KV dependency is: layer N+1 reads layer N+1's just-written cache, which IS on the same stream. The pipelining hides the *writeback to persistent paged storage* behind the next layer's compute, not the cache READ by the next layer.

**Within CUDA Graphs:** stream events are graph-capturable in CUDA 11.4+. Both streams are captured and replayed together.

### 4.7 Calibration utility — bucket boundaries from RULER

**Surface:** `flashquest.runtime.calibrate_buckets.collect_selection_histogram(model, prompts, retention=0.25) → list[int]`.

**Logic:**
1. Run the model in current Phase 7 mode (no Marlin, no graphs) on RULER NIAH 4k prompts (60 prompts).
2. At each decode step, log the actual `top-k page count` per layer per head-group.
3. Build histogram across all (step × layer × head-group) tuples.
4. Choose 4 bucket boundaries that cover ≥95% of decode steps with ≤30% wasted padding on average.
5. Heuristic: pick boundaries at `[p25, p50, p75, p95]` percentiles of the histogram, snap to multiples of 8.

**Output:** `[bucket_1, bucket_2, bucket_3, bucket_4]` saved to `flashquest/runtime/_bucket_calibration.json`. Can be re-run per model via `python -m flashquest.runtime.calibrate_buckets`.

If the histogram is heavily skewed (e.g., 99% of steps fit in 1 bucket), the dispatcher may collapse to 1-2 buckets. The plan accommodates `len(bucket_sizes) ∈ [1, 6]`.

## 5. Compatibility matrix

|  `--kv-bits` | Marlin | Graphs | Status after Phase 8 |
|---|---|---|---|
| 4 (Phase 6 INT4) | ✅ | ✅ | Primary target |
| 8 (Phase 3 INT8) | ✅ | ✅ | Maintained |
| 3 (Phase 7 Turbo) | ✅ | ✅ | Maintained — same dispatcher branches on cache.kv_bits |
| any | ✅ | OFF (`--no-cuda-graphs`) | Fallback for debugging |
| any | OFF (`--no-marlin`) | ✅ | Fallback if Marlin underperforms |

Both flags default ON post-validation. Defaults OFF during Phase 8 task progression until each is validated independently.

## 6. Calibration step (Task 1 of plan)

Before any kernel work, the calibration step grounds the bucket boundaries.

```bash
python -m flashquest.runtime.calibrate_buckets \
    --model unsloth/Llama-3.2-3B-Instruct \
    --kv-bits 4 \
    --retention 0.25 \
    --prompts ruler:niah_single,niah_multikey,niah_multivalue \
    --num-samples 20 \
    --output flashquest/runtime/_bucket_calibration.json
```

Output (example):
```json
{
  "model": "unsloth/Llama-3.2-3B-Instruct",
  "kv_bits": 4,
  "retention": 0.25,
  "histogram": {"4": 12, "8": 41, "12": 33, "16": 18, "20": 8, "24": 3, "32": 1, "...": "..."},
  "bucket_sizes": [8, 16, 24, 32],
  "p95_count": 24,
  "max_count": 38,
  "expected_pad_overhead": 0.18
}
```

The expected_pad_overhead is the wasted compute fraction from padding. ≤30% is the gate; if higher, increase bucket count to 5-6.

## 7. Tests

### 7.1 Parity tests (fast suite)

`tests/test_marlin_linear_parity.py`:
- Random INT4 weights `(out=512, in=512)`. Convert AWQ → Marlin. Compare BF16 reference (`F.linear(x.float(), w_dequant.float())`) vs Marlin output. Max abs err < 2e-2 BF16 (Marlin's internal FP16 + scale-cast loss).
- Edge case: `out_features = 256` (minimum tile).
- Edge case: `out_features = 8192` (large MLP-up shape).

`tests/test_marlin_qkv_fused.py`:
- Build fused QKV with 3 separate `MarlinLinear`s, then with one `FusedMarlinLinear`. Sliced fused output must equal concatenated separate outputs to 1e-3.

`tests/test_marlin_gate_up_fused.py`:
- Same pattern for gate+up. SwiGLU(fused) == SwiGLU(separate) to 1e-3.

`tests/test_cuda_graph_dispatcher.py`:
- Capture graphs for buckets `[8, 16, 24, 32]` on a 2-layer toy decoder. Replay each bucket with a real selection of size 5, 10, 20, 30. Output must equal eager-mode decode output to 1e-3.
- Bucket overflow (selection size 40, max bucket 32): falls back to eager, returns correct output.
- Re-capture: shift context length so `max_pages` increases; re-capture works without segfault.

`tests/test_async_pipeline_parity.py`:
- 2-layer toy decoder. Run with serial layer execution vs. async pipeline. Last hidden state identical to 1e-3.

`tests/test_awq_to_marlin_repack.py`:
- Convert one Llama-3.2-3B layer's AWQ weights to Marlin. Forward through both:
  - AWQ path with mock dequant + BF16 GEMM (BF16 reference)
  - Marlin path
- Output max abs err < 5e-2 BF16 on a single (1, 1, 3072) input.

`tests/test_calibrate_buckets.py`:
- Synthetic histogram input: `{4: 100, 8: 50, 16: 30, 32: 5}`. Run bucket selection logic. Verify p95-coverage and pad-overhead constraints.

### 7.2 Quality gate (slow suite)

`tests/test_phase8_ruler_4k.py` (slow):
- RULER NIAH 4k @ Llama-3.2-3B AWQ-INT4 with `--marlin --cuda-graphs --kv-bits 4`.
- All 3 categories ≥85% (Phase 7 baseline).
- Stretch: 100/100/100 (Phase 6 INT4 baseline).
- Run on 20 prompts per category (60 total).

### 7.3 Performance benches (slow)

`benchmarks/phase8_decode_8k.py`:
- 8k decode tok/s at AWQ-INT4 + INT4 KV + Marlin + graphs. Expect ≥9 tok/s (vs Phase 6 4.94).

`benchmarks/phase8_decode_32k.py`:
- 32k decode tok/s at same config. Expect ≥7 tok/s (vs Phase 6 3.88), stretch ≥10.

`benchmarks/phase8_headtohead.py`:
- Same orchestrator as Phase 7 head-to-head. Backends: flashquest Phase 8 (this), flashquest Phase 6 INT4 (control), llama.cpp Q4_K_M, vLLM AWQ. Cells: 8k decode, 32k decode. Reuse Phase 7's vLLM/llama.cpp 32k+128k cells (they don't change between phases).

`benchmarks/phase8_marlin_microbench.py`:
- Per-layer Marlin GEMM vs AWQ GEMM at decode shapes (1×3072 → 1×{3072, 1024, 8192}). Confirm Marlin > AWQ at batch=1 BEFORE integrating. Gate: Marlin ≥1.3× faster, otherwise drop Marlin from Phase 8.

## 8. Acceptance gates

| Gate | Floor | Stretch | Test/Bench |
|---|---|---|---|
| Marlin microbench batch=1 | ≥1.3× AWQ | ≥1.8× | `phase8_marlin_microbench.py` |
| RULER NIAH 4k all 3 cats | ≥85% (no regression vs Phase 7) | 100/100/100 | `test_phase8_ruler_4k.py` |
| 32k decode tok/s (kv-bits 4) | ≥7 (1.8× current 3.88) | ≥10 (2.6×) | `phase8_decode_32k.py` |
| 8k decode tok/s | ≥9 (1.8× current 4.94) | ≥12 | `phase8_decode_8k.py` |
| Peak VRAM @ 32k | ≤ Phase 7 + 200 MiB | ≤ Phase 7 | bench output |
| Fast suite | 227+ pass, 0 regressions | — | `pytest tests/ -m "not slow"` |
| Compatibility | `--kv-bits {3,4,8}` all still pass | — | bench at each kv-bits |
| Pad overhead (calib) | ≤30% wasted compute | ≤15% | `_bucket_calibration.json` |

If Marlin microbench fails (Gate 1), Phase 8 ships graphs+fusions+pipeline only (~1.5×). Plan reflects this fallback.

## 9. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Marlin batch=1 underperforms AWQ | Medium | Phase 8 ceiling drops 30% | Microbench in Task 2 BEFORE integration. If fails, drop Marlin, ship graphs+fusions only. |
| CUDA Graphs broken in WSL2 | Low (CUDA 12 supports it) | Phase 8 ceiling drops 25% | Sanity-test capture+replay on toy in Task 6 BEFORE Llama integration. If fails, defer to a Phase 8.5. |
| Bucket selection misses real distribution | Medium | Pad overhead > 30%, undermining gain | Calibration in Task 1 sets boundaries from real RULER prompts; test catches it. |
| Selection > max bucket | Low (RULER p95 ~24) | Fallback path activates | Eager fallback for that step; log warning; if sustained, recapture with larger bucket. |
| Async stream data race | Medium | Wrong output | Stream events at layer boundary; parity test (`test_async_pipeline_parity.py`) catches it. |
| RoPE-post audit reveals K is pre-RoPE | Low | One-time fix | Task 5 audits + fixes. |
| AWQ → Marlin conversion mismatch | Medium | Output corruption | Conversion mirrors vLLM's awq_marlin path; parity test catches it. |
| Marlin tile constraint (out % 256 == 0) violated | Low | Some shapes don't fit | Llama-3.2-3B all linear shapes are %256 OK (3072, 1024, 8192). Verified. |
| Graph re-capture overhead exceeds gains | Low | Net regression | Measure recapture frequency; if >1/100 steps, increase max bucket. |
| Marlin scale FP16 cast loses quality | Low | RULER drops below gate | RULER gate in Task 13 catches it. |

## 10. Non-goals (explicitly deferred)

- **Speculative decoding** (any flavor) — Phase 9.
- **DuoAttention head split** — Phase 10. (Note: `head_pattern_layer` parameter already in dispatcher signature; Phase 8 leaves it unused.)
- **Activation sparsity (CATS)** — Phase 10.
- **Lookahead / Jacobi / prompt-lookup** — Phase 11.
- **LayerSkip / early exit** — bonus Phase 12.
- **Token-level Quest (LSH-based)** — bonus Phase 13.
- **Calibrated TurboQuant codebook** — bonus Phase 14.
- **Persistent fused decoder kernel** — bonus Phase 15.
- **Prefill optimization** — Phase 8 leaves prefill as-is (BF16 SDPA). Long-context prefill optimization is a separate axis.
- **Multi-batch / continuous batching** — single-user laptop runtime; not in scope.

## 11. Novel-angle callout

The genuinely new engineering in Phase 8 is the **bucketed CUDA Graph dispatcher under Quest's dynamic top-k page selection**. As of 2026-05, every public CUDA-Graph-using LLM runtime (vLLM, TensorRT-LLM, SGLang, MLC-LLM) captures graphs only with **dense** or **fixed-shape** attention. The page-selection-aware dispatcher (graph cache keyed by quantized top-k count, masked padding) is what lets Phase 8 combine Quest's algorithmic compression with graph-capture's launch-overhead elimination.

If Phase 8's bucketed dispatcher works at the gate, it's worth an independent writeup — the technique generalizes to any sparse paged attention runtime.

## 12. Surface — files to create / modify

### New files
- `src/flashquest/quant/marlin_linear.py` — `MarlinLinear`, `FusedMarlinLinear` nn.Modules
- `src/flashquest/quant/awq_to_marlin.py` — weight repack utility
- `src/flashquest/quant/_marlin/` — vendored CUDA kernel sources (marlin_cuda_kernel.cu, marlin_cuda.cpp, __init__.py)
- `src/flashquest/runtime/__init__.py`
- `src/flashquest/runtime/cuda_graph_dispatcher.py` — `GraphDispatcher`
- `src/flashquest/runtime/calibrate_buckets.py` — calibration utility
- `src/flashquest/runtime/_bucket_calibration.json` — calibration output (committed)
- `tests/test_marlin_linear_parity.py`
- `tests/test_marlin_qkv_fused.py`
- `tests/test_marlin_gate_up_fused.py`
- `tests/test_cuda_graph_dispatcher.py`
- `tests/test_async_pipeline_parity.py`
- `tests/test_awq_to_marlin_repack.py`
- `tests/test_calibrate_buckets.py`
- `tests/test_phase8_ruler_4k.py` (slow)
- `benchmarks/phase8_marlin_microbench.py`
- `benchmarks/phase8_decode_8k.py`
- `benchmarks/phase8_decode_32k.py`
- `benchmarks/phase8_headtohead.py`
- `docs/PHASES/phase-8-notes.md` (post-execution result writeup)

### Modified files
- `pyproject.toml` — add C++ extension build for Marlin (torch.utils.cpp_extension or scikit-build)
- `src/flashquest/eager/llama_persistent_patch.py` — wire Marlin layers + graph dispatcher + async streams into HF Llama forward
- `src/flashquest/cache/persistent_int4.py`, `persistent_int8.py`, `persistent_turbo.py` — verify post-RoPE storage; add `K_already_rotary: bool = True` flag if needed
- `scripts/bench_flashquest.py` — add `--marlin / --no-marlin`, `--cuda-graphs / --no-cuda-graphs`, `--graph-buckets` flags
- `src/flashquest/cli.py` (or wherever `flashquest chat` lives) — same flags
- `README.md` — Phase 8 section under benchmarks (post-execution)
- `DOC.md` — Phase 8 section (post-execution)

## 13. Implementation order (preview — full plan in writing-plans)

Rough sequencing (13-15 tasks):
1. Calibration utility + run on RULER → `_bucket_calibration.json`.
2. Vendor Marlin kernel; build extension; smoke test.
3. `MarlinLinear` wrapper + parity test.
4. AWQ → Marlin repack utility + parity test.
5. RoPE-post cache audit + fix (if needed).
6. CUDA Graph dispatcher (toy 2-layer model first) + parity test.
7. Marlin microbench at decode shapes — gate decision (Marlin in / out).
8. Fused QKV + fused gate+up → `FusedMarlinLinear` + parity tests.
9. Wire Marlin into Llama persistent patch (eager mode, no graphs yet).
10. Integration test: full Llama-3.2-3B forward parity (Marlin vs AWQ).
11. Wire graph dispatcher into Llama persistent patch.
12. Async layer pipeline (two-stream) wired in.
13. Full quality gate: RULER NIAH 4k.
14. Single-cell decode benches (8k, 32k).
15. Head-to-head re-test + writeup.

## 14. References

- **Marlin paper:** "Marlin: Mixed-Precision Auto-Regressive Parallel Inference of Large Language Models" — IST-DASLab, NeurIPS 2024. [arXiv:2408.11743](https://arxiv.org/abs/2408.11743). Code: [github.com/IST-DASLab/marlin](https://github.com/IST-DASLab/marlin).
- **AWQ paper:** "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" — Lin et al., MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978).
- **vLLM AWQ-Marlin path:** [vllm/model_executor/layers/quantization/awq_marlin.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_marlin.py).
- **CUDA Graphs guide:** NVIDIA CUDA C Programming Guide §3.2.6 (CUDA Graphs).
- **PyTorch CUDA Graphs:** [pytorch.org/docs/stable/notes/cuda.html#cuda-graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs).
- **Quest paper:** "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference" — Tang et al., ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774).
- **Phase 7 spec:** `docs/superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md`.
- **Phase 7 notes:** `docs/PHASES/phase-7-notes.md`.

---

## Open question for plan

The plan can decide: do we ship Marlin per-Linear first (Task 9), then add fused QKV/gate+up (Task 8 reordered)? Or is the order above (fusion together) right? Either way the parity gate is the same. The plan-writing step locks the exact order.
