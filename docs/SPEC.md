# flashquest — Specification (Remaining Work)

**Last updated:** 2026-05-01
**Status:** Phases 0–5 shipped (kernel + infra). This SPEC has been pruned to track only **what remains** for v1.0 / v2.
**Target hardware:** NVIDIA RTX 3050 Ti Laptop GPU (GA107, sm_86), Intel i9-12900H, 16GB RAM, WSL2

For what's already shipped, see `DOC.md` (canonical) and `docs/PHASES/phase-{0..5}-notes.md`. The full original SPEC is preserved in git history at tag `phase-4` if you need the full context.

---

## 1. Goal (unchanged)

Make 32k–128k context inference of 3B–8B models *actually usable* on a 4GB-VRAM laptop GPU. Today this is either impossible (OOM) or slow (CPU fallback). We close that gap with a single fused Triton kernel that composes published SOTA techniques into one inference path. Personal artifact + open-source release. Not a paper. Not a production service.

## 2. Hardware constraints (reference)

```
GPU: NVIDIA RTX 3050 Ti Laptop GPU (GA107, sm_86)
  - 16 SMs, 80 tensor cores (3rd gen, BF16/FP16/INT8/INT4)
  - 48 KB shared mem / SM, 2 MB L2
  - 4 GB GDDR6 @ ~192 GB/s, ~3.0–3.3 GB usable
  - No FP8/TMA/WGMMA (ruled out below)
CPU: Intel i9-12900H (AVX2+VNNI, no AVX-512/AMX)
RAM: 16 GB total, ~10 GB available for inference
OS: Windows + WSL2 (CUDA 12.5 driver works natively)
```

Permanently ruled out: FP8 paths, Hopper-only features (TMA, WGMMA, FlashAttention-3, ThunderKittens, DeepGEMM), datacenter-class fits, multi-GPU.

Available and used: BF16/FP16/INT8 tensor cores, `cp.async`, Triton ≥3.x, FlashAttention-2 patterns, Marlin W4A16 (sm_80+), all KIVI/Quest/DuoAttention/StreamingLLM algorithms.

## 3. VRAM budget (reference)

Working budget **3.0 GB**. The 4 GB total minus Windows display tax leaves ~3.0–3.3 GB for the inference process. Concrete model fits at 32 k context with INT8 KV at 10 % retention:

| Model | Weights | KV (INT8 sparse) | Acts + scratch | Total | Fits? |
|---|---|---|---|---|---|
| Llama-3.2-3B AWQ-INT4 | ~2.0 GB | ~70 MB | ~150 MB | ~2.2 GB | ✅ (Phase 5 confirmed) |
| Llama-3.1-8B IQ3-XXS | ~2.3 GB | ~100 MB | ~200 MB | ~2.6 GB | ✅ tight (not yet attempted) |
| Llama-3.1-8B AWQ-INT4 | ~4.5 GB | — | — | exceeds | ❌ over 4 GB before KV (Phase 5 confirmed) |
| Mistral-7B IQ4-XS | ~3.2 GB | ~80 MB | ~200 MB | ~3.5 GB | ❌ needs CPU offload |

The 20× KV reduction (FP16 dense → INT8 + 10 % sparse) is the entire reason this project is feasible. KIVI's INT4 path would compress further but is deferred (see §5.2 below).

## 4. Architecture — what's still missing

The sparse Triton kernel exists today (`flashquest.kernel.flash_attn_sparse_fwd`) and runs the inner loop correctly. But the orchestration around it is in Python and dominates wall time at long context (Phase 5 measured **0.09 tok/s decode at 32 k vs SPEC target ≥4 tok/s**). The remaining architectural changes:

```
1. Criticality + top-k selection — currently Python (fp32 intermediates,
   per-head loop with .item() syncs). Fold into the Triton kernel:
   - Compute per-page channel min/max from the same uint8 K we already touch.
   - QK summary per page in registers, top-k via shared-mem sort.
2. Single fused dispatch per layer — eliminate the host-side
   dequantize_k → page_summary → page_scores → select_pages chain.
3. INT4 KV path — KIVI's full target. Replace per-page channel-wise
   uint8 K + per-token uint8 V with packed 4-bit, dequant on load.
4. Marlin W4A16 projections — only after profiling shows AWQ kernel
   is the bottleneck. Pack-conversion at load time.
5. Logical → physical block table (vLLM PagedAttention layout) —
   needed only if we push to multi-request batching, which we won't
   for v1. Keep contiguous storage for now.
```

## 5. SOTA techniques — remaining items

For each: what we still need to crib and what we'd skip.

### 5.2 KIVI — INT4 KV (currently INT8)

- **Repo:** https://github.com/jy-yuan/KIVI
- **What's done:** asymmetric uint8 KV with per-page channel-wise K, per-token V (Phase 3).
- **What's left:**
  - Packed 4-bit (two values per byte) K and V buffers.
  - Triton kernel-side 4-bit unpack into BF16 registers (Marlin-style nibble splat + scale).
  - Re-validate quality on passkey + (when available) RULER.
- **What we still skip:** their FP16 partial-residual buffer for the most-recent few tokens. Quality cost accepted.
- **Reported:** 2.6× peak memory reduction at 2-bit, 2.35–3.47× throughput.

### 5.5 Marlin — W4A16 GEMM

- **Repo:** https://github.com/IST-DASLab/marlin
- **What's done:** AWQ via `autoawq` provides a W4A16 path that works.
- **What's left (only if profiling justifies):**
  - AWQ → Marlin packing conversion at load time (groups of 128, group-wise scales).
  - Pipelined async loads + tensor core compute pattern in the projection layers.
- **What we still skip:** Sparse-Marlin (2:4 structured sparsity).
- **Trigger to do this:** `nsys` profile shows the AWQ projection kernel is the decode bottleneck. Until then, deferred.

### 5.9 EAGLE-2/3 — speculative decoding

- **Repo:** https://github.com/SafeAILab/EAGLE
- **What's done:** nothing — the kernel composes cleanly because EAGLE wraps the model from the outside.
- **What's left:**
  - Drop-in EAGLE-2 wrapper around our patched HF model.
  - Use their published Llama-3 draft network checkpoints (no draft-net pre-training).
- **What we still skip in v1:** training our own draft network.
- **Reported:** ~3× decode speedup, composes multiplicatively with our memory savings.

### 5.11 PowerInfer — CPU offload (8B / 7B unblocker)

- **Repo:** https://github.com/SJTU-IPADS/PowerInfer
- **What's done:** nothing.
- **What's left (v2 stretch):**
  - Hot/cold layer classification → swap cold layers to system RAM (~10 GB available).
  - Or use llama.cpp's `-ngl` partial offload as a poor-man's substitute.
- **Why it's blocking:** Llama-3.1-8B AWQ-INT4 (~4.5 GB) and Mistral-7B IQ4-XS (~3.2 GB) both fail the §3 VRAM budget without offload.

### 5.10 vLLM PagedAttention — block table (deferred to v2)

- **Repo:** https://github.com/vllm-project/vllm
- **What's done:** page_size=64 storage with contiguous layout.
- **What's left (v2 only):** logical → physical block-table indirection. Useful for multi-request scheduling and for interop with vLLM/SGLang. Single-user laptop inference (our v1 target) does not need it.

### 5.8 MInference — patterns (still skipped per original SPEC)

The vertical-slash and A-shape patterns require per-head pattern *training* on representative inputs. Gain over Quest+DuoAttention is small for our hardware. Skip in v1; reconsider only if a long-context regression appears.

## 6. Phased build plan — what's left

### Phase 6 — Polish & release (current focus, 2–4 weeks)

The original SPEC's Phase 5 (ExLlamaV2 + EAGLE-2 + INT4 KV) and Phase 6 (polish & release) are merged here. The lesson from Phase 5: kernel correctness is fine but the Python wrapper kills perf, so the perf work has to happen **before** any release.

Order, in priority:

1. **Kernel-fused criticality + top-k.** Move `compute_page_summary` + `page_scores` + `select_pages` into the sparse Triton kernel. Target: ≥4 tok/s decode at 32 k on Llama-3.2-3B-AWQ (matches SPEC §1 win row 1's 10 tok/s @ 128 k after extrapolation). Reference impls to crib from: `mit-han-lab/Quest/quest/csrc/quest_attention.cu` (their criticality is already fused in CUDA — port the structure to Triton).
2. **RULER 4 k subset eval.** Wire `niah_single` + `niah_multikey` + `niah_multivalue` from `NVIDIA/RULER` against our patched model. Acceptance: ≥85 % at retention=0.25 vs dense at 4 k. This is the actual quality metric the SPEC commits to.
3. **Demo chat CLI.** `flashquest chat --model llama-3.2-3b-awq --context 32k`. Stream tokens. README §11 acceptance criterion.
4. **Head-to-head benchmark table** vs `llama.cpp -ngl 999` and vLLM at the same context, on the same machine. README §11 acceptance criterion.
5. **INT4 KV.** Phase 6's biggest kernel change. After (1) and (2) ship, re-validate quality at INT4 vs INT8 + dense.
6. **EAGLE-2 wrapper** (drop-in). Test decode tok/s improvement; should compose multiplicatively with the kernel-fused decode.
7. **Marlin W4A16** (only if profile says so).
8. **ExLlamaV2 backend integration.** Optional. Their `exllamav2_ext` extension model has cleaner extension points than HF; would let us ship as a backend rather than a monkeypatch. Defer until (1)–(4) land.

**Phase 6 win conditions:**
- Decode at 32 k ≥4 tok/s on Llama-3.2-3B-AWQ (clears the Phase 5 0.09 tok/s gap).
- RULER 4 k subset ≥85 % vs dense at retention=0.25.
- `flashquest chat` reproducibly streams tokens at the above rate.
- Benchmark table in README shows ≥5× capability gain over `llama.cpp -ngl 999` at 32 k.

**Phase 6 explicit non-goals:**
- 8B at 32 k (hardware-blocked; v2).
- 64 k / 128 k context (revisit after kernel-fused decode lands).
- Full RULER (hours of compute; 4 k subset is enough to gate release).
- Cross-platform packaging (we're WSL2 + CUDA 12.5 only).

### v2 stretch — after Phase 6 ships

These are not gated on Phase 6 and can be approached independently:

- **Llama-3.1-8B at 32–64 k via PowerInfer-style hot/cold offload** (or IQ3-XXS via GGUF interop). Clears SPEC §1 win row 2.
- **Mistral-7B IQ4-XS at 32 k via CPU offload.** Clears SPEC §1 win row 3.
- **vLLM block-table layout** for upstream interop.
- **Colab T4 / A100 cross-validation.** Verifies the kernel is correct on Turing (no `cp.async`) and that perf isn't accidentally tied to a 3050 Ti quirk. Acceptance: numerical equivalence on T4, perf scaling reasonable on A100.
- **LongBench.** Broader task coverage than RULER.
- **Upstream PRs** to ExLlamaV2 or vLLM, or maintain as independent fork.

## 7. Test & evaluation methodology — what still needs to land

### 7.2 Quality benchmarks (the SPEC's actual gate)

- **RULER 4 k subset** (`NVIDIA/RULER`) — the gating metric for Phase 6 release. Specifically `niah_single`, `niah_multikey`, `niah_multivalue`.
- **Full RULER at 8 k / 16 k / 32 k** — v2.
- **LongBench** — v2.
- ~~Wikitext perplexity (Phase 1 done).~~

### 7.3 Performance benchmarks (the SPEC's perf gate)

- Decode tok/s at 8 k / 16 k / 32 k / 64 k / 128 k.
- Prefill latency at same.
- Peak VRAM (`torch.cuda.max_memory_allocated`).
- Baselines for the table: dense FA-2 (have it from Phase 0), llama.cpp CUDA (have it), vLLM (have it at 4 k only), ExLlamaV2 (need to add).
- Profile critical kernels with PyTorch profiler (sm_86 ncu is blocked under WSL2 — see Phase 0 notes).

### 7.4 Cross-validation on Colab (v2)

- Free T4 (sm_75) — verify the algorithm works on a different gen. T4 has tensor cores but no `cp.async`; mostly verifies *correctness*.
- Optional Pro A100 (~$10–15) — same kernel, more capable hardware. Validates no 3050 Ti quirk.

## 8. Open risks (still active)

| Risk | Severity | Mitigation |
|---|---|---|
| Kernel-fused criticality + top-k harder than expected on sm_86 (no shared-mem sort primitive) | High | Quest's CUDA implementation does it; port their structure. Fallback: do the top-k in Triton with a bitonic sort on `BLOCK_N=64`. |
| INT4 KV quality regression on small models | Medium | Stage at INT4 with strict eval gate (RULER 4 k ≥85 %); roll back to INT8 if it falls. |
| EAGLE-2 doesn't compose with our patched LlamaAttention | Low | Their wrapper is at the model level, ours is at the layer level — they should be orthogonal. Verify on a 1B model first. |
| Marlin packing conversion is a rabbit hole | Low | Skip unless profiler demands it. AWQ via autoawq already works. |
| 8B never fits even with PowerInfer offload at 4 GB total VRAM | Medium | Document the wall and target 7B Mistral with offload as the realistic 7B+ story. |

## 9. Naming, license, distribution (unchanged)

- **Working name:** `flashquest`. Open to rename.
- **License:** Apache 2.0.
- **Distribution:** GitHub repo + PyPI package.
- **Acknowledgments:** must credit Quest, KIVI, DuoAttention, StreamingLLM, Marlin, FlashAttention, EAGLE in README and any paper write-ups.

## 10. New directory structure (Phase 6 additions)

```
src/flashquest/
├── kernel/
│   └── sparse_fused.py          # NEW: kernel-fused criticality + top-k
├── runtime/
│   ├── chat.py                  # NEW: flashquest chat CLI
│   └── eagle_wrap.py            # NEW: optional EAGLE-2 speculation
└── eval/
    └── ruler.py                 # NEW: RULER 4k subset harness
benchmarks/
└── phase6_results.json          # NEW: head-to-head table
```

Existing files stay where they are (`src/flashquest/{cache,duo,eager}/...`).

## 11. What success looks like (unchanged — the v1.0 acceptance criteria)

A repo that, when cloned on a similar laptop:
1. `pip install -e .`
2. `flashquest chat --model llama-3.2-3b-awq --context 32k`
3. Tokens stream at ~10 tok/s with 32 k of context loaded, fitting in 4 GB VRAM.
4. README has a benchmark table showing ≥5× capability gain over `llama.cpp -ngl 999` on the same hardware.
5. RULER 4 k subset ≥85 % vs dense at retention=0.25.

That's it. No paper. No Twitter thread. Just a thing that works on a laptop that nobody else's stuff works on.
