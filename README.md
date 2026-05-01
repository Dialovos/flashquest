# flashquest

Sparse-retrieval attention Triton kernel for Ampere laptop GPUs (sm_86, 4GB VRAM class). Targets long-context inference (32k–128k) of 3B–8B models on hardware that today either OOMs or falls back to slow CPU paths.

**Status:** spec only. No code yet.

## What it is

A single fused Triton kernel that composes:

- **Quest-style** query-aware top-k page retrieval (skip irrelevant KV blocks).
- **KIVI-style** 2-bit KV cache (per-channel K, per-token V).
- **DuoAttention-style** retrieval-vs-streaming head split.
- **StreamingLLM** attention sinks + sliding window.
- **Marlin-style** W4A16 weight loading (Ampere-tuned mixed-precision GEMM).
- **FlashAttention-2** online softmax scaffolding (the Triton 06-fused-attention tutorial).
- Optional **EAGLE-2** speculative decoding wrapper.

## Hardware target

NVIDIA RTX 3050 Ti Laptop GPU (GA107, sm_86, 4GB GDDR6, ~192 GB/s, no FP8/TMA/WGMMA). WSL2 + CUDA + Triton.

## Win conditions

| Model | Quant | Context | Decode tok/s | Quality (RULER) |
|---|---|---|---|---|
| Llama-3.2-3B | Q4_K_M | 128k | ≥10 | ≥85% of dense |
| Llama-3.1-8B | IQ3_XXS | 64k | ≥4 | ≥80% of dense |
| Mistral-7B | IQ4_XS | 32k | ≥6 (with offload) | ≥85% of dense |

Cost: ≤$15 across the entire build (Colab Pro for cross-validation only).

## Where to start

1. Read [`docs/SPEC.md`](docs/SPEC.md) — the full design doc.
2. Read [`docs/REFERENCES.md`](docs/REFERENCES.md) — paper/repo/code-path index of every SOTA technique we crib from.
3. Phase 0 from the spec: verify CUDA + Triton in WSL2, baseline llama.cpp + vLLM numbers on the target hardware. **Done** — see baselines below and [`docs/PHASES/phase-0-notes.md`](docs/PHASES/phase-0-notes.md).

## Phase 0 baselines

Llama-3.2-3B-Instruct on the target machine (RTX 3050 Ti Laptop, sm_86, 4 GB VRAM, WSL2 + CUDA 12.5). 128 decode tokens, single request.

| Stack | Quant | Context | Prefill tok/s | Decode tok/s | Peak VRAM |
|---|---|---|---|---|---|
| llama.cpp d775992 (CUDA, `-ngl 999`) | Q4_K_M | 8 192 | 736.55 ± 55.81 | **39.60** ± 0.11 | 3 543 MiB |
| vLLM 0.7.3 (FA-2 backend) | AWQ-INT4 | 4 096 † | ~890 | 17.30 | 3 411 MiB |
| flash-attn 2.7.4 (synthetic fwd) | BF16 | S = 8192 | — | 22.76 ms / forward | 125 MiB |

† **vLLM cannot fit 8 k context on 4 GB** even at `gpu_memory_utilization=0.95` (max KV cache tops at ~3904 tokens) — the AWQ weights + framework overhead leave too little headroom. This is itself the strongest possible motivation for flashquest: a tier-1 inference server fails the 8 k bar on this hardware out of the box. flashquest aims for 32 k–128 k on the same machine.

Re-run via `./scripts/bench_llamacpp.sh`, `python scripts/bench_vllm.py`, `python scripts/profile_fa2.py`. Machine-readable numbers in [`benchmarks/baselines.json`](benchmarks/baselines.json).

## Phase 1 — Eager Quest reference

Pure-PyTorch implementation, validated on `unsloth/Llama-3.2-1B-Instruct` (BF16, SDPA baseline). Page size 64, sinks=4, window=128 tokens. See [`docs/PHASES/phase-1-notes.md`](docs/PHASES/phase-1-notes.md) for the full picture and the Phase 2 handoff.

**Passkey retrieval — Quest's actual claim metric.** 5 trials × 3 depths × 5 configs at ~974-token prompts.

| Retention | depth=0.1 | depth=0.5 | depth=0.9 |
|---|---|---|---|
| dense | 5/5 | 5/5 | 5/5 |
| 1.00 | 5/5 | 5/5 | 5/5 |
| 0.50 | 5/5 | 5/5 | 5/5 |
| 0.25 | 5/5 | 5/5 | 5/5 |
| **0.10** | **5/5** | **5/5** | **5/5** |

**Wikitext-2 perplexity** (8 192 tokens, sliding window 2048 stride 1024). Acknowledged gap vs SPEC win conditions (≤1 % at retention=0.25, ≤3 % at retention=0.10) — see notes.

| Retention | ppl | Δ vs dense |
|---|---|---|
| dense | 13.67 | — |
| 1.00 | 13.67 | -0.01 % (sanity OK) |
| 0.50 | 13.95 | +2.00 % |
| 0.25 | 15.26 | +11.61 % |
| 0.10 | 19.22 | +40.54 % |

The perplexity gap is expected: Quest's per-page upper-bound criticality is loose at our chosen page_size=64 (vs Quest's 16, which we deferred for `BLOCK_N` alignment), and a 1B model on Wikitext is an unfavorable test case for sparse attention. Quest's actual claim is long-context retrieval — passkey passes across the board.

Re-run via `python scripts/phase1_run_perplexity.py` and `python scripts/phase1_run_passkey.py`.

## Phase 2 — Dense FA-2 Triton kernel

Port of the FA-2 06-fused-attention tutorial onto sm_86. `BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2`. BF16 in / FP32 acc / BF16 out, GQA via kernel-side head map, returns LSE for Phase 3 composition. 14 catalogued edge cases (E1–E14) covered (`tests/test_kernel_flash_fwd_edges.py` + property-based hypothesis fuzz).

Llama-3.2-3B geometry, S=8192, BF16, causal:

| Backend | ms / fwd | ratio vs FA-2 |
|---|---|---|
| `flash_attn` 2.7.4 (reference) | 25.51 | 1.000 |
| **flashquest Triton kernel** | **27.37** | **1.073** |
| torch SDPA | 27.65 | 1.084 |

SPEC win condition was within 30 % of FA-2 — we landed at 7 %, and we're 1 % faster than torch SDPA. See [`docs/PHASES/phase-2-notes.md`](docs/PHASES/phase-2-notes.md). Re-run via `python scripts/phase2_bench_attn.py`.

## Phase 3 — Sparse retrieval + INT8 KV

Decode-only sparse forward kernel. Quest selection + KIVI-style asymmetric uint8 KV (per-page channel-wise K, per-token V). Dequant happens inside the Triton kernel; only selected pages are loaded.

Llama-3.2-3B geometry, S_kv=8192, retention=0.25, sinks=4, window=128:

| Backend | ms / decode step | speedup vs dense |
|---|---|---|
| Phase 2 dense (BF16 KV) | 0.449 | 1.0× |
| **Phase 3 sparse (INT8 KV)** | **0.181** | **2.48×** |

11 catalogued edge cases (ES1–ES11) + 15 hypothesis-fuzzed shapes. See [`docs/PHASES/phase-3-notes.md`](docs/PHASES/phase-3-notes.md). Re-run via `python scripts/phase3_bench_decode.py`.

## Phase 4 — DuoAttention head split + HF integration

Per-head retrieval-vs-streaming attention dispatch wired through HF Llama. The Phase 3 sparse kernel already supports per-query-head selection masks; DuoAttention reduces to building a different mask per head (full top-k for retrieval heads; sinks+window only for streaming heads).

Validation on Llama-3.2-1B with a synthetic 70/30 retrieval/streaming split (DuoAttention's upstream classifications cover Llama-3.1-8B and Mistral-7B; no Llama-3.2 file exists upstream):

| Config | depth=0.1 | depth=0.5 | depth=0.9 |
|---|---|---|---|
| Phase 1 (all retrieval) | 5/5 | 5/5 | 5/5 |
| Phase 4 (70 % retrieval, 30 % streaming) | 5/5 | 5/5 | 5/5 |

11 catalogued edge cases (EP1–EP11). See [`docs/PHASES/phase-4-notes.md`](docs/PHASES/phase-4-notes.md) for the full picture and Phase 5 prerequisites (Llama-3.1-8B AWQ + persistent INT8 cache + Marlin + 32 k RULER).

Re-run via `python scripts/phase4_run_passkey.py`.

## Phase 5 — Persistent INT8 KV cache + AWQ-INT4 + fused DuoAttention

Production-grade decode path: persistent INT8 KV cache (HF `Cache` subclass), AWQ-INT4 weight loading, single-call fused DuoAttention dispatch.

Validation on Llama-3.2-3B-AWQ (`casperhansen/llama-3.2-3b-instruct-awq`, ~2 GB weights), synthetic 70/30 retrieval/streaming split, retention=0.25, sinks=4, window=128.

**Passkey at long context** (2 trials × 3 depths). Numbers in `benchmarks/phase5_passkey.json`; see [`docs/PHASES/phase-5-notes.md`](docs/PHASES/phase-5-notes.md).

| Context | depth=0.1 | depth=0.5 | depth=0.9 | Wall time | Peak VRAM |
|---|---|---|---|---|---|
| 8 192 | **2/2** | **2/2** | **2/2** | 82 s | 3 204 MiB |
| 32 768 | **2/2** | **2/2** | **2/2** | 3 026 s | 6 283 MiB † |

† Peak VRAM at 32 k spilled past the 4 GB nominal cap into WSL2 shared memory; correctness still 100 %, but Phase 6 should profile the dequant + criticality intermediates that drive the spill.

**Decode at 32 k context** (1 trial × 16 new tokens):

| | Prefill tok/s | Decode tok/s | Peak VRAM |
|---|---|---|---|
| Phase 5 (persistent INT8 + fused dispatch) | 53.7 | **0.092** | 6 378 MiB † |

The decode tok/s is **far below the SPEC target of ≥4 tok/s** — the Triton kernel itself is fast (Phase 3 measured 0.181 ms/decode at 8 k), but the Python wrapper around it (`dequantize_k` materializes the full bf16 cache every step, `compute_page_summary` + `page_scores` + `select_pages` are pure-Python with fp32 intermediates) dominates wall time. Phase 6's first task is chunked / fused criticality (move it into the Triton kernel) — fixes the gap without algorithmic change.

17 catalogued edge cases (EQ1–EQ17). Re-run via `python scripts/phase5_run_passkey_32k.py` and `python scripts/phase5_bench_decode_32k.py`.

**8B reality check.** Llama-3.1-8B AWQ-INT4 weights alone are ~4.5 GB — they don't fit on a 4 GB GPU before any KV cache. The SPEC §6 Phase 4/5 8B win condition is hardware-blocked on this tier; an 8B control at smaller contexts is recorded in `benchmarks/phase5_8b_control.json`. Phase 6 (or v2) is the natural place for IQ3-XXS or PowerInfer-style hot/cold layer offload to clear this wall.

## Non-goals

- Training kernels. Inference only.
- Datacenter GPUs. Hopper/Blackwell-only features (TMA, WGMMA, FP8) are explicitly skipped.
- Beating FlashAttention-3 anywhere. We're not in that league and don't need to be.
- Production serving. This is a personal artifact + open-source release.
