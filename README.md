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
3. Phase 0 from the spec: verify CUDA + Triton in WSL2, baseline llama.cpp + vLLM numbers on the target hardware.

## Non-goals

- Training kernels. Inference only.
- Datacenter GPUs. Hopper/Blackwell-only features (TMA, WGMMA, FP8) are explicitly skipped.
- Beating FlashAttention-3 anywhere. We're not in that league and don't need to be.
- Production serving. This is a personal artifact + open-source release.
