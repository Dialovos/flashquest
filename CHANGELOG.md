# Changelog

All notable changes are recorded here. Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); version numbers are simple semver. Per-release benchmark numbers reference the **Llama-3.2-3B-Instruct-AWQ** + **RTX 3050 Ti Laptop (4 GB VRAM, sm_86, WSL2 + CUDA 12.5)** target.

## [v1.0] — 2026-05-13

First tagged release. flashquest decodes Llama-3.2-3B at 32 k context on a 4 GB consumer GPU — a capability gap with both **vLLM 0.7.3** (OOMs above ~4 k) and **llama.cpp `-ngl 999`** (aborts at 32 k). Quality holds at RULER NIAH 4 k 100/100/95 (single/multikey/multivalue) at the default settings.

### Composed techniques

- **Quest top-k page selection** with an algebraic per-page score `Q·K_mn + 15·relu(Q)·K_scale` (INT4) — two matmuls per layer, no per-page dequant, no per-token criticality recompute.
- **KIVI-style INT4 paged KV cache** (`PersistentInt4KVCache`) — asymmetric uint8 storage, 2 nibbles per byte along `head_dim`, per-page channel-wise K, per-token V.
- **Fused Triton decode kernel** reads packed INT4 K/V tiles directly, unpacks lo/hi nibbles via `tl.join` + `tl.reshape`, runs online-softmax — no BF16 dequant intermediate.
- **DuoAttention head split** + **StreamingLLM sinks + window** via `patch_llama_for_quest_persistent`, hooked on top of HF `LlamaAttention`.
- **TurboQuant K3-V3 opt-in** via `--kv-bits 3`: Walsh-Hadamard rotation along `head_dim`, 8-codepoint Lloyd-Max codebook inlined as `tl.where` chain, bit-split storage (1-bit MSB plane + 2-bit LSB plane). 25 % smaller KV than INT4; slower decode (2.62 tok/s @ 32 k); RULER multivalue 17/20 (85 %).
- **AWQ-INT4 weights** loaded via `autoawq 0.2.9` + transformers 4.57 compat shim.
- **`flashquest` chat CLI** (single-shot + interactive REPL, `--context-file`, `--retention`, `--kv-bits`) wraps the full chain into a streaming driver.

### Benchmarks at v1.0 defaults (`--kv-bits 4`, `--retention 0.20`)

- 32 k decode: **8.41 tok/s** clean, peak 5478 MiB VRAM. (Phase 10 measurement.)
- 8 k decode: 4.94 tok/s under the Phase 6 head-to-head wrapper (not re-measured at v1.0). The 32 k story is the capability axis; 8 k is for backend comparison only — see README head-to-head table.
- RULER NIAH 4 k (n=20 vs vanilla SDPA dense, all-retrieval head_pattern): single 20/20, multikey 20/20¹, multivalue 19/20.

¹ multikey carried over from Phase 6 task 2 measurement.

### Opt-ins

- `flashquest --retention 0.10` — ~1.24× decode on single-needle retrieval workloads. RULER multivalue regresses to 13/20 (65 %); do not ship as a default for multi-needle tasks.
- `flashquest --kv-bits 3` — TurboQuant K3-V3 storage path (25 % smaller cache, slower decode, multivalue 17/20).

### Non-goals

- Training kernels. Inference only.
- Datacenter GPUs. Hopper/Blackwell-only features (TMA, WGMMA, FP8) are explicitly skipped.
- Beating FlashAttention-3. Not in that league.
- Llama-3.1-8B at 32 k — the AWQ weights alone are ~4.5 GiB; doesn't fit alongside any KV cache on a 4 GB card. Revisit on a 12+ GB GPU.

### Validated stack

Python 3.12 · torch 2.5.1+cu121 · triton 3.1.0 · transformers 4.57.x · autoawq 0.2.9 · CUDA 12.5 · WSL2.

### Known caveats

- Decode-only fused kernel. Prefill uses dense BF16 SDPA on the dequant'd cache (with `enable_gqa=True` to keep SDPA on the Flash backend at long ctx). For the bench, pass `logits_to_keep=1` to skip the 7.83 GiB lm_head allocation that the bench discards anyway.
- Quality validated only on Llama-3.2-3B-Instruct-AWQ. Other Llama-family models with the same head_dim (64 or 128) should work; non-Llama architectures need their own dispatcher patch.
- Peak VRAM may exceed nominal 4095 MiB because PyTorch's allocator overcommits via WSL2 swap. Trust `nvidia-smi` for on-GPU residency.

[v1.0]: https://github.com/Dialovos/flashquest/releases/tag/v1.0
