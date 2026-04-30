# flashquest — Project Doc

Single living document for the project. See `README.md` for the elevator pitch and `docs/SPEC.md` for the full design.

## Overview

Sparse-retrieval attention Triton kernel for Ampere laptop GPUs (sm_86, 4 GB VRAM). Targets long-context inference (32 k – 128 k) of 3 B – 8 B models on a 3050 Ti class GPU. Composes Quest, KIVI, DuoAttention, StreamingLLM, Marlin, FlashAttention-2, and (optionally) EAGLE-2 into one fused kernel.

## Getting started

```bash
git clone <repo>
cd flashquest
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,bench]"
./scripts/vendor_clone.sh                # vendor reference repos to vendor/
python scripts/verify_env.py             # snapshot host env to docs/PHASES/env_snapshot.json
python scripts/verify_triton_int8.py     # confirm sm_86 INT8 mma works
```

To reproduce Phase 0 baselines:
```bash
# 1) build llama.cpp with CUDA (already cmake-configured under vendor/llama.cpp)
nice -n 19 cmake --build vendor/llama.cpp/build --config Release -j 4

# 2) pull the GGUF weights
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models/llama-3.2-3b

# 3) bench
./scripts/bench_llamacpp.sh              # llama.cpp Q4_K_M @ 8k
python scripts/bench_vllm.py             # vLLM AWQ-INT4 @ 4k (8k OOMs on 4 GB)
python scripts/profile_fa2.py            # flash-attn fwd reference
```

## Architecture

See `docs/SPEC.md §4`. Single Triton kernel per attention layer; sparse outer loop over Quest-selected KV blocks; INT8 KV with KIVI-style scales; per-head pattern dispatch (DuoAttention).

## Phases

- **Phase 0 — Setup & baselines** ✅ **complete (2026-04-30, tag `phase-0`)**. Env verified, INT8 mma confirmed on sm_86 (SPEC OQ1 = yes), baselines captured (llama.cpp 39.6 tok/s decode @ 8 k Q4_K_M; vLLM cannot fit 8 k on 4 GB at all — fell back to 4 k @ 17.3 tok/s; flash-attn 22.76 ms / fwd at S = 8192 BF16). WSL2 profiler verdict: nsys captures, importer needs upgrade; ncu blocked by perf-counter perms on consumer drivers (use PyTorch profiler instead). See `docs/PHASES/phase-0-notes.md` and `benchmarks/baselines.json`.
- Phase 1 — Eager Python Quest reference. Not started. Planned: pure-PyTorch top-k + criticality scoring against `vendor/quest/quest/models/QuestAttention.py`, validated on Llama-3.2-1B at 4 k.
- Phase 2 — Dense FA-2 Triton baseline. Not started. Phase-2 target: ≥ 70 % of `flash_attn` 22.76 ms.
- Phase 3 — Sparse retrieval + INT8 KV.
- Phase 4 — DuoAttention split + 8 B model + Marlin W4A16 projections.
- Phase 5 — ExLlamaV2 backend + optional EAGLE-2.
- Phase 6 — Polish & release.

## Configuration

Pinned versions in `pyproject.toml`; full host snapshot in `docs/PHASES/env_snapshot.json` (gitignored, regenerate via `python scripts/verify_env.py`).

Working stack (2026-04-30):
- Python 3.12.3 (system)
- torch 2.5.1+cu121, triton 3.1.0, numpy < 2
- vllm 0.7.3, transformers 4.57.6 (vLLM ≥ 0.20 forces torch 2.11/cu13 — incompatible with our CUDA 12.5 driver)
- flash-attn 2.7.4.post1 (built from source; no prebuilt wheel matched torch 2.5.1)

Model weights live in `~/models/`; vendored upstream repos in `vendor/` (gitignored, repopulate via `scripts/vendor_clone.sh`).

### WSL2 caveats

- WSL2 default RAM cap is 7.6 GiB on this host. For Phase 4 CPU offload of 8 B models, bump via `~/.wslconfig`:
  ```ini
  [wsl2]
  memory=12GB
  ```
  and `wsl --shutdown` from PowerShell.
- nsys 2022.4 captures `.qdstrm` traces but lacks the importer that converts to `.nsys-rep`. Install a newer nsys, or open the `.qdstrm` directly in nsight-systems on Windows.
- ncu is blocked by `ERR_NVGPUCTRPERM` on consumer drivers without Windows-side `NVreg_RestrictProfilingToAdminUsers=0` or root. Use PyTorch profiler / Triton's built-in metrics for Phase 2 kernel tuning.

## References

- `docs/SPEC.md` — full design
- `docs/REFERENCES.md` — papers + code paths index
- `docs/PHASES/phase-N-notes.md` — per-phase journals
- `docs/superpowers/plans/` — implementation plans (one per phase)
- `benchmarks/baselines.json` — machine-readable benchmark history
- `vendor/README.md` — what we vendor and why
