# Phase 0 Notes

**Started:** 2026-04-29
**Completed:** 2026-04-30 (tag `phase-0`)
**Status:** **complete**
**Spec:** [docs/SPEC.md §6 Phase 0](../SPEC.md)

## Phase 0 → Phase 1 handoff

All Phase 0 deliverables landed. Tag: `phase-0`.

### What's true at the end of Phase 0

- Stack: Python 3.12.3, torch 2.5.1+cu121, triton 3.1.0, vllm 0.7.3, transformers 4.57.6, flash-attn 2.7.4.post1, numpy 1.26.4. Pinned in `pyproject.toml`.
- Hardware envelope: 4 GiB VRAM (~3.4–3.5 GiB max usable in practice), CUDA 12.5 driver, sm_86. WSL2 RAM cap 7.6 GiB.
- INT8 mma on sm_86 confirmed working (load-bearing for the kernel design — SPEC OQ1).
- 10 reference repos vendored to `vendor/` (gitignored). Re-clone via `scripts/vendor_clone.sh`.
- Baselines:
  - llama.cpp Q4_K_M @ 8 k: prefill 736.55 / decode 39.60 tok/s, peak 3543 MiB
  - vLLM AWQ-INT4 @ 4 k (8 k OOMs at 4 GB): decode 17.30 tok/s, peak 3411 MiB
  - flash-attn fwd ref @ S=8192 BF16: 22.76 ms warm, 125 MiB
- Profilers: nsys collects `.qdstrm` traces (importer too old to convert); ncu blocked by `ERR_NVGPUCTRPERM`. PyTorch profiler is the working path for Phase 2+ kernel tuning.

### What Phase 1 begins with

- Goal: pure-PyTorch eager Quest-style attention as `attn_implementation="flashquest_eager"` on a HF Llama subclass. Reference: `vendor/quest/quest/models/QuestAttention.py`. Validate on Llama-3.2-1B at 4 k context against dense PyTorch attention.
- Win condition (from SPEC §6 Phase 1): matches dense within 1 % perplexity at top-25 % retention, within 3 % at top-10 %.
- Will need a new plan: `docs/superpowers/plans/2026-MM-DD-phase-1-eager-quest-reference.md`.

### Open items deferred from Phase 0

- **OQ2** (INT8 dequant fused into mma operand path): not answerable without a candidate kernel. Phase 2 design notes will revisit when porting FA-2 onto sm_86.
- **ncu access**: not blocking. If we need fine-grained occupancy / register-pressure data later, options are (a) `NVreg_RestrictProfilingToAdminUsers=0` Windows-side regedit, (b) provide sudo, or (c) profile on a Colab T4 (cross-validation step is already in the plan).
- **WSL2 RAM bump** to 12 GB via `~/.wslconfig`: only relevant when Phase 4 attempts CPU offload of 8 B models. Document but do not change yet.


## Host snapshot

See `env_snapshot.json` (sibling file, written by `scripts/verify_env.py`).

Key facts captured at start (2026-04-29 19:28 local):
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, driver 555.97, CUDA 12.5
- VRAM: 4096 MiB total / 4096 MiB free at idle (WSL2 does not share Windows display VRAM the way native Windows would — spec's "Windows display + browser eats 700MB–1GB" tax does not apply here)
- CPU: Intel i9-12900H, 20 threads (10 P+E cores), AVX2 + AVX-VNNI confirmed
- RAM: WSL2 sees 7.6 GiB / 5.6 GiB free (spec assumed 16 GiB host; **WSL2 default cap is half** — may need `.wslconfig` `memory=12GB` for Phase 4 CPU offload)
- nvcc: 12.0 (toolkit older than driver 12.5; forward-compatible — driver supports any toolkit ≤ its version)
- ncu: 2022.4.1.0 present (older build — may lack some sm_86 metrics; verify in Task 7)
- nsys: 2022.4.2 present
- gcc: 13.3.0 / cmake 3.28.3 — both fine for llama.cpp + flash-attn builds
- Python: 3.12.3 (spec planned 3.10; bumping `.python-version` to 3.12)

## Open questions from SPEC §8

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Triton ≥ 3.x on sm_86 supports `tl.dot` with INT8 operands? | **Yes** — `out_dtype=tl.int32`, exact-zero error at 128×128×128 with BLOCK=64. | `scripts/verify_triton_int8.py` on torch 2.5.1+cu121, triton 3.1.0. |
| 2 | Can we fuse INT8 dequant into the `mma.sync` operand path? | Deferred to Phase 2 | Not answerable without a candidate kernel; revisit when porting FA-2 in Phase 2. |
| 3 | Actual usable VRAM after Windows + browser + IDE? | **~4.0 GB (full)** at WSL2 idle, no Windows GUI tax — better than the spec's 3.0–3.3 GB assumption. To re-confirm under load, capture `nvidia-smi` after Edge + IDE warm. | `env_snapshot.json` nvidia_smi output |
| 4 | Do `ncu` / `nsys` work in WSL2 on this machine? | **Partial.** nsys 2022.4.2 captures `.qdstrm` traces (collection works) but lacks the importer to convert them to `.nsys-rep` with stats — need a newer nsys for postprocessing. ncu is blocked by `ERR_NVGPUCTRPERM` on consumer drivers; requires Windows-side `NVreg_RestrictProfilingToAdminUsers=0` regedit + WSL restart, or root. Workaround for kernel profiling: PyTorch profiler + Triton's own metrics; both work without elevation. | nsys logs (`benchmarks/fa2_profile.qdstrm`), ncu error message in Task 7 ncu-launch attempt |

## Baselines

Llama-3.2-3B-Instruct, RTX 3050 Ti Laptop (sm_86), WSL2.

| Stack | Quant | Context | Prefill tok/s | Decode tok/s | Peak VRAM | Notes |
|---|---|---|---|---|---|---|
| llama.cpp (CUDA, `-ngl 999`) | Q4_K_M | 8 192 | 736.55 ± 55.81 | 39.60 ± 0.11 | 3 543 MiB (86 %) | build d775992 |
| vLLM 0.7.3 | AWQ-INT4 | 4 096 | ~890 (vLLM warmup est.) | 17.30 (vLLM est.) | 3 411 MiB (83 %) | **8 k OOMs at util=0.95**; dropped to 4 k |
| FA-2 (`flash_attn` 2.7.4) — synthetic fwd | BF16 | (S=8192 fwd) | — | 22.76 ms / forward (warm) | 125 MiB | reference target for Phase 2 dense-FA Triton port |

Key findings from baselines:
- **vLLM cannot fit 8 k context on 4 GB** with Llama-3.2-3B-AWQ-INT4, even at `gpu_memory_utilization=0.95` (max KV cache tops at ~3904 tokens). This is itself the strongest possible motivation for flashquest: a SOTA inference server already fails the 8 k bar on this hardware. The flashquest win condition is 128 k.
- llama.cpp wins decode by ~2.3× over vLLM on this hardware (39.6 vs 17.3 tok/s), even at half the context. Single-user / batch=1 is llama.cpp's sweet spot; vLLM's async scheduler + CUDA graph capture cost more than they save here.
- Peak VRAM ~3.4–3.5 GiB for both — both stacks consume nearly the full envelope, leaving ~600 MiB headroom for OS/driver. flashquest's 3.0 GB working budget per SPEC §3 is the right ceiling.
- llama-bench prints `Total VRAM: 4095 MiB` — full 4 GiB available, no Windows display tax in WSL2. **OQ3 answered**: ~4.0 GB usable at idle (vs spec's 3.0–3.3 GB assumption).

## Decisions / deviations from spec

- Python pin: 3.12 (spec said 3.10). 3.12 is the system default; downgrading buys nothing here.
- VRAM budget: assume **3.5–4.0 GB usable** when running benchmarks in WSL2 (vs spec's 3.0–3.3 GB). Recheck after running with browser + IDE open.
- WSL2 RAM is 7.6 GiB, half what spec assumed. Document a `.wslconfig` recommendation in `DOC.md` for users who hit Phase 4 offload limits.
- KIVI repo path drift: SPEC originally cited `quant/triton_quant.py` (which no longer exists upstream). Actual Triton kernel is now `quant/matmul.py`, pack/unpack is `quant/new_pack.py`. SPEC and REFERENCES updated.
- vLLM baseline (Task 6): GGUF Q4_K_M is not directly loadable in vLLM; baseline uses `casperhansen/llama-3.2-3b-instruct-awq` (originally cited `hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4` does not exist on the Hub).
- vLLM baseline context dropped 8 k → 4 k: vLLM cannot fit an 8 k KV cache for the AWQ model on 4 GB even at util=0.95. Documented in baselines.json. This *strengthens* the project motivation: vLLM, a tier-1 inference server, fails the 8 k bar on this hardware out of the box.
- vLLM 0.7.3 + transformers 4.x is the working pair: vLLM 0.20.0 forced torch 2.11/cu13 (incompatible with our CUDA 12.5 driver) and required transformers ≥5; pinning vllm==0.7.3 + transformers<5 keeps the stack on torch 2.5.1+cu121.
- flash-attn 2.7.4.post1 builds and runs cleanly on torch 2.5.1+cu121 + sm_86 (~few minutes from-source compile, no prebuilt wheel matched). 22.76 ms / forward at S=8192, H_q=24, H_kv=8, D=64, BF16 — this is the bar Phase 2's dense Triton port aims at (target: within 30 % per SPEC §6 Phase 2).
- WSL2 profiler verdict: trace-collection (nsys) works; per-kernel metrics (ncu) need elevation we don't have. Plan uses PyTorch profiler + Triton metrics for Phase 2+ — both work without elevation and are sufficient for tuning block sizes / occupancy. ncu can be revisited later if we need a deep occupancy / register-pressure read.
