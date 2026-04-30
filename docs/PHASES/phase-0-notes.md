# Phase 0 Notes

**Started:** 2026-04-29
**Status:** in progress (paused mid-Task 5 for session handoff)
**Spec:** [docs/SPEC.md §6 Phase 0](../SPEC.md)

## Session handoff (2026-04-29)

Tasks 1–4 complete and tagged in commits. Task 5 (llama.cpp baseline) is mid-flight:

- `vendor/llama.cpp/build/` is cmake-configured for sm_86 CUDA. Partial build: ggml-base + ggml-cpu shared libs done (~10% of full build). Resume with:

  ```bash
  cmake --build /home/hoang/code/personal/active/flashquest/vendor/llama.cpp/build --config Release -j $(nproc)
  ```

  cmake/ninja-make track state; this picks up where it stopped. CUDA kernel TUs are the bulk of remaining time (~10–15 min on 12900H).

- `~/models/llama-3.2-3b/` has ~65 MB of the 2 GB Q4_K_M GGUF (`hf` CLI cache + partial symlink). Resume with:

  ```bash
  /home/hoang/code/personal/active/flashquest/.venv/bin/hf download \
    bartowski/Llama-3.2-3B-Instruct-GGUF Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --local-dir $HOME/models/llama-3.2-3b
  ```

  `hf` resumes from cache automatically.

- `scripts/bench_llamacpp.sh`, `scripts/bench_vllm.py`, `scripts/profile_fa2.py` are written and committed but not yet run. vLLM and flash-attn are not yet installed.

### Resume order next session

1. Finish llama.cpp build (above command).
2. Finish model download (above command).
3. Run `./scripts/bench_llamacpp.sh` → fills `benchmarks/llamacpp_8k.txt`.
4. Init `benchmarks/baselines.json` with the result + commit Task 5.
5. Move on to Task 6 (vLLM): `pip install "vllm>=0.6.0"`, run `python scripts/bench_vllm.py`.
6. Task 7 (FA-2 + profilers): `pip install flash-attn --no-build-isolation`, run `scripts/profile_fa2.py`, then `nsys profile ...` and `ncu ...` per plan.
7. Task 8: README baselines table + DOC.md + tag `phase-0`.

### Notes for the next session

- Use absolute paths or stay at repo root in shell commands. Don't `cd` into `vendor/llama.cpp` for foreground commands — cwd persists across Bash calls in this harness.
- vLLM may force-pin its own torch at install time; if it conflicts with torch 2.5.1, accept vLLM's pin (re-verify CUDA after).
- flash-attn first install will compile native CUDA (~10 min on this hardware); use a long timeout or background it.


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
| 4 | Do `ncu` / `nsys` work in WSL2 on this machine? | Tools installed; functional check in Task 7 | (filled in Task 7) |

## Baselines

(filled in across Tasks 5–7; final table mirrored to README.md)

## Decisions / deviations from spec

- Python pin: 3.12 (spec said 3.10). 3.12 is the system default; downgrading buys nothing here.
- VRAM budget: assume **3.5–4.0 GB usable** when running benchmarks in WSL2 (vs spec's 3.0–3.3 GB). Recheck after running with browser + IDE open.
- WSL2 RAM is 7.6 GiB, half what spec assumed. Document a `.wslconfig` recommendation in `DOC.md` for users who hit Phase 4 offload limits.
- KIVI repo path drift: SPEC originally cited `quant/triton_quant.py` (which no longer exists upstream). Actual Triton kernel is now `quant/matmul.py`, pack/unpack is `quant/new_pack.py`. SPEC and REFERENCES updated.
- vLLM baseline (Task 6): GGUF Q4_K_M is not directly loadable in vLLM; baseline will use Llama-3.2-3B AWQ-INT4 instead — same model, comparable bit-width, different quant method.
