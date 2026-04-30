# Phase 0 — Setup & Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a verifiable Triton/CUDA dev environment on the user's RTX 3050 Ti laptop, vendor every reference repo the spec depends on, run baseline llama.cpp / vLLM benchmarks at 8k context on Llama-3.2-3B, and answer the four Phase 0 open questions from SPEC §8 — producing a README baseline table and a phase-0 notes file.

**Architecture:** Pure setup/verification phase. No flashquest code written. We add a Python venv at `.venv/`, a `vendor/` tree (gitignored) for upstream clones, the project skeleton from SPEC §10, and three docs (`README.md` baseline section, `docs/PHASES/phase-0-notes.md`, `DOC.md`).

**Tech Stack:**
- Python 3.10+ (system: `/usr/bin/python3`), venv
- CUDA toolkit (system; verify version), cuDNN
- PyTorch 2.x with CUDA 12.x build, Triton ≥ 3.x
- llama.cpp (CUDA build), vLLM
- nsys / ncu (NVIDIA profilers, WSL2 build)
- HuggingFace Hub CLI for weight pulls

**Hardware:** RTX 3050 Ti Laptop (sm_86, GA107, 4GB VRAM, ~3.0–3.3GB usable), Intel i9-12900H, 16GB RAM, WSL2.

**Cost ceiling:** $0 for Phase 0. Cloud spend ($15 ceiling) is reserved for Phase 5 cross-validation.

---

## File Structure

**Created:**
- `.venv/` (gitignored — already covered by `.gitignore`)
- `pyproject.toml` — project metadata, deps, build config
- `.python-version` — pin interpreter for reproducibility
- `vendor/` — gitignored upstream clones
- `vendor/.gitkeep` — keep dir tracked, contents ignored
- `src/flashquest/__init__.py` — package skeleton
- `src/flashquest/kernel/__init__.py`
- `src/flashquest/cache/__init__.py`
- `src/flashquest/model/__init__.py`
- `src/flashquest/runtime/__init__.py`
- `tests/__init__.py`
- `tests/test_smoke.py` — first sanity test
- `scripts/verify_env.py` — environment introspection script (re-runnable)
- `scripts/bench_llamacpp.sh` — llama.cpp benchmark wrapper
- `scripts/bench_vllm.py` — vLLM benchmark wrapper
- `scripts/profile_fa2.py` — FA-2 reference profile harness
- `benchmarks/baselines.json` — machine-readable baseline numbers
- `docs/PHASES/phase-0-notes.md` — open-question answers, env snapshot
- `DOC.md` — living project doc (per global instructions)

**Modified:**
- `.gitignore` — add `vendor/*` (keep `vendor/.gitkeep`), `benchmarks/*.nsys-rep`
- `README.md` — append baseline numbers table

---

## Task 1: Environment introspection — capture machine state

**Purpose:** Capture an honest snapshot of the host so later phases can reproduce it. Many later decisions (block sizes, KV budget, profiler choice) depend on these numbers.

**Files:**
- Create: `scripts/verify_env.py`
- Create: `docs/PHASES/phase-0-notes.md` (skeleton; answers filled in across later tasks)

- [ ] **Step 1.1: Create `scripts/verify_env.py` to dump environment facts**

```python
#!/usr/bin/env python3
"""Phase 0 environment introspection. Re-runnable, idempotent, no side effects."""
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"<not available: {e}>"


def main() -> None:
    info: dict[str, object] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "cpu": _run(["lscpu"]).split("\n")[:20],
        "memory": _run(["free", "-h"]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvcc": _run(["nvcc", "--version"]),
        "ncu": _run(["ncu", "--version"]) if shutil.which("ncu") else "<absent>",
        "nsys": _run(["nsys", "--version"]) if shutil.which("nsys") else "<absent>",
        "gcc": _run(["gcc", "--version"]).split("\n")[0],
        "cmake": _run(["cmake", "--version"]).split("\n")[0],
        "git": _run(["git", "--version"]),
    }

    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        info["torch"] = "<not installed>"

    try:
        import triton
        info["triton"] = {"version": triton.__version__}
    except ImportError:
        info["triton"] = "<not installed>"

    out_path = Path(__file__).resolve().parents[1] / "docs" / "PHASES" / "env_snapshot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Run before any installs to record raw system state**

Run: `python3 scripts/verify_env.py`
Expected: prints a JSON blob; writes `docs/PHASES/env_snapshot.json`. `nvidia-smi` should show `RTX 3050 Ti Laptop GPU`, compute cap `8.6`. Torch/Triton may report `<not installed>` — that's fine, we install in Task 3.

- [ ] **Step 1.3: Initialize the phase-0 notes doc**

Create `docs/PHASES/phase-0-notes.md`:

```markdown
# Phase 0 Notes

**Started:** 2026-04-29
**Status:** in progress
**Spec:** [docs/SPEC.md §6 Phase 0](../SPEC.md)

## Host snapshot

See `env_snapshot.json` (sibling file, written by `scripts/verify_env.py`).

Key facts captured at start:
- GPU: <fill from nvidia-smi: name, compute cap, total VRAM, free VRAM>
- CUDA: <fill from nvcc>
- ncu / nsys present: <yes/no>
- Free RAM at start: <fill from free -h>

## Open questions from SPEC §8

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Triton ≥ 3.x on sm_86 supports `tl.dot` with INT8 operands? | TBD | (filled in Task 3) |
| 2 | Can we fuse INT8 dequant into the `mma.sync` operand path? | TBD | (filled in Task 7 / deferred to Phase 2) |
| 3 | Actual usable VRAM after Windows + browser + IDE? | TBD | (filled in Task 1) |
| 4 | Do `ncu` / `nsys` work in WSL2 on this machine? | TBD | (filled in Task 7) |

## Baselines

(filled in across Tasks 5–7; final table mirrored to README.md)

## Decisions / deviations from spec

(record any decisions made during Phase 0 that diverge from SPEC.md — e.g. quant choice for vLLM if Q4_K_M unsupported)
```

- [ ] **Step 1.4: Commit the introspection script and notes skeleton**

```bash
git add scripts/verify_env.py docs/PHASES/phase-0-notes.md docs/PHASES/env_snapshot.json
git commit -m "phase 0: env introspection script + notes skeleton"
```

---

## Task 2: Project skeleton + venv

**Purpose:** Lay down the directory structure from SPEC §10 and a clean venv. Keep deps minimal — only what later tasks need.

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `src/flashquest/__init__.py` and submodule `__init__.py` files
- Create: `tests/__init__.py`, `tests/test_smoke.py`
- Modify: `.gitignore` (add `vendor/*`, `!vendor/.gitkeep`, `benchmarks/*.nsys-rep`, `benchmarks/*.ncu-rep`, `docs/PHASES/env_snapshot.json`)

- [ ] **Step 2.1: Pin the interpreter**

Create `.python-version`:
```
3.10
```

(WSL2 Ubuntu 22.04 ships 3.10 as `python3`. If host has 3.11+ available, use that — adjust this file to match.)

- [ ] **Step 2.2: Write `pyproject.toml`**

Create `pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "flashquest"
version = "0.0.0"
description = "Sparse-retrieval attention Triton kernel for Ampere laptop GPUs"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
authors = [{ name = "flashquest contributors" }]

dependencies = [
  # filled in Task 3 once we know which torch/triton versions work
]

[project.optional-dependencies]
dev = [
  "pytest>=8",
  "pytest-cov",
  "ruff",
  "mypy",
]
bench = [
  "transformers>=4.45",
  "accelerate",
  "huggingface_hub[cli]",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra -q"

[tool.ruff]
line-length = 100
target-version = "py310"
```

- [ ] **Step 2.3: Create the package skeleton from SPEC §10**

```bash
mkdir -p src/flashquest/{kernel,cache,model,runtime} tests benchmarks scripts
touch src/flashquest/__init__.py \
      src/flashquest/kernel/__init__.py \
      src/flashquest/cache/__init__.py \
      src/flashquest/model/__init__.py \
      src/flashquest/runtime/__init__.py \
      tests/__init__.py
```

Each `__init__.py` should contain only:
```python
"""flashquest — see docs/SPEC.md."""
```

(Mark only the top-level `src/flashquest/__init__.py` with that line; submodule `__init__.py` files stay empty.)

- [ ] **Step 2.4: Write a smoke test that imports the package**

Create `tests/test_smoke.py`:
```python
def test_import():
    import flashquest  # noqa: F401


def test_submodules_importable():
    from flashquest import kernel, cache, model, runtime  # noqa: F401
```

- [ ] **Step 2.5: Update `.gitignore`**

Append to `.gitignore`:
```
# Vendored upstream clones
vendor/*
!vendor/.gitkeep

# Profiling outputs in repo
benchmarks/*.nsys-rep
benchmarks/*.ncu-rep
benchmarks/*.qdrep

# Generated env snapshots (machine-specific)
docs/PHASES/env_snapshot.json
```

- [ ] **Step 2.6: Create the venv**

Run:
```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
pip install -e ".[dev]"
```
Expected: `pip install -e .` succeeds. `pytest` is now available.

- [ ] **Step 2.7: Run smoke test**

Run: `pytest tests/test_smoke.py -v`
Expected: 2 passing tests.

- [ ] **Step 2.8: Commit skeleton**

```bash
git add pyproject.toml .python-version .gitignore src tests
git commit -m "phase 0: project skeleton + smoke test"
```

---

## Task 3: Install PyTorch + Triton, verify sm_86 INT8 mma path

**Purpose:** Resolve SPEC Open Question 1 — does Triton ≥ 3.x on sm_86 support `tl.dot` with INT8 operands? This unblocks the entire kernel design.

**Files:**
- Modify: `pyproject.toml` (pin torch/triton in `dependencies`)
- Create: `scripts/verify_triton_int8.py`

- [ ] **Step 3.1: Install torch + triton matched to host CUDA**

Capture the host CUDA major from `nvidia-smi` output (top-right "CUDA Version: 12.x") in `docs/PHASES/env_snapshot.json`. Then, with the venv active:

```bash
# Pick the wheel index that matches host CUDA major.
# For CUDA 12.x: cu121 or cu124 wheels both work on Ampere.
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0
pip install triton==3.0.0
```

If `pip install triton==3.0.0` fails (sometimes wheel availability lags), fall back to `pip install triton` (lets pip pick the latest compatible). Record the resolved versions.

- [ ] **Step 3.2: Re-run env script to capture installed versions**

Run: `python scripts/verify_env.py`
Expected: `torch.cuda_available = True`, `device_capability = [8, 6]`, `triton.version` populated.

- [ ] **Step 3.3: Pin resolved versions in `pyproject.toml`**

Open `pyproject.toml` and replace the empty `dependencies = []` block with the resolved versions, e.g.:
```toml
dependencies = [
  "torch==2.4.0",
  "triton>=3.0.0,<4",
  "numpy<2",  # several upstream repos pin <2; keep us compatible
]
```
Use the actual versions reported by `verify_env.py`.

- [ ] **Step 3.4: Write the INT8 `tl.dot` smoke kernel**

Create `scripts/verify_triton_int8.py`:
```python
"""Smoke test: verify Triton tl.dot accepts INT8 operands on sm_86.

Answers SPEC §8 Open Question 1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0).to(tl.int8)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    cc = torch.cuda.get_device_capability(0)
    assert cc == (8, 6), f"Expected sm_86, got sm_{cc[0]}{cc[1]}"

    M, N, K = 128, 128, 128
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device="cuda")
    c = torch.empty((M, N), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _int8_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    torch.cuda.synchronize()

    expected = (a.to(torch.int32) @ b.to(torch.int32))
    max_err = (c - expected).abs().max().item()
    print(f"max abs error: {max_err}")
    assert max_err == 0, "INT8 mma should be exact on integers"
    print("OK: Triton tl.dot with INT8 operands works on sm_86")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.5: Run the INT8 smoke kernel**

Run: `python scripts/verify_triton_int8.py`
Expected: `max abs error: 0` and `OK: Triton tl.dot with INT8 operands works on sm_86`.

If it fails with a Triton compile error mentioning unsupported dtype, downgrade Triton or pick a different `out_dtype` and record the failure mode in `docs/PHASES/phase-0-notes.md`. **Do not silently work around it** — this is a load-bearing capability for the kernel design; if INT8 mma isn't available, the spec needs revision (drop to BF16-only KV).

- [ ] **Step 3.6: Update phase-0 notes — answer Open Question 1**

Edit `docs/PHASES/phase-0-notes.md`, fill row 1 of the Open Questions table:
- Answer: `Yes` (or `No, see below` with explanation)
- Evidence: `scripts/verify_triton_int8.py max_err=0 with triton X.Y.Z, torch A.B.C, sm_86`.

- [ ] **Step 3.7: Commit**

```bash
git add pyproject.toml scripts/verify_triton_int8.py docs/PHASES/phase-0-notes.md
git commit -m "phase 0: pin torch+triton, verify INT8 mma on sm_86"
```

---

## Task 4: Vendor reference implementations

**Purpose:** Get every paper's reference repo locally so later phases can read code paths cited in SPEC §5 and REFERENCES.md without re-cloning. Vendored copies are read-only references; flashquest never imports from them in shipped code.

**Files:**
- Create: `vendor/.gitkeep`
- Create: `vendor/README.md` (an index of what's vendored and why)
- Create: `scripts/vendor_clone.sh`

- [ ] **Step 4.1: Create the vendor index**

Create `vendor/.gitkeep` (empty file).

Create `vendor/README.md`:
```markdown
# Vendored reference repositories

Read-only references for the techniques composed in flashquest. See `docs/SPEC.md §5` and `docs/REFERENCES.md` for what we crib from each.

| Path | Repo | Phase | Why we vendor |
|---|---|---|---|
| `quest/` | mit-han-lab/Quest | P3 | Top-k page-retrieval algorithm + criticality scoring |
| `kivi/` | jy-yuan/KIVI | P3 | Per-channel-K / per-token-V quant; Triton quant kernels |
| `duo-attention/` | mit-han-lab/duo-attention | P4 | Pre-trained head classifications; dispatch pattern |
| `streaming-llm/` | mit-han-lab/streaming-llm | P3 | Sink-token + RoPE-shift trick |
| `marlin/` | IST-DASLab/marlin | P4 | W4A16 GEMM, used as-is for projections |
| `eagle/` | SafeAILab/EAGLE | P5 | Speculative decoding wrapper |
| `triton/` | triton-lang/triton | P2+ | `python/tutorials/06-fused-attention.py` is our FA-2 scaffold |
| `flash-attention/` | Dao-AILab/flash-attention | P0 | Reference numbers + profiling target |
| `block-sparse-attention/` | mit-han-lab/Block-Sparse-Attention | P4 | Per-head pattern API reference |

Contents are gitignored. Re-clone with `scripts/vendor_clone.sh`.
```

- [ ] **Step 4.2: Write the clone script (idempotent)**

Create `scripts/vendor_clone.sh`:
```bash
#!/usr/bin/env bash
# Idempotent vendor sync. Re-running pulls latest on each repo's default branch.
# Pin SHAs once Phase 1 begins to avoid upstream drift breaking phase reproducibility.
set -euo pipefail

VENDOR_DIR="$(git rev-parse --show-toplevel)/vendor"
mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"

clone_or_pull() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "==> Updating $dir"
    git -C "$dir" fetch --depth=1 origin
    git -C "$dir" reset --hard origin/HEAD
  else
    echo "==> Cloning $dir"
    git clone --depth=1 "$url" "$dir"
  fi
}

clone_or_pull https://github.com/mit-han-lab/Quest.git                  quest
clone_or_pull https://github.com/jy-yuan/KIVI.git                       kivi
clone_or_pull https://github.com/mit-han-lab/duo-attention.git          duo-attention
clone_or_pull https://github.com/mit-han-lab/streaming-llm.git          streaming-llm
clone_or_pull https://github.com/IST-DASLab/marlin.git                  marlin
clone_or_pull https://github.com/SafeAILab/EAGLE.git                    eagle
clone_or_pull https://github.com/triton-lang/triton.git                 triton
clone_or_pull https://github.com/Dao-AILab/flash-attention.git          flash-attention
clone_or_pull https://github.com/mit-han-lab/Block-Sparse-Attention.git block-sparse-attention

echo "Done. Vendored repos in $VENDOR_DIR"
```

```bash
chmod +x scripts/vendor_clone.sh
```

- [ ] **Step 4.3: Run the vendor clone script**

Run: `./scripts/vendor_clone.sh`
Expected: 9 repos cloned under `vendor/`. Disk usage ~1–2 GB (Triton repo is the heaviest).

- [ ] **Step 4.4: Verify the canonical files cited in SPEC §5 exist**

Run:
```bash
for f in \
  vendor/quest/quest/models/QuestAttention.py \
  vendor/kivi/quant/triton_quant.py \
  vendor/duo-attention/duo_attn/patch/llama.py \
  vendor/streaming-llm/streaming_llm/kv_cache.py \
  vendor/streaming-llm/streaming_llm/pos_shift/modify_llama.py \
  vendor/marlin/marlin/__init__.py \
  vendor/triton/python/tutorials/06-fused-attention.py
do
  if [ -f "$f" ]; then echo "OK $f"; else echo "MISSING $f"; fi
done
```
Expected: every line `OK ...`. If any path has changed upstream, update `vendor/README.md` and SPEC §5 with the new path. Record the deviation in `docs/PHASES/phase-0-notes.md`.

- [ ] **Step 4.5: Commit (vendor contents are gitignored, only metadata is committed)**

```bash
git add vendor/.gitkeep vendor/README.md scripts/vendor_clone.sh
git commit -m "phase 0: vendor reference repos (clone script + index)"
```

---

## Task 5: llama.cpp baseline at 8k context

**Purpose:** Establish the SPEC's primary tok/s baseline. llama.cpp's CUDA build with `-ngl 999` is the bar everything else is measured against, since it's what 4GB-laptop users currently run.

**Files:**
- Create: `scripts/bench_llamacpp.sh`
- Modify: `benchmarks/baselines.json` (created here, appended to in later tasks)

- [ ] **Step 5.1: Build llama.cpp with CUDA**

```bash
cd vendor
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```
Expected: build completes; `build/bin/llama-cli` and `build/bin/llama-bench` exist.

If build fails missing `nvcc`: install CUDA toolkit (`sudo apt install nvidia-cuda-toolkit` on WSL2 Ubuntu), re-run. Record toolkit version into env snapshot.

- [ ] **Step 5.2: Pull Llama-3.2-3B-Instruct Q4_K_M GGUF**

```bash
mkdir -p ~/models/llama-3.2-3b
huggingface-cli download \
  bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf \
  --local-dir ~/models/llama-3.2-3b
```

(`bartowski` is the most reliable GGUF mirror. If gated/missing, fall back to `unsloth/Llama-3.2-3B-Instruct-GGUF` or any reputable Q4_K_M conversion. Record the source.)

Expected: ~2.0 GB file at `~/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf`.

- [ ] **Step 5.3: Write the benchmark wrapper**

Create `scripts/bench_llamacpp.sh`:
```bash
#!/usr/bin/env bash
# Run llama.cpp benchmark at 8k context. Records tok/s + peak VRAM.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LLAMA_BIN="$REPO_ROOT/vendor/llama.cpp/build/bin/llama-bench"
MODEL="${MODEL:-$HOME/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"
OUT="$REPO_ROOT/benchmarks/llamacpp_8k.txt"

mkdir -p "$REPO_ROOT/benchmarks"

# -p 8192 prefill, -n 128 decode tokens, -ngl 999 = all layers on GPU
"$LLAMA_BIN" \
  -m "$MODEL" \
  -p 8192 -n 128 \
  -ngl 999 \
  -t 6 \
  -r 3 \
  -o md \
  | tee "$OUT"

# Peak VRAM: capture nvidia-smi at the tail of the run via nvidia-smi --query-gpu
echo "---" >> "$OUT"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv >> "$OUT"
```

```bash
chmod +x scripts/bench_llamacpp.sh
```

- [ ] **Step 5.4: Run the benchmark**

Run: `./scripts/bench_llamacpp.sh`
Expected: a markdown table from `llama-bench` showing prefill (`pp8192`) and decode (`tg128`) tok/s, written to `benchmarks/llamacpp_8k.txt`.

If it OOMs: drop `-ngl 999` to `-ngl 22` (offloads later layers) and record the partial-offload number; note this is a different operating point from the SPEC win condition.

- [ ] **Step 5.5: Record results into machine-readable baselines file**

Create `benchmarks/baselines.json`:
```json
{
  "host": "<copy GPU name + driver version from env_snapshot.json>",
  "date": "2026-04-29",
  "model": "Llama-3.2-3B-Instruct-Q4_K_M",
  "context": 8192,
  "results": {
    "llama_cpp": {
      "version": "<git-rev from vendor/llama.cpp>",
      "build": "CUDA, -ngl 999",
      "prefill_tok_s": 0.0,
      "decode_tok_s": 0.0,
      "peak_vram_mb": 0,
      "raw_log": "benchmarks/llamacpp_8k.txt"
    }
  }
}
```
Fill in the actual numbers from the run. Compute peak VRAM from `nvidia-smi memory.used` at run-tail (post-load).

- [ ] **Step 5.6: Commit**

```bash
git add scripts/bench_llamacpp.sh benchmarks/baselines.json benchmarks/llamacpp_8k.txt
git commit -m "phase 0: llama.cpp Q4_K_M baseline at 8k context"
```

---

## Task 6: vLLM baseline at 8k context

**Purpose:** Second baseline from a different stack (PagedAttention + FA-2 backend). Sets the reference for what an "optimized server" path achieves on the same model. Note: vLLM does not load Q4_K_M GGUF; we use BF16 or AWQ — record the actual quant chosen.

**Files:**
- Create: `scripts/bench_vllm.py`

- [ ] **Step 6.1: Install vLLM (in the project venv)**

```bash
. .venv/bin/activate
pip install "vllm>=0.6.0"
```

vLLM brings its own torch pin; if it conflicts with the torch we installed in Task 3, accept vLLM's pin (vLLM's perf-relevant CUDA paths assume its bundled torch). Re-run `python scripts/verify_env.py` after install to confirm CUDA is still functional.

- [ ] **Step 6.2: Decide quant (write decision into phase-0 notes)**

vLLM cannot load `Llama-3.2-3B-Instruct-Q4_K_M.gguf`. Options ordered by closeness to the GGUF baseline:
1. AWQ INT4 (`hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4`) — closest match, similar bits.
2. BF16 (`meta-llama/Llama-3.2-3B-Instruct`) — uncompressed; needs ~6.4 GB just for weights — **will not fit in 4 GB VRAM, expect OOM**.
3. GPTQ INT4 — equivalent to AWQ for our purposes.

Pick (1). Add a `Decisions / deviations` entry in `docs/PHASES/phase-0-notes.md`:
> vLLM baseline uses AWQ INT4 instead of Q4_K_M because vLLM does not load GGUF. Same model, comparable bit-width, different quant method — note when comparing against the llama.cpp number.

- [ ] **Step 6.3: Write the bench harness**

Create `scripts/bench_vllm.py`:
```python
"""vLLM decode + prefill timing at 8k context, single request."""
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main() -> None:
    model_id = "hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4"
    llm = LLM(
        model=model_id,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=False,
        swap_space=0,
    )

    # Build an 8k-token prompt (~ 7000 input tokens leaving headroom for generation)
    prompt = ("The quick brown fox jumps over the lazy dog. " * 1000)[:30000]

    # Warm-up
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.0))
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    n_in = len(out.prompt_token_ids)
    n_out = len(out.outputs[0].token_ids)
    elapsed = t1 - t0
    decode_tok_s = n_out / elapsed  # crude — includes prefill in total time

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = {
        "model": model_id,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "elapsed_s": elapsed,
        "tok_s_total": (n_in + n_out) / elapsed,
        "decode_tok_s_approx": decode_tok_s,
        "peak_vram_mb": peak_mb,
    }
    print(json.dumps(result, indent=2))

    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "vllm_8k.json"
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run**

Run: `python scripts/bench_vllm.py`
Expected: prints a JSON result, writes `benchmarks/vllm_8k.json`. If it OOMs at 0.85 utilization, drop to 0.75 and re-run. Record the utilization needed.

- [ ] **Step 6.5: Append to baselines.json**

Update `benchmarks/baselines.json`'s `results` object with a `vllm` entry containing the numbers from `vllm_8k.json` plus a `quant: "awq-int4"` field and the deviation note.

- [ ] **Step 6.6: Commit**

```bash
git add scripts/bench_vllm.py benchmarks/baselines.json benchmarks/vllm_8k.json
git commit -m "phase 0: vLLM AWQ baseline at 8k context"
```

---

## Task 7: FA-2 reference profile + WSL2 profiler check

**Purpose:** Resolve SPEC Open Question 4 (do `nsys` / `ncu` work in WSL2?) and capture a reference FA-2 trace we'll compare against in Phase 2. We use the upstream FlashAttention package on a synthetic shape — the goal is profiler validation, not a real benchmark.

**Files:**
- Create: `scripts/profile_fa2.py`

- [ ] **Step 7.1: Install flash-attn**

```bash
. .venv/bin/activate
pip install flash-attn --no-build-isolation
```

This compiles native CUDA on first install (~10 min on this hardware). If it fails with a CUDA-version mismatch, install the prebuilt wheel matching `torch.version.cuda` (`pip install flash-attn==<ver> --find-links https://github.com/Dao-AILab/flash-attention/releases`).

- [ ] **Step 7.2: Write the profile harness**

Create `scripts/profile_fa2.py`:
```python
"""Run a single FA-2 forward on a representative shape; print timing."""
import time

import torch
from flash_attn import flash_attn_func


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Llama-3.2-3B shape: 24 heads, head_dim=64; pick seq=8192, batch=1
    B, H_q, H_kv, S_q, S_kv, D = 1, 24, 8, 8192, 8192, 64
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, H_kv, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, H_kv, D, dtype=dtype, device=device)

    # Warm-up
    for _ in range(3):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / n_iter * 1000
    print(f"FA-2 fwd (B={B}, S={S_q}, H_q={H_q}, H_kv={H_kv}, D={D}) avg: {avg_ms:.3f} ms")
    print(f"output sum (sanity): {out.float().sum().item():.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.3: Plain run (sanity)**

Run: `python scripts/profile_fa2.py`
Expected: prints an avg-ms number. Record it.

- [ ] **Step 7.4: Profile with `nsys`**

Run:
```bash
nsys profile --output benchmarks/fa2_profile --force-overwrite=true \
    --stats=true python scripts/profile_fa2.py | tail -80
```
Expected: writes `benchmarks/fa2_profile.nsys-rep` and prints kernel summary. The summary should show FA-2 kernel symbols (e.g. `flash_fwd_kernel`).

If `nsys` is missing: `sudo apt install nvidia-nsight-systems` on WSL2, or download the matching version from NVIDIA's site. Record the install path.

- [ ] **Step 7.5: Profile with `ncu` (one kernel only — ncu is slow)**

Run:
```bash
ncu --target-processes all \
    --kernel-name regex:flash_fwd \
    --launch-count 1 \
    -o benchmarks/fa2_ncu \
    python scripts/profile_fa2.py
```
Expected: writes `benchmarks/fa2_ncu.ncu-rep`. View summary:
```bash
ncu --import benchmarks/fa2_ncu.ncu-rep --details all | head -80
```

If `ncu` requires elevated privileges in WSL2, set:
```bash
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
```
and restart WSL. Record this fix in `docs/PHASES/phase-0-notes.md`.

- [ ] **Step 7.6: Update phase-0 notes — answer Open Questions 3 and 4**

In `docs/PHASES/phase-0-notes.md`:
- Q3 (usable VRAM): from `nvidia-smi memory.free` immediately after a fresh boot of the WSL2 session (no inference running). Capture both with the user's normal Windows session running and at idle. Record both numbers.
- Q4 (nsys/ncu in WSL2): `Yes` if both produced reports, `Yes with caveat: <fix>` if Step 7.5 needed the modprobe option.

(Q2 — INT8 dequant fused into mma operand path — punt to Phase 2 design notes; not answerable without a candidate kernel.)

- [ ] **Step 7.7: Commit**

```bash
git add scripts/profile_fa2.py docs/PHASES/phase-0-notes.md
# .nsys-rep and .ncu-rep are gitignored under benchmarks/
git commit -m "phase 0: FA-2 reference profile + WSL2 nsys/ncu validation"
```

---

## Task 8: Update README + DOC.md, close out Phase 0

**Purpose:** Land the Phase 0 deliverable per SPEC §6: a baseline numbers table in the README and a living project doc.

**Files:**
- Modify: `README.md` (append a Baselines section)
- Create: `DOC.md` (per global CLAUDE.md instructions)
- Modify: `docs/PHASES/phase-0-notes.md` (set status to `complete`)

- [ ] **Step 8.1: Append Baselines section to README.md**

Append to `README.md` (under the existing "Where to start" section):

```markdown
## Phase 0 baselines

Baselines on the target machine (RTX 3050 Ti Laptop, sm_86, 4 GB VRAM, WSL2). Llama-3.2-3B-Instruct, 8k context, 128 decode tokens, single request.

| Stack | Quant | Prefill tok/s | Decode tok/s | Peak VRAM | Notes |
|---|---|---|---|---|---|
| llama.cpp (CUDA, `-ngl 999`) | Q4_K_M | <fill> | <fill> | <fill> MB | `benchmarks/llamacpp_8k.txt` |
| vLLM (FA-2 backend) | AWQ-INT4 | — | <fill> | <fill> MB | combined tok/s reported; `benchmarks/vllm_8k.json` |
| FA-2 reference (synthetic, fwd only) | BF16 | — | — | — | <fill> ms / forward at S=8192, H_q=24, H_kv=8, D=64 |

Re-run via `./scripts/bench_llamacpp.sh` and `python scripts/bench_vllm.py`. See `docs/PHASES/phase-0-notes.md` for environment caveats.
```

Fill in the `<fill>` slots from `benchmarks/baselines.json` and `benchmarks/vllm_8k.json`.

- [ ] **Step 8.2: Create DOC.md (living project doc)**

Create `DOC.md`:
```markdown
# flashquest — Project Doc

Single living document for the project. See `README.md` for the elevator pitch and `docs/SPEC.md` for the full design.

## Overview

Sparse-retrieval attention Triton kernel for Ampere laptop GPUs (sm_86). Targets long-context inference (32k–128k) of 3B–8B models on 4 GB VRAM. Composes Quest, KIVI, DuoAttention, StreamingLLM, Marlin, FlashAttention-2, and (optionally) EAGLE-2 into one fused kernel.

## Getting started

```bash
git clone <repo>
cd flashquest
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,bench]"
./scripts/vendor_clone.sh        # vendor reference repos
python scripts/verify_env.py     # snapshot env
python scripts/verify_triton_int8.py  # confirm sm_86 INT8 mma
```

## Architecture

See `docs/SPEC.md §4`. Single Triton kernel per attention layer; sparse outer loop over Quest-selected KV blocks; INT8 KV with KIVI-style scales; per-head pattern dispatch (DuoAttention).

## Phases

- **Phase 0 — Setup & baselines** (status: <update at end of Phase 0>) — env verified, baselines captured. See `docs/PHASES/phase-0-notes.md`.
- Phase 1 — Eager Python reference. Not started.
- Phase 2 — Dense FA-2 Triton baseline. Not started.
- Phase 3 — Sparse retrieval + INT8 KV. Not started.
- Phase 4 — DuoAttention split + 8B model. Not started.
- Phase 5 — Integration + speculation. Not started.
- Phase 6 — Polish & release. Not started.

## Configuration

- Python: see `.python-version` and `pyproject.toml`.
- CUDA / Triton versions: see `docs/PHASES/env_snapshot.json`.
- Model weights: not in repo. Pull GGUFs to `~/models/`.
- Vendored reference repos: `vendor/` (gitignored). Re-clone via `scripts/vendor_clone.sh`.

## References

- `docs/SPEC.md` — full design
- `docs/REFERENCES.md` — papers + code paths index
- `docs/PHASES/phase-N-notes.md` — per-phase journals
- `benchmarks/baselines.json` — machine-readable benchmark history
```

- [ ] **Step 8.3: Mark phase-0 notes complete**

In `docs/PHASES/phase-0-notes.md` change `**Status:** in progress` to `**Status:** complete (2026-MM-DD)` with the actual finish date. Confirm all four open-question rows have `Answer` and `Evidence` populated (Q2 is `Deferred to Phase 2`).

- [ ] **Step 8.4: Final smoke + commit**

```bash
. .venv/bin/activate
pytest tests/ -v
python scripts/verify_env.py > /dev/null
python scripts/verify_triton_int8.py
```
Expected: smoke tests pass, env script writes snapshot, INT8 kernel reports OK.

- [ ] **Step 8.5: Commit**

```bash
git add README.md DOC.md docs/PHASES/phase-0-notes.md
git commit -m "phase 0: README baselines table + DOC.md + phase notes complete"
```

- [ ] **Step 8.6: Tag the phase**

```bash
git tag -a phase-0 -m "Phase 0 complete: env verified, baselines captured"
```

---

## Self-review checklist (run after writing the plan, before execution)

- [x] **Spec coverage:** every Phase 0 bullet from SPEC §6 maps to a task — env verify (T1, T3), llama.cpp baseline (T5), vLLM baseline (T6), reference impls (T4), profilers (T7), README baseline table (T8). The four open questions from SPEC §8 are all addressed (Q1→T3, Q2→deferred with note, Q3→T7, Q4→T7).
- [x] **No placeholders:** every step has actual commands or actual code. The `<fill>` markers in T8.1 are explicit data-entry slots, not unwritten content.
- [x] **Type / name consistency:** the JSON schema for `benchmarks/baselines.json` defined in T5.5 is extended (not replaced) in T6.5. Script paths cited in README (T8.1) match the script paths created earlier (`bench_llamacpp.sh`, `bench_vllm.py`).
- [x] **Reversibility:** every task ends with a commit. If any task fails the host (e.g. vLLM install conflicts), prior commits remain valid Phase-0 progress.

## Phase 0 → Phase 1 handoff

When this plan completes:
- `phase-0` git tag exists
- `benchmarks/baselines.json` has `llama_cpp` and `vllm` entries with real numbers
- `docs/PHASES/phase-0-notes.md` has answers to OQ 1, 3, 4 (OQ 2 marked deferred)
- `vendor/` has all 9 reference repos cloned
- INT8 mma confirmed on sm_86 (or, if not, SPEC revised — flag this loudly)

Phase 1 can begin with: writing the eager-Python Quest-style attention against `vendor/quest/quest/models/QuestAttention.py` as the reference, validated against HF Llama-3.2-1B at 4k context. That gets its own plan in `docs/superpowers/plans/` when we get there.
