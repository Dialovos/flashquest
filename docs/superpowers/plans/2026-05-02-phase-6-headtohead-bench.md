# Phase 6 Task 4 — Head-to-head Benchmark Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a reproducible head-to-head benchmark of flashquest vs llama.cpp vs vLLM at 8 k / 32 k / 128 k context on Llama-3.2-3B (each backend's native-strongest quant), produce a JSON + markdown table for README, and faithfully record the SPEC §11.4 "≥5× capability gain" gate result whether it passes or fails.

**Architecture:** Three per-backend bench scripts (one new for flashquest; two existing modified to take ctx args + handle OOM gracefully) plus one orchestrator that runs all 9 (backend, ctx) cells sequentially with a 10-min hard cap per cell, merges the per-cell JSONs, and renders the markdown table. No `src/flashquest/` code — pure benchmarking glue.

**Tech Stack:** Python `subprocess`, existing `flashquest.runtime.awq_load` + `cache.persistent_int8` + `eager.llama_persistent_patch`, llama.cpp `llama-bench` binary, vLLM 0.7.3.

**Reference:** `docs/superpowers/specs/2026-05-02-phase-6-headtohead-bench-design.md`. Existing patterns: `scripts/phase6_bench_decode_32k_v2.py` (flashquest decode bench), `scripts/bench_llamacpp.sh` (llama.cpp baseline), `scripts/bench_vllm.py` (vLLM baseline), `scripts/phase6_run_ruler_4k.py` (orchestrator pattern).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `scripts/bench_flashquest.py` | Single-cell flashquest decode bench (parametric ctx, JSON out) | Create |
| `scripts/bench_llamacpp.sh` | Existing — extend to take `CTX` env var | Modify |
| `scripts/bench_vllm.py` | Existing — accept `--max-model-len`, catch OOM | Modify |
| `scripts/phase6_run_headtohead.py` | Orchestrator: 3 backends × 3 contexts; merge → JSON + markdown | Create |
| `benchmarks/phase6_headtohead.json` | Run output | Create at end |
| `benchmarks/phase6_headtohead.md` | Generated table for README inclusion | Create at end |
| `tests/test_phase6_headtohead.py` | Unit tests for orchestrator helpers (parsing, rendering, dry-run) | Create |
| `docs/PHASES/phase-6-notes.md` | Append task 4 section | Modify at end |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 4, embed table | Modify at end |

---

## Task 1: `bench_flashquest.py` — single-cell decode bench

**Files:**
- Create: `scripts/bench_flashquest.py`

- [ ] **Step 1: Write the script**

```python
"""Single-cell flashquest decode bench. Writes a per-cell JSON to --out.

Mirrors scripts/phase6_bench_decode_32k_v2.py but parametric over --ctx-len
and catches torch.cuda.OutOfMemoryError to record `oom: true` instead of
crashing the orchestrator.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.cache.persistent_int8 import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, required=True)
    p.add_argument("--n-decode", type=int, default=32)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    record = {
        "backend": "flashquest",
        "quant": "AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25",
        "ctx_len": args.ctx_len,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "peak_vram_mib": None,
        "wall_s": None,
        "oom": False,
        "error": None,
    }

    t_start = time.perf_counter()
    try:
        torch.cuda.reset_peak_memory_stats()
        model, tok = load_awq_model(args.model)
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        pattern = torch.ones(
            cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool,
        )
        cache = PersistentInt8KVCache(
            batch_size=1,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=args.ctx_len + args.n_decode + 128,
            page_size=args.page_size,
            device="cuda",
        )
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=args.retention, num_sinks=args.num_sinks,
            window_pages=args.window_pages, page_size=args.page_size,
        )

        ids = torch.randint(0, cfg.vocab_size, (1, args.ctx_len), device="cuda")

        # Prefill
        torch.cuda.synchronize()
        t_pf0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=True)
        torch.cuda.synchronize()
        t_pf1 = time.perf_counter()
        record["prefill_tok_s"] = args.ctx_len / (t_pf1 - t_pf0)

        # Decode
        next_ids = torch.tensor([[0]], device="cuda")
        torch.cuda.synchronize()
        t_dec0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.n_decode):
                out = model(input_ids=next_ids, use_cache=True)
                next_ids = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()
        t_dec1 = time.perf_counter()
        record["decode_tok_s"] = args.n_decode / (t_dec1 - t_dec0)
        record["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    except torch.cuda.OutOfMemoryError as exc:
        record["oom"] = True
        record["error"] = f"torch.cuda.OutOfMemoryError: {exc}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_s"] = time.perf_counter() - t_start
        try:
            _free()
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke at ctx=256 (1-2 min wall)**

Run:
```
source .venv/bin/activate && python scripts/bench_flashquest.py --ctx-len 256 --n-decode 4 --out /tmp/fq_smoke.json
```
Expected: writes `/tmp/fq_smoke.json` with `oom: false`, `decode_tok_s` non-null, exits 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_flashquest.py
git commit -m "phase 6 task 4: bench_flashquest.py — single-cell decode bench"
```

---

## Task 2: Modify `bench_llamacpp.sh` to accept `CTX` env

**Files:**
- Modify: `scripts/bench_llamacpp.sh`

- [ ] **Step 1: Read the current file**

Open `scripts/bench_llamacpp.sh`. Note the hardcoded `-p 8192` and the `OUT="$REPO_ROOT/benchmarks/llamacpp_8k.txt"` line.

- [ ] **Step 2: Replace hardcoded 8192 with `${CTX:-8192}`**

Edit `scripts/bench_llamacpp.sh`. Replace:

```bash
OUT="$REPO_ROOT/benchmarks/llamacpp_8k.txt"
```

with:

```bash
CTX="${CTX:-8192}"
OUT="${OUT:-$REPO_ROOT/benchmarks/llamacpp_${CTX}.txt}"
```

And replace the line `-p 8192 -n 128 \` with:

```bash
-p "$CTX" -n 128 \
```

- [ ] **Step 3: Smoke at CTX=256 (~30 s wall)**

Run:
```
CTX=256 OUT=/tmp/llamacpp_smoke.txt bash scripts/bench_llamacpp.sh
```
Expected: writes `/tmp/llamacpp_smoke.txt`, llama-bench prints a markdown table, exits 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_llamacpp.sh
git commit -m "phase 6 task 4: bench_llamacpp.sh — accept CTX + OUT env vars"
```

---

## Task 3: Modify `bench_vllm.py` to accept `--max-model-len` + catch OOM

**Files:**
- Modify: `scripts/bench_vllm.py`

- [ ] **Step 1: Rewrite `bench_vllm.py`**

Replace the entire file with:

```python
"""vLLM single-request decode bench, parametric over --max-model-len.

Catches OOM during LLM(...) construction and llm.generate(...) and writes a
per-cell JSON record so the orchestrator can keep going.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def _record_skeleton(model_id: str, max_model_len: int) -> dict:
    return {
        "backend": "vLLM 0.7.3",
        "quant": "AWQ-INT4, FP16 KV",
        "ctx_len": max_model_len,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "peak_vram_mib": None,
        "wall_s": None,
        "oom": False,
        "error": None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    record = _record_skeleton(args.model, args.max_model_len)
    t_start = time.perf_counter()

    import torch  # imported here so OOM during import is caught below

    try:
        from vllm import LLM, SamplingParams

        torch.cuda.reset_peak_memory_stats()

        llm = LLM(
            model=args.model,
            quantization="awq",
            dtype="float16",
            gpu_memory_utilization=0.95,
            max_model_len=args.max_model_len,
            enforce_eager=False,
            swap_space=0,
        )

        # Build a prompt that uses ~80% of max_model_len to leave decode headroom.
        target_in = max(64, int(args.max_model_len * 0.8))
        prompt = ("The quick brown fox jumps over the lazy dog. " * (target_in // 8 + 1))
        # Truncate by tokens via the model's tokenizer to stay just below max_model_len.
        tok = llm.get_tokenizer()
        ids = tok.encode(prompt)[:target_in]
        prompt = tok.decode(ids, skip_special_tokens=True)

        # Warm-up
        llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
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

        record["decode_tok_s"] = n_out / elapsed if elapsed > 0 else None
        record["prefill_tok_s"] = n_in / elapsed if elapsed > 0 else None
        record["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:  # type: ignore[attr-defined]
        msg = str(exc).lower()
        if "out of memory" in msg or "kv cache" in msg or "no available" in msg:
            record["oom"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_s"] = time.perf_counter() - t_start
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke at max_model_len=512 (~1 min wall)**

Run:
```
source .venv/bin/activate && python scripts/bench_vllm.py --max-model-len 512 --out /tmp/vllm_smoke.json
```
Expected: writes `/tmp/vllm_smoke.json`. If vLLM is installed, `oom: false`, `decode_tok_s` non-null. If vLLM isn't usable any more (env drift since Phase 0), `error` field captured cleanly without crashing.

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_vllm.py
git commit -m "phase 6 task 4: bench_vllm.py — accept --max-model-len + catch OOM"
```

---

## Task 4: Orchestrator skeleton + `--dry-run`

**Files:**
- Create: `scripts/phase6_run_headtohead.py`
- Create: `tests/test_phase6_headtohead.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_phase6_headtohead.py`:

```python
"""Phase 6 task 4 — head-to-head orchestrator tests."""
import json
import sys
from pathlib import Path


# Add scripts/ to path so we can import the orchestrator module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase6_run_headtohead as H


def test_planned_matrix_default():
    """3 backends × 3 contexts = 9 cells in canonical order."""
    matrix = H.planned_matrix()
    assert len(matrix) == 9
    backends = {b for b, _ in matrix}
    ctxs = {c for _, c in matrix}
    assert backends == {"flashquest", "llamacpp", "vllm"}
    assert ctxs == {8192, 32768, 131072}
    # All flashquest cells come before all llamacpp before all vllm:
    backend_order = [b for b, _ in matrix]
    assert backend_order == (
        ["flashquest"] * 3 + ["llamacpp"] * 3 + ["vllm"] * 3
    )


def test_dry_run_prints_matrix(capsys):
    """--dry-run prints all 9 cells and returns 0 without launching anything."""
    rc = H.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flashquest @ 8192" in out
    assert "vllm @ 131072" in out
    assert "9 cells" in out


def test_render_markdown_table_has_three_backend_rows():
    """render_markdown produces a table with one row per backend."""
    cells = [
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 8192,
         "decode_tok_s": 7.2, "peak_vram_mib": 4500, "oom": False, "error": None},
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 32768,
         "decode_tok_s": 5.14, "peak_vram_mib": 6379, "oom": False, "error": None},
        {"backend": "flashquest", "quant": "AWQ-INT4 + INT8", "ctx_len": 131072,
         "decode_tok_s": None, "peak_vram_mib": None, "oom": True, "error": "OOM"},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 8192,
         "decode_tok_s": 39.6, "peak_vram_mib": 3543, "oom": False, "error": None},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 32768,
         "decode_tok_s": 12.0, "peak_vram_mib": 3700, "oom": False, "error": None},
        {"backend": "llama.cpp", "quant": "Q4_K_M", "ctx_len": 131072,
         "decode_tok_s": None, "peak_vram_mib": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 8192,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 32768,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
        {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4", "ctx_len": 131072,
         "decode_tok_s": None, "oom": True, "error": "OOM"},
    ]
    md = H.render_markdown(cells)
    # Header row + separator + 3 data rows
    body_lines = [l for l in md.splitlines() if l.startswith("|")]
    assert len(body_lines) >= 5  # header + sep + 3 rows
    assert "flashquest" in md
    assert "llama.cpp" in md
    assert "vLLM" in md
    assert "5.14" in md
    assert "OOM" in md
```

- [ ] **Step 2: Run tests to verify ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_phase6_headtohead.py -v
```
Expected: ImportError on `phase6_run_headtohead`.

- [ ] **Step 3: Implement orchestrator (skeleton + dry-run + render only — no run_one yet)**

Create `scripts/phase6_run_headtohead.py`:

```python
"""Phase 6 task 4 — head-to-head bench orchestrator.

Runs flashquest + llama.cpp + vLLM at 8 k / 32 k / 128 k sequentially,
collates per-cell JSONs into benchmarks/phase6_headtohead.json, and writes
a markdown table to benchmarks/phase6_headtohead.md.

One backend at a time; 10 min hard cap per cell; nice -n 19 for shell-driven
cells. Resumable via --skip-existing.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CELL_DIR = REPO_ROOT / "benchmarks" / "phase6_cells"
RESULTS_JSON = REPO_ROOT / "benchmarks" / "phase6_headtohead.json"
RESULTS_MD = REPO_ROOT / "benchmarks" / "phase6_headtohead.md"

BACKENDS = ["flashquest", "llamacpp", "vllm"]
CTXS = [8192, 32768, 131072]
CELL_TIMEOUT_S = 600  # 10 min hard cap


def planned_matrix() -> list[tuple[str, int]]:
    return [(b, c) for b in BACKENDS for c in CTXS]


def _cell_path(backend: str, ctx: int) -> Path:
    return CELL_DIR / f"{backend}_{ctx}.json"


def _free_gpu() -> None:
    """2 s sleep + cuda.empty_cache + gc, between cells."""
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    time.sleep(2.0)


def render_markdown(cells: list[dict]) -> str:
    """Render the head-to-head cells as a markdown table.

    One row per backend; columns: 8 k tok/s, 32 k tok/s, 128 k fits?,
    peak VRAM at the largest fit context.
    """
    by_backend: dict[str, dict[int, dict]] = {}
    for cell in cells:
        by_backend.setdefault(cell["backend"], {})[cell["ctx_len"]] = cell

    lines = [
        "| Backend | Quant | 8 k decode tok/s | 32 k decode tok/s | 128 k fits? | Peak VRAM @ max fit |",
        "|---|---|---|---|---|---|",
    ]
    for backend, by_ctx in by_backend.items():
        c8 = by_ctx.get(8192, {})
        c32 = by_ctx.get(32768, {})
        c128 = by_ctx.get(131072, {})

        def cell_fmt(c: dict) -> str:
            if not c:
                return "—"
            if c.get("oom"):
                return "OOM"
            if c.get("error"):
                return "✗"
            v = c.get("decode_tok_s")
            return f"{v:.2f}" if v is not None else "—"

        c128_str = "✓" if c128 and not c128.get("oom") and not c128.get("error") else "✗"
        # Peak VRAM at the largest fit context
        peak = "n/a"
        for c in (c128, c32, c8):
            if c and not c.get("oom") and not c.get("error") and c.get("peak_vram_mib"):
                peak = f"{c['peak_vram_mib']} MiB"
                break

        quant = (c32 or c8 or c128 or {}).get("quant", "")
        lines.append(
            f"| {backend} | {quant} | {cell_fmt(c8)} | {cell_fmt(c32)} | {c128_str} | {peak} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="print planned matrix and exit")
    p.add_argument("--skip-existing", action="store_true",
                   help="skip cells whose JSON already exists")
    args = p.parse_args(argv)

    matrix = planned_matrix()

    if args.dry_run:
        print(f"{len(matrix)} cells planned:")
        for b, c in matrix:
            print(f"  - {b} @ {c}")
        return 0

    # Real run is added in Task 5.
    sys.stderr.write("error: full run not yet implemented (use --dry-run)\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_phase6_headtohead.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase6_run_headtohead.py tests/test_phase6_headtohead.py
git commit -m "phase 6 task 4: orchestrator skeleton + render_markdown + dry-run"
```

---

## Task 5: Orchestrator `run_one` (real subprocess driver)

**Files:**
- Modify: `scripts/phase6_run_headtohead.py`

- [ ] **Step 1: Add `run_one` and wire it into `main`**

In `scripts/phase6_run_headtohead.py`, replace the `main(...)` body's tail
(everything from `# Real run is added in Task 5.` to the end of `main`) with:

```python
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    cells: list[dict] = []
    for backend, ctx in matrix:
        cell_path = _cell_path(backend, ctx)
        if args.skip_existing and cell_path.exists():
            print(f"[skip] {backend} @ {ctx} (cached at {cell_path})")
            cells.append(json.loads(cell_path.read_text()))
            continue
        print(f"[run]  {backend} @ {ctx} → {cell_path}", flush=True)
        cell = run_one(backend, ctx, cell_path)
        cells.append(cell)
        _free_gpu()

    RESULTS_JSON.write_text(json.dumps(_build_results(cells), indent=2))
    RESULTS_MD.write_text(render_markdown(cells))
    print(f"\nWrote {RESULTS_JSON}")
    print(f"Wrote {RESULTS_MD}\n")
    print(render_markdown(cells))
    return 0
```

Add the helpers above `main`:

```python
def _build_results(cells: list[dict]) -> dict:
    return {
        "host": {"gpu": "NVIDIA GeForce RTX 3050 Ti Laptop GPU",
                 "vram_mib": 4095, "cuda": "12.5", "wsl2": True},
        "model": "Llama-3.2-3B-Instruct",
        "date": time.strftime("%Y-%m-%d"),
        "headline_metric": "decode tok/s at ctx=32768",
        "capability_axis": "max ctx that decodes (≥1 tok/s, no OOM)",
        "results": cells,
    }


def run_one(backend: str, ctx: int, out_path: Path) -> dict:
    """Drive a per-backend bench script, parse its JSON / log output."""
    if backend == "flashquest":
        cmd = [
            "nice", "-n", "19",
            sys.executable, str(REPO_ROOT / "scripts" / "bench_flashquest.py"),
            "--ctx-len", str(ctx),
            "--n-decode", "32",
            "--out", str(out_path),
        ]
    elif backend == "llamacpp":
        log_path = out_path.with_suffix(".llamacpp.txt")
        cmd = [
            "bash", str(REPO_ROOT / "scripts" / "bench_llamacpp.sh"),
        ]
        env_overrides = {"CTX": str(ctx), "OUT": str(log_path)}
    elif backend == "vllm":
        cmd = [
            "nice", "-n", "19",
            sys.executable, str(REPO_ROOT / "scripts" / "bench_vllm.py"),
            "--max-model-len", str(ctx),
            "--out", str(out_path),
        ]
    else:
        raise ValueError(f"unknown backend: {backend}")

    t0 = time.perf_counter()
    try:
        if backend == "llamacpp":
            import os
            env = {**os.environ, **env_overrides}
            res = subprocess.run(
                cmd, env=env, capture_output=True, text=True,
                timeout=CELL_TIMEOUT_S,
            )
        else:
            res = subprocess.run(
                cmd, capture_output=True, text=True, timeout=CELL_TIMEOUT_S,
            )
    except subprocess.TimeoutExpired:
        record = {
            "backend": backend, "ctx_len": ctx,
            "decode_tok_s": None, "peak_vram_mib": None,
            "wall_s": time.perf_counter() - t0,
            "oom": False, "error": f"timeout (>{CELL_TIMEOUT_S}s)",
        }
        out_path.write_text(json.dumps(record, indent=2))
        return record

    if backend == "llamacpp":
        # Parse llama-bench markdown output → JSON cell.
        record = _parse_llamacpp_log(log_path, ctx, res.returncode, res.stderr)
        out_path.write_text(json.dumps(record, indent=2))
        return record

    if out_path.exists():
        return json.loads(out_path.read_text())

    return {
        "backend": backend, "ctx_len": ctx,
        "decode_tok_s": None, "peak_vram_mib": None,
        "wall_s": time.perf_counter() - t0,
        "oom": False,
        "error": f"subprocess returned {res.returncode}: {res.stderr[-500:]}",
    }


def _parse_llamacpp_log(log_path: Path, ctx: int, rc: int, stderr: str) -> dict:
    """Extract prefill/decode tok/s + peak VRAM from llama-bench markdown."""
    record = {
        "backend": "llama.cpp",
        "quant": "Q4_K_M, FP16 KV",
        "ctx_len": ctx,
        "decode_tok_s": None, "prefill_tok_s": None, "peak_vram_mib": None,
        "wall_s": None, "oom": False, "error": None,
    }
    if rc != 0 or not log_path.exists():
        msg = (stderr or "").lower()
        if "out of memory" in msg or "cuda" in msg and "fail" in msg:
            record["oom"] = True
        record["error"] = f"rc={rc}: {(stderr or '')[-400:]}"
        return record

    text = log_path.read_text()
    # Lines like "| llama 3B Q4_K_M | ... | pp8192      | 736.55 ± 55.81 |"
    pp_match = re.search(rf"pp{ctx}\s*\|\s*([0-9.]+)\s*", text)
    tg_match = re.search(r"tg128\s*\|\s*([0-9.]+)\s*", text)
    if pp_match:
        record["prefill_tok_s"] = float(pp_match.group(1))
    if tg_match:
        record["decode_tok_s"] = float(tg_match.group(1))
    # nvidia-smi tail: "VRAM at run-tail (nvidia-smi):" then a CSV row.
    smi_match = re.search(r"(\d+)\s*MiB,\s*\d+\s*MiB", text)
    if smi_match:
        record["peak_vram_mib"] = int(smi_match.group(1))
    if record["decode_tok_s"] is None:
        record["error"] = "could not parse llama-bench output"
    return record
```

- [ ] **Step 2: Add a unit test for the llama-bench parser**

Append to `tests/test_phase6_headtohead.py`:

```python
def test_parse_llamacpp_log(tmp_path):
    """_parse_llamacpp_log extracts pp + tg + smi from an llama-bench markdown."""
    log = tmp_path / "llamacpp_8192.txt"
    log.write_text(
        "| model | size | params | backend | ngl | test | t/s |\n"
        "|---|---|---|---|---|---|---|\n"
        "| llama 3B Q4_K_M | 1.87 GiB | 3.21 B | CUDA | 999 | pp8192      | 736.55 ± 55.81 |\n"
        "| llama 3B Q4_K_M | 1.87 GiB | 3.21 B | CUDA | 999 | tg128       | 39.60 ± 0.11   |\n"
        "---\n"
        "VRAM at run-tail (nvidia-smi):\n"
        "memory.used [MiB], memory.free [MiB]\n"
        "3543 MiB, 552 MiB\n"
    )
    rec = H._parse_llamacpp_log(log, ctx=8192, rc=0, stderr="")
    assert rec["prefill_tok_s"] == 736.55
    assert rec["decode_tok_s"] == 39.60
    assert rec["peak_vram_mib"] == 3543
    assert rec["oom"] is False


def test_parse_llamacpp_log_oom(tmp_path):
    """A non-zero rc with 'out of memory' in stderr → oom=true."""
    log = tmp_path / "fake.txt"
    log.write_text("")
    rec = H._parse_llamacpp_log(
        log, ctx=131072, rc=1,
        stderr="ggml_cuda_compute_forward: out of memory",
    )
    assert rec["oom"] is True
```

- [ ] **Step 3: Run the new tests + dry-run**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_phase6_headtohead.py -v && \
  python scripts/phase6_run_headtohead.py --dry-run
```
Expected: 5 PASS; dry-run prints `9 cells planned:` followed by all 9 entries.

- [ ] **Step 4: Commit**

```bash
git add scripts/phase6_run_headtohead.py tests/test_phase6_headtohead.py
git commit -m "phase 6 task 4: orchestrator run_one + llama-bench parser + tests"
```

---

## Task 6: Orchestrator end-to-end smoke (tiny ctx)

**Files:** none (run-only)

- [ ] **Step 1: Patch `CTXS` to a tiny value temporarily, dry-run-disabled**

Run a one-shot smoke at ctx=256 by overriding the matrix in-place. From shell:

```
source .venv/bin/activate && python -c "
import scripts.phase6_run_headtohead as H
H.CTXS = [256]
import sys
sys.exit(H.main([]))
"
```

Expected: prints `[run]` lines for `flashquest @ 256`, `llamacpp @ 256`, `vllm @ 256`. Each takes <2 min wall (model load dominates). Final `benchmarks/phase6_headtohead.json` has 3 cells; `benchmarks/phase6_headtohead.md` is a 1-row-per-backend markdown table.

If any cell errors (e.g. vLLM env broken), record but do NOT fix in this task — Task 7's full run will document the gap.

- [ ] **Step 2: Inspect the smoke output**

Run:
```
cat benchmarks/phase6_headtohead.json && echo "---" && cat benchmarks/phase6_headtohead.md
```
Expected: JSON with 3 cells; markdown table with three rows (flashquest, llama.cpp, vLLM).

- [ ] **Step 3: Remove the smoke output (keep the cell directory clean for the real run)**

Run:
```
rm -rf benchmarks/phase6_headtohead.json benchmarks/phase6_headtohead.md benchmarks/phase6_cells
```
Expected: smoke artifacts gone; the next real run starts from a clean directory.

- [ ] **Step 4: No commit (smoke output discarded)**

Skip — nothing to commit.

---

## Task 7: Full run (3 backends × 3 contexts) in background

**Files:** none (run-only; results inform Task 8 docs)

- [ ] **Step 1: Launch the full run in background**

Run via Bash `run_in_background`:
```
source .venv/bin/activate && python -u scripts/phase6_run_headtohead.py 2>&1 | tee /tmp/phase6_headtohead.log
```

Expected: returns the background task ID immediately. Wall ≈ 1-2 hr.

- [ ] **Step 2: Monitor via Monitor tool**

Arm the Monitor on `/tmp/phase6_headtohead.log` with a tight regex covering progress + failure signals:

```
tail -F /tmp/phase6_headtohead.log 2>/dev/null | grep --line-buffered -E "\[run\]|\[skip\]|decode_tok_s|oom|Error|Traceback|Wrote|rc=|timeout"
```

Each `[run]` line marks a new cell starting; `decode_tok_s` lines mark per-cell completion; `Wrote` marks final results.

- [ ] **Step 3: When the monitor reports `Wrote benchmarks/phase6_headtohead.json`, inspect**

Run:
```
cat benchmarks/phase6_headtohead.json | python -m json.tool | head -80
echo "---"
cat benchmarks/phase6_headtohead.md
```
Expected: 9 cells, each with `decode_tok_s` (when fits) or `oom: true` / `error` (when not). Markdown table renders.

- [ ] **Step 4: Sanity-check flashquest @ 32 k**

The cell for `flashquest @ 32768` should have `decode_tok_s ≈ 5.14 ± 1`. If it deviates significantly (say <3 or >7), there is a regression — diagnose before continuing.

Run:
```
python -c "
import json
cells = json.loads(open('benchmarks/phase6_headtohead.json').read())['results']
fq32 = next(c for c in cells if c['backend']=='flashquest' and c['ctx_len']==32768)
print(fq32)
assert 3.0 <= fq32['decode_tok_s'] <= 7.5, f'regression: {fq32}'
print('OK')
"
```

- [ ] **Step 5: Commit results**

```bash
git add benchmarks/phase6_headtohead.json benchmarks/phase6_headtohead.md \
        benchmarks/phase6_cells/
git commit -m "phase 6 task 4: head-to-head run — 3 backends × 3 contexts"
```

(If any cells errored unrelated to the SPEC story — e.g. vLLM env broke — capture the per-cell `error` field; the JSON record stands as the honest result.)

---

## Task 8: Update notes + DOC + README + SPEC; tag

**Files:**
- Modify: `docs/PHASES/phase-6-notes.md`
- Modify: `DOC.md`
- Modify: `README.md`
- Modify: `docs/SPEC.md`

- [ ] **Step 1: Read the generated table once**

Run:
```
cat benchmarks/phase6_headtohead.md
```
Note the actual numbers — Task 8's docs use them verbatim.

- [ ] **Step 2: Append task 4 section to `docs/PHASES/phase-6-notes.md`**

Append at the end of file:

```markdown

---

# Phase 6 Task 4 Notes

**Started:** 2026-05-02
**Status:** **complete (tag `phase-6-task-4`)**.
**Spec:** [../superpowers/specs/2026-05-02-phase-6-headtohead-bench-design.md](../superpowers/specs/2026-05-02-phase-6-headtohead-bench-design.md)
**Plan:** [../superpowers/plans/2026-05-02-phase-6-headtohead-bench.md](../superpowers/plans/2026-05-02-phase-6-headtohead-bench.md)

## Summary

Three backends × three contexts (8 k / 32 k / 128 k) head-to-head on
Llama-3.2-3B-Instruct, RTX 3050 Ti Laptop, 4 GB VRAM, WSL2:
- flashquest = AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25
- llama.cpp = Q4_K_M GGUF, FP16 KV, `-ngl 999`
- vLLM 0.7.3 = AWQ-INT4, FP16 KV

Headline metric: decode tok/s at 32 k. Capability axis: max ctx that
decodes (≥1 tok/s, no OOM). Per-cell OOM caught + recorded; 10 min hard
cap. One backend at a time (no parallel torch processes on a 4 GB VRAM
laptop). Results: `benchmarks/phase6_headtohead.json` +
`benchmarks/phase6_headtohead.md`.

## Result

(Copy the rendered `benchmarks/phase6_headtohead.md` table here verbatim.)

## SPEC §11.4 gate

(Edit this section based on the measured numbers.)

If the headline ratio (flashquest 32 k tok/s × max-ctx-fit) ≥ 5× llama.cpp
on the capability axis: gate cleared.

If not: gate annotated with the post-§11 path — Phase 6 task 5 (INT4 KV)
+ kernel-fused criticality + TurboQuant, expected to close the gap.

## v2 follow-ups

- ExLlamaV2 fourth row (SPEC §6 task 8).
- Multi-request throughput.
- Latency percentiles.
- Larger model (Llama-3.1-8B-AWQ if it can be made to fit via more aggressive
  KV compression).
```

Replace the placeholder paragraphs with the measured numbers + verdict.

- [ ] **Step 3: Append task 4 row to `DOC.md`**

Edit `DOC.md`. Find:
```markdown
- Phase 6 task 4+ — head-to-head benchmark table, INT4 KV, EAGLE-2, Marlin, ExLlamaV2, TurboQuant (post-§11 follow-up).
```
Replace with:
```markdown
- **Phase 6 task 4 — head-to-head benchmark table** ✅ **complete (tag `phase-6-task-4`)**. `scripts/bench_flashquest.py` + parametrized `bench_llamacpp.sh` + parametrized `bench_vllm.py` + `scripts/phase6_run_headtohead.py` orchestrator. Three backends × three contexts (8 k / 32 k / 128 k) on Llama-3.2-3B-Instruct, RTX 3050 Ti Laptop, 4 GB VRAM, WSL2. Headline metric: decode tok/s at 32 k. Capability axis: max ctx that decodes. Each backend's native-strongest quant (flashquest=AWQ-INT4 + INT8 paged KV + Quest, llama.cpp=Q4_K_M, vLLM=AWQ-INT4). Results: `benchmarks/phase6_headtohead.{json,md}`. See `docs/PHASES/phase-6-notes.md` for the table + verdict.
- Phase 6 task 5+ — INT4 KV, EAGLE-2, Marlin, ExLlamaV2, TurboQuant (post-§11 follow-up).
```

- [ ] **Step 4: Embed table in `README.md`**

Edit `README.md`. Find the "## Phase 6 task 3 — `flashquest chat` CLI" section. Insert *after* it (before "## Non-goals"):

```markdown
## Phase 6 task 4 — head-to-head benchmark

Llama-3.2-3B-Instruct, RTX 3050 Ti Laptop (sm_86, 4 GB VRAM, WSL2 + CUDA 12.5).
128 decode tokens, single request, each backend's native-strongest quant.

(Paste the contents of benchmarks/phase6_headtohead.md here.)

(Paragraph: capability ratio interpretation — what the table actually shows.
If the SPEC §11.4 ≥5× gate cleared, say so. If not, name what closes the gap
— Phase 6 task 5 INT4 KV + kernel-fused criticality + TurboQuant.)

Re-run via `python scripts/phase6_run_headtohead.py`. Per-cell JSONs in
`benchmarks/phase6_cells/`.
```

Replace the placeholder paragraphs with the measured numbers + verdict.

- [ ] **Step 5: Tick task 4 in `docs/SPEC.md`**

Edit `docs/SPEC.md`. Find:
```markdown
4. **Head-to-head benchmark table** vs `llama.cpp -ngl 999` and vLLM at the same context, on the same machine. README §11 acceptance criterion.
```
Replace with:
```markdown
4. **Head-to-head benchmark table.** ✅ **DONE (2026-05-02, tag `phase-6-task-4`).** `scripts/bench_flashquest.py` + parametrized `bench_llamacpp.sh` + parametrized `bench_vllm.py` + `scripts/phase6_run_headtohead.py` orchestrator. Three backends × three contexts (8 k / 32 k / 128 k). Headline = decode tok/s at 32 k; capability axis = max ctx that decodes. Each backend's native-strongest quant. Results: `benchmarks/phase6_headtohead.{json,md}`. (Edit this line with the gate verdict — cleared, or annotated with the post-§11 path if the ≥5× gate didn't clear.) See `docs/PHASES/phase-6-notes.md`.
```

Also annotate `docs/SPEC.md` §11 acceptance bullet 4 if the ≥5× gate didn't clear:

Find:
```markdown
4. README has a benchmark table showing ≥5× capability gain over `llama.cpp -ngl 999` on the same hardware.
```
If gate cleared, leave it. If not, replace with:
```markdown
4. README has a benchmark table showing ≥X× capability gain over `llama.cpp -ngl 999` on the same hardware (measured 2026-05-02, see `benchmarks/phase6_headtohead.md`). The full ≥5× gate is gated on Phase 6 task 5+ (INT4 KV + kernel-fused criticality + TurboQuant — see `docs/PHASES/phase-6-notes.md` task 4 verdict for the per-axis breakdown).
```

- [ ] **Step 6: Commit and tag**

```bash
git add docs/PHASES/phase-6-notes.md DOC.md README.md docs/SPEC.md
git commit -m "phase 6 task 4: complete — DOC + README + SPEC + notes with measured table"
git tag phase-6-task-4
git log --oneline phase-6-task-3..HEAD
```

---

## Self-review (post-write)

**1. Spec coverage** — every section of `2026-05-02-phase-6-headtohead-bench-design.md` maps to a task:

- §Architecture / `bench_flashquest.py` → Task 1
- §Architecture / `bench_llamacpp.sh` modify → Task 2
- §Architecture / `bench_vllm.py` modify → Task 3
- §Architecture / orchestrator → Tasks 4 + 5
- §Components / `_parse_llamacpp_log` → Task 5 (with unit tests)
- §Data flow / results JSON → Task 5 (`_build_results`)
- §Data flow / markdown table → Tasks 4 + 5 (`render_markdown`)
- §Edge cases ER1-ER8 → Tasks 1 (ER3 OOM catch), 3 (ER1 OOM catch + ER6 vLLM-broken), 5 (ER2 llama.cpp non-zero rc, ER4 timeout, ER7 dry-run, ER8 skip-existing)
- §Validation gates / per-script smoke → Tasks 1, 2, 3 step "Smoke at ctx=256"
- §Validation gates / dry-run → Task 4
- §Validation gates / full run within 2 hr → Task 7 (10-min hard cap is enforced; 3 × 3 × <10 min ≤ 90 min)
- §Validation gates / flashquest 32 k matches Phase 6 task 1 → Task 7 step 4 (5.14 ± 1 sanity assert)
- §Validation gates / table renders → Task 7 (manual inspection) + Task 8 (README embed)

**2. Placeholder scan** — Task 8 contains `(Copy the rendered ...)`, `(Edit this section based on the measured numbers.)`, and `(Paste the contents of ...)` which are deliberate — the actual numbers come from Task 7's measured run. The pattern is "paste-from-this-file" not "TBD". Acceptable: the table is auto-generated, not authored. Engineer instructions are explicit about which file to copy from.

**3. Type consistency** — every per-cell JSON across Tasks 1, 3, 5 has the same shape: `{backend, quant, ctx_len, decode_tok_s, prefill_tok_s, peak_vram_mib, wall_s, oom, error}`. `render_markdown` and `_build_results` consume that shape. `_parse_llamacpp_log` produces it. `planned_matrix()` returns `list[tuple[str, int]]`; consumers in tests + `main` agree. `CTXS` and `BACKENDS` constants are referenced from one place.

No issues found.
