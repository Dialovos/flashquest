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
import os
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
CELL_TIMEOUT_S = 1800

# Canonical backend names used across all per-cell records, keyed by the
# orchestrator's short name. Fallback (timeout / non-zero rc) records reuse these
# so the renderer doesn't show duplicate rows for the same backend.
CANONICAL_BACKEND = {
    "flashquest": "flashquest",
    "llamacpp": "llama.cpp",
    "vllm": "vLLM 0.7.3",
}
CANONICAL_QUANT = {
    "flashquest": "AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25",
    "llamacpp": "Q4_K_M, FP16 KV",
    "vllm": "AWQ-INT4, FP16 KV",
}


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
    backend_order: list[str] = []
    for cell in cells:
        b = cell["backend"]
        if b not in by_backend:
            backend_order.append(b)
            by_backend[b] = {}
        by_backend[b][cell["ctx_len"]] = cell

    lines = [
        "| Backend | Quant | 8 k decode tok/s | 32 k decode tok/s | 128 k fits? | Peak VRAM @ max fit |",
        "|---|---|---|---|---|---|",
    ]
    for backend in backend_order:
        by_ctx = by_backend[backend]
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
        if "out of memory" in msg or ("cuda" in msg and "fail" in msg):
            record["oom"] = True
        record["error"] = f"rc={rc}: {(stderr or '')[-400:]}"
        return record

    text = log_path.read_text()
    pp_match = re.search(rf"pp{ctx}\s*\|\s*([0-9.]+)\s*", text)
    tg_match = re.search(r"tg128\s*\|\s*([0-9.]+)\s*", text)
    if pp_match:
        record["prefill_tok_s"] = float(pp_match.group(1))
    if tg_match:
        record["decode_tok_s"] = float(tg_match.group(1))
    smi_match = re.search(r"(\d+)\s*MiB,\s*\d+\s*MiB", text)
    if smi_match:
        record["peak_vram_mib"] = int(smi_match.group(1))
    if record["decode_tok_s"] is None:
        record["error"] = "could not parse llama-bench output"
    return record


def run_one(backend: str, ctx: int, out_path: Path) -> dict:
    """Drive a per-backend bench script, parse its JSON / log output."""
    env_overrides: dict[str, str] = {}
    log_path: Path | None = None
    if backend == "flashquest":
        kv_bits = os.environ.get("KV_BITS", "4")
        cmd = [
            "nice", "-n", "19",
            sys.executable, str(REPO_ROOT / "scripts" / "bench_flashquest.py"),
            "--ctx-len", str(ctx),
            "--n-decode", "32",
            "--kv-bits", kv_bits,
            "--out", str(out_path),
        ]
    elif backend == "llamacpp":
        log_path = out_path.with_suffix(".llamacpp.txt")
        cmd = ["bash", str(REPO_ROOT / "scripts" / "bench_llamacpp.sh")]
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
        env = {**os.environ, **env_overrides} if env_overrides else None
        res = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=CELL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        record = {
            "backend": CANONICAL_BACKEND[backend],
            "quant": CANONICAL_QUANT[backend],
            "ctx_len": ctx,
            "decode_tok_s": None, "peak_vram_mib": None,
            "wall_s": time.perf_counter() - t0,
            "oom": False, "error": f"timeout (>{CELL_TIMEOUT_S}s)",
        }
        out_path.write_text(json.dumps(record, indent=2))
        return record

    if backend == "llamacpp":
        record = _parse_llamacpp_log(log_path, ctx, res.returncode, res.stderr)
        record["wall_s"] = time.perf_counter() - t0
        out_path.write_text(json.dumps(record, indent=2))
        return record

    if out_path.exists():
        return json.loads(out_path.read_text())

    return {
        "backend": CANONICAL_BACKEND[backend],
        "quant": CANONICAL_QUANT[backend],
        "ctx_len": ctx,
        "decode_tok_s": None, "peak_vram_mib": None,
        "wall_s": time.perf_counter() - t0,
        "oom": False,
        "error": f"subprocess returned {res.returncode}: {res.stderr[-500:]}",
    }


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


if __name__ == "__main__":
    sys.exit(main())
