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
CELL_TIMEOUT_S = 600


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

    sys.stderr.write("error: full run not yet implemented (use --dry-run)\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
