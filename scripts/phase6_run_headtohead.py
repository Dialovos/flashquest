"""Run context-matched validation cells; preserve the historical phase6 results."""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from statistics import median

from bench_common import (
    DEFAULT_MODEL,
    SCHEMA_VERSION,
    content_hash,
    is_oom,
    new_record,
    provenance,
    run_config,
    write_benchmark_record,
    write_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ["flashquest", "llamacpp", "vllm"]
CTXS = [8192, 32768, 131072]
CELL_TIMEOUT_S = 1800


def planned_matrix(backends=BACKENDS, contexts=CTXS) -> list[tuple[str, int]]:
    return [(b, c) for b in backends for c in contexts]


def cell_config(backend: str, ctx: int, args) -> dict:
    settings = {
        "flashquest": {"model": args.model, "kv_bits": args.kv_bits,
                       "retention": args.retention, "page_size": 64,
                       "num_sinks": 4, "window_pages": 2},
        "vllm": {"model": args.model, "kv_cache_dtype": args.vllm_kv_cache_dtype},
        "llamacpp": {"model": Path(os.environ.get("MODEL", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")).name,
                     "kv_k": args.llamacpp_kv, "kv_v": args.llamacpp_kv,
                     "n_gpu_layers": os.environ.get("NGL", "999"),
                     "threads": os.environ.get("THREADS", "6")},
    }
    return run_config(ctx, args.n_decode, args.reps, args.seed, **settings[backend])


def cell_record(backend: str, config: dict) -> dict:
    if backend == "flashquest":
        mode = "TurboQuant K3-V3" if config["kv_bits"] == 3 else f"INT{config['kv_bits']}"
        quant = f"AWQ-INT4 + {mode} KV, retention={config['retention']:g}"
        name = "flashquest"
    elif backend == "llamacpp":
        quant = f"Q4_K_M, {config['kv_k']}/{config['kv_v']} KV"
        name = "llama.cpp"
    else:
        quant = f"AWQ-INT4, {config['kv_cache_dtype']} KV"
        name = "vLLM"
    return new_record(name, quant, config)


def render_markdown(cells: list[dict]) -> str:
    def fmt(value):
        return f"{value:.2f}" if value is not None else "—"

    lines = [
        "| Backend | Input tokens | Quant | Decode tok/s | Prefill tok/s | Peak allocated MiB | Status |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for cell in cells:
        status = "OOM" if cell.get("oom") else "error" if cell.get("error") else "measured"
        if cell.get("decode_tok_s") is None and status == "measured":
            status = "unmeasured"
        lines.append(f"| {cell['backend']} | {cell['ctx_len']} | {cell['quant']} | "
                     f"{fmt(cell.get('decode_tok_s'))} | {fmt(cell.get('prefill_tok_s'))} | "
                     f"{fmt(cell.get('peak_allocated_mib'))} | {status} |")
    lines.append("\nAllocator memory is not physical GPU residency; — means unmeasured.")
    return "\n".join(lines)


def _parse_llamacpp_log(log_path: Path, config: dict, rc: int, stderr: str) -> dict:
    """Accept only JSON tests with the exact input depth and cache precision."""
    record = cell_record("llamacpp", config)
    if rc != 0:
        record.update(oom=is_oom(stderr), error=f"rc={rc}: {stderr[-500:]}")
        return record
    try:
        decode = json.loads(log_path.read_text(encoding="utf-8"))
        prefill = json.loads(log_path.with_suffix(".prefill.json").read_text(encoding="utf-8"))

        def matching(rows, n_prompt, n_gen, depth):
            if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
                raise ValueError("expected a JSON array of llama-bench tests")
            matches = [r for r in rows if
                       r.get("n_prompt") == n_prompt and r.get("n_gen") == n_gen and
                       r.get("n_depth") == depth and r.get("type_k") == config["kv_k"] and
                       r.get("type_v") == config["kv_v"]]
            if len(matches) != 1:
                raise ValueError("expected one llama-bench JSON test with matching context and KV types")
            row = matches[0]
            samples = row.get("samples_ts", [])
            if len(samples) != config["reps"] or any(not math.isfinite(v) or v <= 0 for v in samples):
                raise ValueError("missing or invalid llama-bench timing repetitions")
            return row, samples

        dec, dec_samples = matching(decode, 0, config["n_decode"] - 1, config["ctx_len"])
        _, pf_samples = matching(prefill, config["ctx_len"], 0, 0)
        record["decode_tok_s"] = median(dec_samples)
        record["prefill_tok_s"] = median(pf_samples)
        record["samples"] = {"decode_tok_s": dec_samples, "prefill_tok_s": pf_samples}
        record["versions"] = {"llamacpp_commit": dec.get("build_commit"),
                              "llamacpp_build": dec.get("build_number")}
        record["prompt_kind"] = "llama-bench built-in tokens (not shared input IDs)"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        record["error"] = f"invalid llama-bench result: {exc}"
    return record


def _command(backend: str, ctx: int, output: Path, args) -> tuple[list[str], dict]:
    if backend == "llamacpp":
        raw = output.with_suffix(".llamacpp.json")
        env = {"CTX": str(ctx), "OUT": str(raw), "N_DECODE": str(args.n_decode),
               "REPS": str(args.reps), "KV_K": args.llamacpp_kv, "KV_V": args.llamacpp_kv}
        return ["bash", str(REPO_ROOT / "scripts" / "bench_llamacpp.sh")], env
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / f"bench_{backend}.py"),
           "--model", args.model, "--ctx-len", str(ctx), "--n-decode", str(args.n_decode),
           "--reps", str(args.reps), "--seed", str(args.seed), "--out", str(output)]
    if backend == "flashquest":
        cmd += ["--kv-bits", str(args.kv_bits), "--retention", str(args.retention)]
    else:
        cmd += ["--kv-cache-dtype", args.vllm_kv_cache_dtype]
    return cmd, {}


def _run_command(cmd: list[str], env: dict, timeout: int):
    # A vLLM cell can spawn GPU workers. Stop the entire process group on timeout.
    with subprocess.Popen(cmd, env={**os.environ, **env}, cwd=REPO_ROOT,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
            raise
        return process.returncode, stdout, stderr


def run_one(backend: str, ctx: int, out_path: Path, args) -> dict:
    config = cell_config(backend, ctx, args)
    record = cell_record(backend, config)
    record["provenance"] = provenance()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = REPO_ROOT / "artifacts" / "benchmarks" / uuid.uuid4().hex
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # A fresh child output prevents a failed rerun from accepting old successes.
    with tempfile.TemporaryDirectory(prefix=".cell-", dir=raw_dir) as directory:
        output = Path(directory) / "result.json"
        cmd, env = _command(backend, ctx, output, args)
        try:
            rc, stdout, stderr = _run_command(cmd, env, args.timeout)
            (raw_dir / "cell.log").write_text(stdout + "\n" + stderr, encoding="utf-8")
            if backend == "llamacpp":
                raw = output.with_suffix(".llamacpp.json")
                record = _parse_llamacpp_log(raw, config, rc, stderr)
                for source in (raw, raw.with_suffix(".prefill.json")):
                    if source.exists():
                        target = raw_dir / source.name
                        target.write_bytes(source.read_bytes())
            elif output.exists():
                child = json.loads(output.read_text(encoding="utf-8"))
                if (not isinstance(child, dict) or child.get("schema_version") != SCHEMA_VERSION
                        or child.get("config") != config):
                    raise ValueError("child result configuration does not match this run")
                record = child
                if rc != 0 and not record.get("error"):
                    record["error"] = f"subprocess returned {rc}: {stderr[-500:]}"
            else:
                record.update(oom=is_oom(stderr),
                              error=f"subprocess returned {rc} without a result: {stderr[-500:]}")
        except subprocess.TimeoutExpired:
            record["error"] = f"timeout (>{args.timeout}s)"
            (raw_dir / "cell.log").write_text(record["error"] + "\n", encoding="utf-8")
        except (OSError, ValueError) as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = time.perf_counter() - t0
    record.setdefault("provenance", provenance())
    return write_benchmark_record(out_path, record, raw_dir=raw_dir)


def reusable_cell(path: Path, config: dict, identity: dict | None = None) -> dict | None:
    # The legacy backend adapters cannot resolve complete identity before launch.
    # Until WP3 adds that probe, rerunning is safer than accepting stale evidence.
    if identity is None:
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        samples = record.get("samples") if isinstance(record, dict) else None
        if isinstance(samples, list):
            complete = len(samples) == config["reps"] and all(
                s.get("input_tokens") == config["ctx_len"] and
                s.get("output_tokens") == config["n_decode"] and
                isinstance(s.get("decode_tok_s"), (int, float)) and
                math.isfinite(s["decode_tok_s"]) and s["decode_tok_s"] > 0 for s in samples)
        elif isinstance(samples, dict):
            complete = all(len(samples.get(key, [])) == config["reps"] and
                           all(isinstance(v, (int, float)) and math.isfinite(v) and v > 0
                               for v in samples[key]) for key in ("decode_tok_s", "prefill_tok_s"))
        else:
            complete = False
        if (isinstance(record, dict) and record.get("schema_version") == SCHEMA_VERSION
                and record.get("identity") == identity
                and record.get("run_identity") == content_hash(identity) and complete
                and record.get("config") == config
                and not record.get("error") and not record.get("oom")
                and math.isfinite(record.get("decode_tok_s") or 0)
                and (record.get("decode_tok_s") or 0) > 0):
            return record
    except (OSError, ValueError, TypeError, AttributeError, KeyError):
        pass
    return None


def _host() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return {"gpus": result.stdout.strip().splitlines()}
    except (OSError, subprocess.SubprocessError):
        return {"gpus": None}


def parse_args(argv: Sequence[str] | None = None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-existing", action="store_true",
                   help="requires resolved run identity; legacy adapters currently rerun")
    p.add_argument("--backends", nargs="+", choices=BACKENDS, default=BACKENDS)
    p.add_argument("--contexts", nargs="+", type=int, default=CTXS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-decode", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retention", type=float, default=0.20)
    p.add_argument("--kv-bits", type=int, choices=[3, 4, 8], default=4)
    p.add_argument("--llamacpp-kv", choices=["q4_0", "q8_0", "f16"], default="q4_0")
    p.add_argument("--vllm-kv-cache-dtype", choices=["auto", "fp8"], default="fp8")
    p.add_argument("--output-dir", type=Path, default=REPO_ROOT / "benchmarks" / "validation")
    p.add_argument("--timeout", type=int, default=CELL_TIMEOUT_S)
    args = p.parse_args(argv)
    if (min(args.contexts) < 1 or args.n_decode < 2 or args.reps < 1 or args.timeout < 1
            or not 0 < args.retention <= 1):
        p.error("positive contexts/reps/timeout, n-decode >= 2, and 0 < retention <= 1 required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    matrix = planned_matrix(args.backends, args.contexts)
    if args.dry_run:
        print(f"{len(matrix)} cells planned; {args.reps} repetitions, {args.n_decode - 1} decode steps:")
        for backend, ctx in matrix:
            record = cell_record(backend, cell_config(backend, ctx, args))
            print(f"  - {backend} @ {ctx}: {record['quant']}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for backend, ctx in matrix:
        path = args.output_dir / f"{backend}_{ctx}.json"
        cached = reusable_cell(path, cell_config(backend, ctx, args)) if args.skip_existing else None
        print(f"[{'skip' if cached else 'run'}] {backend} @ {ctx}", flush=True)
        cells.append(cached or run_one(backend, ctx, path, args))
        # Save partial progress after each cell for interrupted matrices.
        write_record(args.output_dir / "results.json", {
            "schema_version": SCHEMA_VERSION, "host": _host(),
            "date": time.strftime("%Y-%m-%d"), "results": cells,
        })
        (args.output_dir / "results.md").write_text(render_markdown(cells) + "\n", encoding="utf-8")
    print(render_markdown(cells))
    return int(any(c.get("error") or c.get("decode_tok_s") is None for c in cells))


if __name__ == "__main__":
    raise SystemExit(main())
