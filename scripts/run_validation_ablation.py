"""Balanced sparse/all-pages blocks with strict identities and memory observation."""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from bench_common import (
    DEFAULT_MODEL,
    REPO_ROOT,
    SCHEMA_VERSION,
    canonical_identity,
    content_hash,
    export_identity,
    file_hash,
    make_identity,
    model_identity,
    provenance,
    validate_export,
    write_record,
)
from bench_flashquest import BENCH_PROTOCOL
from gpu_memory import observe_command, resolve_device
from phase6_run_ruler_4k_int4 import screen_verdict


def schedule(contexts: list[int], seeds: list[int], retention: float) -> list[dict]:
    cells = []
    for context_index, context in enumerate(contexts):
        for seed_index, seed in enumerate(seeds):
            arms = [("sparse", retention), ("all-pages", 1.0)]
            first_arm_offset = 1 if context == 32768 else context_index
            if (first_arm_offset + seed_index) % 2:
                arms.reverse()
            for order, (arm, ratio) in enumerate(arms):
                cells.append({"cell_id": f"c{context}-s{seed}-{arm}", "ctx_len": context,
                              "seed": seed, "arm": arm, "retention": ratio, "block_order": order})
    return cells


def quality_prerequisites(paths: list[Path], contexts: list[int], retention: float,
                          model: dict) -> list[dict]:
    evidence = []
    covered = set()
    for path in paths:
        record = json.loads(path.read_text())
        identity = canonical_identity(record["identity"])
        if content_hash(identity) != record["run_identity"] or record["status"] != "complete":
            raise ValueError("quality record is incomplete or has an invalid identity")
        cfg = identity["config"]
        args = SimpleNamespace(tasks=cfg["tasks"], seeds=cfg["seeds"],
                               n_samples=cfg["n_samples"], retentions=cfg["retentions"])
        expected_cells = {(task, seed, arm) for task in args.tasks for seed in args.seeds
                          for arm in ["dense", *[f"int4-r{r:g}" for r in args.retentions]]}
        if ({(c["task"], c["seed"], c["arm"]) for c in record["cells"]} != expected_cells or
                len(record["cells"]) != len(expected_cells) or
                any(c["total"] != args.n_samples or len(c["samples"]) != args.n_samples or
                    c["hits"] != sum(s["hit"] for s in c["samples"]) for c in record["cells"]) or
                screen_verdict(record["cells"], args) != record["screen"]):
            raise ValueError("quality cells or screen are inconsistent")
        if (identity["model"] != model or cfg["ctx_len"] in covered or
                cfg["tasks"] != ["single", "multikey", "multivalue"] or cfg["n_samples"] < 20 or
                retention not in cfg["retentions"] or record["screen"]["status"] != "pass" or
                any(not task["arms"][f"int4-r{retention:g}"]["screen_pass"]
                    for task in record["screen"]["tasks"].values())):
            raise ValueError("quality evidence does not cover this model/retention/context")
        covered.add(cfg["ctx_len"])
        evidence.append({"run_identity": record["run_identity"], "sha256": file_hash(path),
                         "ctx_len": cfg["ctx_len"], "path": path.relative_to(REPO_ROOT).as_posix()})
    if covered != set(contexts):
        raise ValueError("quality evidence must cover every scheduled context")
    return sorted(evidence, key=lambda row: row["ctx_len"])


def checked_cell(path: Path, expected: dict, run: dict, reps: int, n_decode: int) -> dict:
    record = json.loads(path.read_text())
    identity = canonical_identity(record["identity"])
    if content_hash(identity) != record["run_identity"] or record["config"] != identity["config"]:
        raise ValueError("benchmark identity/config mismatch")
    for section in ("model", "source", "environment"):
        if identity[section] != run["identity"][section]:
            raise ValueError("benchmark differs from the frozen schedule identity")
    cfg = record["config"]
    for key in ("ctx_len", "seed", "retention"):
        if cfg[key] != expected[key]:
            raise ValueError("benchmark differs from its scheduled cell")
    if (record.get("error") or record.get("oom") or record["schema_version"] != SCHEMA_VERSION or
            identity["protocol_sha256"] != content_hash(BENCH_PROTOCOL) or
            cfg["kv_bits"] != 4 or cfg["page_size"] != 64 or cfg["num_sinks"] != 4 or
            cfg["window_pages"] != 2 or cfg["reps"] != reps or cfg["n_decode"] != n_decode or
            cfg["generation"] != "manual-greedy-fixed-output-count-v1" or
            cfg["warmup_steps"] != max(n_decode - 1, 65) or
            cfg["cache_capacity"] != cfg["ctx_len"] + cfg["warmup_steps"] + 1 or
            len(record["samples"]) != reps):
        raise ValueError("benchmark is incomplete or uses another measurement protocol")
    for sample in record["samples"]:
        if sample["input_tokens"] != cfg["ctx_len"] or sample["output_tokens"] != n_decode:
            raise ValueError("benchmark token counts differ")
        for key in ("prefill_s", "decode_s", "request_s", "decode_tok_s", "prefill_tok_s", "end_to_end_tok_s"):
            if not isinstance(sample[key], (int, float)) or not math.isfinite(sample[key]) or sample[key] <= 0:
                raise ValueError("invalid benchmark timing")
        derived = {"decode_tok_s": (n_decode - 1) / sample["decode_s"],
                   "prefill_tok_s": cfg["ctx_len"] / sample["prefill_s"],
                   "end_to_end_tok_s": n_decode / sample["request_s"],
                   "request_s": sample["prefill_s"] + sample["decode_s"]}
        if any(not math.isclose(sample[key], value, rel_tol=1e-8) for key, value in derived.items()):
            raise ValueError("benchmark rates differ from their timed counts")
    if any(not math.isclose(record[key], median(s[key] for s in record["samples"]), rel_tol=1e-8)
           for key in ("decode_tok_s", "prefill_tok_s", "end_to_end_tok_s")):
        raise ValueError("benchmark medians differ from samples")
    memory_protocol = cfg["memory_protocol"]
    schedule_config = run["identity"]["config"]
    if memory_protocol != {"version": 1, "method": "nvidia-smi-query-linux-proc-monotonic",
                           "interval_s": schedule_config["interval_ms"] / 1000,
                           "physical_device_index": schedule_config["physical_device_index"]}:
        raise ValueError("benchmark memory protocol differs")
    validate_export(record)
    return record


def export_schedule(run: dict, result: dict) -> dict:
    exported = {**result, **run, "schema_version": SCHEMA_VERSION}
    exported["identity"] = export_identity(run["identity"])
    validate_export(exported)
    return exported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", nargs="+", type=int, default=[8192, 32768])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--retention", type=float, default=.20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--quality", nargs="+", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--interval-ms", type=float, default=50)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--n-decode", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--attempt", type=int, default=0, help="fresh whole-block attempt; preserve earlier outputs")
    parser.add_argument("--retry-of", type=Path, help="prior schedule.json when retrying a failed block")
    args = parser.parse_args(argv)
    if (min(args.contexts) < 1 or len(set(args.contexts)) != len(args.contexts) or
            min(args.seeds) < 0 or len(set(args.seeds)) != len(args.seeds) or
            len(args.seeds) % 2 or not 0 < args.retention < 1 or args.gpu_index < 0 or
            not 10 <= args.interval_ms <= 1000 or not 0 < args.timeout <= 7200 or
            args.reps < 3 or args.n_decode < 2 or args.attempt < 0):
        parser.error("unique positive contexts, an even number of unique nonnegative seeds, valid retention/budgets required")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    model = model_identity(args.model, args.revision)
    device = resolve_device(args.gpu_index)
    quality = quality_prerequisites(args.quality, args.contexts, args.retention, model)
    cells = schedule(args.contexts, args.seeds, args.retention)
    protocol = {"version": 1, "kind": "balanced-int4-ablation", "cells": cells,
                "quality": quality, "bench_protocol": BENCH_PROTOCOL,
                "order_unit": "whole-arm-warmup-and-repetitions", "minimum_median_ratio": 1.10,
                "minimum_each_seed_ratio": 1.0, "claims": "pilot ablation; no competitor or capacity claim"}
    config = {"contexts": args.contexts, "seeds": args.seeds, "retention": args.retention,
              "reps": args.reps, "n_decode": args.n_decode, "interval_ms": args.interval_ms,
              "timeout_s": args.timeout, "physical_device_index": args.gpu_index,
              "attempt": args.attempt, "retry_of": None,
              "physical_device": {k: v for k, v in device.items() if k != "uuid"}}
    if args.retry_of:
        prior = json.loads(args.retry_of.read_text())
        prior_identity = canonical_identity(prior["identity"])
        prior_config = prior_identity["config"]
        if (content_hash(prior_identity) != prior["run_identity"] or
                args.attempt <= prior_config["attempt"] or
                {k: v for k, v in config.items() if k not in {"attempt", "retry_of"}} !=
                {k: v for k, v in prior_config.items() if k not in {"attempt", "retry_of"}}):
            raise ValueError("retry must preserve the block settings and increase its attempt number")
        config["retry_of"] = prior["run_identity"]
    run = make_identity(config, model, protocol, provenance())
    directory = REPO_ROOT / "benchmarks" / "validation" / "ablation" / run["run_identity"]
    out = directory / "schedule.json"
    result = {"protocol": protocol, "status": "incomplete", "cells": []}
    if out.exists():
        previous = json.loads(out.read_text())
        if not args.resume or canonical_identity(previous["identity"]) != run["identity"]:
            raise ValueError("existing schedule requires an identical --resume")
        result["cells"] = previous["cells"]
        if [entry["cell"] for entry in result["cells"]] != cells[:len(result["cells"])]:
            raise ValueError("resume cells do not preserve the frozen order")
        for entry in result["cells"]:
            if entry["status"] != "complete":
                raise ValueError("resume cannot mix a failed attempt into a balanced block; start a fresh block")
            path = REPO_ROOT / entry["result"]["path"]
            if file_hash(path) != entry["result"]["sha256"]:
                raise ValueError("resume cell changed")
            checked_cell(path, entry["cell"], run, args.reps, args.n_decode)
    write_record(REPO_ROOT / "benchmarks" / "validation" / "protocols" /
                 f"{content_hash(protocol)}.json", protocol)
    write_record(out, export_schedule(run, result))
    for cell in cells[len(result["cells"]):]:
        raw_dir = REPO_ROOT / "artifacts" / "ablation" / run["run_identity"] / uuid.uuid4().hex
        output = directory / "cells" / f"{cell['cell_id']}.json"
        if output.exists():
            raise ValueError("orphaned cell result; preserve it and start a fresh block")
        command = [sys.executable, str(REPO_ROOT / "scripts" / "bench_flashquest.py"),
                   "--model", args.model, "--revision", model["revision"], "--ctx-len", str(cell["ctx_len"]),
                   "--seed", str(cell["seed"]), "--retention", str(cell["retention"]),
                   "--reps", str(args.reps), "--n-decode", str(args.n_decode), "--out", str(output)]
        observed = observe_command(command, raw_dir, device, timeout_s=args.timeout, interval_s=args.interval_ms / 1000)
        entry = {"cell": cell, **observed}
        memory = entry["memory"]
        memory["raw_series"]["path"] = (raw_dir / "memory-series.json").relative_to(REPO_ROOT).as_posix()
        if observed["returncode"] != 0 and entry["status"] == "complete":
            entry["status"] = "backend-error"
        if output.exists():
            entry["result"] = {"path": output.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(output)}
            if entry["status"] == "complete":
                try:
                    checked_cell(output, cell, run, args.reps, args.n_decode)
                except (KeyError, TypeError, ValueError):
                    entry["status"] = "invalid-result"
        else:
            entry["status"] = "missing-result"
        if (memory["sampler_failed"] or memory["concurrent_compute_workload"] or
                memory["ownership_check_dropouts"] or not memory["device_sample_count"]):
            entry["status"] = "invalid-telemetry"
        result["cells"].append(entry)
        result["status"] = "complete" if len(result["cells"]) == len(cells) else "incomplete"
        if entry["status"] != "complete":
            result["status"] = "execution-error"
        write_record(out, export_schedule(run, result))
        print(f"{cell['cell_id']}: {entry['status']}", flush=True)
        if result["status"] == "execution-error":
            return 1
    print(f"Schedule: {out.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
