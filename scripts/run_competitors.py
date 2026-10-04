"""Observe pinned competitor cells sequentially; preserve failures and native boundaries."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import uuid
from pathlib import Path
from statistics import median

from bench_common import (
    REPO_ROOT,
    canonical_identity,
    content_hash,
    export_identity,
    file_hash,
    make_identity,
    provenance,
    synthetic_token_ids,
    validate_export,
    write_record,
)
from bench_competitor import PROTOCOL, backend_identity, verified_awq_model, verify_runtime
from competitor_backend import add_backend_arguments
from competitor_niah import load_manifest
from gpu_memory import observe_command, resolve_device, summarize


def tokenizer_size(snapshot):
    from transformers import AutoTokenizer
    return len(AutoTokenizer.from_pretrained(snapshot, local_files_only=True))


def entry_status(entry):
    if entry["observation_status"] != "complete":
        return entry["observation_status"]
    if "backend_status" not in entry:
        return "missing-result"
    if entry["returncode"] != 0 and entry["backend_status"] == "complete":
        return "backend-error-after-result"
    return entry["backend_status"]


def schedule_status(entries, cells):
    if len(entries) != len(cells):
        return "incomplete"
    return "complete" if all(entry_status(e) == "complete" and e["telemetry_status"] == "complete"
                              for e in entries) else "complete-with-failures"


def checked_result(record, spec, schedule, args, quality):
    """Validate child semantics as well as bytes, including during resume."""
    identity = canonical_identity(record["identity"])
    frozen = canonical_identity(schedule["identity"])
    cfg = identity["config"]
    if (record["run_identity"] != content_hash(identity) or
            identity["protocol_sha256"] != content_hash(record["protocol"]) or
            cfg["backend_identity"] != frozen["config"]["backend_fingerprints"][spec["backend"]] or
            any(identity[k] != frozen[k] for k in ("source", "environment"))):
        raise ValueError("child identity/backend/source/environment differs from schedule")
    expected_model = frozen["model"]
    if spec["backend"] == "llamacpp":
        hashes = {Path(args.gguf).name: frozen["config"]["gguf_sha256"]}
        expected_model = {"model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                          "revision": "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
                          "files": hashes, "content_sha256": content_hash(hashes)}
    if identity["model"] != expected_model:
        raise ValueError("child model differs from frozen artifacts")
    common = {"backend": spec["backend"], "kv_dtype": spec["kv_dtype"], "ctx_len": spec["ctx_len"],
              "gpu_utilization": args.gpu_utilization if spec["backend"] == "vllm" else None}
    if args.mode == "performance":
        expected = {**common, "n_decode": args.n_decode, "reps": args.reps, "seeds": [spec["seed"]],
                    "capacity": spec["ctx_len"] + args.n_decode, "max_batch_tokens": 2048,
                    "llama_ubatch": 512 if spec["backend"] == "llamacpp" else None,
                    "generation": PROTOCOL["generation"], "ready_timeout_s": args.ready_timeout,
                    "request_timeout_s": args.ready_timeout,
                    "engine_settings": {"offload_gb": 0, "prefix_cache": False, "batch": 1,
                                        "temperature": 0, "ignore_eos": True, "speculation": False,
                                        "vllm_worker_observation": "vllm0.30-after-first-worker-execution",
                                        "llama_flash_attention": "on", "llama_fit": "off"}}
        expected_protocol = PROTOCOL
        examples = None
    else:
        reference = quality[str(spec["ctx_len"])]
        path = REPO_ROOT / reference["path"]
        if file_hash(path) != reference["sha256"]:
            raise ValueError("frozen quality evidence changed")
        original, original_identity, manifest, examples = load_manifest(path)
        if original["run_identity"] != reference["run_identity"] or original_identity["model"] != frozen["model"]:
            raise ValueError("frozen quality model/run differs")
        expected = {**common, "capacity": original_identity["config"]["cache_capacity"],
                    "max_new_tokens": 128, "examples": len(examples), "limit_per_task": None}
        expected_protocol = {"version": 1, "quality_run": original["run_identity"], "manifest": manifest,
                             "max_new_tokens": 128, "scorer": "all expected substrings in decoded answer",
                             "generation": "greedy EOS or output limit; exact input IDs; no wrapping",
                             "timing": "native phases and client times; quality may stop at EOS"}
    if any(cfg.get(k) != value for k, value in expected.items()) or record["protocol"] != expected_protocol:
        raise ValueError("child settings/protocol differ from scheduled cell")
    ref = record["raw_evidence"]
    raw_path = REPO_ROOT / ref["path"]
    subtree = REPO_ROOT / "artifacts" / ("competitors" if args.mode == "performance" else "competitor-quality")
    if not raw_path.resolve().is_relative_to(subtree.resolve()) or file_hash(raw_path) != ref["sha256"]:
        raise ValueError("child raw evidence missing/changed/outside artifact directory")
    raw = json.loads(raw_path.read_text())
    exported = {**raw, "identity": export_identity(raw["identity"]), "raw_evidence": ref,
                "samples": [{k: v for k, v in s.items() if k not in {"generated", "expected"}}
                            for s in raw["samples"]]}
    if exported != record:
        raise ValueError("child raw/public evidence differs")
    validate_export(record)
    if record["status"] != "complete":
        return record
    verify_runtime(record["runtime"], spec["backend"], spec["kv_dtype"], cfg["capacity"])
    samples = record["samples"]
    if len(samples) != (args.reps if examples is None else len(examples)):
        raise ValueError("incomplete competitor sample set")
    if examples is None:
        if {s["repetition"] for s in samples} != set(range(args.reps)) or any(s["seed"] != spec["seed"] for s in samples):
            raise ValueError("competitor sample seed/repetition differs")
        if len({s["input_sha256"] for s in samples}) != 1:
            raise ValueError("competitor repetitions use different input IDs")
        if any(s["input_sha256"] != spec["input_sha256"] for s in samples):
            raise ValueError("competitor input differs from frozen exact synthetic IDs")
    elif [s["example_id"] for s in samples] != [e["example_id"] for e in examples]:
        raise ValueError("competitor quality examples differ from exact manifest")
    for index, sample in enumerate(samples):
        nin = spec["ctx_len"] if examples is None else len(examples[index]["input_ids"])
        nout = sample["output_tokens"]
        if (not isinstance(sample["input_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", sample["input_sha256"]) or
                sample["input_tokens"] != nin or type(nout) is not int or not 1 <= nout <= (args.n_decode if examples is None else 128) or
                (examples is None and (nout != args.n_decode or sample["termination"] != "length")) or
                sample["decode_steps"] != nout - 1):
            raise ValueError("competitor input/output counts differ")
        for field in ("prefill_s", "client_request_s"):
            value = sample[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("invalid competitor timing")
        decode = sample["decode_s"]
        if nout > 1:
            if isinstance(decode, bool) or not isinstance(decode, (int, float)) or not math.isfinite(decode) or decode <= 0:
                raise ValueError("invalid competitor decode timing")
        elif decode is not None or sample["decode_tok_s"] is not None:
            raise ValueError("single-token response has no later decode phase")
        duration = sample["prefill_s"] + (decode or 0)
        derived = {"prefill_tok_s": nin / sample["prefill_s"], "native_request_s": duration,
                   "native_output_tok_s": nout / duration}
        if nout > 1:
            derived["decode_tok_s"] = (nout - 1) / decode
        if any(not math.isclose(sample[k], v, rel_tol=1e-8) for k, v in derived.items()):
            raise ValueError("competitor rates differ from measured counts")
        native = sample["native_numeric_metrics"]
        if spec["backend"] == "vllm":
            boundary = "scheduled-to-first / first-to-last-token"
            native_prefill = native["time_to_first_token_ms"] / 1000
            native_decode = native["generation_time_ms"] / 1000 if nout > 1 else None
            if sample["queue_ms"] != native["queue_time_ms"]:
                raise ValueError("competitor queue metric differs")
            allowed_stops = {"stop", "length"}
        else:
            boundary = "native prompt eval / native generation forwards"
            native_prefill = native["prompt_ms"] / 1000
            native_decode = native["predicted_ms"] / 1000 if nout > 1 else None
            if native["prompt_n"] != nin or native["predicted_n"] != nout:
                raise ValueError("competitor native token counts differ")
            allowed_stops = {"eos", "length"}
        if (sample["timing_boundary"] != boundary or sample["termination"] not in allowed_stops or
                not math.isclose(sample["prefill_s"], native_prefill, rel_tol=1e-8) or
                (nout > 1 and not math.isclose(decode, native_decode, rel_tol=1e-8))):
            raise ValueError("competitor phases differ from backend-native metrics")
        if examples is not None:
            example, private_sample = examples[index], raw["samples"][index]
            if (any(sample[k] != example[k] for k in ("task", "seed", "index", "input_sha256")) or
                    private_sample["expected"] != example["expected"] or
                    type(sample["hit"]) is not bool or
                    sample["hit"] != all(v in private_sample["generated"] for v in example["expected"]) or
                    sample["generated_sha256"] != content_hash(private_sample["generated"])):
                raise ValueError("competitor quality scorer/input differs")
    if examples is None and any(not math.isclose(record[k], median(s[k] for s in samples), rel_tol=1e-8)
                                for k in ("decode_tok_s", "prefill_tok_s")):
        raise ValueError("competitor medians differ from samples")
    if examples is not None:
        mapping = record["token_mapping"]
        if spec["backend"] == "vllm":
            if mapping != {"status": "same-pinned-HF-tokenizer", "manifest_sha256": content_hash(examples)}:
                raise ValueError("competitor tokenizer mapping unverified")
        elif (mapping["status"] != "complete" or mapping["checked_prompts"] != len(examples) or
              mapping["input_manifest_sha256"] != content_hash(examples)):
            raise ValueError("competitor GGUF mapping incomplete")
    return record


def checked_observation(memory, args, request_count):
    """A phase shorter than the sample interval can have zero samples, but must have a window."""
    if (memory["sampler_failed"] or memory["concurrent_compute_workload"] or memory["ownership_check_dropouts"] or
            memory.get("phase_status") != "validated-windows" or memory["device_sample_count"] <= 0):
        return False
    path = REPO_ROOT / memory["raw_series"]["path"]
    if not path.resolve().is_relative_to((REPO_ROOT / "artifacts" / "competitor-observation").resolve()) or file_hash(path) != memory["raw_series"]["sha256"]:
        raise ValueError("competitor memory evidence changed")
    raw = json.loads(path.read_text())
    expected = summarize(raw["samples"], raw["baseline"], args.interval, raw["markers"],
                         raw["start_ns"], raw["end_ns"], memory["concurrent_compute_workload"])
    if (any(memory.get(k) != v for k, v in expected.items() if k != "mem_available_after_mib") or
            memory["ownership_checks"] <= 0 or
            memory["device_sample_count"] + memory["device_dropouts"] != memory["sample_count"]):
        raise ValueError("competitor memory summary differs from raw samples")
    for phase, count in {"load": 1, "request": request_count, **({"warmup": 1} if args.mode == "performance" else {})}.items():
        if memory["phases"].get(phase, {}).get("window_count") != count:
            return False
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # The scheduler chooses backend/dtype, but reuses adapter setup options.
    add_backend_arguments(p, select_backend=False)
    p.add_argument("--mode", choices=["performance", "quality"], default="performance")
    p.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-decode", type=int, default=128)
    p.add_argument("--quality", type=Path, nargs="+", default=[])
    p.add_argument("--only", nargs="+", default=["llamacpp:q4_0", "vllm:fp8", "llamacpp:f16", "vllm:auto"])
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--interval", type=float, default=.05)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    if (not args.only or len(set(args.only)) != len(args.only) or len(set(args.contexts)) != len(args.contexts) or
            any(c < 1 for c in args.contexts) or args.reps < 1 or args.n_decode < 2 or not args.seeds or
            len(set(args.seeds)) != len(args.seeds)):
        p.error("invalid or duplicate measurement dimensions")
    backend_fingerprints = {}
    for setting in args.only:
        args.backend, args.kv_dtype = setting.split(":")
        if args.backend not in backend_fingerprints:
            backend_fingerprints[args.backend] = backend_identity(args)
    gguf_sha256 = file_hash(Path(args.gguf)) if "llamacpp" in backend_fingerprints else None
    vocabulary_size = tokenizer_size(args.awq_model_path) if args.mode == "performance" else None
    quality = {}
    for path in args.quality:
        record = json.loads(path.read_text())
        context = canonical_identity(record["identity"])["config"]["ctx_len"]
        quality[str(context)] = {"path": path.resolve().relative_to(REPO_ROOT).as_posix(),
                            "sha256": file_hash(path), "run_identity": record["run_identity"]}
    if args.mode == "quality" and set(quality) != set(map(str, args.contexts)):
        p.error("one complete quality manifest per context required")
    cells = []
    for i, context in enumerate(args.contexts):
        # Each context gets a Latin rotation of the backend order across input seeds.
        blocks = list(enumerate(args.seeds)) if args.mode == "performance" else [(i, None)]
        for block, seed in blocks:
            order = args.only[block % len(args.only):] + args.only[:block % len(args.only)]
            for setting in order:
                backend, dtype = setting.split(":")
                if backend not in {"llamacpp", "vllm"}:
                    p.error("unsupported backend")
                cells.append({"ctx_len": context, "backend": backend, "kv_dtype": dtype,
                              **({"seed": seed, "input_sha256": content_hash(synthetic_token_ids(vocabulary_size, context, seed))}
                                 if seed is not None else {})})
    protocol = {"version": 1, "mode": args.mode, "ordered_cells": cells,
                "native_boundary_comparison": "phase rates described separately; no client/native interchangeable ratio",
                "timing": "same-request native metrics and client wall clock", "warmup_per_input_seed": 1,
                "quality": quality, "output_count": args.n_decode, "seeds": args.seeds,
                "reps": args.reps, "memory_interval_s": args.interval,
                "residency": "GPU placement configuration; absence of OS fallback unmeasured"}
    config = {"mode": args.mode, "contexts": args.contexts, "only": args.only,
              "seeds": args.seeds, "reps": args.reps, "n_decode": args.n_decode,
              "timeout_s": args.timeout, "ready_timeout_s": args.ready_timeout,
              "gpu_utilization": args.gpu_utilization, "gpu_index": args.gpu_index,
              "attempt": args.attempt, "quality": quality,
              "tokenizer_size": vocabulary_size,
              "backend_fingerprints": backend_fingerprints, "gguf_sha256": gguf_sha256}
    pinned_awq = verified_awq_model(args.awq_model_path)
    schedule = {**make_identity(config, pinned_awq, protocol, provenance()),
                "protocol": protocol, "status": "running", "cells": []}
    schedule["identity"] = export_identity(schedule["identity"])
    directory = REPO_ROOT / "benchmarks" / "validation" / "competitors" / schedule["run_identity"]
    path = directory / "schedule.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if not args.resume or previous["identity"] != schedule["identity"]:
            p.error("preserve existing schedule; identical resume or fresh attempt required")
        if (previous["run_identity"] != schedule["run_identity"] or previous["protocol"] != protocol or
                previous["status"] not in {"running", "incomplete", "complete", "complete-with-failures"}):
            p.error("saved schedule identity/protocol/status changed")
        schedule = previous
        for i, entry in enumerate(schedule["cells"]):
            if i >= len(cells) or {k: entry[k] for k in cells[i]} != cells[i]:
                p.error("saved competitor cell changed")
            if "path" in entry:
                saved = REPO_ROOT / entry["path"]
                if saved.resolve() != (directory / "cells" / f"{i:02d}.json").resolve() or file_hash(saved) != entry["sha256"]:
                    p.error("saved competitor cell changed")
                result = checked_result(json.loads(saved.read_text()), cells[i], schedule, args, quality)
                if result["run_identity"] != entry["run_identity"] or result["status"] != entry["backend_status"]:
                    p.error("saved competitor cell identity/status differs")
                if entry["telemetry_status"] == "complete" and not checked_observation(entry["memory"], args, len(result["samples"])):
                    p.error("saved observation is incomplete")
            elif entry["status"] == "complete":
                p.error("saved successful cell has no evidence")
            if entry["status"] != entry_status(entry):
                p.error("saved status contradicts observed child outcome")
        if schedule["status"].startswith("complete"):
            if schedule["status"] != schedule_status(schedule["cells"], cells):
                p.error("saved terminal schedule is incomplete or contradictory")
            print(json.dumps({"run_identity": schedule["run_identity"], "status": schedule["status"]}))
            return int(schedule["status"] != "complete")
    write_record(path, schedule)
    device = resolve_device(args.gpu_index)
    for i, spec in enumerate(cells):
        if i < len(schedule["cells"]):
            continue
        output = directory / "cells" / f"{i:02d}.json"
        private = REPO_ROOT / "artifacts" / "competitor-observation" / schedule["run_identity"] / uuid.uuid4().hex
        child = "bench_competitor.py" if args.mode == "performance" else "competitor_niah.py"
        command = [sys.executable, str(REPO_ROOT / "scripts" / child), "--backend", spec["backend"],
                   "--kv-dtype", spec["kv_dtype"], "--awq-model-path", args.awq_model_path,
                   "--vllm-python", args.vllm_python, "--llama-server", args.llama_server or "unavailable",
                   "--gguf", args.gguf, "--gpu-utilization", str(args.gpu_utilization),
                   "--ready-timeout", str(args.ready_timeout), "--out", str(output)]
        if args.mode == "performance":
            command += ["--ctx-len", str(spec["ctx_len"]), "--seeds", str(spec["seed"]),
                        "--reps", str(args.reps), "--n-decode", str(args.n_decode)]
        else:
            command += ["--quality", str(REPO_ROOT / quality[str(spec["ctx_len"])]["path"])]
        observed = observe_command(command, private, device, timeout_s=args.timeout, interval_s=args.interval)
        memory = observed["memory"]
        memory["raw_series"]["path"] = (private / "memory-series.json").relative_to(REPO_ROOT).as_posix()
        entry = {**spec, "returncode": observed["returncode"], "observation_status": observed["status"], "memory": memory}
        if output.exists():
            result = checked_result(json.loads(output.read_text()), spec, schedule, args, quality)
            entry.update(path=output.relative_to(REPO_ROOT).as_posix(), sha256=file_hash(output),
                         run_identity=result["run_identity"], backend_status=result["status"],
                         status=observed["status"] if observed["status"] != "complete" else
                                result["status"] if observed["returncode"] == 0 or result["status"] != "complete" else "backend-error-after-result")
        else:
            entry.update(status=observed["status"] if observed["status"] != "complete" else "missing-result")
        count = args.reps if args.mode == "performance" else len(result["samples"]) if output.exists() else 0
        entry["telemetry_status"] = "complete" if checked_observation(memory, args, count) else "invalid"
        schedule["cells"].append(entry)
        validate_export(schedule)
        write_record(path, schedule)
        print(json.dumps({"cell": i, **spec, "status": entry["status"]}), flush=True)
        if observed["status"] == "concurrent-workload":
            break
    schedule["status"] = schedule_status(schedule["cells"], cells)
    write_record(path, schedule)
    print(json.dumps({"run_identity": schedule["run_identity"], "status": schedule["status"]}), flush=True)
    return int(schedule["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
