"""Pinned native server benchmark on exact IDs; native and client times stay separate."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
import uuid
from pathlib import Path
from statistics import median

from bench_common import (
    REPO_ROOT,
    content_hash,
    error_code,
    export_identity,
    file_hash,
    make_identity,
    model_identity,
    provenance,
    synthetic_token_ids,
    validate_export,
    write_record,
)
from competitor_backend import (
    Server,
    add_backend_arguments,
    command_for,
    completion_payload,
    llama_runtime_directory,
    marker,
    parse_completion,
)

PROTOCOL = {"version": 1, "batch": 1, "concurrency": 1, "warmup": "one full request per seed",
            "generation": "greedy fixed output; first output from prefill",
            "input": "exact seeded IDs matching FlashQuest",
            "timing": "native same-request phases; client wall clock separately",
            "cache_reuse": False, "offload_gb": 0}


def verified_awq_model(snapshot):
    pinned = model_identity("casperhansen/llama-3.2-3b-instruct-awq", "272b3bde867b606760447deb9a4d2719fbdfd3ae")
    actual = model_identity(snapshot, pinned["revision"])
    if actual["files"] != pinned["files"]:
        raise ValueError("loaded AWQ/tokenizer artifacts differ from the pinned model")
    return pinned


def backend_identity(args):
    if args.backend == "vllm":
        code = "import json,importlib.metadata as m,torch;print(json.dumps({'packages':{d.metadata['Name']:d.version for d in m.distributions()},'cuda':torch.version.cuda}))"
        info = json.loads(subprocess.run([args.vllm_python, "-c", code], capture_output=True,
                                        text=True, check=True, timeout=60).stdout)
        info["interpreter_sha256"] = file_hash(Path(args.vllm_python).resolve())
        return info
    binary = Path(args.llama_server).resolve()
    version = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                             check=True, timeout=20)
    build = re.search(r"([0-9]+)\s*\(([a-f0-9]{7,40})\)", version.stdout + version.stderr)
    version_text = version.stdout + version.stderr
    current_build = re.search(r"build (\d+), commit ([a-f0-9]+)", version_text)
    return {"server_sha256": file_hash(binary), "build": current_build.group(0) if current_build else build.group(0) if build else "unknown",
            "cuda_runtime_sha256": [{"sha256": file_hash(p), "name": p.name}
                                      for p in sorted(llama_runtime_directory(binary).glob("*.so*")) if p.is_file()],
            "library_sha256": [{"sha256": file_hash(p), "name": p.name}
                                  for p in sorted(binary.parent.glob("*.so*")) if p.is_file()]}


def resolved_runtime(backend, log):
    if backend == "llamacpp":
        match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", log)
        cache = re.findall(r"(CUDA\d+) KV buffer size\s*=\s*([\d.]+) MiB", log)
        realized = re.search(r"llama_kv_cache:.*\(\s*(\d+) cells.*K\s*\(([^)]+)\).*V\s*\(([^)]+)\)", log)
        return {"weight_placement": "all-model-layers-GPU" if match and match[1] == match[2] else "unverified",
                "offloaded_layers": int(match[1]) if match else None,
                "total_layers": int(match[2]) if match else None,
                "gpu_kv_buffers_mib": [{"device": d, "mib": float(n)} for d, n in cache],
                "flash_attention_enabled": bool(re.search(r"[Ff]lash [Aa]ttention.*enabled|flash_attn\s*=\s*(?:1|enabled)", log)),
                "cache_capacity_tokens": int(realized[1]) if realized else None,
                "cache_k_dtype": realized[2] if realized else None,
                "cache_v_dtype": realized[3] if realized else None,
                "os_fallback": "unmeasured"}
    attention = re.findall(r"Using ([A-Za-z0-9_]+) attention backend", log)
    capacity = re.search(r"GPU KV cache size:\s*([\d,]+) tokens", log)
    weight_kernels = sorted(set(re.findall(r"Using ([A-Za-z0-9_]+) for AutoAWQMarlinLinearMethod", log)))
    return {"weight_kernels": weight_kernels, "kernel_verification": "resolved-selection-log" if weight_kernels else "unverified",
            "attention_backend": sorted(set(attention)),
            "allocated_kv_tokens": int(capacity[1].replace(",", "")) if capacity else None,
            "kv_scale_policy": "checkpoint-or-unit-default; values in worker observation", "cpu_offload_gb": 0,
            "os_fallback": "unmeasured"}


def verify_runtime(runtime, backend, dtype, capacity):
    if backend == "llamacpp":
        if (runtime["weight_placement"] != "all-model-layers-GPU" or
                not runtime["flash_attention_enabled"] or not runtime["gpu_kv_buffers_mib"] or
                runtime["cache_k_dtype"] != dtype or runtime["cache_v_dtype"] != dtype or
                (runtime["cache_capacity_tokens"] or 0) < capacity):
            raise ValueError("actual full GPU model/cache precision/FA not verified")
    else:
        observed = runtime["worker_observation"]
        # Pinned vLLM encodes FP8 in uint8 tensors; config selects E4M3 interpretation.
        expected_dtype = "torch.uint8" if dtype == "fp8" else "torch.float16"
        if (observed["cache_config_dtype"] != dtype or observed["logical_cache_dtype"] != expected_dtype or
                observed["cache_storage_dtypes"] != [expected_dtype] or
                observed["cache_tensor_devices"] != ["cuda:0"] or observed["cache_tensor_bytes"] <= 0 or
                observed["model_parameter_devices"] != ["cuda:0"] or not runtime["weight_kernels"] or
                (runtime["allocated_kv_tokens"] or 0) < capacity or observed["cpu_offload_gb"] != 0 or
                observed["offload_group_size"] != 0 or
                "AutoAWQMarlinLinearMethod" not in observed["quantization_methods"] or
                not observed["kv_scales"] or any(not math.isfinite(s[k]) or s[k] <= 0
                                                  for s in observed["kv_scales"] for k in ("k", "v"))):
            raise ValueError("realized vLLM cache/device/kernel verification failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(p)
    p.add_argument("--ctx-len", type=int, required=True)
    p.add_argument("--n-decode", type=int, default=128)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.ctx_len < 1 or args.n_decode < 2 or args.reps < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error("invalid measurement dimensions")
    if args.out.exists():
        p.error("output already exists; preserve previous attempt")
    model = verified_awq_model(args.awq_model_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.awq_model_path, local_files_only=True)
    backend = backend_identity(args)
    if args.backend == "llamacpp":
        hashes = {Path(args.gguf).name: file_hash(Path(args.gguf))}
        model = {"model": "bartowski/Llama-3.2-3B-Instruct-GGUF",
                 "revision": "5ab33fa94d1d04e903623ae72c95d1696f09f9e8", "files": hashes,
                 "content_sha256": content_hash(hashes)}
    config = {"backend": args.backend, "kv_dtype": args.kv_dtype, "ctx_len": args.ctx_len,
              "n_decode": args.n_decode, "reps": args.reps, "seeds": args.seeds,
              "capacity": args.ctx_len + args.n_decode, "gpu_utilization": args.gpu_utilization if args.backend == "vllm" else None,
              "max_batch_tokens": 2048, "llama_ubatch": 512 if args.backend == "llamacpp" else None,
              "backend_identity": backend, "generation": PROTOCOL["generation"],
              "ready_timeout_s": args.ready_timeout, "request_timeout_s": args.ready_timeout,
              "engine_settings": {"offload_gb": 0, "prefix_cache": False, "batch": 1,
                                  "temperature": 0, "ignore_eos": True, "speculation": False,
                                  "vllm_worker_observation": "vllm0.30-after-first-worker-execution",
                                  "llama_flash_attention": "on", "llama_fit": "off"}}
    record = {**make_identity(config, model, PROTOCOL, provenance()), "protocol": PROTOCOL,
              "status": "running", "samples": [], "runtime": None}
    directory = REPO_ROOT / "artifacts" / "competitors" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        with Server(command_for(args, capacity=config["capacity"]), directory, timeout=args.ready_timeout) as server:
            for seed in args.seeds:
                ids = synthetic_token_ids(len(tokenizer), args.ctx_len, seed)
                payload = completion_payload(args.backend, ids, args.n_decode, performance=True)
                route = "/v1/completions" if args.backend == "vllm" else "/completion"
                marker("warmup_start", seed)
                parse_completion(args.backend, server.request(route, payload), len(ids), args.n_decode, performance=True)
                marker("warmup_end", seed)
                for repetition in range(args.reps):
                    marker("request_start", seed * args.reps + repetition)
                    start = time.perf_counter()
                    response = server.request(route, payload)
                    elapsed = time.perf_counter() - start
                    marker("request_end", seed * args.reps + repetition)
                    sample = parse_completion(args.backend, response, len(ids), args.n_decode, performance=True)
                    sample.pop("text")
                    sample.pop("tokens")
                    sample.update(seed=seed, repetition=repetition, input_sha256=content_hash(ids), client_request_s=elapsed)
                    record["samples"].append(sample)
            record["runtime"] = resolved_runtime(args.backend, (directory / "server.log").read_text())
            if args.backend == "vllm":
                record["runtime"]["worker_observation"] = json.loads((directory / "runtime.json").read_text())
            verify_runtime(record["runtime"], args.backend, args.kv_dtype, config["capacity"])
        record["status"] = "complete"
        record["decode_tok_s"] = median(s["decode_tok_s"] for s in record["samples"])
        record["prefill_tok_s"] = median(s["prefill_tok_s"] for s in record["samples"])
    except Exception as exc:  # noqa: BLE001
        record["status"] = error_code(exc)
        (directory / "error.log").write_text(f"{type(exc).__name__}: {exc}\n")
    raw = directory / "raw.json"
    write_record(raw, record)
    record["identity"] = export_identity(record["identity"])
    record["raw_evidence"] = {"path": raw.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(raw)}
    validate_export(record)
    write_record(args.out, record)
    print(json.dumps({"run_identity": record["run_identity"], "status": record["status"]}))
    return int(record["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
