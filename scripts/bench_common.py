"""Shared measurement and result conventions for the validation benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import tempfile
from pathlib import Path
from statistics import median

SCHEMA_VERSION = 3
DEFAULT_MODEL = "casperhansen/llama-3.2-3b-instruct-awq"
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "triton", "transformers", "autoawq", "accelerate", "huggingface_hub", "vllm")
_ABSOLUTE_PATH = re.compile(r"(?:^|[\s\"'=(:])/(?!v1/)[^\s]+|[A-Za-z]:[\\/]|file://")


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--ctx-len", type=int, required=True,
                        help="input tokens already in the cache when decode starts")
    parser.add_argument("--n-decode", type=int, default=128,
                        help="output tokens; the first comes from prefill")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)


def validate_run_arguments(parser: argparse.ArgumentParser, args) -> None:
    if args.ctx_len < 1 or args.n_decode < 2 or args.reps < 1:
        parser.error("ctx-len >= 1, n-decode >= 2, and reps >= 1 are required")


def synthetic_token_ids(vocab_size: int, ctx_len: int, seed: int) -> list[int]:
    """Use identical input IDs for FlashQuest and vLLM, without retokenization."""
    rng = random.Random(seed)
    return [rng.randrange(vocab_size) for _ in range(ctx_len)]


def run_config(ctx_len: int, n_decode: int, reps: int, seed: int, **settings) -> dict:
    return {"ctx_len": ctx_len, "n_decode": n_decode, "reps": reps, "seed": seed,
            **settings}


def new_record(backend: str, quant: str, config: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "backend": backend,
        "quant": quant,
        "config": config,
        "ctx_len": config["ctx_len"],
        "decode_steps": config["n_decode"] - 1,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "end_to_end_tok_s": None,
        # PyTorch allocator counters are not physical GPU residency.
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
        "samples": [],
        "wall_s": None,
        "oom": False,
        "error": None,
    }


def summarize_samples(record: dict) -> None:
    """Report median throughput; retain every sample for inspection."""
    samples = record["samples"]
    if not samples:
        raise ValueError("no completed benchmark samples")
    for key in ("decode_tok_s", "prefill_tok_s", "end_to_end_tok_s"):
        values = [s[key] for s in samples if s.get(key) is not None]
        record[key] = median(values) if values else None
    for key in ("peak_allocated_mib", "peak_reserved_mib"):
        values = [s[key] for s in samples if s.get(key) is not None]
        record[key] = max(values) if values else None


def vllm_sample(output, ctx_len: int, n_decode: int, elapsed: float) -> dict:
    """Separate scheduled prefill and first-to-last-token decode intervals.

    Requires vLLM RequestMetrics (V0 engine). Never substitute end-to-end
    request throughput when a version omits the phase timestamps.
    """
    n_in = len(output.prompt_token_ids)
    n_out = len(output.outputs[0].token_ids)
    if n_in != ctx_len or n_out != n_decode:
        raise ValueError(f"expected {ctx_len} input/{n_decode} output tokens; got {n_in}/{n_out}")
    metrics = output.metrics
    scheduled = getattr(metrics, "first_scheduled_time", None)
    first = getattr(metrics, "first_token_time", None)
    last = getattr(metrics, "last_token_time", None)
    if scheduled is None or first is None or last is None or not scheduled < first < last:
        raise ValueError("vLLM phase timestamps unavailable; use a RequestMetrics-capable V0 engine")
    if elapsed <= 0:
        raise ValueError("non-positive request duration")
    prefill_s, decode_s = first - scheduled, last - first
    return {"input_tokens": n_in, "output_tokens": n_out,
            "prefill_s": prefill_s, "decode_s": decode_s, "request_s": elapsed,
            "prefill_tok_s": n_in / prefill_s,
            "decode_tok_s": (n_out - 1) / decode_s,
            "end_to_end_tok_s": n_out / elapsed}


def write_record(out: Path, record: dict) -> None:
    """Atomically save JSON; an identified run cannot overwrite a different run."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and record.get("run_identity"):
        previous = json.loads(out.read_text(encoding="utf-8"))
        if previous.get("run_identity") != record["run_identity"]:
            raise ValueError("output belongs to a different run identity")
    payload = json.dumps(record, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".write-", suffix=".json", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, out)
    finally:
        Path(name).unlink(missing_ok=True)


def content_hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_identity(root: Path = REPO_ROOT) -> dict:
    """Hash relevant contents, including new files, without exporting local paths."""
    paths = [p for base in ("src", "scripts") for p in (root / base).rglob("*.py")]
    paths += [root / name for name in ("pyproject.toml", "requirements-validation.txt", "requirements-vllm.txt",
                                      "data/PaulGrahamEssays.json", "scripts/bench_llamacpp.sh")]
    hashes = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(paths) if p.is_file()}
    def git(*args):
        try:
            result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True,
                                    text=True, timeout=10)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--", "src", "scripts",
                              "pyproject.toml", "requirements-validation.txt", "requirements-vllm.txt",
                              "data/PaulGrahamEssays.json")), "files": hashes,
            "content_sha256": content_hash(hashes)}


def provenance() -> dict:
    """Collect environment metadata without importing any GPU runtime."""
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    gpus = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            name, total, driver, capability = (s.strip() for s in line.split(","))
            gpus.append({"name": name, "total_mib": float(total), "driver": driver,
                         "compute_capability": capability})
    except (OSError, ValueError, subprocess.SubprocessError):
        gpus = None
    return {"python": platform.python_version(), "os": platform.system(),
            "kernel": platform.release(), "wsl": "microsoft" in platform.release().lower(),
            "packages": versions, "gpus": gpus}


def model_identity(name: str, revision: str | None = None) -> dict:
    """Resolve immutable local/HF artifacts before accepting any cached evidence.

    Local metadata only is read when name is a directory. Remote access is explicit
    production execution, never import-time, and does not require account credentials.
    """
    directory = Path(name)
    if directory.is_dir():
        files = [p for p in directory.rglob("*") if p.is_file() and
                 (p.suffix in {".safetensors", ".bin", ".gguf"} or
                  p.name in {"config.json", "tokenizer.json", "tokenizer_config.json",
                             "special_tokens_map.json", "generation_config.json"})]
        hashes = {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(files)}
        if "config.json" not in hashes or not any(
                p.endswith((".safetensors", ".bin", ".gguf")) for p in hashes):
            raise ValueError("local model lacks configuration or weight artifacts")
        return {"model": "local-model", "revision": revision, "files": hashes,
                "content_sha256": content_hash(hashes)}
    from huggingface_hub import HfApi, hf_hub_download

    info = HfApi(token=False).model_info(name, revision=revision, files_metadata=True)
    hashes = {}
    for sibling in info.siblings:
        filename = sibling.rfilename
        if filename.endswith((".safetensors", ".bin")):
            lfs = sibling.lfs
            if lfs is None:
                raise ValueError("model weights lack immutable content metadata")
            hashes[filename] = lfs.sha256
        elif filename in {"config.json", "tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "generation_config.json"}:
            path = hf_hub_download(name, filename, revision=info.sha, token=False,
                                   cache_dir=REPO_ROOT / "artifacts" / "hf-cache")
            hashes[filename] = file_hash(Path(path))
    if "config.json" not in hashes or not any(p.endswith((".bin", ".safetensors")) for p in hashes):
        raise ValueError("model identity lacks configuration or weight content")
    return {"model": name, "revision": info.sha, "files": hashes,
            "content_sha256": content_hash(hashes)}


def make_identity(config: dict, model: dict, protocol: dict, environment: dict,
                  *, source: dict | None = None) -> dict:
    identity = {"schema_version": SCHEMA_VERSION, "config": config, "model": model,
                "protocol_sha256": content_hash(protocol), "environment": environment,
                "source": source if source is not None else source_identity()}
    return {"run_identity": content_hash(identity), "identity": identity}


def export_identity(identity: dict) -> dict:
    """Give file fingerprints explicit types; retain the canonical identity hash.

    Filename-to-digest mappings can resemble credentials to secret scanners when
    a filename contains 'token' or 'key'. Public evidence uses path/sha256 entries;
    raw evidence and run hashing continue to use the original canonical mappings.
    """
    identity = canonical_identity(identity)
    exported = dict(identity)
    for section in ("model", "source"):
        metadata = dict(identity[section])
        metadata["files"] = [{"sha256": digest, "path": path}
                             for path, digest in sorted(metadata["files"].items())]
        exported[section] = metadata
    return exported


def canonical_identity(identity: dict) -> dict:
    """Recover the hashed identity from either public or original file mappings."""
    canonical = dict(identity)
    for section in ("model", "source"):
        metadata = dict(identity[section])
        entries = metadata["files"]
        if isinstance(entries, list):
            files = {}
            for entry in entries:
                if (set(entry) != {"sha256", "path"} or
                        not isinstance(entry["path"], str) or entry["path"] in files or
                        not isinstance(entry["sha256"], str) or
                        not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                    raise ValueError("invalid file fingerprint entry")
                files[entry["path"]] = entry["sha256"]
            metadata["files"] = files
        if content_hash(metadata["files"]) != metadata["content_sha256"]:
            raise ValueError("file fingerprints differ from their content identity")
        canonical[section] = metadata
    return canonical


def validate_export(value) -> None:
    """Fail closed on paths, identifying fields, non-finite numbers and unknown types."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {"email", "hostname", "username", "serial", "credentials",
                               "traceback", "stdout", "stderr", "model_filename"}:
                raise ValueError("private or raw field in export")
            validate_export(key)
            validate_export(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            validate_export(item)
    elif isinstance(value, str):
        without_urls = re.sub(r"https?://[^\s]+", "", value)
        if _ABSOLUTE_PATH.search(without_urls) or re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value):
            raise ValueError("local path or email in export")
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite metric in export")
    elif value is not None and not isinstance(value, (bool, int)):
        raise ValueError("unsupported export type")


def error_code(exc: Exception) -> str:
    if is_oom(str(exc)):
        return "out_of_memory"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (ImportError, ModuleNotFoundError)):
        return "missing_dependency"
    return "execution_error"


def export_benchmark(record: dict) -> dict:
    """Allowlist normalized timing evidence; raw backend messages stay local."""
    keys = {"schema_version", "backend", "quant", "ctx_len", "decode_steps", "config",
            "decode_tok_s", "prefill_tok_s", "end_to_end_tok_s", "peak_allocated_mib",
            "peak_reserved_mib", "samples", "wall_s", "oom", "versions", "prompt_kind",
            "provenance", "identity", "run_identity", "raw_evidence"}
    config_keys = {"ctx_len", "n_decode", "reps", "seed", "model", "kv_bits", "retention",
                   "page_size", "num_sinks", "window_pages", "kv_cache_dtype", "kv_k",
                   "kv_v", "n_gpu_layers", "threads"}
    config_keys |= {"revision", "model_cache", "warmup_steps", "cache_capacity", "input_sha256",
                    "generation", "runtime", "memory_protocol"}
    sample_keys = {"input_tokens", "output_tokens", "prefill_s", "decode_s", "request_s",
                   "prefill_tok_s", "decode_tok_s", "end_to_end_tok_s",
                   "peak_allocated_mib", "peak_reserved_mib"}
    result = {key: value for key, value in record.items() if key in keys}
    if "identity" in result:
        result["identity"] = export_identity(result["identity"])
    result["config"] = {key: value for key, value in record.get("config", {}).items()
                        if key in config_keys}
    model = result["config"].get("model")
    if model and (_ABSOLUTE_PATH.search(model) or Path(model).exists()):
        result["config"]["model"] = "local-model"
    if isinstance(result.get("samples"), list):
        result["samples"] = [{key: value for key, value in sample.items() if key in sample_keys}
                             for sample in result["samples"]]
    elif isinstance(result.get("samples"), dict):
        result["samples"] = {key: value for key, value in result["samples"].items()
                             if key in {"decode_tok_s", "prefill_tok_s"}}
    result["error"] = ("out_of_memory" if record.get("oom") else "backend_error") if record.get("error") else None
    validate_export(result)
    return result


def write_benchmark_record(out: Path, record: dict, *, raw_dir: Path | None = None) -> dict:
    """Retain original evidence privately and write a sanitized measurement record."""
    import uuid

    raw_dir = raw_dir or REPO_ROOT / "artifacts" / "benchmarks" / uuid.uuid4().hex
    raw_path = raw_dir / "raw.json"
    write_record(raw_path, record)
    record = {**record, "raw_evidence": {"path": raw_path.relative_to(REPO_ROOT).as_posix(),
                                        "sha256": file_hash(raw_path)}}
    exported = export_benchmark(record)
    write_record(out, exported)
    return exported


def is_oom(message: str) -> bool:
    # A generic CUDA failure or a reference to a KV cache isn't evidence of OOM.
    return any(s in message.lower() for s in
               ("out of memory", "no available memory", "no available blocks",
                "no available memory for the cache blocks"))
