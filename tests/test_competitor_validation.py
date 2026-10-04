"""Independent CPU checks for frozen competitor inputs and public evidence."""
import copy
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_common as C
import bench_competitor as B
import competitor_niah as N
import gpu_memory as M
import phase6_run_ruler_4k_int4 as Q
import run_competitors as R

MODEL_FILES = {"config.json": "a" * 64, "model.safetensors": "b" * 64}
MODEL = {"model": "validation-model", "revision": "c" * 40, "files": MODEL_FILES,
         "content_sha256": C.content_hash(MODEL_FILES)}
SOURCE = {"commit": "d" * 40, "dirty": False, "files": {},
          "content_sha256": C.content_hash({})}
ENVIRONMENT = {"python": "test", "packages": {}}
BACKEND = {"version": "test", "binary_sha256": "e" * 64}


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    for module in (N, R):
        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        monkeypatch.setattr(module, "provenance", lambda: copy.deepcopy(ENVIRONMENT))
        monkeypatch.setattr(module, "backend_identity", lambda args: copy.deepcopy(BACKEND))
        monkeypatch.setattr(module, "verified_awq_model", lambda path: copy.deepcopy(MODEL))
    monkeypatch.setattr(C, "source_identity", lambda: copy.deepcopy(SOURCE))
    monkeypatch.setattr(R, "resolve_device", lambda index: "test-device")
    monkeypatch.setattr(R, "tokenizer_size", lambda path: 6, raising=False)
    return tmp_path


def quality_fixture(root, context=8192, *, examples_mutator=None):
    examples = [{"example_id": f"{task}:1:{index}", "task": task, "seed": 1,
                 "index": index, "input_ids": [0, 3, index + 4],
                 "input_sha256": C.content_hash([0, 3, index + 4]),
                 "expected": ["red"] if task == "single" else ["red", "blue"]}
                for task in Q.TASKS for index in range(2)]
    if examples_mutator:
        examples_mutator(examples)
    cfg = {"ctx_len": context, "actual_max_prompt_tokens": 3, "cache_capacity": 131,
           "max_new_tokens": 128, "n_samples": 2, "tasks": list(Q.TASKS), "seeds": [1],
           "retentions": [.2, 1.0], "kv_bits": 4, "page_size": 64, "num_sinks": 4,
           "window_pages": 2, "manifest_sha256": C.content_hash(examples),
           "runtime": {"weight_kernel": "test", "dtype": "torch.float16"}}
    run = C.make_identity(cfg, MODEL, Q.SCREEN_PROTOCOL, ENVIRONMENT, source=SOURCE)
    cells = []
    for arm, retention in (("dense", None), ("int4-r0.2", .2), ("int4-r1", 1.0)):
        for task in Q.TASKS:
            selected = [e for e in examples if e["task"] == task]
            samples = [{"example_id": e["example_id"], "input_sha256": e["input_sha256"],
                        "prompt_tokens": len(e["input_ids"]), "expected": e["expected"],
                        "generated": "red blue", "hit": True, "output_tokens": 2,
                        "termination": "eos"} for e in selected]
            cells.append({"task": task, "seed": 1, "arm": arm, "retention": retention,
                          "samples": samples, "hits": 2, "total": 2, "wall_s": .01})
    raw = {"schema_version": C.SCHEMA_VERSION, **run, "protocol": Q.SCREEN_PROTOCOL,
           "manifest": {"sha256": C.content_hash(examples), "count": len(examples),
                        "min_prompt_tokens": 3, "max_prompt_tokens": 3},
           "status": "complete", "error": None, "cells": cells,
           "screen": Q.screen_verdict(cells, SimpleNamespace(**cfg))}
    raw_path = root / "artifacts/quality" / run["run_identity"] / "raw.json"
    C.write_record(raw_path, raw)
    C.write_record(raw_path.parent / "manifest.json", {"run_identity": run["run_identity"],
                                                       "examples": examples})
    public_path = root / "benchmarks/validation/quality" / run["run_identity"] / "quality.json"
    C.write_record(public_path, Q.export_result(raw, raw_path))
    return public_path, raw_path


def rewrite_quality(path, raw_path, *, raw_mutator=None, examples_mutator=None,
                    public_mutator=None):
    public, raw = json.loads(path.read_text()), json.loads(raw_path.read_text())
    if examples_mutator:
        manifest_path = raw_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        examples_mutator(manifest["examples"])
        C.write_record(manifest_path, manifest)
        raw["manifest"]["sha256"] = C.content_hash(manifest["examples"])
        public["manifest"]["sha256"] = raw["manifest"]["sha256"]
    if raw_mutator:
        raw_mutator(raw)
    # Corruption fixtures deliberately bypass the production overwrite guard.
    raw_path.write_text(json.dumps(raw) + "\n")
    public["raw_evidence"]["sha256"] = C.file_hash(raw_path)
    if public_mutator:
        public_mutator(public)
    C.write_record(path, public)


def command_value(command, flag):
    return command[command.index(flag) + 1]


def native_response(backend, text, input_count, output_count, *, performance):
    if backend == "vllm":
        return {"choices": [{"text": text, "token_ids": [3] * output_count,
                             "finish_reason": "length" if performance else "stop"}],
                "usage": {"prompt_tokens": input_count, "completion_tokens": output_count},
                "metrics": {"time_to_first_token_ms": 100, "generation_time_ms": 200,
                            "queue_time_ms": 0}}
    return {"content": text, "tokens": [3] * output_count, "tokens_predicted": output_count,
            "stop_type": "limit" if performance else "eos",
            "timings": {"prompt_n": input_count, "prompt_ms": 100,
                        "predicted_n": output_count, "predicted_ms": 200}}


def vllm_runtime(dtype="auto", capacity=131):
    observed = {"method": "vllm0.30-worker-after-first-request-v1",
                "cache_config_dtype": dtype,
                "logical_cache_dtype": "torch.uint8" if dtype == "fp8" else "torch.float16",
                "cache_tensor_devices": ["cuda:0"], "cache_tensor_bytes": 4096,
                "cache_storage_dtypes": ["torch.uint8"] if dtype == "fp8" else ["torch.float16"],
                "model_parameter_devices": ["cuda:0"], "model_tensor_devices": ["cuda:0"],
                "quantization_methods": ["AutoAWQMarlinLinearMethod"],
                "kv_scales": [{"k": 1.0, "v": 1.0}], "cpu_offload_gb": 0,
                "offload_group_size": 0, "offload_backend": "auto"}
    return {"weight_kernels": ["MarlinLinearKernel"], "kernel_verification": "resolved-selection-log",
            "attention_backend": ["TRITON_ATTN"], "allocated_kv_tokens": capacity,
            "kv_scale_policy": "checkpoint-or-unit-default; values in worker observation",
            "cpu_offload_gb": 0, "os_fallback": "unmeasured", "worker_observation": observed}


def observed_memory(private, *, performance, repetitions, interval_s):
    phases = [("load", None)] + ([("warmup", 0)] if performance else [])
    phases += [("request", i) for i in range(repetitions)]
    markers, samples = [], []
    for index, (phase, repetition) in enumerate(phases):
        start = index * 100_000_000
        end = start + 90_000_000
        markers.extend([{"event": f"{phase}_start", "monotonic_ns": start, "repetition": repetition},
                        {"event": f"{phase}_end", "monotonic_ns": end, "repetition": repetition}])
        for offset in (10_000_000, 60_000_000):
            samples.append({"query_start_ns": start + offset, "query_end_ns": start + offset + 1_000_000,
                            "device_used_mib": 200.0, "process_tree_rss_mib": 20.0,
                            "mem_available_mib": 500.0, "sm_clock_mhz": 1000.0,
                            "temperature_c": 40.0, "ownership_checked": True,
                            "ownership_check_failed": False})
    baseline = {"device_used_mib": 100.0, "mem_available_mib": 500.0}
    end_ns = len(phases) * 100_000_000
    memory = M.summarize(samples, baseline, interval_s, markers, 0, end_ns, False)
    memory.update(sampler_failed=False, physical_device={"index": 0, "name": "test GPU",
                                                        "total_mib": 1000, "driver": "test"},
                  backend_visible_device="cuda:0")
    series = private / "memory-series.json"
    C.write_record(series, {"baseline": baseline, "samples": samples, "markers": markers,
                            "start_ns": 0, "end_ns": end_ns})
    memory["raw_series"] = {"sha256": C.file_hash(series)}
    return memory


def install_scheduler(monkeypatch, root, *, mode="performance", mutation=None, result_mutation=None):
    commands = []

    def observe(command, private, device, **kwargs):
        assert device == "test-device"
        commands.append(command)
        backend = command_value(command, "--backend")
        dtype = command_value(command, "--kv-dtype")
        if mode == "performance":
            context = int(command_value(command, "--ctx-len"))
            seed = int(command_value(command, "--seeds"))
            output_count = int(command_value(command, "--n-decode"))
            repetitions = int(command_value(command, "--reps"))
            cfg = {"backend": backend, "kv_dtype": dtype, "ctx_len": context,
                   "n_decode": output_count, "capacity": context + output_count,
                   "reps": repetitions, "seeds": [seed],
                   "gpu_utilization": .7, "max_batch_tokens": 2048, "llama_ubatch": None,
                   "backend_identity": copy.deepcopy(BACKEND), "generation": B.PROTOCOL["generation"],
                   "ready_timeout_s": 300, "request_timeout_s": 300,
                   "engine_settings": {"offload_gb": 0, "prefix_cache": False, "batch": 1,
                                       "temperature": 0, "ignore_eos": True, "speculation": False,
                                       "vllm_worker_observation": "vllm0.30-after-first-worker-execution",
                                       "llama_flash_attention": "on", "llama_fit": "off"}}
            protocol = copy.deepcopy(B.PROTOCOL)
            samples = []
            for repetition in range(repetitions):
                sample = N.parse_completion(backend, native_response(backend, "red", context,
                                            output_count, performance=True), context, output_count,
                                            performance=True)
                sample.pop("text")
                sample.pop("tokens")
                sample.update(seed=seed, repetition=repetition,
                              input_sha256=C.content_hash(C.synthetic_token_ids(6, context, seed)),
                              client_request_s=.31)
                samples.append(sample)
        else:
            quality_path = Path(command_value(command, "--quality"))
            quality, original, ref, examples = N.load_manifest(quality_path)
            cfg = {"backend": backend, "kv_dtype": dtype,
                   "ctx_len": original["config"]["ctx_len"],
                   "capacity": original["config"]["cache_capacity"],
                   "max_new_tokens": 128, "examples": len(examples), "limit_per_task": None,
                   "backend_identity": copy.deepcopy(BACKEND), "gpu_utilization": .7}
            protocol = {"version": 1, "quality_run": quality["run_identity"], "manifest": ref,
                        "max_new_tokens": 128, "scorer": "all expected substrings in decoded answer",
                        "generation": "greedy EOS or output limit; exact input IDs; no wrapping",
                        "timing": "native phases and client times; quality may stop at EOS"}
            samples = []
            for example in examples:
                sample = N.parse_completion(backend, native_response(backend, "red blue", 3, 2,
                                            performance=False), 3, 128, performance=False)
                sample.pop("text")
                sample.pop("tokens")
                sample.update(example_id=example["example_id"], task=example["task"],
                              seed=example["seed"], index=example["index"],
                              input_sha256=example["input_sha256"], hit=True,
                              generated_sha256=C.content_hash("red blue"), client_request_s=.31,
                              generated="red blue", expected=example["expected"])
                samples.append(sample)
        model, source, environment = copy.deepcopy(MODEL), copy.deepcopy(SOURCE), copy.deepcopy(ENVIRONMENT)
        parts = {"config": cfg, "model": model, "source": source,
                 "environment": environment, "protocol": protocol}
        if mutation:
            mutation(parts)
        result = {**C.make_identity(parts["config"], parts["model"], parts["protocol"],
                                   parts["environment"], source=parts["source"]),
                  "protocol": parts["protocol"], "status": "complete", "samples": samples,
                  "runtime": vllm_runtime(dtype, cfg["capacity"])}
        if mode == "quality":
            result["token_mapping"] = {"status": "same-pinned-HF-tokenizer",
                                       "manifest_sha256": C.content_hash(examples)}
        else:
            result["decode_tok_s"] = samples[0]["decode_tok_s"]
            result["prefill_tok_s"] = samples[0]["prefill_tok_s"]
        raw_path = root / "artifacts" / ("competitors" if mode == "performance" else "competitor-quality") / result["run_identity"] / "raw.json"
        C.write_record(raw_path, result)
        result["identity"] = C.export_identity(result["identity"])
        result["samples"] = [{k: v for k, v in sample.items() if k not in {"generated", "expected"}}
                             for sample in samples]
        result["raw_evidence"] = {"path": raw_path.relative_to(root).as_posix(),
                                  "sha256": C.file_hash(raw_path)}
        if result_mutation:
            result_mutation(result)
        C.write_record(Path(command_value(command, "--out")), result)
        return {"returncode": 0, "status": "complete",
                "memory": observed_memory(private, performance=mode == "performance",
                                          repetitions=len(samples), interval_s=kwargs["interval_s"])}

    monkeypatch.setattr(R, "observe_command", observe)
    return commands


def scheduler_args(root, *, mode="performance", quality=None, contexts=(8192,)):
    result = ["run_competitors", "--awq-model-path", "unused-model", "--mode", mode,
              "--only", "vllm:auto", "--contexts", *map(str, contexts)]
    if quality:
        result += ["--quality", *map(str, quality)]
    return result


def test_schedule_quality_context_strings_roundtrip(isolated, monkeypatch):
    root = isolated
    paths = [quality_fixture(root, context)[0] for context in (8192, 32768)]
    commands = install_scheduler(monkeypatch, root, mode="quality")
    monkeypatch.setattr(sys, "argv", scheduler_args(root, mode="quality", quality=paths,
                                                   contexts=(8192, 32768)))
    assert R.main() == 0
    schedule = json.loads(next(root.glob("benchmarks/validation/competitors/*/schedule.json")).read_text())
    canonical = C.canonical_identity(schedule["identity"])
    assert C.content_hash(canonical) == schedule["run_identity"]
    assert canonical["protocol_sha256"] == C.content_hash(schedule["protocol"])
    assert set(canonical["config"]["quality"]) == {"8192", "32768"}
    assert canonical["config"]["quality"] == schedule["protocol"]["quality"]
    assert [command_value(c, "--quality") for c in commands] == list(map(str, paths))


@pytest.mark.parametrize("field", ["backend", "source", "environment", "model"])
def test_schedule_rejects_frozen_backend_source_environment_model_change(isolated, monkeypatch, field):
    def mutate(parts):
        if field == "backend":
            parts["config"]["backend_identity"]["version"] = "changed"
        elif field == "source":
            parts["source"]["commit"] = "f" * 40
        elif field == "environment":
            parts["environment"]["python"] = "changed"
        else:
            parts["model"]["revision"] = "f" * 40
    install_scheduler(monkeypatch, isolated, mutation=mutate)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


@pytest.mark.parametrize("field,value", [("ctx_len", 32768), ("backend", "llamacpp"),
                                       ("kv_dtype", "fp8"), ("n_decode", 64),
                                       ("reps", 1), ("seeds", [9])])
def test_schedule_rejects_child_measurement_config_drift(isolated, monkeypatch, field, value):
    install_scheduler(monkeypatch, isolated,
                      mutation=lambda parts: parts["config"].update({field: value}))
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


def test_schedule_rejects_child_protocol_hash_mismatch(isolated, monkeypatch):
    install_scheduler(monkeypatch, isolated,
                      result_mutation=lambda result: result["protocol"].update(warmup="changed"))
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


@pytest.mark.parametrize("target", ["missing_sample", "duplicate_repetition", "wrong_seed",
                                    "output_count", "phase_rate", "input_hash"])
def test_schedule_rejects_coherently_rehashed_invalid_performance_samples(isolated, monkeypatch, target):
    def mutation(result):
        raw_path = isolated / result["raw_evidence"]["path"]
        raw = json.loads(raw_path.read_text())
        for record in (raw, result):
            if target == "missing_sample":
                record["samples"].pop()
            elif target == "duplicate_repetition":
                record["samples"][1]["repetition"] = 0
            elif target == "wrong_seed":
                record["samples"][0]["seed"] = 9
            elif target == "output_count":
                record["samples"][0]["output_tokens"] = 64
            elif target == "phase_rate":
                record["samples"][0]["prefill_tok_s"] = 1.0
            else:
                for sample in record["samples"]:
                    sample["input_sha256"] = "invalid-digest"
        C.write_record(raw_path, raw)
        result["raw_evidence"]["sha256"] = C.file_hash(raw_path)

    install_scheduler(monkeypatch, isolated, result_mutation=mutation)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


def test_schedule_rejects_valid_sha_for_wrong_exact_seeded_input(isolated, monkeypatch):
    def mutation(result):
        raw_path = isolated / result["raw_evidence"]["path"]
        raw = json.loads(raw_path.read_text())
        cfg = C.canonical_identity(result["identity"])["config"]
        wrong_hash = C.content_hash(C.synthetic_token_ids(6, cfg["ctx_len"], cfg["seeds"][0] + 100))
        for record in (raw, result):
            for sample in record["samples"]:
                sample["input_sha256"] = wrong_hash
        C.write_record(raw_path, raw)
        result["raw_evidence"]["sha256"] = C.file_hash(raw_path)

    install_scheduler(monkeypatch, isolated, result_mutation=mutation)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


@pytest.mark.parametrize("target", ["prefill", "decode", "boundary"])
def test_schedule_binds_coherent_phase_rates_to_native_numeric_evidence(isolated, monkeypatch, target):
    def mutation(result):
        raw_path = isolated / result["raw_evidence"]["path"]
        raw = json.loads(raw_path.read_text())
        for record in (raw, result):
            for sample in record["samples"]:
                if target == "boundary":
                    sample["timing_boundary"] = "different interval"
                    continue
                sample[f"{target}_s"] *= 2
                sample["prefill_tok_s"] = sample["input_tokens"] / sample["prefill_s"]
                sample["decode_tok_s"] = sample["decode_steps"] / sample["decode_s"]
                sample["native_request_s"] = sample["prefill_s"] + sample["decode_s"]
                sample["native_output_tok_s"] = sample["output_tokens"] / sample["native_request_s"]
            for field in ("prefill_tok_s", "decode_tok_s"):
                record[field] = record["samples"][0][field]
        C.write_record(raw_path, raw)
        result["raw_evidence"]["sha256"] = C.file_hash(raw_path)

    install_scheduler(monkeypatch, isolated, result_mutation=mutation)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated))
    with pytest.raises(ValueError):
        R.main()


@pytest.mark.parametrize("target", ["example_id", "input_hash", "false_hit", "generated_hash", "termination"])
def test_quality_schedule_reconstructs_self_hashed_samples_from_manifest_and_answers(isolated, monkeypatch, target):
    path, _ = quality_fixture(isolated)

    def mutation(result):
        raw_path = isolated / result["raw_evidence"]["path"]
        raw = json.loads(raw_path.read_text())
        for record in (raw, result):
            sample = record["samples"][0]
            if target == "example_id":
                sample["example_id"] = "single:1:9"
            elif target == "input_hash":
                sample["input_sha256"] = "0" * 64
            elif target == "false_hit":
                sample["hit"] = False
            elif target == "termination":
                sample["termination"] = "running"
            else:
                sample["generated_sha256"] = "0" * 64
        C.write_record(raw_path, raw)
        result["raw_evidence"]["sha256"] = C.file_hash(raw_path)

    install_scheduler(monkeypatch, isolated, mode="quality", result_mutation=mutation)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated, mode="quality", quality=[path]))
    with pytest.raises(ValueError):
        R.main()


@pytest.mark.parametrize("target", ["quality_run", "manifest", "examples", "limit_per_task"])
def test_quality_schedule_rejects_child_manifest_or_subset_drift(isolated, monkeypatch, target):
    path, _ = quality_fixture(isolated)

    def mutation(parts):
        if target == "quality_run":
            parts["protocol"]["quality_run"] = "0" * 64
        elif target == "manifest":
            parts["protocol"]["manifest"]["sha256"] = "0" * 64
        else:
            parts["config"][target] = 1

    install_scheduler(monkeypatch, isolated, mode="quality", mutation=mutation)
    monkeypatch.setattr(sys, "argv", scheduler_args(isolated, mode="quality", quality=[path]))
    with pytest.raises(ValueError):
        R.main()


def test_resume_revalidates_child_semantics_after_matching_file_hash(isolated, monkeypatch):
    install_scheduler(monkeypatch, isolated)
    argv = scheduler_args(isolated)
    monkeypatch.setattr(sys, "argv", argv)
    assert R.main() == 0
    path = next(isolated.glob("benchmarks/validation/competitors/*/schedule.json"))
    schedule = json.loads(path.read_text())
    entry = schedule["cells"][0]
    child_path = isolated / entry["path"]
    child = json.loads(child_path.read_text())
    identity = C.canonical_identity(child["identity"])
    identity["config"]["ctx_len"] = 32768
    child["identity"] = C.export_identity(identity)
    child["run_identity"] = C.content_hash(identity)
    child_path.write_text(json.dumps(child) + "\n")
    entry["sha256"] = C.file_hash(child_path)
    entry["run_identity"] = child["run_identity"]
    C.write_record(path, schedule)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises((ValueError, SystemExit)):
        R.main()


@pytest.mark.parametrize("target", ["run_identity", "protocol", "terminal_status", "partial_complete",
                                    "extra_cell", "cell_order", "entry_status", "timeout_status", "cell_path"])
def test_resume_rejects_inconsistent_terminal_schedule_container(isolated, monkeypatch, target):
    commands = install_scheduler(monkeypatch, isolated)
    argv = scheduler_args(isolated)
    monkeypatch.setattr(sys, "argv", argv)
    assert R.main() == 0
    path = next(isolated.glob("benchmarks/validation/competitors/*/schedule.json"))
    schedule = json.loads(path.read_text())
    if target == "run_identity":
        schedule["run_identity"] = "0" * 64
    elif target == "protocol":
        schedule["protocol"]["version"] = 99
    elif target == "terminal_status":
        schedule["status"] = "complete-with-failures"
    elif target == "partial_complete":
        schedule["cells"] = schedule["cells"][:1]
    elif target == "extra_cell":
        schedule["cells"].append(copy.deepcopy(schedule["cells"][0]))
    elif target == "cell_order":
        schedule["cells"][0], schedule["cells"][1] = schedule["cells"][1], schedule["cells"][0]
    elif target == "entry_status":
        schedule["cells"][0]["status"] = "out_of_memory"
    elif target == "timeout_status":
        schedule["cells"][0].update(observation_status="timeout", returncode=-15)
    else:
        entry = schedule["cells"][0]
        original = isolated / entry["path"]
        duplicate = original.with_name("99.json")
        duplicate.write_bytes(original.read_bytes())
        entry["path"] = duplicate.relative_to(isolated).as_posix()
    path.write_text(json.dumps(schedule) + "\n")
    launched_count = len(commands)
    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    with pytest.raises((ValueError, SystemExit)):
        R.main()
    assert len(commands) == launched_count


def test_manifest_roundtrip_returns_exact_original_ids(isolated):
    path, raw_path = quality_fixture(isolated)
    public, identity, ref, examples = N.load_manifest(path)
    assert identity == C.canonical_identity(public["identity"])
    assert ref["sha256"] == C.content_hash(examples)
    assert ref["file_sha256"] == C.file_hash(raw_path.parent / "manifest.json")
    assert len(examples) == 6 and examples[0]["input_ids"] == [0, 3, 4]


@pytest.mark.parametrize("target", ["public_identity", "raw_hash", "manifest_hash", "input_hash"])
def test_manifest_rejects_identity_and_byte_hash_changes(isolated, target):
    path, raw_path = quality_fixture(isolated)
    if target == "public_identity":
        public = json.loads(path.read_text())
        public["run_identity"] = "0" * 64
        path.write_text(json.dumps(public))
    elif target == "raw_hash":
        raw_path.write_text(raw_path.read_text() + " ")
    else:
        manifest_path = raw_path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["examples"][0]["input_ids"] = [0, 3, 5]
        C.write_record(manifest_path, manifest)
        if target == "input_hash":
            rewrite_quality(path, raw_path, raw_mutator=lambda raw: raw["manifest"].update(
                sha256=C.content_hash(manifest["examples"])))
    with pytest.raises(ValueError):
        N.load_manifest(path)


@pytest.mark.parametrize("target", ["raw_identity", "raw_identity_config", "raw_protocol", "raw_status",
                                    "public_manifest", "count", "duplicate_id"])
def test_manifest_rejects_rehashed_but_mismatched_evidence(isolated, target):
    path, raw_path = quality_fixture(isolated)
    kwargs = {}
    if target == "raw_identity":
        kwargs["raw_mutator"] = lambda raw: raw.update(run_identity="0" * 64)
    elif target == "raw_identity_config":
        kwargs["raw_mutator"] = lambda raw: raw["identity"]["config"].update(ctx_len=32768)
    elif target == "raw_protocol":
        kwargs["raw_mutator"] = lambda raw: raw["protocol"].update(scorer="different")
    elif target == "raw_status":
        kwargs["raw_mutator"] = lambda raw: raw.update(status="incomplete")
    elif target == "public_manifest":
        kwargs["public_mutator"] = lambda public: public["manifest"].update(sha256="0" * 64)
    elif target == "count":
        kwargs["raw_mutator"] = lambda raw: raw["manifest"].update(count=7)
    else:
        kwargs["examples_mutator"] = lambda examples: examples[1].update(example_id=examples[0]["example_id"])
    rewrite_quality(path, raw_path, **kwargs)
    with pytest.raises(ValueError):
        N.load_manifest(path)


@pytest.mark.parametrize("target", ["task", "seed", "index", "negative_token", "boolean_token"])
def test_manifest_rejects_self_hashed_but_invalid_example_metadata(isolated, target):
    def mutate(examples):
        if target in {"task", "seed", "index"}:
            examples[0][target] = "multikey" if target == "task" else 2
        else:
            examples[0]["input_ids"][1] = -1 if target == "negative_token" else True
            examples[0]["input_sha256"] = C.content_hash(examples[0]["input_ids"])

    path, _ = quality_fixture(isolated, examples_mutator=mutate)
    with pytest.raises(ValueError):
        N.load_manifest(path)


def test_verified_awq_compares_actual_local_weight_and_tokenizer_hashes(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    remote.mkdir()
    local.mkdir()
    values = {"config.json": b"{}", "tokenizer.json": b'{"vocab":"original"}',
              "generation_config.json": b'{"eos_token_id":1}', "model.safetensors": b"weight-data"}
    for name, data in values.items():
        (remote / name).write_bytes(data)
        (local / name).write_bytes(data)
    calls = []

    class HfApi:
        def __init__(self, *, token):
            assert token is False

        def model_info(self, name, *, revision, files_metadata):
            calls.append(name)
            assert revision == "272b3bde867b606760447deb9a4d2719fbdfd3ae" and files_metadata
            return SimpleNamespace(sha=revision, siblings=[SimpleNamespace(
                rfilename=name, lfs=SimpleNamespace(sha256=C.file_hash(remote / name))
                if name.endswith(".safetensors") else None) for name in values])

    def download(name, filename, *, revision, token, cache_dir):
        assert token is False and revision == "272b3bde867b606760447deb9a4d2719fbdfd3ae"
        return str(remote / filename)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=HfApi, hf_hub_download=download))
    pinned = B.verified_awq_model(str(local))
    assert calls and pinned["files"]["model.safetensors"] == C.file_hash(local / "model.safetensors")
    for name in ("tokenizer.json", "model.safetensors"):
        (local / name).write_bytes(b"changed")
        with pytest.raises(ValueError, match="artifacts differ"):
            B.verified_awq_model(str(local))
        (local / name).write_bytes(values[name])


def gguf_string(value):
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def gguf_bytes(entries, version=3):
    result = b"GGUF" + struct.pack("<IQQ", version, 0, len(entries))
    for name, kind, value in entries:
        result += gguf_string(name) + struct.pack("<I", kind) + value
    return result


def test_gguf_metadata_reads_scalar_strings_and_vocabulary_only(tmp_path):
    tokens = ["BOS", "EOS", "▁word"]
    payload = gguf_bytes([("architecture", 8, gguf_string("llama")),
                          ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 0)),
                          ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) +
                           b"".join(map(gguf_string, tokens)))])
    path = tmp_path / "metadata.gguf"
    path.write_bytes(payload + b"tensor-payload-is-not-metadata")
    assert N.gguf_metadata(path) == {"architecture": "llama", "tokenizer.ggml.bos_token_id": 0,
                                     "tokenizer.ggml.tokens": tokens}


@pytest.mark.parametrize("payload", [b"GGUF", gguf_bytes([], version=1),
                                     b"GGUF" + struct.pack("<IQQ", 3, 0, 1),
                                     gguf_bytes([("value", 4, b"\x01")]),
                                     gguf_bytes([("text", 8, struct.pack("<Q", 5) + b"ab")]),
                                     gguf_bytes([("array", 9, struct.pack("<IQ", 4, 2) + struct.pack("<I", 1))])])
def test_gguf_metadata_rejects_unsupported_or_truncated_inputs(tmp_path, payload):
    path = tmp_path / "truncated.gguf"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        N.gguf_metadata(path)


class Tokenizer:
    vocabulary = ("B", "E", "T", "A", "C", "D")
    all_special_ids = (0, 1, 2)
    bos_token_id = 0

    def __init__(self, directory):
        self.name_or_path = str(directory)

    def convert_ids_to_tokens(self, index):
        return self.vocabulary[index]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return "".join(self.vocabulary[index] for index in ids)


def mapping_fixture(tmp_path):
    model = tmp_path / "tokenizer"
    model.mkdir()
    (model / "generation_config.json").write_text('{"eos_token_id":[1,2]}')
    tokenizer = Tokenizer(model)
    metadata = {"tokenizer.ggml.tokens": list(tokenizer.vocabulary),
                "tokenizer.ggml.bos_token_id": 0, "tokenizer.ggml.eos_token_id": 1}
    (tmp_path / "server.log").write_text("EOG token = 2\n")
    calls = []

    def request(route, payload):
        assert route == "/detokenize" and set(payload) == {"tokens"}
        calls.append(payload["tokens"])
        return {"content": tokenizer.decode(payload["tokens"], skip_special_tokens=False,
                                             clean_up_tokenization_spaces=False)}

    server = SimpleNamespace(directory=tmp_path, request=request)
    examples = [{"input_ids": [0, 3, 4]}, {"input_ids": [0, 3, 5]}]
    return metadata, tokenizer, examples, server, calls


def test_gguf_mapping_checks_every_used_special_id_and_every_prompt(tmp_path):
    metadata, tokenizer, examples, server, calls = mapping_fixture(tmp_path)
    result = N.validate_gguf_tokens(metadata, tokenizer, examples, server)
    assert result["status"] == "complete" and result["checked_prompts"] == 2
    assert result["checked_used_ids"] == 4 and result["checked_special_ids"] == 3
    assert result["eos_ids"] == [1, 2] and result["input_manifest_sha256"] == C.content_hash(examples)
    assert calls == [e["input_ids"] for e in examples]


@pytest.mark.parametrize("target", ["used_id", "unused_special_id", "bos", "eos", "second_full_prompt"])
def test_gguf_mapping_rejects_vocab_full_prompt_and_eos_differences(tmp_path, target):
    metadata, tokenizer, examples, server, _ = mapping_fixture(tmp_path)
    if target == "used_id":
        metadata["tokenizer.ggml.tokens"][5] = "different"
    elif target == "unused_special_id":
        metadata["tokenizer.ggml.tokens"][1] = "different"
    elif target == "bos":
        metadata["tokenizer.ggml.bos_token_id"] = 3
    elif target == "eos":
        metadata["tokenizer.ggml.eos_token_id"] = 4
    else:
        original_request = server.request
        server.request = lambda route, payload: {"content": "wrong"} if payload["tokens"][-1] == 5 else original_request(route, payload)
    with pytest.raises(ValueError):
        N.validate_gguf_tokens(metadata, tokenizer, examples, server)


def install_quality_server(monkeypatch, tokenizer, *, answers=None):
    calls = []
    answers = iter(answers or ["red blue"] * 6)

    class FakeServer:
        def __init__(self, command, directory, **kwargs):
            self.directory = directory

        def __enter__(self):
            (self.directory / "server.log").write_text("test server\n")
            C.write_record(self.directory / "runtime.json", vllm_runtime()["worker_observation"])
            return self

        def __exit__(self, *args):
            pass

        def request(self, route, payload):
            calls.append((route, copy.deepcopy(payload)))
            text = next(answers)
            return {"choices": [{"text": text, "token_ids": [3, 4], "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": len(payload["prompt"]), "completion_tokens": 2},
                    "metrics": {"time_to_first_token_ms": 100, "generation_time_ms": 200,
                                "queue_time_ms": 0}}

    def from_pretrained(path, *, local_files_only):
        assert local_files_only is True
        return tokenizer

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    monkeypatch.setattr(N, "Server", FakeServer)
    monkeypatch.setattr(N, "command_for", lambda args, **kwargs: ["mock-server"])
    monkeypatch.setattr(N, "resolved_runtime", lambda *args: {
        k: v for k, v in vllm_runtime().items() if k != "worker_observation"})
    monkeypatch.setattr(N, "marker", lambda *args: None)
    return calls


def test_quality_answers_reconstruct_public_outcomes_and_exact_requests(isolated, monkeypatch):
    path, _ = quality_fixture(isolated)
    answers = ["red", "missing", "red blue", "red", "blue", "red and blue"]
    calls = install_quality_server(monkeypatch, Tokenizer(isolated), answers=answers)
    output = isolated / "result.json"
    monkeypatch.setattr(sys, "argv", ["competitor_niah", "--backend", "vllm", "--kv-dtype", "auto",
                                      "--awq-model-path", "unused-model", "--quality", str(path),
                                      "--out", str(output)])
    assert N.main() == 0
    public = json.loads(output.read_text())
    raw_path = isolated / public["raw_evidence"]["path"]
    raw = json.loads(raw_path.read_text())
    assert public["raw_evidence"]["sha256"] == C.file_hash(raw_path)
    assert [s["hit"] for s in public["samples"]] == [True, False, True, False, False, True]
    assert len(calls) == len(raw["samples"]) == 6
    for sample, exported, answer, (_, payload) in zip(raw["samples"], public["samples"], answers, calls, strict=True):
        assert sample["hit"] == all(value in answer for value in sample["expected"])
        assert exported["generated_sha256"] == C.content_hash(answer)
        assert {k: v for k, v in sample.items() if k not in {"generated", "expected"}} == exported
        assert "generated" not in exported and "expected" not in exported
        assert payload["ignore_eos"] is False and payload["max_tokens"] == 128
        assert C.content_hash(payload["prompt"]) == exported["input_sha256"]
    canonical = C.canonical_identity(public["identity"])
    assert canonical == raw["identity"] and C.content_hash(canonical) == public["run_identity"]
    assert canonical["protocol_sha256"] == C.content_hash(public["protocol"])


@pytest.mark.parametrize("dtype", ["auto", "fp8"])
def test_runtime_accepts_verified_gpu_cache_precision(dtype):
    B.verify_runtime(vllm_runtime(dtype), "vllm", dtype, 131)


@pytest.mark.parametrize("target,value", [("cache_config_dtype", "fp8"),
                                         ("logical_cache_dtype", "torch.float32"),
                                         ("cache_tensor_devices", ["cpu"]),
                                         ("cache_tensor_bytes", 0),
                                         ("model_parameter_devices", ["cpu"]),
                                         ("offload_group_size", 1)])
def test_runtime_rejects_resolved_precision_or_device_mismatch(target, value):
    runtime = vllm_runtime()
    runtime["worker_observation"][target] = value
    with pytest.raises(ValueError):
        B.verify_runtime(runtime, "vllm", "auto", 131)


def test_runtime_rejects_unresolved_weight_kernel():
    runtime = vllm_runtime()
    runtime["weight_kernels"] = []
    with pytest.raises(ValueError):
        B.verify_runtime(runtime, "vllm", "auto", 131)


def test_quality_mapping_mismatch_is_explicitly_unsupported(isolated, monkeypatch):
    path, _ = quality_fixture(isolated)
    metadata, tokenizer, _, _, _ = mapping_fixture(isolated)
    install_quality_server(monkeypatch, tokenizer)
    gguf = isolated / "model.gguf"
    gguf.write_bytes(b"mock-GGUF")
    metadata["tokenizer.ggml.eos_token_id"] = 4
    monkeypatch.setattr(N, "gguf_metadata", lambda path: metadata)
    output = isolated / "unsupported.json"
    monkeypatch.setattr(sys, "argv", ["competitor_niah", "--backend", "llamacpp", "--kv-dtype", "q4_0",
                                      "--awq-model-path", "unused-model", "--quality", str(path),
                                      "--gguf", str(gguf), "--out", str(output)])
    assert N.main() == 1
    public = json.loads(output.read_text())
    assert public["status"] == "unsupported" and public["samples"] == []
