"""Matched dense/all-pages/sparse INT4 retrieval pilot with resumable evidence.

Importing this script never initializes CUDA. Raw prompts/output stay under
artifacts; exported results contain allowlisted evidence and content hashes.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bench_common import (
    DEFAULT_MODEL,
    REPO_ROOT,
    SCHEMA_VERSION,
    content_hash,
    error_code,
    export_identity,
    file_hash,
    make_identity,
    model_identity,
    provenance,
    source_identity,
    validate_export,
    write_record,
)

TASKS = ("single", "multikey", "multivalue")
SCREEN_PROTOCOL = {
    "version": 1, "kind": "retrieval-pilot", "scorer": "all-expected-substrings",
    "ratio_floor": 0.85, "dense_zero": "inconclusive", "generation": "greedy-eos-or-limit",
    "claims": "small retrieval screen only; no general quality or capacity claim",
}


@dataclass
class Runtime:
    loader: Callable
    cache_factory: Callable
    patcher: Callable
    evaluator: Callable
    attention_modules: Callable
    details: Callable
    cleanup: Callable
    manifest_builder: Callable


def build_manifest(tokenizer, args) -> list[dict]:
    from flashquest.eval.niah import make_prompt

    examples = []
    for task in args.tasks:
        for seed in args.seeds:
            for index in range(args.n_samples):
                prompt, expected = make_prompt(task, args.ctx_len, tokenizer, seed=seed * 10000 + index)
                ids = list(tokenizer(prompt).input_ids)
                if not ids or any(not isinstance(token, int) or token < 0 for token in ids):
                    raise ValueError("invalid manifest input tokens")
                examples.append({"example_id": f"{task}:{seed}:{index}", "task": task,
                                 "seed": seed, "index": index, "input_ids": ids,
                                 "input_sha256": content_hash(ids), "expected": list(expected)})
    return examples


def production_runtime() -> Runtime:
    import torch
    from transformers.models.llama.modeling_llama import LlamaAttention

    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.eval.runner import run_niah
    from flashquest.runtime.awq_load import load_awq_model

    def patch(model, cache, retention, args):
        pattern = torch.ones(model.config.num_hidden_layers, model.config.num_key_value_heads,
                             dtype=torch.bool)
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern, retention=retention,
            num_sinks=args.num_sinks, window_pages=args.window_pages, page_size=args.page_size,
        )

    def cleanup():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def details(model):
        import sys

        gemm = sys.modules.get("awq.modules.linear.gemm")
        weight_kernel = ("awq_ext" if getattr(gemm, "awq_ext", None) is not None else
                         "triton" if getattr(gemm, "TRITON_AVAILABLE", False) else "unknown")
        devices = sorted({str(t.device) for t in [*model.parameters(), *model.buffers()]})
        if not devices or any(not device.startswith("cuda") for device in devices):
            raise ValueError("model contains tensors outside the selected CUDA device")
        return {"cuda": torch.version.cuda, "weight_kernel": weight_kernel,
                "weight_modules": sorted({type(m).__name__ for m in model.modules()
                                           if type(m).__module__.startswith("awq.")}),
                "devices": devices, "dtype": str(model.dtype),
                "dense_attention": model.config._attn_implementation}

    def load(name, **kwargs):
        return load_awq_model(name, cache_dir=str(REPO_ROOT / "artifacts" / "hf-cache"), **kwargs)

    return Runtime(load, PersistentInt4KVCache, patch, run_niah,
                   lambda model: [m for m in model.modules() if isinstance(m, LlamaAttention)],
                   details, cleanup, build_manifest)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", help="immutable HF commit; resolved before model loading")
    parser.add_argument("--ctx-len", type=int, default=4096, help="nominal prompt budget")
    parser.add_argument("--n-samples", type=int, default=20, help="examples per task and seed")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    retention = parser.add_mutually_exclusive_group()
    retention.add_argument("--retentions", nargs="+", type=float)
    retention.add_argument("--retention", type=float, help="legacy single-retention alias")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seeds", nargs="+", type=int)
    seeds.add_argument("--seed", type=int, help="legacy single-seed alias")
    parser.add_argument("--num-sinks", type=int, default=4)
    parser.add_argument("--window-pages", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--out", type=Path, help="defaults to quality/<identity>/quality.json")
    parser.add_argument("--resume", action="store_true", help="resume an identical incomplete run")
    parser.add_argument("--require-screen-pass", action="store_true")
    parser.add_argument("--confirmation-protocol", type=Path,
                        help="hash-addressed frozen protocol; reject settings drift before loading")
    args = parser.parse_args(argv)
    args.retentions = args.retentions or ([args.retention, 1.0] if args.retention is not None
                                        else [0.20, 1.0])
    args.retentions = list(dict.fromkeys(args.retentions))
    if 1.0 not in args.retentions:
        args.retentions.append(1.0)
    args.seeds = args.seeds if args.seeds is not None else [args.seed if args.seed is not None else 0]
    if (args.ctx_len < 512 or not 1 <= args.n_samples <= 10000 or args.max_new_tokens < 1
            or args.page_size < 1 or args.num_sinks < 0 or args.window_pages < 0
            or any(not math.isfinite(r) or not 0 < r <= 1 for r in args.retentions)
            or any(s < 0 for s in args.seeds) or len(set(args.seeds)) != len(args.seeds)
            or len(set(args.tasks)) != len(args.tasks)):
        parser.error("invalid budgets, retentions, repeated tasks/seeds, or negative settings")
    return args


def screen_verdict(cells: list[dict], args) -> dict:
    results = {}
    complete = True
    for task in args.tasks:
        selected = {arm: [cell for cell in cells if cell["task"] == task and cell["arm"] == arm]
                    for arm in ["dense", *[f"int4-r{r:g}" for r in args.retentions]]}
        if any(len(group) != len(args.seeds) for group in selected.values()):
            complete = False
            continue
        dense = sum(cell["hits"] for cell in selected["dense"])
        arms = {}
        for retention in args.retentions:
            key = f"int4-r{retention:g}"
            hits = sum(cell["hits"] for cell in selected[key])
            arms[key] = {"hits": hits, "ratio": hits / dense if dense else None,
                         "screen_pass": hits >= 0.85 * dense if dense else None}
        results[task] = {"dense_hits": dense, "total": args.n_samples * len(args.seeds),
                         "baseline_status": "positive-inspect-absolute-quality" if dense else
                         "inconclusive-zero-hits", "arms": arms}
    if not complete:
        status = "incomplete"
    elif any(task["dense_hits"] == 0 for task in results.values()):
        status = "inconclusive"
    else:
        sparse = [value["screen_pass"] for task in results.values()
                  for arm, value in task["arms"].items() if arm != "int4-r1"]
        status = "pass" if sparse and all(sparse) else "fail"
    return {"status": status, "tasks": results}


def validate_cell(cell, examples, max_new_tokens):
    from flashquest.eval.niah import score

    samples = cell.get("samples", [])
    if len(samples) != len(examples) or cell.get("total") != len(examples):
        raise ValueError("incomplete quality cell")
    for sample, example in zip(samples, examples, strict=True):
        if (sample.get("example_id") != example["example_id"] or
                sample.get("prompt_tokens") != len(example["input_ids"]) or
                not isinstance(sample.get("hit"), bool) or
                sample.get("input_sha256") != example["input_sha256"] or
                not isinstance(sample.get("output_tokens"), int) or
                not 1 <= sample["output_tokens"] <= max_new_tokens or
                sample.get("termination") not in {"eos", "limit"} or
                not isinstance(sample.get("generated"), str)):
            raise ValueError("quality sample does not match manifest")
        if sample.get("expected") != example["expected"] or sample["hit"] != score(sample["generated"], example["expected"]):
            raise ValueError("quality answer or hit differs from manifest/scorer")
    if cell.get("hits") != sum(s["hit"] for s in samples):
        raise ValueError("quality hit count differs from samples")


def export_result(result: dict, raw_path: Path) -> dict:
    exported = {key: result[key] for key in ("schema_version", "run_identity", "identity",
                                            "protocol", "status", "error", "screen", "manifest")}
    exported["identity"] = export_identity(result["identity"])
    exported["raw_evidence"] = {"path": raw_path.relative_to(REPO_ROOT).as_posix(),
                                "sha256": file_hash(raw_path)}
    exported["cells"] = []
    for cell in result["cells"]:
        output = {k: cell[k] for k in ("task", "seed", "arm", "retention", "hits", "total", "wall_s")}
        output["samples"] = [{k: s[k] for k in ("example_id", "input_sha256", "prompt_tokens", "hit",
                                                "output_tokens", "termination")} |
                             {"generated_sha256": content_hash(s["generated"])} for s in cell["samples"]]
        exported["cells"].append(output)
    validate_export(exported)
    return exported


def run_quality(args, runtime: Runtime, resolved_model: dict, environment: dict) -> tuple[dict, Path]:
    model = cache = reset_cache = None
    originals = []
    result = None
    out_path = raw_path = None
    execution_started = False
    try:
        protocol = SCREEN_PROTOCOL
        frozen_source = None
        if args.confirmation_protocol is not None:
            from validation_stats import load_protocol, validate_run_settings

            protocol = load_protocol(args.confirmation_protocol)
            validate_run_settings(args, resolved_model, protocol)
            frozen_source = source_identity()
            if frozen_source.get("dirty") is not False:
                raise ValueError("confirmation requires a clean source snapshot before loading")
        model, tokenizer = runtime.loader(args.model, revision=resolved_model["revision"])
        actual_revision = getattr(model.config, "_commit_hash", None)
        if actual_revision is not None and actual_revision != resolved_model["revision"]:
            raise ValueError("loaded model revision differs from resolved identity")
        runtime_info = runtime.details(model)
        manifest = runtime.manifest_builder(tokenizer, args)
        if not manifest:
            raise ValueError("empty manifest")
        max_prompt = max(len(e["input_ids"]) for e in manifest)
        capacity = max_prompt + args.max_new_tokens
        if capacity > model.config.max_position_embeddings:
            raise ValueError("input plus output exceeds model capacity")
        config = {"model": resolved_model["model"], "revision": resolved_model["revision"],
                  "ctx_len": args.ctx_len, "actual_max_prompt_tokens": max_prompt,
                  "cache_capacity": capacity, "n_samples": args.n_samples, "tasks": args.tasks,
                  "seeds": args.seeds, "retentions": args.retentions, "kv_bits": 4,
                  "page_size": args.page_size, "num_sinks": args.num_sinks,
                  "window_pages": args.window_pages, "max_new_tokens": args.max_new_tokens,
                  "manifest_sha256": content_hash(manifest), "runtime": runtime_info}
        run = make_identity(config, resolved_model, protocol, environment, source=frozen_source)
        validate_export(run)
        directory = REPO_ROOT / "artifacts" / "quality" / run["run_identity"]
        raw_path = directory / "raw.json"
        out_path = args.out or (REPO_ROOT / "benchmarks" / "validation" / "quality" /
                                run["run_identity"] / "quality.json")
        result = {"schema_version": SCHEMA_VERSION, **run, "protocol": protocol,
                  "manifest": {"sha256": content_hash(manifest), "count": len(manifest),
                               "min_prompt_tokens": min(len(e["input_ids"]) for e in manifest),
                               "max_prompt_tokens": max_prompt},
                  "status": "incomplete", "error": None, "cells": [], "screen": {"status": "incomplete"}}
        if out_path.exists():
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if prior.get("run_identity") != run["run_identity"]:
                raise ValueError("output belongs to a different run identity")
            if not args.resume:
                raise ValueError("existing result requires --resume")
        if raw_path.exists():
            if not args.resume:
                raise ValueError("existing raw result requires --resume")
            previous = json.loads(raw_path.read_text(encoding="utf-8"))
            if previous.get("identity") != run["identity"]:
                raise ValueError("raw result identity differs")
            result["cells"] = previous["cells"]
        groups = {(task, seed): [e for e in manifest if e["task"] == task and e["seed"] == seed]
                  for task in args.tasks for seed in args.seeds}
        seen = set()
        for cell in result["cells"]:
            key = (cell["arm"], cell["task"], cell["seed"])
            if (key in seen or (cell["task"], cell["seed"]) not in groups or
                    cell["arm"] not in ["dense", *[f"int4-r{r:g}" for r in args.retentions]]):
                raise ValueError("duplicate or unexpected resume cell")
            validate_cell(cell, groups[(cell["task"], cell["seed"])], args.max_new_tokens)
            seen.add(key)
        write_record(directory / "manifest.json", {"run_identity": run["run_identity"],
                                                   "examples": manifest})
        write_record(REPO_ROOT / "benchmarks" / "validation" / "protocols" /
                     f"{content_hash(protocol)}.json", protocol)

        def save():
            result["screen"] = screen_verdict(result["cells"], args)
            write_record(raw_path, result)
            write_record(out_path, export_result(result, raw_path))

        execution_started = True
        save()
        arms = [("dense", None), *[(f"int4-r{r:g}", r) for r in args.retentions]]
        for arm, retention in arms:
            pending = [(task, seed) for task in args.tasks for seed in args.seeds
                       if (arm, task, seed) not in seen]
            if not pending:
                continue
            if retention is not None:
                if cache is None:
                    modules = runtime.attention_modules(model)
                    if len(modules) != model.config.num_hidden_layers:
                        raise ValueError("attention module count differs from model config")
                    originals = [(module, module.forward) for module in modules]
                    cfg = model.config
                    cache = runtime.cache_factory(
                        batch_size=1, num_layers=cfg.num_hidden_layers,
                        num_kv_heads=cfg.num_key_value_heads,
                        head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads,
                        max_seq_len=capacity, page_size=args.page_size, device=str(model.device),
                    )
                runtime.patcher(model, cache, retention, args)

                def reset_cache(_index, request_cache=cache):
                    request_cache._seen_tokens = [0] * request_cache.num_layers
            for task, seed in pending:
                examples = groups[(task, seed)]
                start = time.perf_counter()
                evaluated = runtime.evaluator(
                    model, tokenizer, task=task, n_samples=len(examples), ctx_len=args.ctx_len,
                    seed=seed, max_new_tokens=args.max_new_tokens, pre_sample=reset_cache,
                    examples=examples,
                )
                for sample, example in zip(evaluated["samples"], examples, strict=True):
                    sample["input_sha256"] = example["input_sha256"]
                cell = {"arm": arm, "retention": retention, "task": task, "seed": seed,
                        "hits": evaluated["hits"], "total": evaluated["total"],
                        "samples": evaluated["samples"], "wall_s": time.perf_counter() - start}
                validate_cell(cell, examples, args.max_new_tokens)
                result["cells"].append(cell)
                print(f"{arm} {task} seed={seed}: {cell['hits']}/{cell['total']}", flush=True)
                save()
        result["status"] = "complete"
        save()
        return result, out_path
    except Exception as exc:
        if execution_started:
            result["status"] = "execution_error"
            result["error"] = error_code(exc)
            (raw_path.parent / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
            save()
        raise
    finally:
        for module, forward in originals:
            module.forward = forward
        originals.clear()
        reset_cache = None
        cache = model = None
        runtime.cleanup()


def main(argv=None, *, runtime=None, resolved_model=None, environment=None) -> int:
    args = parse_args(argv)
    try:
        resolved_model = resolved_model if resolved_model is not None else model_identity(args.model, args.revision)
        environment = environment if environment is not None else provenance()
        runtime = runtime if runtime is not None else production_runtime()
        result, path = run_quality(args, runtime, resolved_model, environment)
        label = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path.name
        print(f"Screen: {result['screen']['status']}; result: {label}")
        return int(args.require_screen_pass and result["screen"]["status"] != "pass")
    except Exception as exc:  # noqa: BLE001 — CLI reports backend/import failures
        print(f"Quality execution failed: {error_code(exc)} ({type(exc).__name__})", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
