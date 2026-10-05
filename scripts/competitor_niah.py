"""Run deterministic competitors on an existing exact-ID quality manifest."""
from __future__ import annotations

import argparse
import json
import re
import struct
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from bench_common import (
    REPO_ROOT,
    canonical_identity,
    content_hash,
    error_code,
    export_identity,
    file_hash,
    make_identity,
    provenance,
    validate_export,
    write_record,
)
from bench_competitor import backend_identity, resolved_runtime, verified_awq_model, verify_runtime
from competitor_backend import (
    Server,
    UnsupportedConfiguration,
    add_backend_arguments,
    command_for,
    compile_workers,
    completion_payload,
    marker,
    parse_completion,
)

ANSWER_DECODER = {"tokenizer": "same-pinned-HF-tokenizer", "input": "returned-output-token-IDs",
                  "skip_special_tokens": True, "clean_up_tokenization_spaces": True}
PRIVATE_SAMPLE_FIELDS = {"generated", "expected", "generated_ids", "native_text"}
TOKEN_MAPPING_METHOD = "every-vocabulary-token-piece plus used/special-ID coverage and every full decoded prompt"
QUALITY_STOP_POLICY = {"eos_source": "pinned-HF-generation-config",
                       "stop_or_eos": "terminal EOS; no earlier EOS",
                       "length": "exact output cap; no EOS",
                       "precedence": "EOS overrides length in pinned vLLM0.30 and llama.cpp b11382"}


def decode_answer(tokenizer, token_ids):
    """Match FlashQuest's pinned HF answer decoder, independently of server text."""
    if (not isinstance(token_ids, list) or not token_ids or
            any(type(x) is not int or not 0 <= x < len(tokenizer) for x in token_ids)):
        raise ValueError("invalid generated IDs for the pinned answer tokenizer")
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)


def generation_eos_ids(tokenizer):
    config = json.loads((Path(tokenizer.name_or_path) / "generation_config.json").read_text())
    ids = config["eos_token_id"]
    ids = ids if isinstance(ids, list) else [ids]
    if not ids or any(type(x) is not int or not 0 <= x < len(tokenizer) for x in ids):
        raise ValueError("invalid pinned generation EOS policy")
    return set(ids)


def validate_quality_stop(token_ids, termination, eos_ids, max_new_tokens):
    """Verify quality's first-EOS-or-cap behavior from the returned sampled IDs."""
    if (type(max_new_tokens) is not int or max_new_tokens < 1 or
            not isinstance(token_ids, list) or not token_ids or len(token_ids) > max_new_tokens or
            any(type(token) is not int or token < 0 for token in token_ids)):
        raise ValueError("invalid quality output token count/IDs")
    if any(token in eos_ids for token in token_ids[:-1]):
        raise ValueError("quality output continued after an earlier EOS")
    if termination in {"stop", "eos"}:
        if token_ids[-1] not in eos_ids:
            raise ValueError("quality stop lacks the pinned terminal EOS")
    elif termination == "length":
        # Both pinned backends give EOS precedence when EOS also lands at the cap.
        if len(token_ids) != max_new_tokens or token_ids[-1] in eos_ids:
            raise ValueError("quality length stop differs from the exact cap/no-EOS policy")
    else:
        raise ValueError("unsupported quality termination")


def gguf_metadata(path):
    """Read only GGUF metadata, including the complete vocabulary, without dependencies."""
    sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
    with Path(path).open("rb") as f:
        def unpack(fmt):
            n = struct.calcsize("<" + fmt)
            data = f.read(n)
            if len(data) != n:
                raise ValueError("truncated GGUF metadata")
            return struct.unpack("<" + fmt, data)[0]

        def string():
            n = unpack("Q")
            if n > 64 * 2**20:
                raise ValueError("oversized GGUF metadata string")
            value = f.read(n)
            if len(value) != n:
                raise ValueError("truncated GGUF string")
            return value.decode("utf-8")

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = unpack("I"), unpack("Q")
                if count > 2**24:
                    raise ValueError("oversized GGUF metadata array")
                return [value(element) for _ in range(count)]
            return unpack(sizes[kind])

        if f.read(4) != b"GGUF" or unpack("I") not in (2, 3):
            raise ValueError("unsupported GGUF format")
        unpack("Q")  # tensor count; tensor payload is never read here
        count = unpack("Q")
        if count > 10000:
            raise ValueError("oversized GGUF metadata")
        result = {}
        for _ in range(count):
            name, kind = string(), unpack("I")
            result[name] = value(kind)
        return result


def validate_gguf_tokens(metadata, tokenizer, examples, server):
    vocabulary = metadata["tokenizer.ggml.tokens"]
    used = {x for e in examples for x in e["input_ids"]}
    expected_eos = generation_eos_ids(tokenizer)
    # all_special_ids omits some added control tokens in this pinned HF artifact.
    special = (set(tokenizer.all_special_ids) | expected_eos |
               {index for index, token in tokenizer.added_tokens_decoder.items() if token.special})
    # Generated ordinary IDs need the same vocabulary contract as prompt IDs.
    if len(vocabulary) != len(tokenizer):
        raise UnsupportedConfiguration("GGUF vocabulary size differs from the answer tokenizer")
    checked = set(range(len(tokenizer))) | used | special
    mismatched = [x for x in checked if type(x) is not int or not 0 <= x < len(vocabulary) or
                  vocabulary[x] != tokenizer.convert_ids_to_tokens(x)]
    if mismatched:
        raise UnsupportedConfiguration("GGUF token-ID vocabulary differs from the manifest tokenizer")
    if metadata.get("tokenizer.ggml.bos_token_id") != tokenizer.bos_token_id:
        raise UnsupportedConfiguration("GGUF BOS ID differs")
    eos = {metadata.get(f"tokenizer.ggml.{name}_token_id") for name in ("eos", "eot", "eom")}
    eos.discard(None)
    # Llama BPE also registers EOG tokens from vocabulary names; inspect resolved loader output.
    eos.update(map(int, re.findall(r"EOG token\s*=\s*(\d+)", (server.directory / "server.log").read_text())))
    if eos != expected_eos:
        raise UnsupportedConfiguration("GGUF EOS policy differs from the quality model")
    for example in examples:
        ids = example["input_ids"]
        decoded = server.request("/detokenize", {"tokens": ids})["content"]
        expected = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != expected:
            raise UnsupportedConfiguration("GGUF full decoded prompt differs")
    return {"status": "complete", "checked_used_ids": len(used), "checked_special_ids": len(special),
            "checked_vocabulary_ids": len(vocabulary),
            "checked_prompts": len(examples), "eos_ids": sorted(eos),
            "input_manifest_sha256": content_hash(examples),
            "method": TOKEN_MAPPING_METHOD}


def load_manifest(quality_path):
    quality = json.loads(quality_path.read_text())
    identity = canonical_identity(quality["identity"])
    if quality["run_identity"] != content_hash(identity) or quality["status"] != "complete":
        raise ValueError("quality evidence identity/status invalid")
    raw_ref = quality["raw_evidence"]
    raw_path = REPO_ROOT / raw_ref["path"]
    expected_path = REPO_ROOT / "artifacts" / "quality" / quality["run_identity"] / "raw.json"
    if raw_path.resolve() != expected_path.resolve() or not raw_path.resolve().is_relative_to((REPO_ROOT / "artifacts" / "quality").resolve()):
        raise ValueError("quality raw evidence path is outside its run directory")
    if file_hash(raw_path) != raw_ref["sha256"]:
        raise ValueError("quality raw evidence changed")
    raw = json.loads(raw_path.read_text())
    if (raw["run_identity"] != quality["run_identity"] or raw["status"] != "complete" or
            canonical_identity(raw["identity"]) != identity or raw["manifest"] != quality["manifest"] or
            raw["protocol"] != quality["protocol"] or content_hash(raw["protocol"]) != identity["protocol_sha256"]):
        raise ValueError("raw/public quality evidence differs")
    ref = raw["manifest"]
    path = raw_path.parent / "manifest.json"
    manifest = json.loads(path.read_text())
    examples = manifest["examples"]
    if content_hash(examples) != ref["sha256"]:
        raise ValueError("quality manifest changed")
    cfg = identity["config"]
    expected_ids = {f"{task}:{seed}:{index}" for task in cfg["tasks"] for seed in cfg["seeds"]
                    for index in range(cfg["n_samples"])}
    if (len(examples) != len(expected_ids) or ref["count"] != len(examples) or
            {e["example_id"] for e in examples} != expected_ids or
            cfg["manifest_sha256"] != content_hash(examples) or
            max(len(e["input_ids"]) for e in examples) != cfg["actual_max_prompt_tokens"] or
            cfg["cache_capacity"] != cfg["actual_max_prompt_tokens"] + cfg["max_new_tokens"]):
        raise ValueError("quality manifest count/input capacity differs from configuration")
    if manifest["run_identity"] != quality["run_identity"]:
        raise ValueError("quality manifest identity differs")
    for e in examples:
        if (e["task"] not in cfg["tasks"] or type(e["seed"]) is not int or e["seed"] not in cfg["seeds"] or
                type(e["index"]) is not int or not 0 <= e["index"] < cfg["n_samples"] or
                e["example_id"] != f'{e["task"]}:{e["seed"]}:{e["index"]}' or
                not e["input_ids"] or any(type(t) is not int or t < 0 for t in e["input_ids"]) or
                len(e["input_ids"]) > cfg["ctx_len"] or
                not isinstance(e["expected"], list) or not e["expected"] or
                any(not isinstance(s, str) or not s for s in e["expected"]) or
                content_hash(e["input_ids"]) != e["input_sha256"]):
            raise ValueError("manifest input hash/count mismatch")
    from phase6_run_ruler_4k_int4 import export_result, screen_verdict, validate_cell
    expected_cells = {(task, seed, arm) for task in cfg["tasks"] for seed in cfg["seeds"]
                      for arm in ["dense", *[f"int4-r{r:g}" for r in cfg["retentions"]]]}
    if (len(raw["cells"]) != len(expected_cells) or
            {(c["task"], c["seed"], c["arm"]) for c in raw["cells"]} != expected_cells):
        raise ValueError("quality evidence cells are incomplete")
    for cell in raw["cells"]:
        matched = [e for e in examples if e["task"] == cell["task"] and e["seed"] == cell["seed"]]
        validate_cell(cell, matched, cfg["max_new_tokens"])
    if export_result(raw, raw_path) != quality or screen_verdict(raw["cells"], SimpleNamespace(**cfg)) != raw["screen"]:
        raise ValueError("quality public samples/scorer/screen differ from raw evidence")
    return quality, identity, {**ref, "file_sha256": file_hash(path), "path": path.relative_to(REPO_ROOT).as_posix()}, examples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_backend_arguments(p)
    p.add_argument("--quality", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit-per-task", type=int)
    args = p.parse_args()
    if args.out.exists():
        p.error("preserve existing attempt; choose a fresh output")
    if args.limit_per_task is not None and args.limit_per_task < 1:
        p.error("limit-per-task must be positive")
    quality, original, manifest_ref, examples = load_manifest(args.quality.resolve())
    if verified_awq_model(args.awq_model_path) != original["model"]:
        raise ValueError("competitor tokenizer differs from the quality manifest model")
    if args.limit_per_task:
        examples = [e for e in examples if e["index"] < args.limit_per_task]
    protocol = {"version": 2, "quality_run": quality["run_identity"], "manifest": manifest_ref,
                "max_new_tokens": 128, "scorer": "all expected substrings in decoded answer",
                "answer_decoder": ANSWER_DECODER,
                "termination_validation": QUALITY_STOP_POLICY,
                "generation": "greedy EOS or output limit; exact input IDs; no wrapping",
                "timing": "native phases and client times; quality may stop at EOS"}
    config = {"backend": args.backend, "kv_dtype": args.kv_dtype,
              "ctx_len": original["config"]["ctx_len"], "capacity": original["config"]["cache_capacity"],
              "max_new_tokens": 128, "examples": len(examples), "limit_per_task": args.limit_per_task,
              "compile_workers": compile_workers(args.backend),
              "backend_identity": backend_identity(args), "gpu_utilization": args.gpu_utilization if args.backend == "vllm" else None}
    model = original["model"]
    if args.backend == "llamacpp":
        hashes = {Path(args.gguf).name: file_hash(Path(args.gguf))}
        model = {"model": "bartowski/Llama-3.2-3B-Instruct-GGUF", "revision": "5ab33fa94d1d04e903623ae72c95d1696f09f9e8",
                 "files": hashes, "content_sha256": content_hash(hashes)}
    record = {**make_identity(config, model, protocol, provenance()), "protocol": protocol,
              "status": "running", "samples": [], "token_mapping": None, "runtime": None}
    directory = REPO_ROOT / "artifacts" / "competitor-quality" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.awq_model_path, local_files_only=True)
        eos_ids = generation_eos_ids(tokenizer)
        with Server(command_for(args, capacity=config["capacity"]), directory, timeout=args.ready_timeout) as server:
            if args.backend == "llamacpp":
                record["token_mapping"] = validate_gguf_tokens(gguf_metadata(args.gguf), tokenizer, examples, server)
            else:
                # Exact tokenizer artifact fingerprints and server token-count checks apply to every request.
                record["token_mapping"] = {"status": "same-pinned-HF-tokenizer", "manifest_sha256": content_hash(examples)}
            route = "/v1/completions" if args.backend == "vllm" else "/completion"
            for i, example in enumerate(examples):
                ids = example["input_ids"]
                marker("request_start", i)
                start = time.perf_counter()
                response = server.request(route, completion_payload(args.backend, ids, 128, performance=False))
                elapsed = time.perf_counter() - start
                marker("request_end", i)
                sample = parse_completion(args.backend, response, len(ids), 128, performance=False)
                native_text, generated_ids = sample.pop("text"), sample.pop("tokens")
                validate_quality_stop(generated_ids, sample["termination"], eos_ids, 128)
                answer = decode_answer(tokenizer, generated_ids)
                hit = all(value in answer for value in example["expected"])
                sample.update(example_id=example["example_id"], task=example["task"], seed=example["seed"],
                              index=example["index"], input_sha256=example["input_sha256"], hit=hit,
                              generated_sha256=content_hash(answer), native_text_sha256=content_hash(native_text),
                              generated_ids_sha256=content_hash(generated_ids), client_request_s=elapsed)
                record["samples"].append({**sample, "generated": answer, "expected": example["expected"],
                                          "native_text": native_text, "generated_ids": generated_ids})
                write_record(directory / "raw.json", record)
            record["runtime"] = resolved_runtime(args.backend, (directory / "server.log").read_text())
            if args.backend == "vllm":
                record["runtime"]["worker_observation"] = json.loads((directory / "runtime.json").read_text())
            verify_runtime(record["runtime"], args.backend, args.kv_dtype, config["capacity"])
        record["status"] = "complete"
    except Exception as exc:  # noqa: BLE001
        record["status"] = "unsupported" if isinstance(exc, UnsupportedConfiguration) else error_code(exc)
        (directory / "error.log").write_text(f"{type(exc).__name__}: {exc}\n")
    raw_path = directory / "raw.json"
    write_record(raw_path, record)
    exported = {**record, "identity": export_identity(record["identity"]),
                "samples": [{k: v for k, v in sample.items() if k not in PRIVATE_SAMPLE_FIELDS} for sample in record["samples"]],
                "raw_evidence": {"path": raw_path.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(raw_path)}}
    validate_export(exported)
    write_record(args.out, exported)
    print(json.dumps({"run_identity": record["run_identity"], "status": record["status"], "examples": len(record["samples"])}))
    return int(record["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
