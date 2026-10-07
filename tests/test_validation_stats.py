"""Confirmatory uncertainty and evidence-integrity checks without a GPU."""
import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_common as C
import phase6_run_ruler_4k_int4 as Q
import validation_stats as S

PROTOCOL_PATH = (C.REPO_ROOT / "benchmarks/validation/protocols" /
                 "9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json")
PILOT_PATH = (C.REPO_ROOT / "benchmarks/validation/quality" /
              "dd8f60de24c2957a2120473d9ff3a194b64b22d85b2adfecca3aab485092f104/quality.json")


@pytest.fixture
def protocol():
    return S.load_protocol(PROTOCOL_PATH)


def quality_fixture(root, protocol, context, *, dense_misses=0, sparse_misses=0,
                    complete=True, source_marker="a", weight_kernel="test"):
    """Synthetic local/public evidence obeying the actual exporter/scorer contract."""
    model = C.canonical_identity(json.loads(PILOT_PATH.read_text())["identity"])["model"]
    retention = dict(S.CONTEXT_RETENTIONS)[context]
    examples = [{"example_id": f"{task}:{seed}:{i}", "task": task, "seed": seed, "index": i,
                 "input_ids": [context, seed, i + 1],
                 "input_sha256": C.content_hash([context, seed, i + 1]), "expected": ["answer"]}
                for task in S.TASKS for seed in S.SEEDS for i in range(20)]
    files = {row["path"]: row["sha256"] for row in protocol["input_sources"]}
    source = {"commit": source_marker * 40, "dirty": False, "files": files,
              "content_sha256": C.content_hash(files)}
    cfg = {"model": model["model"], "revision": model["revision"], "ctx_len": context,
           "actual_max_prompt_tokens": 3, "cache_capacity": 131, "n_samples": 20,
           "tasks": S.TASKS, "seeds": S.SEEDS, "retentions": [retention, 1.0],
           "kv_bits": 4, "page_size": 64, "num_sinks": 4, "window_pages": 2,
           "max_new_tokens": 128, "manifest_sha256": C.content_hash(examples),
           "runtime": {"weight_kernel": weight_kernel, "dtype": "torch.float16"}}
    run = C.make_identity(cfg, model, protocol, {"python": "test", "packages": {}}, source=source)
    cells = []
    for arm in ["dense", f"int4-r{retention:g}", "int4-r1"]:
        for task in S.TASKS:
            for seed in S.SEEDS:
                samples = []
                for example in [e for e in examples if e["task"] == task and e["seed"] == seed]:
                    position = (seed - 1) * 20 + example["index"]
                    misses = dense_misses if arm == "dense" else sparse_misses if arm != "int4-r1" else 0
                    hit = position >= misses
                    samples.append({"example_id": example["example_id"],
                                    "input_sha256": example["input_sha256"], "prompt_tokens": 3,
                                    "hit": hit, "expected": example["expected"],
                                    "generated": "answer" if hit else "miss",
                                    "output_tokens": 2, "termination": "eos"})
                cells.append({"task": task, "seed": seed, "arm": arm,
                              "retention": None if arm == "dense" else
                              retention if arm != "int4-r1" else 1.0,
                              "hits": sum(s["hit"] for s in samples), "total": 20,
                              "wall_s": .01, "samples": samples})
    if not complete:
        cells = cells[:1]
    raw = {"schema_version": C.SCHEMA_VERSION, **run, "protocol": protocol,
           "manifest": {"sha256": C.content_hash(examples), "count": 300,
                        "min_prompt_tokens": 3, "max_prompt_tokens": 3},
           "status": "complete" if complete else "incomplete", "error": None, "cells": cells,
           "screen": Q.screen_verdict(cells, SimpleNamespace(**cfg))}
    raw_path = root / "artifacts/quality" / run["run_identity"] / "raw.json"
    C.write_record(raw_path, raw)
    C.write_record(raw_path.parent / "manifest.json", {"run_identity": run["run_identity"],
                                                       "examples": examples})
    public = {key: raw[key] for key in ("schema_version", "run_identity", "protocol", "manifest",
                                        "status", "error", "screen")}
    public["identity"] = C.export_identity(raw["identity"])
    public["raw_evidence"] = {"path": raw_path.relative_to(root).as_posix(),
                              "sha256": C.file_hash(raw_path)}
    public["cells"] = []
    for cell in cells:
        exported = {key: cell[key] for key in
                    ("task", "seed", "arm", "retention", "hits", "total", "wall_s")}
        exported["samples"] = [{key: sample[key] for key in
                                ("example_id", "input_sha256", "prompt_tokens", "hit",
                                 "output_tokens", "termination")} |
                               {"generated_sha256": C.content_hash(sample["generated"])}
                               for sample in cell["samples"]]
        public["cells"].append(exported)
    path = root / "benchmarks/validation/quality" / run["run_identity"] / "quality.json"
    C.write_record(path, public)
    return path, raw_path


def test_import_never_loads_gpu_runtime():
    result = subprocess.run([sys.executable, "-c",
                             ("import sys; sys.path.insert(0, sys.argv[1]); "
                              "import validation_stats; assert 'torch' not in sys.modules"),
                             str(C.REPO_ROOT / "scripts")], capture_output=True, text=True,
                            check=False)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("n", [1, 20, 100])
def test_exact_zero_all_success_and_symmetry(n):
    alpha = .05 / 18
    assert S.exact_bounds(0, n, alpha) == pytest.approx((0, 1 - alpha ** (1 / n)), abs=1e-13)
    assert S.exact_bounds(n, n, alpha) == pytest.approx((alpha ** (1 / n), 1), abs=1e-13)
    k = n // 2
    lower, upper = S.exact_bounds(k, n, alpha)
    inverse = S.exact_bounds(n - k, n, alpha)
    assert (lower, upper) == pytest.approx((1 - inverse[1], 1 - inverse[0]), abs=1e-13)


def test_binomial_bound_matches_scipy_documented_independent_reference():
    # Official scipy.stats.binomtest example, one-sided 95% bound for 3/15.
    assert S.exact_bounds(3, 15, .05)[0] == pytest.approx(.05684686759024681, abs=1e-12)
    # Independent direct finite-sum inversion checks on both nonzero tails.
    lower, upper = S.exact_bounds(4, 15, .025)
    tail = sum(math.comb(15, k) * lower ** k * (1 - lower) ** (15 - k) for k in range(4, 16))
    cdf = sum(math.comb(15, k) * upper ** k * (1 - upper) ** (15 - k) for k in range(5))
    assert tail == pytest.approx(.025, abs=1e-12)
    assert cdf == pytest.approx(.025, abs=1e-12)


def test_paired_uncertainty_is_nonzero_and_opposite_outcomes_have_direction():
    identical = S.paired_difference([True] * 20, [True] * 20, .05 / 18)
    assert identical["difference"] == 0 and identical["difference_lower"] < -.10
    worse = S.paired_difference([True] * 100, [False] * 100, .05 / 18)
    better = S.paired_difference([False] * 100, [True] * 100, .05 / 18)
    assert worse["losses"] == 100 and worse["gains"] == 0 and worse["difference_lower"] == -1
    assert better["gains"] == 100 and better["losses"] == 0 and better["difference_lower"] > .8


@pytest.mark.parametrize("args", [(0, 0, .025), (-1, 20, .025), (21, 20, .025),
                                 (1, 20, 0), (1, 20, float("nan")), (True, 20, .025)])
def test_invalid_binomial_inputs_fail(args):
    with pytest.raises(ValueError):
        S.exact_bounds(*args)


def test_frozen_protocol_rejects_changed_margin_and_pilot_seed(tmp_path, protocol):
    for key, value in [("noninferiority_margin", .2), ("seeds", [0, 1, 2, 3, 4])]:
        changed = copy.deepcopy(protocol)
        changed[key] = value
        path = tmp_path / f"{C.content_hash(changed)}.json"
        C.write_record(path, changed)
        with pytest.raises(ValueError, match="frozen decisions"):
            S.load_protocol(path)
    path = tmp_path / "changed.json"
    C.write_record(path, protocol)
    with pytest.raises(ValueError, match="filename"):
        S.load_protocol(path)


def test_confirmation_setting_drift_fails_before_loader(tmp_path, protocol, monkeypatch):
    called = []
    runtime = Q.Runtime(lambda *a, **kw: called.append("loader"), None, None, None, None, None,
                        lambda: called.append("cleanup"), None)
    args = Q.parse_args(["--ctx-len", "4096", "--seeds", "1", "2", "3", "4", "5",
                         "--confirmation-protocol", str(PROTOCOL_PATH), "--retentions", ".25", "1"])
    model = C.canonical_identity(json.loads(PILOT_PATH.read_text())["identity"])["model"]
    with pytest.raises(ValueError, match="frozen confirmation"):
        Q.run_quality(args, runtime, model, {})
    assert called == ["cleanup"]
    args.retentions = [.20, 1.0]
    monkeypatch.setattr(Q, "source_identity", lambda: {"dirty": True})
    with pytest.raises(ValueError, match="clean source"):
        Q.run_quality(args, runtime, model, {})
    assert "loader" not in called


@pytest.mark.parametrize("misses, expected", [(0, "pass"), (2, "pass"), (3, "fail"), (21, "fail")])
def test_all_nine_endpoints_apply_simultaneous_gate(tmp_path, protocol, misses, expected):
    paths = [quality_fixture(tmp_path, protocol, context, sparse_misses=misses)[0]
             for context, _ in S.CONTEXT_RETENTIONS]
    result = S.summarize(paths, protocol, root=tmp_path)
    assert result["status"] == expected and result["complete"] and result["completed_endpoints"] == 9
    assert all(e["sparse_minus_dense"]["n"] == 100 for e in result["endpoints"])
    assert all(e["sparse_minus_dense"]["bound_alpha"] == .05 / 18 for e in result["endpoints"])
    C.validate_export(result)


def test_missing_partial_and_weak_baseline_are_inconclusive(tmp_path, protocol):
    path, _ = quality_fixture(tmp_path, protocol, 4096)
    assert S.summarize([path], protocol, root=tmp_path)["status"] == "inconclusive"
    paths = [path, quality_fixture(tmp_path, protocol, 8192)[0],
             quality_fixture(tmp_path, protocol, 32768, complete=False)[0]]
    result = S.summarize(paths, protocol, root=tmp_path)
    assert result["status"] == "inconclusive" and result["completed_endpoints"] == 6
    weak_root = tmp_path / "weak"
    paths = [quality_fixture(weak_root, protocol, context, dense_misses=21)[0]
             for context, _ in S.CONTEXT_RETENTIONS]
    result = S.summarize(paths, protocol, root=weak_root)
    assert result["status"] == "inconclusive" and result["complete"]
    assert all(e["status"] == "inconclusive-weak-dense-baseline" for e in result["endpoints"])


@pytest.mark.parametrize("kind", ["hit", "input", "generated", "missing-cell", "identity"])
def test_public_evidence_tampering_is_rejected(tmp_path, protocol, kind):
    path, _ = quality_fixture(tmp_path, protocol, 4096)
    record = json.loads(path.read_text())
    if kind == "hit":
        record["cells"][0]["samples"][0]["hit"] = False
    elif kind == "input":
        record["cells"][0]["samples"][0]["input_sha256"] = "b" * 64
    elif kind == "generated":
        record["cells"][0]["samples"][0]["generated_sha256"] = "b" * 64
    elif kind == "missing-cell":
        record["cells"].pop()
    else:
        record["identity"]["config"]["seeds"] = [0, 1, 2, 3, 4]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        S.checked_quality(path, protocol, tmp_path)


def test_manifest_and_raw_hashes_are_required(tmp_path, protocol):
    path, raw_path = quality_fixture(tmp_path, protocol, 4096)
    raw_path.write_text(raw_path.read_text() + " ")
    with pytest.raises(ValueError, match="raw quality evidence hash"):
        S.checked_quality(path, protocol, tmp_path)
    record = json.loads(path.read_text())
    record["raw_evidence"]["sha256"] = C.file_hash(raw_path)
    path.write_text(json.dumps(record))
    manifest_path = raw_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["examples"][0]["input_ids"].append(42)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        S.checked_quality(path, protocol, tmp_path)


def test_contexts_from_different_source_snapshots_do_not_join(tmp_path, protocol):
    first = quality_fixture(tmp_path, protocol, 4096)[0]
    second = quality_fixture(tmp_path, protocol, 8192, source_marker="b")[0]
    with pytest.raises(ValueError, match="model/source/environment"):
        S.summarize([first, second], protocol, root=tmp_path)


def test_contexts_with_different_realized_kernels_do_not_join(tmp_path, protocol):
    first = quality_fixture(tmp_path, protocol, 4096)[0]
    second = quality_fixture(tmp_path, protocol, 8192, weight_kernel="other")[0]
    with pytest.raises(ValueError, match="realized runtime"):
        S.summarize([first, second], protocol, root=tmp_path)


def test_private_raw_evidence_reference_is_rejected_before_read(tmp_path, protocol):
    path, _ = quality_fixture(tmp_path, protocol, 4096)
    record = json.loads(path.read_text())
    record["raw_evidence"]["path"] = ".env"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="run directory"):
        S.checked_quality(path, protocol, tmp_path)


def test_summary_identity_and_private_export_boundary(tmp_path, protocol):
    path, _ = quality_fixture(tmp_path, protocol, 4096)
    result = S.summarize([path], protocol, root=tmp_path)
    identity = result.pop("run_identity")
    assert C.content_hash(result) == identity
    rendered = json.dumps(result)
    assert "generated" not in rendered and "input_ids" not in rendered and str(tmp_path) not in rendered
