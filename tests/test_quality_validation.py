"""CPU checks for paired quality evidence, resume integrity and cache lifetimes."""
import gc
import json
import subprocess
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_common as C
import phase6_run_ruler_4k_int4 as Q

MODEL_ID = {"model": "test/model", "revision": "a" * 40, "files": {"config.json": "b" * 64}}
ENV = {"python": "test", "packages": {}, "gpus": None}


class Attention:
    def forward(self):
        return "dense"


class Cache:
    def __init__(self, **kwargs):
        self.num_layers = kwargs["num_layers"]
        self._seen_tokens = [0] * self.num_layers


@pytest.fixture
def fixture_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(C, "source_identity", lambda: {"content_sha256": "c" * 64})
    attn = Attention()
    model = SimpleNamespace(
        attn=attn, device="cuda:0",
        config=SimpleNamespace(max_position_embeddings=2048, num_hidden_layers=1,
                               num_key_value_heads=1, hidden_size=128, num_attention_heads=2),
    )
    state = {"calls": [], "caches": [], "cleanup": 0, "fail_at": None}

    def manifest(_tok, args):
        return [{"example_id": f"{task}:{seed}:{i}", "task": task, "seed": seed,
                 "index": i, "input_ids": [seed + 1, i + 2, 3],
                 "input_sha256": C.content_hash([seed + 1, i + 2, 3]), "expected": ["1234567"]}
                for task in args.tasks for seed in args.seeds for i in range(args.n_samples)]

    def factory(**kwargs):
        cache = Cache(**kwargs)
        state["caches"].append(weakref.ref(cache))
        return cache

    def patch(model, cache, retention, _args):
        def forward():
            return retention, cache
        model.attn.forward = forward

    def evaluate(model, _tok, *, examples, pre_sample, **kwargs):
        if state["fail_at"] == len(state["calls"]):
            raise RuntimeError("failure with /private/local/path")
        state["calls"].append((kwargs["task"], kwargs["seed"], [e["input_ids"] for e in examples]))
        samples = []
        for index, example in enumerate(examples):
            if pre_sample:
                pre_sample(index)
                _, cache = model.attn.forward()
                assert cache._seen_tokens == [0]
                cache._seen_tokens = [123]
            samples.append({"example_id": example["example_id"], "prompt_tokens": 3,
                            "expected": example["expected"], "generated": "1234567 /private/path",
                            "hit": True, "output_tokens": 2, "termination": "eos"})
        return {"hits": len(samples), "total": len(samples), "samples": samples}

    def cleanup():
        gc.collect()
        state["cleanup"] += 1

    runtime = Q.Runtime(lambda *a, **kw: (model, object()), factory, patch, evaluate,
                        lambda model: [model.attn], lambda model: {"weight_kernel": "fake"},
                        cleanup, manifest)
    return runtime, model, state


def args_for(tmp_path, *extra):
    return Q.parse_args(["--tasks", "single", "--n-samples", "2", "--out",
                         str(tmp_path / "result.json"), *extra])


def test_script_import_is_gpu_runtime_free():
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    result = subprocess.run([sys.executable, "-c",
                             ("import sys; sys.path.insert(0, sys.argv[1]); "
                              "import phase6_run_ruler_4k_int4; assert 'torch' not in sys.modules"),
                             scripts], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_three_arms_share_inputs_and_reuse_one_cache(tmp_path, fixture_runtime):
    runtime, model, state = fixture_runtime
    original = model.attn.forward
    args = args_for(tmp_path, "--seeds", "0", "1")
    result, path = Q.run_quality(args, runtime, MODEL_ID, ENV)
    assert result["status"] == "complete" and result["screen"]["status"] == "pass"
    assert len(result["cells"]) == 6 and len(state["caches"]) == 1
    assert state["calls"][:2] == state["calls"][2:4] == state["calls"][4:6]
    assert model.attn.forward == original
    gc.collect()
    assert state["caches"][0]() is None
    exported = json.loads(path.read_text())
    assert "generated" not in exported["cells"][0]["samples"][0]
    assert "/private/path" not in path.read_text()
    raw = tmp_path / exported["raw_evidence"]["path"]
    assert "generated" in json.loads(raw.read_text())["cells"][0]["samples"][0]


def test_failure_preserves_completed_cells_and_resume_skips_them(tmp_path, fixture_runtime):
    runtime, model, state = fixture_runtime
    original = model.attn.forward
    args = args_for(tmp_path)
    state["fail_at"] = 1
    with pytest.raises(RuntimeError):
        Q.run_quality(args, runtime, MODEL_ID, ENV)
    failed = json.loads(args.out.read_text())
    assert failed["status"] == "execution_error" and failed["error"] == "execution_error"
    assert len(failed["cells"]) == 1 and model.attn.forward == original
    assert "/private" not in args.out.read_text()
    state["fail_at"] = None
    args.resume = True
    completed, _ = Q.run_quality(args, runtime, MODEL_ID, ENV)
    assert len(completed["cells"]) == 3 and len(state["calls"]) == 3
    assert completed["error"] is None
    assert state["cleanup"] == 2


def test_collision_preserves_prior_result(tmp_path, fixture_runtime):
    runtime, _, _ = fixture_runtime
    args = args_for(tmp_path)
    Q.run_quality(args, runtime, MODEL_ID, ENV)
    original = args.out.read_bytes()
    args.resume = True
    args.retentions = [0.25, 1.0]
    with pytest.raises(ValueError, match="different run identity"):
        Q.run_quality(args, runtime, MODEL_ID, ENV)
    assert args.out.read_bytes() == original


def test_corrupt_resume_cell_is_rejected_without_rewriting(tmp_path, fixture_runtime):
    runtime, _, _ = fixture_runtime
    args = args_for(tmp_path)
    _, path = Q.run_quality(args, runtime, MODEL_ID, ENV)
    exported = json.loads(path.read_text())
    raw_path = tmp_path / exported["raw_evidence"]["path"]
    raw = json.loads(raw_path.read_text())
    raw["cells"][0]["samples"][0]["input_sha256"] = "wrong"
    raw_path.write_text(json.dumps(raw))
    before = raw_path.read_bytes()
    args.resume = True
    with pytest.raises(ValueError, match="manifest"):
        Q.run_quality(args, runtime, MODEL_ID, ENV)
    assert raw_path.read_bytes() == before


def test_capacity_check_happens_before_evaluation(tmp_path, fixture_runtime):
    runtime, model, state = fixture_runtime
    model.config.max_position_embeddings = 4
    with pytest.raises(ValueError, match="capacity"):
        Q.run_quality(args_for(tmp_path), runtime, MODEL_ID, ENV)
    assert not state["calls"] and not state["caches"] and state["cleanup"] == 1


@pytest.mark.parametrize("hits, expected", [(17, "pass"), (16, "fail"), (0, "fail")])
def test_screen_boundaries(hits, expected):
    args = Q.parse_args(["--tasks", "single"])
    cells = [{"task": "single", "arm": arm, "hits": n}
             for arm, n in [("dense", 20), ("int4-r0.2", hits), ("int4-r1", 20)]]
    assert Q.screen_verdict(cells, args)["status"] == expected
    cells[0]["hits"] = 0
    assert Q.screen_verdict(cells, args)["status"] == "inconclusive"


def test_aliases_defaults_and_conflicts():
    assert Q.parse_args([]).retentions == [0.20, 1.0]
    args = Q.parse_args(["--retention", "0.25", "--seed", "3"])
    assert args.retentions == [0.25, 1.0] and args.seeds == [3]
    with pytest.raises(SystemExit):
        Q.parse_args(["--seed", "0", "--seeds", "1"])


def test_actual_manifest_records_tokens_once(fixture_runtime, monkeypatch):
    from flashquest.eval import niah

    calls = []
    def prompt(task, context, tokenizer, seed):
        calls.append((task, seed))
        return f"prompt-{seed}", ["1234567"]
    monkeypatch.setattr(niah, "make_prompt", prompt)
    class Tokenizer:
        def __call__(self, text):
            return SimpleNamespace(input_ids=[1, len(text)])
    args = Q.parse_args(["--tasks", "single", "--n-samples", "2", "--seeds", "0", "3"])
    manifest = Q.build_manifest(Tokenizer(), args)
    assert calls == [("single", 0), ("single", 1), ("single", 30000), ("single", 30001)]
    assert len({e["example_id"] for e in manifest}) == 4
    assert all(e["input_sha256"] == C.content_hash(e["input_ids"]) for e in manifest)


def test_identity_changes_with_source_model_environment_and_protocol():
    source = {"files": {"new-file.py": "one"}}
    baseline = C.make_identity({}, MODEL_ID, {}, ENV, source=source)["run_identity"]
    for config, model, protocol, env, code in [
        ({"timing": "new"}, MODEL_ID, {}, ENV, source),
        ({}, {**MODEL_ID, "revision": "other"}, {}, ENV, source),
        ({}, MODEL_ID, {"margin": 0.10}, ENV, source),
        ({}, MODEL_ID, {}, {"packages": {"torch": "new"}}, source),
        ({}, MODEL_ID, {}, ENV, {"files": {"new-file.py": "two"}}),
    ]:
        assert C.make_identity(config, model, protocol, env, source=code)["run_identity"] != baseline


def test_source_identity_hashes_untracked_changes(tmp_path):
    (tmp_path / "scripts").mkdir()
    untracked = tmp_path / "scripts" / "new-file.py"
    untracked.write_text("one")
    before = C.source_identity(tmp_path)
    untracked.write_text("two")
    after = C.source_identity(tmp_path)
    assert before["content_sha256"] != after["content_sha256"]
    assert "scripts/new-file.py" in after["files"]


def test_local_model_identity_tracks_weights_and_tokenizer(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    (model_dir / "tokenizer.json").write_text("{\"version\":1}")
    weights = model_dir / "model.safetensors"
    weights.write_bytes(b"first-test-weights")
    first = C.model_identity(str(model_dir))
    weights.write_bytes(b"second-test-weights")
    second = C.model_identity(str(model_dir))
    assert first["content_sha256"] != second["content_sha256"]
    (model_dir / "tokenizer.json").write_text("{\"version\":2}")
    third = C.model_identity(str(model_dir))
    assert second["content_sha256"] != third["content_sha256"]
    assert third["model"] == "local-model"
    C.validate_export(third)


def test_reference_scorer_rejects_tampered_resumed_hit(tmp_path, fixture_runtime):
    runtime, _, _ = fixture_runtime
    args = args_for(tmp_path)
    _, path = Q.run_quality(args, runtime, MODEL_ID, ENV)
    raw_path = tmp_path / json.loads(path.read_text())["raw_evidence"]["path"]
    raw = json.loads(raw_path.read_text())
    raw["cells"][0]["samples"][0]["generated"] = "wrong answer"
    raw_path.write_text(json.dumps(raw))
    args.resume = True
    with pytest.raises(ValueError, match="scorer"):
        Q.run_quality(args, runtime, MODEL_ID, ENV)


@pytest.mark.parametrize("record", [
    {"nested": {"config": "/tmp/model"}}, {"model_filename": "safe.gguf"},
    {"config": {"path": "C:\\Users\\example\\model.gguf"}}, {"rate": float("nan")},
    {"email": "private@example.test"},
])
def test_export_rejects_nested_private_paths_and_raw_fields(record):
    with pytest.raises(ValueError):
        C.validate_export(record)


def test_safe_export_keeps_public_links_and_drops_raw_fields():
    C.validate_export({"source": "https://example.org/primary", "path": "artifacts/raw.json"})
    record = C.new_record("flashquest", "INT4", {"ctx_len": 4096, "n_decode": 128,
                                                "model": "/tmp/private-model", "unexpected": "private"})
    record.update(error="traceback at /tmp/private-code", raw={"private": "value"})
    exported = C.export_benchmark(record)
    assert exported["config"]["model"] == "local-model"
    assert "unexpected" not in exported["config"] and "raw" not in exported
    assert exported["error"] == "backend_error"


def test_atomic_write_refuses_another_identity(tmp_path):
    path = tmp_path / "result.json"
    C.write_record(path, {"run_identity": "first", "value": 1})
    with pytest.raises(ValueError):
        C.write_record(path, {"run_identity": "second", "value": 2})
    assert json.loads(path.read_text())["value"] == 1
    assert not list(tmp_path.glob(".write-*"))


def test_provenance_survives_missing_packages_and_gpu_tools(monkeypatch):
    def missing(_package):
        raise C.importlib.metadata.PackageNotFoundError
    monkeypatch.setattr(C.importlib.metadata, "version", missing)
    def no_gpu(*args, **kwargs):
        raise FileNotFoundError
    monkeypatch.setattr(C.subprocess, "run", no_gpu)
    result = C.provenance()
    assert result["gpus"] is None and all(v is None for v in result["packages"].values())
