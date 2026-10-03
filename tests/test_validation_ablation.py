"""CPU checks for paired schedules, strict joins and practical pilot gates."""
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bench_common as C
import run_validation_ablation as A
import summarize_validation as S

METADATA = {"files": {}, "content_sha256": C.content_hash({})}
MODEL = {**METADATA, "model": "test/model", "revision": "a" * 40}


def test_schedule_balances_each_context_and_keeps_pairs_together():
    cells = A.schedule([8192, 32768], [0, 1, 2, 3], .20)
    assert len(cells) == 16
    assert [c["arm"] for c in cells[::2]] == ["sparse", "all-pages"] * 2 + ["all-pages", "sparse"] * 2
    for offset in range(0, len(cells), 2):
        first, second = cells[offset:offset + 2]
        assert (first["ctx_len"], first["seed"]) == (second["ctx_len"], second["seed"])
        assert {first["retention"], second["retention"]} == {.20, 1.0}
    assert A.schedule([32768], [0, 1, 2, 3], .25)[0]["arm"] == "all-pages"


def build_cell(cell, run, rate):
    config = C.run_config(cell["ctx_len"], 8, 3, cell["seed"], model=MODEL["model"],
                          kv_bits=4, retention=cell["retention"], page_size=64, num_sinks=4,
                          window_pages=2, revision=MODEL["revision"], warmup_steps=65,
                          cache_capacity=cell["ctx_len"] + 66,
                          input_sha256=C.content_hash([cell["ctx_len"], cell["seed"]]),
                          generation="manual-greedy-fixed-output-count-v1", runtime={"devices": ["cuda:0"]},
                          memory_protocol={"version": 1, "method": "nvidia-smi-query-linux-proc-monotonic",
                                           "interval_s": .05, "physical_device_index": 0})
    record = C.new_record("flashquest", "INT4", config)
    for sample_rate in [rate * .95, rate, rate * 1.05]:
        decode_s = 7 / sample_rate
        request_s = .5 + decode_s
        record["samples"].append({"input_tokens": cell["ctx_len"], "output_tokens": 8,
                                  "prefill_s": .5, "decode_s": decode_s, "request_s": request_s,
                                  "prefill_tok_s": cell["ctx_len"] / .5, "decode_tok_s": sample_rate,
                                  "end_to_end_tok_s": 8 / request_s,
                                  "peak_allocated_mib": 100, "peak_reserved_mib": 150})
    C.summarize_samples(record)
    record.update(C.make_identity(config, MODEL, A.BENCH_PROTOCOL, {}, source=METADATA))
    return C.export_benchmark(record)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(S, "REPO_ROOT", tmp_path)
    cells = A.schedule([8192], [0, 1, 2, 3], .20)
    protocol = {"cells": cells, "minimum_median_ratio": 1.10, "minimum_each_seed_ratio": 1.0}
    config = {"contexts": [8192], "seeds": [0, 1, 2, 3], "retention": .20,
              "reps": 3, "n_decode": 8, "interval_ms": 50, "physical_device_index": 0}
    run = C.make_identity(config, MODEL, protocol, {}, source=METADATA)
    result = {"protocol": protocol, "status": "complete", "cells": []}
    for cell in cells:
        path = tmp_path / f"{cell['cell_id']}.json"
        record = build_cell(cell, run, 120 if cell["arm"] == "sparse" else 100)
        C.write_record(path, record)
        memory = {"sampler_failed": False, "concurrent_compute_workload": False,
                  "ownership_check_dropouts": 0, "device_sample_count": 5,
                  "ownership_checks": 1, "sample_count": 5, "device_baseline_mib": 50,
                  "phase_status": "validated-windows", "device_sampled_peak_mib": 200,
                  "device_baseline_adjusted_peak_mib": 150, "process_tree_rss_sampled_peak_mib": 100,
                  "actual_interval_median_ms": 50, "actual_interval_max_ms": 80,
                  "device_dropouts": 0,
                  "phases": {phase: {"device_sampled_peak_mib": None, "window_count": count,
                                      "sample_count": 0, "process_tree_rss_sampled_peak_mib": None}
                             for phase, count in (("load", 1), ("warmup", 1), ("prefill", 3), ("decode", 3))},
                  "raw_series": {"path": "artifacts/not-in-checkout.json", "sha256": C.content_hash([])}}
        result["cells"].append({"cell": cell, "status": "complete", "memory": memory,
                                "result": {"path": path.name, "sha256": C.file_hash(path)}})
    saved = A.export_schedule(run, result)
    path = tmp_path / "schedule.json"
    C.write_record(path, saved)
    return path, saved, run


def rewrite(path, saved):
    C.write_record(path, saved)


def test_summary_uses_paired_seed_medians_and_spread(evidence):
    path, _, _ = evidence
    result = S.summarize(path)["contexts"][0]
    assert result["paired_seed_count"] == 4
    assert result["median_decode_ratio"] == pytest.approx(1.2)
    assert result["practical_screen_pass"] is True
    assert result["pairs"][0]["sparse"]["decode_sample_min_tok_s"] < 120


def test_summary_accepts_relative_cli_path(evidence, monkeypatch):
    path, _, _ = evidence
    monkeypatch.chdir(path.parent)
    assert S.summarize(Path("schedule.json"))["contexts"][0]["practical_screen_pass"] is True


def test_one_slower_seed_fails_practical_screen(evidence):
    path, saved, run = evidence
    entry = next(c for c in saved["cells"] if c["cell"]["arm"] == "sparse")
    output = path.parent / entry["result"]["path"]
    C.write_record(output, build_cell(entry["cell"], run, 99))
    entry["result"]["sha256"] = C.file_hash(output)
    rewrite(path, saved)
    result = S.summarize(path)["contexts"][0]
    assert result["median_decode_ratio"] == pytest.approx(1.2)
    assert result["practical_screen_pass"] is False


def test_incomplete_schedule_has_no_gate(evidence):
    path, saved, _ = evidence
    saved["cells"] = saved["cells"][:2]
    saved["status"] = "incomplete"
    rewrite(path, saved)
    assert S.summarize(path)["contexts"][0]["practical_screen_pass"] is None


def test_completion_status_requires_every_cell(evidence):
    path, saved, _ = evidence
    saved["cells"] = saved["cells"][:2]
    rewrite(path, saved)
    with pytest.raises(ValueError, match="completion differs"):
        S.summarize(path)


def test_memory_coverage_and_local_raw_hash_are_verified(evidence):
    path, saved, _ = evidence
    saved["cells"][0]["memory"]["device_sample_count"] = 6
    rewrite(path, saved)
    with pytest.raises(ValueError, match="coverage"):
        S.summarize(path)
    saved["cells"][0]["memory"]["device_sample_count"] = 5
    rewrite(path, saved)
    series = path.parent / "artifacts" / "not-in-checkout.json"
    series.parent.mkdir()
    series.write_text("[]")
    assert S.summarize(path)["contexts"][0]["practical_screen_pass"] is True
    series.write_text("{}")
    with pytest.raises(ValueError, match="raw memory series changed"):
        S.summarize(path)


def test_changed_cell_hash_is_rejected(evidence):
    path, saved, _ = evidence
    output = path.parent / saved["cells"][0]["result"]["path"]
    output.write_text(output.read_text() + " ")
    with pytest.raises(ValueError, match="changed after observation"):
        S.summarize(path)


def test_different_input_ids_cannot_join(evidence):
    path, saved, _ = evidence
    entry = saved["cells"][0]
    output = path.parent / entry["result"]["path"]
    record = json.loads(output.read_text())
    identity = C.canonical_identity(record["identity"])
    identity["config"]["input_sha256"] = C.content_hash("different input IDs")
    record.update(identity=C.export_identity(identity), run_identity=C.content_hash(identity), config=identity["config"])
    output.write_text(json.dumps(record))
    entry["result"]["sha256"] = C.file_hash(output)
    rewrite(path, saved)
    with pytest.raises(ValueError, match="different inputs"):
        S.summarize(path)


def test_inconsistent_rates_are_rejected(evidence):
    path, saved, run = evidence
    output = path.parent / saved["cells"][0]["result"]["path"]
    record = json.loads(output.read_text())
    record["samples"][0]["output_tokens"] = 9
    C.write_record(output, record)
    with pytest.raises(ValueError, match="token counts"):
        A.checked_cell(output, saved["cells"][0]["cell"], run, 3, 8)
    record["samples"][0]["output_tokens"] = 8
    record["samples"][0]["decode_tok_s"] *= 2
    C.write_record(output, record)
    with pytest.raises(ValueError, match="timed counts"):
        A.checked_cell(output, saved["cells"][0]["cell"], run, 3, 8)


def test_reordered_resume_is_rejected(evidence):
    path, saved, _ = evidence
    saved["cells"][0], saved["cells"][1] = saved["cells"][1], saved["cells"][0]
    rewrite(path, saved)
    with pytest.raises(ValueError, match="balanced order"):
        S.summarize(path)


def test_quality_prerequisite_recomputes_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "REPO_ROOT", tmp_path)
    cfg = {"ctx_len": 8192, "tasks": ["single", "multikey", "multivalue"], "seeds": [0],
           "n_samples": 20, "retentions": [.20, 1.]}
    record = {**C.make_identity(cfg, MODEL, {}, {}, source=METADATA), "status": "complete",
              "cells": [{"task": task, "seed": 0, "arm": arm, "hits": 20, "total": 20,
                         "samples": [{"hit": True}] * 20}
                        for task in cfg["tasks"] for arm in ["dense", "int4-r0.2", "int4-r1"]]}
    record["screen"] = A.screen_verdict(record["cells"], SimpleNamespace(**cfg))
    record["identity"] = C.export_identity(record["identity"])
    path = tmp_path / "quality.json"
    C.write_record(path, record)
    assert A.quality_prerequisites([path], [8192], .20, MODEL)[0]["ctx_len"] == 8192
    monkeypatch.chdir(tmp_path)
    assert A.quality_prerequisites([Path("quality.json")], [8192], .20, MODEL)[0]["path"] == "quality.json"
    with pytest.raises(ValueError, match="every scheduled context"):
        A.quality_prerequisites([path], [8192, 32768], .20, MODEL)
    record["screen"]["tasks"]["single"]["arms"]["int4-r0.2"]["screen_pass"] = False
    C.write_record(path, record)
    with pytest.raises(ValueError, match="inconsistent"):
        A.quality_prerequisites([path], [8192], .20, MODEL)


def test_partial_resume_keeps_order_and_does_not_repeat_completed_cells(evidence, monkeypatch):
    path, saved, _ = evidence
    monkeypatch.setattr(C, "source_identity", lambda: METADATA)
    monkeypatch.setattr(A, "model_identity", lambda *args: MODEL)
    monkeypatch.setattr(A, "resolve_device", lambda index: {"index": 0, "uuid": "test-device",
                                                          "name": "test GPU", "total_mib": 1000, "driver": "test"})
    monkeypatch.setattr(A, "quality_prerequisites", lambda *args: [])
    monkeypatch.setattr(A, "provenance", dict)
    calls = []
    fail_after = [2]
    def observe(command, raw_dir, device, **kwargs):
        if len(calls) == fail_after[0]:
            raise RuntimeError("interrupted before starting the next cell")
        def argument(flag):
            return command[command.index(flag) + 1]
        cell = {"ctx_len": int(argument("--ctx-len")), "seed": int(argument("--seed")),
                "retention": float(argument("--retention"))}
        calls.append(cell)
        output = Path(argument("--out"))
        C.write_record(output, build_cell(cell, None, 120 if cell["retention"] < 1 else 100))
        raw_dir.mkdir(parents=True)
        series = raw_dir / "memory-series.json"
        C.write_record(series, {"samples": []})
        memory = deepcopy(saved["cells"][0]["memory"])
        memory["raw_series"] = {"sha256": C.file_hash(series)}
        return {"status": "complete", "returncode": 0, "memory": memory}
    monkeypatch.setattr(A, "observe_command", observe)
    argv = ["--contexts", "8192", "--seeds", "0", "1", "2", "3", "--revision", MODEL["revision"],
            "--model", MODEL["model"], "--quality", str(path), "--n-decode", "8"]
    with pytest.raises(RuntimeError, match="interrupted"):
        A.main(argv)
    schedules = list((path.parent / "benchmarks").glob("validation/ablation/*/schedule.json"))
    assert len(schedules) == 1
    partial = json.loads(schedules[0].read_text())
    assert len(partial["cells"]) == 2 and partial["status"] == "incomplete"
    hashes = [cell["result"]["sha256"] for cell in partial["cells"]]
    fail_after[0] = 100
    assert A.main([*argv, "--resume"]) == 0
    completed = json.loads(schedules[0].read_text())
    assert len(calls) == 8 and completed["status"] == "complete"
    assert hashes == [cell["result"]["sha256"] for cell in completed["cells"][:2]]
    assert S.summarize(schedules[0])["contexts"][0]["practical_screen_pass"] is True
    original_observer = A.observe_command
    def timed_out(*args, **kwargs):
        observed = original_observer(*args, **kwargs)
        observed["status"] = "timeout"
        observed["returncode"] = -15
        observed["memory"]["sampler_failed"] = True
        return observed
    monkeypatch.setattr(A, "observe_command", timed_out)
    assert A.main([*argv, "--attempt", "1", "--retry-of", str(schedules[0])]) == 1
    retries = list((path.parent / "benchmarks").glob("validation/ablation/*/schedule.json"))
    failed = next(json.loads(p.read_text()) for p in retries if p != schedules[0])
    assert failed["status"] == "execution-error" and failed["cells"][0]["status"] == "timeout"
    assert failed["cells"][0]["failure_categories"] == ["timeout", "invalid-telemetry"]
    assert failed["identity"]["config"]["retry_of"] == completed["run_identity"]
