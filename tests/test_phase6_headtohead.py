"""Regression checks for comparable context depth, timing, and cached results."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import phase6_run_headtohead as H
from bench_common import content_hash, is_oom, synthetic_token_ids, vllm_sample, write_record


def test_planned_matrix_and_dry_run(capsys):
    assert H.planned_matrix() == [(b, c) for b in H.BACKENDS for c in H.CTXS]
    assert H.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert "9 cells" in output and "vllm @ 131072" in output
    assert "q4_0/q4_0 KV" in output and "fp8 KV" in output
    assert "127 decode steps" in output


def test_dense_ablation_and_custom_contexts(capsys):
    assert H.main(["--dry-run", "--backends", "flashquest", "--contexts", "4096",
                   "--retention", "1.0"]) == 0
    output = capsys.readouterr().out
    assert "1 cells" in output and "retention=1" in output
    assert H.parse_args([]).output_dir == H.REPO_ROOT / "benchmarks" / "validation"


def llama_fixture(tmp_path):
    config = H.cell_config("llamacpp", 8192, H.parse_args([]))
    raw = tmp_path / "llama.json"
    row = {"n_prompt": 0, "n_gen": 127, "n_depth": 8192,
           "type_k": "q4_0", "type_v": "q4_0", "avg_ts": 99,
           "samples_ts": [10.0, 30.0, 20.0], "build_commit": "abc123"}
    raw.write_text(json.dumps([row]))
    prefill = {**row, "n_prompt": 8192, "n_gen": 0, "n_depth": 0,
               "samples_ts": [100.0, 300.0, 200.0]}
    raw.with_suffix(".prefill.json").write_text(json.dumps([prefill]))
    return raw, config, row


def test_llamacpp_context_matched_median_and_unmeasured_memory(tmp_path):
    raw, config, _ = llama_fixture(tmp_path)
    record = H._parse_llamacpp_log(raw, config, 0, "")
    assert record["error"] is None
    assert record["decode_tok_s"] == 20 and record["prefill_tok_s"] == 200
    assert record["peak_allocated_mib"] is None
    assert record["versions"]["llamacpp_commit"] == "abc123"


@pytest.mark.parametrize("change", [
    {"n_depth": 0}, {"n_gen": 128}, {"type_k": "f16"},
    {"type_v": "f16"}, {"samples_ts": [20.0]},
    {"samples_ts": [10.0, float("nan"), 20.0]},
])
def test_llamacpp_rejects_wrong_depth_precision_or_repetitions(tmp_path, change):
    raw, config, row = llama_fixture(tmp_path)
    raw.write_text(json.dumps([{**row, **change}]))
    record = H._parse_llamacpp_log(raw, config, 0, "")
    assert record["error"] and record["decode_tok_s"] is None


def test_old_markdown_decode_and_tail_memory_are_not_accepted(tmp_path):
    raw, config, _ = llama_fixture(tmp_path)
    raw.write_text("| pp8192 | 736.55 |\n| tg128 | 39.60 |\n3543 MiB, 552 MiB\n")
    record = H._parse_llamacpp_log(raw, config, 0, "")
    assert record["error"] and record["decode_tok_s"] is None
    assert record["peak_allocated_mib"] is None


@pytest.mark.parametrize("bad_json", [{"test": "wrong"}, ["not a test"]])
def test_malformed_llamacpp_shape_is_a_cell_error(tmp_path, bad_json):
    raw, config, _ = llama_fixture(tmp_path)
    raw.write_text(json.dumps(bad_json))
    assert H._parse_llamacpp_log(raw, config, 0, "")["error"]


def test_llamacpp_shell_runs_decode_at_depth(tmp_path):
    model = tmp_path / "test.gguf"
    model.touch()
    binary = tmp_path / "llama-bench"
    calls = tmp_path / "calls.jsonl"
    binary.write_text(f"#!{sys.executable}\n" +
                     "import json, os, sys\n"
                     "with open(os.environ['CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                     "print('[]')\n")
    binary.chmod(0o755)
    raw = tmp_path / "result.json"
    subprocess.run(["bash", str(H.REPO_ROOT / "scripts" / "bench_llamacpp.sh")],
                   cwd=H.REPO_ROOT, check=True, capture_output=True,
                   env={**os.environ, "MODEL": str(model), "LLAMA_BIN": str(binary),
                        "CTX": "8192", "N_DECODE": "128", "REPS": "3",
                        "KV_K": "q4_0", "KV_V": "q4_0", "OUT": str(raw),
                        "CALLS": str(calls)})
    prefill, decode = [json.loads(line) for line in calls.read_text().splitlines()]
    def option(argv, key):
        return argv[argv.index(key) + 1]
    assert (option(prefill, "-p"), option(prefill, "-n"), option(prefill, "-d")) == ("8192", "0", "0")
    assert (option(decode, "-p"), option(decode, "-n"), option(decode, "-d")) == ("0", "127", "8192")
    assert option(decode, "-ctk") == "q4_0" and option(decode, "-ctv") == "q4_0"
    assert option(decode, "-fa") == "on" and option(decode, "-r") == "3"
    assert raw.with_suffix(".prefill.json").exists()


def request_output():
    return SimpleNamespace(prompt_token_ids=[1] * 8192,
                           outputs=[SimpleNamespace(token_ids=[2] * 128)],
                           metrics=SimpleNamespace(first_scheduled_time=2.0,
                                                   first_token_time=12.0,
                                                   last_token_time=22.0))


def test_vllm_decode_excludes_prefill_and_first_output_token():
    sample = vllm_sample(request_output(), 8192, 128, 25.0)
    assert sample["prefill_tok_s"] == 819.2
    assert sample["decode_tok_s"] == 12.7
    assert sample["end_to_end_tok_s"] == 5.12


@pytest.mark.parametrize("bad_metrics", [None, SimpleNamespace(first_token_time=12.0)])
def test_missing_vllm_phase_metrics_are_errors(bad_metrics):
    output = request_output()
    output.metrics = bad_metrics
    with pytest.raises(ValueError, match="timestamps unavailable"):
        vllm_sample(output, 8192, 128, 25.0)


def test_wrong_vllm_token_count_is_an_error():
    with pytest.raises(ValueError, match="expected"):
        vllm_sample(request_output(), 32768, 128, 25.0)


def test_seeded_inputs_match_without_touching_global_rng():
    assert synthetic_token_ids(128256, 256, 0) == synthetic_token_ids(128256, 256, 0)
    assert synthetic_token_ids(128256, 256, 0) != synthetic_token_ids(128256, 256, 1)


def test_skip_existing_requires_matching_successful_config(tmp_path):
    config = H.cell_config("flashquest", 8192, H.parse_args([]))
    record = H.cell_record("flashquest", config)
    record["decode_tok_s"] = 10.0
    identity = {"config": config, "source": "fixed", "model": "revision"}
    record.update(identity=identity, run_identity=content_hash(identity),
                  samples=[{"input_tokens": 8192, "output_tokens": 128,
                            "decode_tok_s": 10.0}] * 3)
    path = tmp_path / "result.json"
    write_record(path, record)
    assert H.reusable_cell(path, config) is None
    assert H.reusable_cell(path, config, identity) == record
    assert H.reusable_cell(path, config, {**identity, "source": "changed"}) is None
    assert H.reusable_cell(path, {**config, "retention": 1.0}, identity) is None
    record["error"] = "failed"
    write_record(path, record)
    assert H.reusable_cell(path, config, identity) is None


def test_failed_rerun_cannot_reuse_stale_success(tmp_path, monkeypatch):
    args = H.parse_args([])
    config = H.cell_config("flashquest", 8192, args)
    record = H.cell_record("flashquest", config)
    record["decode_tok_s"] = 10.0
    path = tmp_path / "result.json"
    write_record(path, record)
    monkeypatch.setattr(H, "_run_command", lambda *args: (1, "", "broken dependency"))
    result = H.run_one("flashquest", 8192, path, args)
    assert result["error"] and result["decode_tok_s"] is None
    assert json.loads(path.read_text())["error"]


def test_unmeasured_cells_never_render_as_fitting():
    config = H.cell_config("flashquest", 131072, H.parse_args([]))
    md = H.render_markdown([H.cell_record("flashquest", config)])
    assert "unmeasured" in md and "✓" not in md
    assert "Peak allocated" in md and "physical GPU residency" in md


@pytest.mark.parametrize("message, expected", [
    ("CUDA out of memory", True), ("no available memory for the cache blocks", True),
    ("CUDA initialization failed", False), ("unsupported kv cache dtype", False),
])
def test_oom_labels_require_memory_evidence(message, expected):
    assert is_oom(message) is expected
