"""Native response contracts, exact inputs, and owned server cleanup."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from competitor_backend import Server, completion_payload, parse_completion


@pytest.mark.parametrize("explicit_jit", [None, "0", "false", ""])
def test_vllm_build_policy_overrides_inherited_and_explicit_environment(tmp_path, monkeypatch, explicit_jit):
    import competitor_backend as adapter

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *unused):
            pass

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 12345)

    python = tmp_path / "backend" / "bin" / "python"
    toolkit = python.parent.parent / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    (toolkit / "bin").mkdir(parents=True)
    (toolkit / "bin" / "nvcc").touch()
    monkeypatch.setattr(adapter.socket, "socket", FakeSocket)
    monkeypatch.setenv("MAX_JOBS", "64")
    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "0")
    monkeypatch.setenv("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    explicit = {"MAX_JOBS": "99"}
    if explicit_jit is not None:
        explicit["FLASHINFER_DISABLE_JIT"] = explicit_jit
        explicit["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
    server = Server([str(python), "-m", "vllm.entrypoints.openai.api_server"], tmp_path,
                    env=explicit)
    assert server.env["MAX_JOBS"] == "2" == str(adapter.compile_workers("vllm"))
    assert adapter.compile_workers("llamacpp") is None
    assert server.env["FLASHINFER_DISABLE_JIT"] == "1"
    assert "FLASHINFER_DISABLE_VERSION_CHECK" not in server.env
    assert adapter.flashinfer_disable_jit("vllm") is True
    assert adapter.flashinfer_disable_jit("llamacpp") is None
    assert adapter.os.environ["FLASHINFER_DISABLE_JIT"] == "0"
    assert adapter.os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert str(python.parent) in server.env["PATH"].split(adapter.os.pathsep)
    assert server.env["CUDA_HOME"] == str(toolkit)


def response(backend, nout=8):
    if backend == "vllm":
        return {"choices": [{"text": "answer", "token_ids": list(range(nout)), "finish_reason": "length"}],
                "usage": {"prompt_tokens": 1024, "completion_tokens": nout},
                "metrics": {"time_to_first_token_ms": 100, "generation_time_ms": 700, "queue_time_ms": 2}}
    return {"content": "answer", "tokens": list(range(nout)), "tokens_predicted": nout,
            "stop_type": "limit", "timings": {"prompt_n": 1024, "prompt_ms": 100,
                                                "predicted_n": nout, "predicted_ms": 700}}


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_same_request_native_counts_and_units(backend):
    r = parse_completion(backend, response(backend), 1024, 8, performance=True)
    assert r["prefill_s"] == .1 and r["decode_s"] == .7
    assert r["decode_steps"] == 7 and r["decode_tok_s"] == 10
    assert r["termination"] == "length"


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_one_output_quality_has_no_decode_interval(backend):
    r = parse_completion(backend, response(backend, 1), 1024, 128, performance=False)
    assert r["decode_s"] is None and r["decode_tok_s"] is None


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_count_and_token_id_mismatches_rejected(backend):
    r = response(backend)
    with pytest.raises(ValueError):
        parse_completion(backend, r, 1023, 8, performance=True)
    with pytest.raises(ValueError):
        parse_completion(backend, r, 1024, 9, performance=True)
    (r["choices"][0] if backend == "vllm" else r)["token_ids" if backend == "vllm" else "tokens"] = None
    with pytest.raises(ValueError):
        parse_completion(backend, r, 1024, 8, performance=True)


def test_missing_metrics_and_nonfinite_duration_rejected():
    r = response("vllm")
    r["metrics"] = None
    with pytest.raises(TypeError):
        parse_completion("vllm", r, 1024, 8, performance=True)
    r = response("vllm")
    r["metrics"]["generation_time_ms"] = float("nan")
    with pytest.raises(ValueError):
        parse_completion("vllm", r, 1024, 8, performance=True)


def test_llama_eos_and_output_counter_contract():
    r = response("llamacpp")
    r["stop_type"] = "eos"
    assert parse_completion("llamacpp", r, 1024, 128, performance=False)["termination"] == "eos"
    r["timings"]["predicted_n"] = 7
    with pytest.raises(ValueError, match="native output count"):
        parse_completion("llamacpp", r, 1024, 128, performance=False)


@pytest.mark.parametrize("backend", ["vllm", "llamacpp"])
def test_payload_preserves_exact_ids_and_generation_policy(backend):
    ids = [128000, 20, 31]
    payload = completion_payload(backend, ids, 128, performance=False)
    assert payload["prompt"] == ids and payload["ignore_eos"] is False
    assert payload["temperature"] == 0
    assert completion_payload(backend, ids, 128, performance=True)["ignore_eos"] is True


def test_failed_readiness_cleans_owned_escaped_worker(tmp_path):
    pid_path = tmp_path / "worker.json"
    worker = "import time;time.sleep(30)"
    code = f"import subprocess,sys,time,json; p=subprocess.Popen([sys.executable,'-c',{worker!r}],start_new_session=True);open({str(pid_path)!r},'w').write(json.dumps(p.pid));time.sleep(30)"
    from gpu_memory import processes
    with subprocess.Popen([sys.executable, "-c", worker], start_new_session=True) as foreign:
        try:
            with pytest.raises(TimeoutError), Server([sys.executable, "-c", code], tmp_path, timeout=.6):
                pass
            pid = json.loads(pid_path.read_text())
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z"
            assert foreign.poll() is None and foreign.pid in processes()
        finally:
            foreign.terminate()
