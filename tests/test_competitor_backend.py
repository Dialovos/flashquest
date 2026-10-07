"""Native response contracts, exact inputs, and owned server cleanup."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from competitor_backend import Server, completion_payload, parse_completion

LOOPBACK_SERVER = """
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys

class Handler(BaseHTTPRequestHandler):
    def respond(self):
        health = self.path == "/health"
        body = (sys.argv[3] if health else sys.argv[4]).encode()
        self.send_response(int(sys.argv[2]) if health else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = respond
    do_POST = respond

HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


@pytest.fixture
def loopback_server(tmp_path):
    def make(status=200, health_body="", api_body="{}", timeout=2):
        return Server([sys.executable, "-u", "-c", LOOPBACK_SERVER, "{port}", str(status),
                       health_body, api_body], tmp_path, timeout=timeout)
    return make


@pytest.mark.parametrize("body", ["", '{"status":"ok"}'])
def test_readiness_accepts_native_http_200_with_empty_or_json_body(loopback_server, monkeypatch, body):
    # A bad inherited proxy would prevent readiness unless loopback stays direct.
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("no_proxy", "")
    monkeypatch.setenv("NO_PROXY", "")
    server = loopback_server(health_body=body)
    with server:
        assert server.process.poll() is None
        server.check_health(timeout=.5)
    assert server.process.poll() is not None


def test_readiness_rejects_http_503_and_cleans_owned_server(loopback_server):
    server = loopback_server(status=503, timeout=.8)
    with pytest.raises(TimeoutError, match="server readiness timeout"), server:
        pass
    assert server.process.poll() is not None
    assert "503" in (server.directory / "server.log").read_text()


@pytest.mark.parametrize("route", ["/v1/completions", "/detokenize"])
@pytest.mark.parametrize("body", ["", "invalid JSON"])
def test_nonhealth_requests_still_require_json(loopback_server, route, body):
    with loopback_server(api_body=body) as server, pytest.raises(json.JSONDecodeError):
        server.request(route, {"tokens": [3]}, timeout=.5)


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


@pytest.mark.parametrize("body", [b"ASCII\r\n", b"valid caf\xc3\xa9\r\n",
                                   b"token bytes \xdb\xff\x80\x00\r\nUTF-8 \xc3\xa9\n"])
def test_native_log_decode_preserves_every_byte(tmp_path, body):
    from competitor_backend import read_native_log
    path = tmp_path / "server.log"
    path.write_bytes(body)
    decoded = read_native_log(path)
    assert decoded.encode("utf-8", errors="surrogateescape") == body
    assert path.read_bytes() == body


@pytest.mark.parametrize("body", [b'{"content":"\xff"}', b'{"content":'])
@pytest.mark.parametrize("route", ["/completion", "/v1/completions", "/detokenize"])
def test_native_log_codec_does_not_relax_api_json(monkeypatch, body, route):
    import io
    server = Server.__new__(Server)
    monkeypatch.setattr(server, "_open", lambda *args: io.BytesIO(body))
    with pytest.raises((UnicodeDecodeError, json.JSONDecodeError)):
        server.request(route, {"tokens": [3]})


@pytest.mark.parametrize("diagnostic,error", [
    (b"CUDA out of memory", "CUDA out of memory during server startup"),
    (b"torch.OutOfMemoryError", "CUDA out of memory during server startup"),
    (b"unrelated startup failure", "server exited before readiness"),
])
def test_binary_startup_log_preserves_diagnostic_classification(tmp_path, monkeypatch, diagnostic, error):
    import competitor_backend as adapter
    body = b"generated bytes \xdb\xff\r\n" + diagnostic + b"\n"
    server = Server.__new__(Server)
    server.directory, server.timeout, server.command, server.env, server.known = tmp_path, 1, ["mock-server"], {}, {}
    cleanup = []

    class Process:
        pid = 12345

        def poll(self):
            return 1

    def launch(command, *, stdout, **kwargs):
        stdout.buffer.write(body)
        stdout.flush()
        return Process()

    def close(*unused):
        cleanup.append(True)
        server.log.close()

    class Thread:
        def __init__(self, **kwargs):
            pass

        def start(self):
            pass

    monkeypatch.setattr(adapter.subprocess, "Popen", launch)
    monkeypatch.setattr(adapter, "processes", lambda: {12345: {"start_ticks": 1}})
    monkeypatch.setattr(adapter.threading, "Thread", Thread)
    monkeypatch.setattr(adapter, "marker", lambda *args: None)
    monkeypatch.setattr(server, "__exit__", close)
    with pytest.raises(RuntimeError, match=error):
        server.__enter__()
    assert cleanup == [True]
    assert (tmp_path / "server.log").read_bytes() == body
