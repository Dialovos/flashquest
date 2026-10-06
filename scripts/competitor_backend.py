"""Loopback native completion adapters with bounded owned-worker cleanup."""
from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from gpu_memory import owned_processes, processes

VLLM_COMPILE_WORKERS = 2


def read_native_log(path):
    """Decode diagnostic bytes reversibly; structured JSON keeps strict decoding."""
    return Path(path).read_bytes().decode("utf-8", errors="surrogateescape")


def compile_workers(backend):
    """Bound FlashInfer/Torch extension ninja jobs in the vLLM subprocess."""
    return VLLM_COMPILE_WORKERS if backend == "vllm" else None


def flashinfer_disable_jit(backend):
    """Require matching precompiled FlashInfer kernels for the vLLM subprocess."""
    return True if backend == "vllm" else None


class UnsupportedConfiguration(ValueError):
    """The exact input/model/cache configuration cannot be compared."""


def positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"missing numeric {label}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {label}")
    return float(value)


def completion_payload(backend, ids, output_count, *, performance):
    if not ids or any(type(x) is not int or x < 0 for x in ids):
        raise ValueError("invalid exact token input")
    if backend == "vllm":
        return {"model": "validation-model", "prompt": ids, "max_tokens": output_count,
                "temperature": 0.0, "seed": 0, "n": 1, "stream": False,
                "ignore_eos": performance, "add_special_tokens": False,
                "return_token_ids": True,
                **({"stop_token_ids": [128001, 128008, 128009]} if not performance else {})}
    return {"prompt": ids, "n_predict": output_count, "temperature": 0.0,
            "seed": 0, "stream": False, "ignore_eos": performance,
            "cache_prompt": False, "return_tokens": True,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def parse_completion(backend, response, expected_input, expected_output, *, performance):
    if backend == "vllm":
        if len(response.get("choices", [])) != 1:
            raise ValueError("response must contain one stream")
        choice = response["choices"][0]
        usage = response["usage"]
        nin, nout = usage["prompt_tokens"], usage["completion_tokens"]
        metrics = response.get("metrics") or {}
        prefill = positive(metrics.get("time_to_first_token_ms"), "TTFT") / 1000
        decode = positive(metrics.get("generation_time_ms"), "generation") / 1000 if nout > 1 else None
        queue = metrics.get("queue_time_ms")
        if queue is not None and (not math.isfinite(queue) or queue < 0):
            raise ValueError("invalid queue time")
        text, tokens = choice["text"], choice.get("token_ids")
        termination = choice["finish_reason"]
        boundary = "scheduled-to-first / first-to-last-token"
        numeric = {k: metrics.get(k) for k in ("time_to_first_token_ms", "generation_time_ms", "queue_time_ms")}
    else:
        timings = response["timings"]
        nin, nout = timings["prompt_n"], response["tokens_predicted"]
        prefill = positive(timings["prompt_ms"], "prompt") / 1000
        # b11382 serializes n_gen including the first output, but times n_gen-1 forwards.
        if timings["predicted_n"] != nout:
            raise ValueError("llama-server native output count differs from response")
        steps = nout - 1
        decode = positive(timings["predicted_ms"], "generation") / 1000 if steps else None
        queue = None
        text, tokens = response["content"], response.get("tokens")
        termination = {"limit": "length", "eos": "eos"}.get(response["stop_type"], response["stop_type"])
        boundary = "native prompt eval / native generation forwards"
        numeric = {k: timings[k] for k in ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms")}
    if type(nin) is not int or nin != expected_input or type(nout) is not int or nout < 1:
        raise ValueError("backend token counts differ from requested input")
    if performance and nout != expected_output:
        raise ValueError("backend did not produce the fixed output count")
    if nout > expected_output:
        raise ValueError("output exceeds frozen limit")
    if not isinstance(text, str):
        raise TypeError("missing generated text")
    if not isinstance(tokens, list) or len(tokens) != nout or any(type(x) is not int or x < 0 for x in tokens):
        raise ValueError("generated token IDs disagree with usage")
    return {"input_tokens": nin, "output_tokens": nout, "decode_steps": nout - 1,
            "prefill_s": prefill, "decode_s": decode,
            "prefill_tok_s": nin / prefill,
            "decode_tok_s": (nout - 1) / decode if decode else None,
            "native_request_s": prefill + (decode or 0),
            "native_output_tok_s": nout / (prefill + (decode or 0)),
            "queue_ms": queue, "native_numeric_metrics": numeric,
            "timing_boundary": boundary, "termination": termination,
            "text": text, "tokens": tokens}


def marker(event, repetition=None):
    path = os.environ.get("FLASHQUEST_MARKERS")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": event, "monotonic_ns": time.monotonic_ns(),
                                "repetition": repetition}) + "\n")


def llama_runtime_directory(binary):
    candidates = [p for p in Path(binary).resolve().parent.parent.glob("cudart-*")
                  if p.is_dir() and (p / "libcudart.so.12").exists()]
    if len(candidates) != 1:
        raise ValueError("a unique pinned llama.cpp CUDA runtime directory is required")
    return candidates[0]


class Server:
    """Server inherits its cell group; tracked identities also cover escaped workers."""
    def __init__(self, command, directory: Path, *, timeout=300, env=None):
        self.directory, self.timeout = directory, timeout
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.command = [str(x).replace("{port}", str(self.port)) for x in command]
        self.env = {**os.environ, "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1",
                    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                    "PYTHONPATH": str(Path(__file__).resolve().parent),
                    "FLASHQUEST_VLLM_RUNTIME": str(directory / "runtime.json"), **(env or {})}
        if Path(command[0]).name == "llama-server":
            runtime_dir = llama_runtime_directory(command[0])
            self.env["LD_LIBRARY_PATH"] = str(runtime_dir) + os.pathsep + self.env.get("LD_LIBRARY_PATH", "")
        if "vllm.entrypoints.openai.api_server" in command:
            # Enforce the recorded build policy and matching cached-kernel versions.
            self.env["MAX_JOBS"] = str(VLLM_COMPILE_WORKERS)
            self.env["FLASHINFER_DISABLE_JIT"] = "1"
            self.env.pop("FLASHINFER_DISABLE_VERSION_CHECK", None)
            toolkits = [p for p in Path(command[0]).absolute().parent.parent.glob("lib/python*/site-packages/nvidia/cu13")
                        if (p / "bin" / "nvcc").exists()]
            if len(toolkits) != 1:
                raise ValueError("the isolated vLLM CUDA toolkit is unavailable")
            self.env["CUDA_HOME"] = str(toolkits[0])
            self.env["PATH"] = str(Path(command[0]).absolute().parent) + os.pathsep + str(toolkits[0] / "bin") + os.pathsep + self.env.get("PATH", "")
        self.known = {}
        self.stop = threading.Event()

    def track(self):
        while not self.stop.wait(0.1):
            owned_processes(processes(), self.process.pid, self.start, self.known)

    def _open(self, route, payload=None, timeout=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(f"http://127.0.0.1:{self.port}{route}", data=data,
                                         headers={"Content-Type": "application/json"})
        # Never route loopback traffic through a user's external proxy.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout or self.timeout)

    def request(self, route, payload=None, timeout=None):
        with self._open(route, payload, timeout) as f:
            return json.load(f)

    def check_health(self, timeout=None):
        """Native readiness is HTTP 200; vLLM's successful response has no JSON body."""
        with self._open("/health", timeout=timeout) as f:
            if f.status != 200:
                raise urllib.error.URLError(f"backend health returned HTTP {f.status}")

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.log = (self.directory / "server.log").open("w", encoding="utf-8")
        marker("load_start")
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=subprocess.STDOUT,
                                         env=self.env)
        self.start = processes().get(self.process.pid, {}).get("start_ticks")
        if self.start is None:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.log.close()
            raise RuntimeError("server identity unavailable")
        self.known[self.process.pid] = self.start
        self.thread = threading.Thread(target=self.track, daemon=True)
        self.thread.start()
        deadline = time.monotonic() + self.timeout
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    self.log.flush()
                    log = read_native_log(self.directory / "server.log")
                    if "CUDA out of memory" in log or "torch.OutOfMemoryError" in log:
                        raise RuntimeError("CUDA out of memory during server startup")
                    raise RuntimeError("server exited before readiness")
                try:
                    self.check_health(timeout=1)
                    marker("load_end")
                    return self
                except (OSError, urllib.error.URLError):
                    time.sleep(0.2)
            raise TimeoutError("server readiness timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *unused):
        self.stop.set()
        self.thread.join(timeout=2)
        owned_processes(processes(), self.process.pid, self.start, self.known)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            snapshot = processes()
            for pid, start in list(self.known.items()):
                if snapshot.get(pid, {}).get("start_ticks") == start:
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
            if sig == signal.SIGTERM:
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        self.process.wait(timeout=5)
        self.log.close()


def command_for(args, *, capacity):
    if args.backend == "vllm":
        return [args.vllm_python, "-m", "vllm.entrypoints.openai.api_server",
                "--host", "127.0.0.1", "--port", "{port}", "--model", args.awq_model_path,
                "--served-model-name", "validation-model", "--dtype", "float16",
                "--quantization", "awq_marlin", "--kv-cache-dtype", args.kv_dtype,
                "--worker-cls", "vllm_validation_worker.ValidationWorker",
                "--max-model-len", str(capacity), "--max-num-seqs", "1",
                "--max-num-batched-tokens", "2048", "--gpu-memory-utilization", str(args.gpu_utilization),
                "--cpu-offload-gb", "0", "--no-enable-prefix-caching", "--enable-per-request-metrics",
                "--generation-config", "vllm", "--no-enable-log-requests"]
    return [args.llama_server, "--host", "127.0.0.1", "--port", "{port}", "--model", args.gguf,
            "--ctx-size", str(capacity), "--parallel", "1", "--gpu-layers", "999",
            "--flash-attn", "on", "--cache-type-k", args.kv_dtype, "--cache-type-v", args.kv_dtype,
            "--batch-size", "2048", "--ubatch-size", "512", "--threads", "6", "--fit", "off",
            "--cache-ram", "0", "--no-webui", "--log-verbosity", "5"]


def add_backend_arguments(parser, *, select_backend=True):
    if select_backend:
        parser.add_argument("--backend", choices=["vllm", "llamacpp"], required=True)
        parser.add_argument("--kv-dtype", required=True)
    parser.add_argument("--vllm-python", default=".venv/backends/vllm/bin/python")
    parser.add_argument("--awq-model-path", required=True)
    parser.add_argument("--llama-server")
    parser.add_argument("--gguf", default="artifacts/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--gpu-utilization", type=float, default=0.70)
    parser.add_argument("--ready-timeout", type=float, default=300)
