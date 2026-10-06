# Pinned competitor validation

The validation adapters use loopback-only native completion servers with exact
input token arrays. Native prefill/decode and client elapsed times remain separate;
they have different boundaries from FlashQuest's synchronized forwards. The
historical V0 adapter and head-to-head records are not current competitor evidence.

Use a separate environment for vLLM:

```bash
uv venv --python .venv/bin/python .venv/backends/vllm
uv pip install --python .venv/backends/vllm/bin/python -r requirements-vllm.txt
uv --no-cache pip check --python .venv/backends/vllm/bin/python
```

This resolves vLLM 0.30.0, Torch 2.13.0/CUDA 13.0, Triton 3.7.1, Transformers
5.18.0 and NumPy 2.3.5 on the available Linux machine. Every result fingerprints
the full isolated package set and interpreter. Keep the existing Torch 2.5.1 AWQ
environment for FlashQuest. See [vLLM installation](https://docs.vllm.ai/en/v0.30.0/getting_started/installation/gpu/)
and [same-request metrics](https://docs.vllm.ai/en/v0.30.0/features/per_request_metrics/)
for the pinned upstream contracts.

The compiler components are pinned together: nvcc, NVVM and CRT 13.0.88,
with CCCL 13.0.85. An unconstrained NVVM 13.4.92 installation generated PTX 9.4
that the 13.0 assembler rejected. Pinning NVVM resolves that version mismatch,
but a minimal compile still fails because CUDA 13.0 headers conflict with this
host's glibc 2.43 `rsqrt` exception declarations. Compilation is not validated.
The installed environment and both failure logs remain local setup evidence.
See the [CUDA 13.0 Update 1 component table](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-toolkit-release-notes/index.html).

The bounded setup uses the matching precompiled
`flashinfer-jit-cache==0.6.18.post1+cu130` from the
[official CUDA 13.0 index](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/),
with `flashinfer-python==0.6.18.post1` and version checking enabled.
The adapters force `FLASHINFER_DISABLE_JIT=1` in the vLLM child and bind
that effective policy in the child/schedule identities. Missing or unloadable
precompiled modules remain explicit failures; this route does not establish
successful host compilation or native execution. Both FP16/E4M3 prefill modules
pass CPU load and installed wheel-record hashes; 199 packages pass dependency
checks. Accept runtime execution only after
successful intended-precision requests, runtime observations and cleanup.

The external llama.cpp CUDA 12.8 binary release is
[b11382](https://github.com/ggml-org/llama.cpp/releases/tag/b11382), commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6`. Its binaries and shared libraries are
fingerprinted in results. Set `LLAMA_SERVER` to that external executable; there is
no vendored repository. Set `AWQ_SNAPSHOT` to the pinned snapshot under
`artifacts/hf-cache/`. The public
[Q4_K_M GGUF](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/tree/5ab33fa94d1d04e903623ae72c95d1696f09f9e8)
is stored at `artifacts/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf`;
its SHA256 is `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`.
This is the same model family with a different weight quantization from AWQ.

For a fresh short performance smoke:

```bash
.venv/bin/python scripts/run_competitors.py \
  --awq-model-path "$AWQ_SNAPSHOT" --llama-server "$LLAMA_SERVER" \
  --contexts 1024 --seeds 0 --reps 1 --n-decode 8
```

The longer performance block defaults to contexts 8192/32768, seeds 0–3, one
full warmup per seed and three repetitions. Each context uses a Latin rotation
of backend order across four input seeds, with a fresh server for each cell.
Exact synthetic input hashes are frozen and checked against every repetition.
Output count 128 includes the first output
from prefill, with 127 later forwards. No prefix cache, speculation or CPU weight
offload is requested. vLLM uses a declared 0.70 memory utilization and 2048-token
prefill chunks; llama.cpp uses a 512-token physical microbatch, full layer offload,
Flash Attention and explicit Q4 or FP16 K/V. Those allocation policies differ.

For matching quality, pass complete quality records for both contexts:

```bash
.venv/bin/python scripts/run_competitors.py --mode quality \
  --awq-model-path "$AWQ_SNAPSHOT" --llama-server "$LLAMA_SERVER" \
  --quality benchmarks/validation/quality/<8k-identity>/quality.json \
            benchmarks/validation/quality/<32k-identity>/quality.json
```

Substitute actual identities; all raw manifests and decoded answers remain ignored.
The GGUF adapter checks every used/special token mapping and every full decoded
prompt, plus resolved EOG behavior. Quality stops at EOS or 128 outputs; performance
forces its output count. Failures and unavailable measurements remain in the schedule.

The worker observer captures actual vLLM model/cache devices, logical/storage KV
types and scale values after its first worker execution, which may be an internal
startup warmup. It does not claim input-calibrated FP8 scales. Authoritative kernel-selection
logs identify the realized weight implementation. Physical-device memory and owned
worker-tree RSS cover each cell; load, warmup and whole-request windows are separate.
Native metric durations do not supply aligned prefill/decode timestamps, so isolated
server phase memory stays unmeasured. Configured placement and sampling on 12 GB do
not establish no OS fallback or physical fit on 4 GB.

Schedules and cells are written to `benchmarks/validation/competitors/<identity>/`.
Local server/error logs, raw records and telemetry series remain under `artifacts/`.
Never overwrite an attempt; identical completed schedules may be resumed, and a
changed backend, source, model or setting requires a fresh identity. Independent audit
is recorded in [independent-roadmap-audit.md](independent-roadmap-audit.md).

Native readiness checks HTTP 200, through the same direct-loopback opener as
requests. Pinned vLLM returns an empty successful health response; llama.cpp may
return JSON. Completion and detokenize responses still require valid JSON.
Transport errors and HTTP 503 keep readiness polling within its deadline, with
owned-worker cleanup on timeout. The interrupted ninth attempt remains evidence
of the former JSON-only readiness mismatch, not a failed native inference result.

Native diagnostic logs may contain arbitrary generated token bytes. Read them
with reversible UTF-8 `surrogateescape`; raw files stay unchanged, and re-encoding
recovers every byte. Structured HTTP responses and worker JSON retain strict
decoding, as do runtime precision, device, kernel and EOS checks. Full performance
attempt 1 remains failed because two cells encountered the former strict log
decoder after requests completed. A corrected source requires fresh complete
measurements; historical failed cells remain ineligible.
