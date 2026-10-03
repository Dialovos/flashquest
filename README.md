# flashquest

Experimental sparse attention runtime for memory-constrained GPUs: AWQ weights, packed INT4/TurboQuant KV, and Quest page selection.

The current direction is to validate whether this integration offers a useful memory/quality/speed tradeoff against dense quantized KV. The broad combination builds on [Quest](https://arxiv.org/abs/2406.10774), [KIVI](https://arxiv.org/abs/2402.02750), and [TurboQuant](https://arxiv.org/abs/2504.19874). Reusing quantization metadata for page scoring is an engineering contribution candidate; research novelty and a competitive advantage remain unproven.

The implementation includes persistent packed caches and fused Triton decode kernels. Historical reports label the hardware as an RTX 3050 Ti Laptop under WSL2, but the original comparison runner hardcoded that metadata. Their allocator measurements exceed 4 GB, and the saved competitor timings need correction, so they do not establish a GPU-resident 32 k capability advantage on a 4 GB device.

Run the CLI examples below, or use the quality harness for matched retrieval examples. CPU validation checks: `python -m pytest tests/test_phase6_headtohead.py tests/test_quality_validation.py`. GPU checks: `python -m pytest tests/test_bench_flashquest.py tests/test_sparse_int4.py tests/test_persistent_int4.py tests/test_page_scores_int8.py -m 'not slow'`.

2026-10-03 local validation: 48 CPU checks and 26 targeted GPU checks passed on Linux with an RTX 4080 Laptop GPU (12 GB). The pinned AWQ model passed the initial 4k retrieval screen at retention 0.20: 20/20 single, 20/20 multikey, and 18/20 multivalue; dense and all-pages INT4 scored 20/20 throughout. Expanded quality, competitive speed, novelty, and 4 GB capacity remain unproven.

See the [research roadmap](roadmap.md) for the remaining experiments, priorities, and decision criteria.

## Install

```bash
pip install -e ".[bench,dev]"
```

For the validated pilot stack, constrain the installation with
`pip install -e ".[bench,dev]" -c requirements-validation.txt`.
The local working stack uses Torch 2.5.1 (CUDA 12.4), Triton 3.1.0,
Transformers 4.57.6, AutoAWQ 0.2.9, and Accelerate 1.15.0. AutoAWQ's Triton
GEMM path was verified; installing a separate CUDA extension was unnecessary.

Dependency ranges are in `pyproject.toml`. Historical working stack: Python 3.12, torch 2.5.1+cu121, triton 3.1.0, transformers 4.57.x, autoawq 0.2.9. Built and tested under WSL2 + CUDA 12.5.

## Quick start — CLI

```bash
# Single-shot, default kv-bits=4 (KIVI INT4)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 \
           --prompt "Summarise: <your prompt here>"

# Interactive REPL with context file
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 -i \
           --context-file my_doc.txt

# TurboQuant K3-V3 (smaller cache, slower decode)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq --kv-bits 3 --context 32768 -i

# Tighten retention for single-needle workloads (faster, lossier multi-needle)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq --retention 0.10 --context 32768 -i
```

`--kv-bits 4` and `--retention 0.20` are the runtime defaults. The saved quality results below distinguish measurements at 0.20 and 0.25; they do not establish all three quality scores at the default setting. `--kv-bits 3` enables TurboQuant K3-V3, whose packed K/V payload is 25 % smaller than INT4; metadata and staging reduce the total-cache saving.

At retention 0.10, the saved 4 k multivalue retrieval result drops to 13/20. Treat lower retention as a workload-specific experiment.

## Quick start — library

```python
import torch
from flashquest.cache import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model

model, tokenizer = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
cfg = model.config
head_dim = cfg.hidden_size // cfg.num_attention_heads

cache = PersistentInt4KVCache(
    batch_size=1,
    num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads,
    head_dim=head_dim,
    max_seq_len=32_768,
    page_size=64,
    device="cuda",
)
# All-retrieval head pattern. Quality depends on retention and context length.
pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
patch_llama_for_quest_persistent(
    model, cache=cache, head_pattern=pattern,
    retention=0.20, num_sinks=4, window_pages=2, page_size=64,
)

ids = tokenizer("...", return_tensors="pt").input_ids.cuda()
out = model.generate(ids, max_new_tokens=128, use_cache=True)
print(tokenizer.decode(out[0]))
```

For TurboQuant K3-V3, swap `PersistentInt4KVCache` for `PersistentTurboKVCache` (same constructor signature, `kv_bits=3`). The dispatcher branches automatically.

## Architecture

- **Quest top-k page selection.** Per-page channel-wise (`K_scale`, `K_mn`) lets us bound the per-page max QK score algebraically: `Σ_d max(Q[d]·K_min[p,d], Q[d]·K_max[p,d])`. We rewrite that bound as `Q·K_mn + 255·relu(Q)·K_scale` (INT8) or `15·relu(Q)·K_scale` (INT4) — two matmuls per layer, no dequant, no per-token criticality. `retention=0.20` (default) reads ~one page in five.
- **Paged INT8/INT4 KV cache.** KIVI-style asymmetric quantization: per-page channel-wise K, per-token V. INT8 packs 1 byte/value, INT4 packs 2 nibbles/byte. Persistent across decode steps; partial pages live in BF16 staging until they fill.
- **TurboQuant K3-V3 (opt-in).** Per-token Walsh-Hadamard rotation along `head_dim`, fixed 8-codepoint Lloyd-Max codebook, bit-split storage (1-bit MSB plane @ 8/byte + 2-bit LSB plane @ 4/byte). 25 % smaller packed K/V payload than INT4, before metadata and staging. Two non-paper adjustments were needed on Llama-3.2-3B: per-token RMS scale (paper uses max-abs) and V at 3-bit (paper's K3-V2 multivalue regressed too far).
- **Fused Triton sparse decode kernel.** One CTA per `(batch, query head)`, decode-only `S_q=1`. Reads packed K/V tiles directly — no BF16 dequant intermediate, no GMEM codebook gather (the TurboQuant codebook is inlined as a `tl.where` chain over compile-time constants). Online-softmax accumulation, same numerics as FlashAttention.
- **Dispatcher.** `make_quest_persistent_forward` branches on `cache.kv_bits ∈ {3, 4, 8}` and routes to the right dequant + sparse kernel. INT8 is the original Phase 3 baseline; INT4 is the v1 default; TurboQuant is the storage opt-in.

## Benchmarks and validation

### Current quality — matched 4k pilot

Pinned Llama-3.2-3B-Instruct-AWQ revision
`272b3bde867b606760447deb9a4d2719fbdfd3ae`, seed 0, 20 examples per task.
All arms use the same 60 input manifests. Dense uses the same AWQ weights with
native FP16 KV; the INT4 arms share one persistent cache and fused attention path.

| Arm | Single | Multikey | Multivalue |
| --- | --- | --- | --- |
| Dense FP16 KV | 20/20 | 20/20 | 20/20 |
| All-pages INT4, retention 1.0 | 20/20 | 20/20 | 20/20 |
| Sparse INT4, retention 0.20 | 20/20 | 20/20 | 18/20 |

The sparse arm passes the existing per-task screen of at least 85% of dense hits.
The two multivalue misses reached the 128-token generation limit. Actual input
lengths are 3,839–3,963 tokens; “4k” is the nominal budget. This small retrieval
pilot supports proceeding at 0.20; it does not establish statistical equivalence,
general language quality, competitive throughput, or target-device capacity.

[Saved per-example evidence](benchmarks/validation/quality/dd8f60de24c2957a2120473d9ff3a194b64b22d85b2adfecca3aab485092f104/quality.json)
includes model/tokenizer content, source/environment identity, protocol, actual
lengths, generated-output hashes, and all 180 outcomes. Raw prompts/answers stay
under ignored `artifacts/quality/`. The separate 1024-token two-example smoke
completed every arm but sparse multivalue scored 1/2; it remains local diagnostic
evidence, not a passing quality result.

```bash
# New runs resolve immutable model identity and choose a content-based output path.
python scripts/phase6_run_ruler_4k_int4.py \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --ctx-len 4096 --n-samples 20 --seeds 0 --retentions 0.20 1.0

# Add --resume to continue an identical run. Changed inputs/code/models get a new identity.
```

The harness writes atomic progress after each task/arm, rejects mismatched or
corrupt resume cells, and restores original attention methods before freeing the
cache. Explicit output paths cannot overwrite another run identity. Execution
errors return nonzero; `--require-screen-pass` also returns nonzero on a failed
or inconclusive screen. Pilot seed 0 is reserved for tuning; confirmation uses
fresh seeds after its protocol is frozen.

### Historical quality — RULER NIAH 4 k subset

Llama-3.2-3B-Instruct-AWQ, all-retrieval head pattern, 20 examples per task. These small retrieval subsets do not establish general long-context quality. A dash means no saved result for that exact configuration.

| Cache mode | retention | niah_single | niah_multikey | niah_multivalue |
|---|---|---|---|---|
| INT4 | 0.25 | 20/20 | 20/20 | 20/20 |
| INT4 (runtime default) | 0.20 | — | — | 19/20 |
| INT4 | 0.10 | 20/20 | — | 13/20 |
| TurboQuant K3-V3 | 0.25 | 20/20 | 20/20 | 17/20 |

Sources: [INT4 0.25](benchmarks/phase6_ruler_4k_int4.json), [retention sweep](benchmarks/phase10_retention/), and [TurboQuant](benchmarks/phase7_ruler_4k_turbo.json).

### Historical throughput — 32 k input

| Saved run | retention | decode tok/s | prefill tok/s | Peak PyTorch allocated MiB |
|---|---|---|---|---|
| [INT4 head-to-head cell](benchmarks/phase6_cells_int4/flashquest_32768.json) | 0.25 | 3.88 | 66.1 | 5478 |
| [INT4 fused single cell](benchmarks/phase6_decode_int4_fused.json) | 0.25 | 3.32 | 65.0 | 5478 |
| [TurboQuant single cell](benchmarks/phase7_decode_turbo_32k.json) | 0.25 | 2.62 | 77.6 | 6105 |

These are historical single runs. Allocated memory is not physical GPU residency or evidence of automatic offload. Earlier README claims of 8.41 and 9.91 tok/s have no matching saved measurement and are excluded.

The old head-to-head tables are not a reliable comparison: llama.cpp's `-p CTX -n 128` timed separate prefill and empty-context decode tests; vLLM mixed prefill and decode in its throughput calculation. The original files remain under `benchmarks/` for provenance.

### Reproduce corrected comparisons

The runner writes new results to `benchmarks/validation/`, preserving the historical phase files. It measures 128 generated tokens as 127 post-prefill decode steps, warms FlashQuest/vLLM, and reports median throughput across three repetitions. FlashQuest and vLLM share seeded synthetic input IDs; llama-bench uses its own token fixture at the same cache depth. These are throughput workloads, not quality evaluations.

```bash
# Inspect the bounded 8 k / 32 k matrix first.
python scripts/phase6_run_headtohead.py --contexts 8192 32768 --dry-run

# External llama.cpp build and matching GGUF; vLLM is installed separately.
export LLAMA_BIN=/path/to/llama-bench
export MODEL=/path/to/Llama-3.2-3B-Instruct-Q4_K_M.gguf
python scripts/phase6_run_headtohead.py --contexts 8192 32768

# All-pages INT4 ablation: isolate the effect of page selection in this runtime.
python scripts/phase6_run_headtohead.py --backends flashquest \
    --contexts 8192 32768 --retention 1.0 \
    --output-dir benchmarks/validation-dense

# Three-arm quality evidence with an immutable output directory.
python scripts/phase6_run_ruler_4k_int4.py --retentions 0.20 1.0 --seeds 0
```

The all-pages ablation uses the same fused kernel and page-selection machinery; it is not an independently optimized dense implementation. Compare it with llama.cpp Q4 KV and vLLM FP8 KV before drawing a performance conclusion. llama.cpp requires a build supporting `-d`, quantized K/V, and Flash Attention. The vLLM adapter still targets V0 `RequestMetrics`; migrating it to a pinned current release with same-request server metrics is pending in the implementation plan. Do not use the legacy adapter as evidence against a current competitor. `--llamacpp-kv f16` and `--vllm-kv-cache-dtype auto` retain FP16 baseline options.

Quality resume checks source/model/environment/protocol identity and matched sample counts. The matrix's `--skip-existing` currently reruns legacy adapters because complete identity cannot be resolved before launch; backend identity integration remains pending. Raw benchmark logs and original backend JSON stay under ignored `artifacts/benchmarks/`; exported records contain normalized evidence and error categories. Timeout/error cells remain failures. GPU name and total memory come from the machine running the matrix. PyTorch allocated/reserved bytes are labeled separately; physical residency and competitor peak memory require additional measurement.

The next research gate is longer-context quality at retention 0.20, then balanced repeated sparse/all-pages comparisons at 8 k and 32 k. Validate on an actual 4 GB GPU before claiming that capacity target.

## Non-goals

- Training kernels. Inference only.
- Datacenter GPUs. Hopper/Blackwell-only features (TMA, WGMMA, FP8) are explicitly skipped.
- Beating FlashAttention-3. Not in that league and don't need to be.
- 8 B at 32 k. Llama-3.1-8B AWQ is ~4.5 GiB; doesn't fit alongside any KV cache on a 4 GB card. Revisit on a 12+ GB GPU.

## Caveats

- Historical model runs used WSL2 + CUDA 12.5. Allocator counters, physical GPU residency, and system memory are different measurements; the saved records do not prove GPU-only residency or a managed offload path.
- Decode-only fused kernel. Prefill uses dense BF16 SDPA on the dequant'd cache (with `enable_gqa=True` to keep SDPA on the Flash backend at long ctx). For the bench, pass `logits_to_keep=1` to skip the 7.83 GiB lm_head allocation that the bench discards anyway.
- Quality validated only on Llama-3.2-3B-Instruct-AWQ. Other Llama-family models with the same head_dim (64 or 128) should work; non-Llama architectures need their own dispatcher patch.
