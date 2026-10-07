# flashquest

Sparse-attention decode runtime for Llama-3.2-3B: fused INT4/TurboQuant KV + Quest top-k page
selection, written in Triton.

**Status: research closed (2026-10-06). Kept up as an engineering reference.**

The pitch was a 3B model decoding at 32k context on a 4 GB laptop GPU that has no business doing
so. A proper validation says it doesn't hold up. On the same GPU, llama.cpp and vLLM decode faster,
llama.cpp fits 32k in less memory, and both match or beat flashquest's retrieval. The code still
works; it just isn't a win. Full write-up: [research decision](docs/research-decision.md).

- **Run:** `pip install -e ".[bench,dev]"`, then
  `flashquest --model casperhansen/llama-3.2-3b-instruct-awq --context 8192 -i`
  ([more](#quick-start--cli))
- **Test:** `python -m pytest -m 'not slow'` runs 606 tests. You need a CUDA GPU and two small local
  fixtures ([details](#tests)).

## Results at a glance

Everything here ran on one RTX 4080 Laptop GPU (12 GB) with Llama-3.2-3B-Instruct
([device details](#validation-device)). Each run decodes 127 tokens after the prompt. Numbers are
medians over 4 input seeds × 3 timed runs. Full report:
[descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md).

**Speed and memory**

| Configuration | Decode, 8k prompt | Decode, 32k prompt | Prefill, 32k | KV cache, 32k | Sampled GPU peak, 32k |
| --- | ---: | ---: | ---: | ---: | ---: |
| flashquest, sparse INT4 | 37.1 tok/s | 35.0 tok/s | 7.18 s | 991 MiB | 7,006 MiB |
| flashquest, all pages INT4 | 38.1 tok/s | 19.2 tok/s | 7.18 s | 991 MiB | 7,006 MiB |
| llama.cpp, Q4_0 KV | 130.9 tok/s | 85.0 tok/s | 7.51 s | 1,016 MiB | 3,368 MiB |
| llama.cpp, FP16 KV | 120.3 tok/s | 66.2 tok/s | 7.19 s | 3,612 MiB | 5,878 MiB |
| vLLM, FP8 KV | 130.7 tok/s | 90.1 tok/s | 6.22 s | ≈5.7 GiB pool | 8,980 MiB |
| vLLM, FP16 KV | 116.7 tok/s | 64.6 tok/s | 6.39 s | ≈5.7 GiB pool | 8,588 MiB |

**Retrieval quality**: hits out of 100 on single / multikey / multivalue needle tasks. Every
configuration answers the same 300 prompts.

| Configuration | 8k | 32k |
| --- | --- | --- |
| AWQ model, dense FP16 KV (flashquest's reference) | 100 / 98 / 97 | 100 / 91 / 91 |
| flashquest, sparse INT4 | 100 / 96 / 89 | 100 / 88 / 89 |
| flashquest, all pages INT4 | 100 / 98 / 96 | 100 / 90 / 90 |
| llama.cpp, Q4_0 KV | 100 / 100 / 95 | 100 / 90 / 94 |
| llama.cpp, FP16 KV | 100 / 100 / 97 | 100 / 97 / 96 |
| vLLM, FP8 KV | 100 / 98 / 97 | 100 / 90 / 91 |
| vLLM, FP16 KV | 100 / 98 / 97 | 100 / 91 / 91 |

What that means:

- **It's slower.** llama.cpp (Q4_0 KV) and vLLM (FP8 KV) decode at roughly 2.4–3.5× flashquest's
  rate, and their FP16 KV setups still come in around 1.8–3.5×. Each engine times a slightly
  different span, so treat those as ballpark ratios, not exact ones. The runs also don't pin down
  which part of flashquest causes the gap.
- **Page selection works, but only at long context.** At 32k, reading a quarter of the pages
  decodes 1.82× faster than reading all of them, on every seed. At 8k it's a wash (0.98×). The
  all-pages mode still pays for page scoring and top-k, so it's an internal control rather than a
  tuned dense baseline.
- **Quality is close but not proven equal.** I froze the rule before the main run: sparse has to
  stay within 10 points of dense on all nine checks (3 tasks × 4k/8k/32k, 100 examples each). Five
  passed. Multivalue at every length and multikey at 32k missed, with actual drops of 2–8 points.
  So "not shown to be as good," not "proven 10+ points worse."
- **Same-size cache, bigger everything else.** The KV cache is basically llama.cpp's size (991 vs.
  1,016 MiB at 32k). The extra memory is runtime overhead: the 32k prefill alone goes past 4 GB.
  vLLM peaks higher, but that's the pool it reserves up front, not what it actually needs.
- **The metadata trick saves a little.** Scoring pages straight from the INT4 quantization
  metadata skips separate min/max summaries. That's 12.5% of the packed K payload (6.25% of K+V)
  at about the same scoring speed. Selections sometimes differ from exact summaries.
- **It's not new.** Prior work already pairs sparse page selection with low-bit KV caches and uses
  compressed keys as retrieval indexes ([prior-work map](docs/contribution-prior-work.md)).

llama.cpp runs a Q4_K_M GGUF; flashquest and vLLM share the same AWQ checkpoint. flashquest uses
retention 0.20 at 8k and 0.25 at 32k.

**About the 4 GB claim.** v1.0 said it could decode 32k on a 4 GB GPU. That's withdrawn: those
runs allocated about 5.5 GB, and the current 32k peak is over 4 GB too. I chose not to test on
4 GB hardware, so that's a call, not a measured failure. An 8k run (3.4 GB peak) on a 4 GB card is
still untested.

## Validation device

Everything above ran on an Alienware m16 R1 running Ubuntu directly, with no WSL. The RTX 3050 Ti +
WSL2 setup only applies to the withdrawn v1.0 numbers.

| Component | Specification |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 **Laptop** GPU; Ada, compute capability 8.9 |
| GPU memory | 12 GB class; `nvidia-smi` reports 12,282 MiB |
| CPU | Intel Core i9-13900HX; 24 cores, 32 threads |
| RAM reported by Linux | About 6.9 GiB usable; 4 GiB swap |
| OS | Ubuntu 26.04.1 LTS, x86-64; Linux `7.0.0-38-generic` |
| NVIDIA driver | `595.91.07` |
| flashquest Python stack | Python 3.12.14; Torch 2.5.1+cu124 (CUDA 12.4); Triton 3.1.0 |

vLLM runs in its own environment with its own CUDA 13 runtime and pins; see
[competitor validation](docs/competitor-validation.md).

## Install

```bash
pip install -e ".[bench,dev]"
# Exact validated stack: add -c requirements-validation.txt
```

Tested with Python 3.12, Torch 2.5.1 (CUDA 12.4), Triton 3.1.0, Transformers 4.57.6, AutoAWQ 0.2.9
and Accelerate 1.15.0. v1.0 was built on an Ampere laptop GPU under WSL2. The vLLM comparison has
its own requirements file ([`requirements-vllm.txt`](requirements-vllm.txt)).

## Quick start — CLI

```bash
# Single-shot, default kv-bits=4 (KIVI INT4)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 \
           --prompt "Summarise: <your prompt here>"

# Interactive REPL with a context file
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 -i \
           --context-file my_doc.txt

# TurboQuant K3-V3 (smaller cache, slower decode)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq --kv-bits 3 --context 32768 -i

# Lower retention: faster, but worse on multi-needle retrieval
flashquest --model casperhansen/llama-3.2-3b-instruct-awq --retention 0.10 --context 32768 -i
```

Defaults are `--kv-bits 4` and `--retention 0.20`. For 32k, use 0.25: that's the setting that
passed the quality screen, and 0.20 didn't. `--kv-bits 3` turns on TurboQuant K3-V3, whose packed
K/V payload is 25% smaller than INT4 before metadata and staging. Plan on about 7 GB of GPU memory
for a 32k prompt; that's what the benchmark harness peaked at.

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

For TurboQuant K3-V3, swap `PersistentInt4KVCache` for `PersistentTurboKVCache` (same constructor,
`kv_bits=3`). The dispatcher picks the right kernels on its own.

## How it works

- **Quest-style top-k page selection.** Each page's channel-wise quantization parameters
  (`K_scale`, `K_mn`) give the endpoints for the score `Q·K_mn + levels·relu(Q)·K_scale`, where
  `levels` is 255 for INT8 and 15 for INT4. That's two matmuls: no separate min/max summaries and
  no key dequantization. Rounded affine ranges can drift from the original key extrema, so the score
  isn't a guaranteed upper bound on the original keys. Retention picks a fraction of the completed
  pages, and sink pages, recent pages and the unquantized tail always get included.
- **Paged INT8/INT4 KV cache.** KIVI-style asymmetric quantization: per-page channel-wise K and
  per-token V. INT4 packs two values per byte. Partial pages stay in BF16 until they fill.
- **TurboQuant K3-V3 (opt-in).** Per-token Walsh-Hadamard rotation, a fixed 8-point Lloyd-Max
  codebook and bit-split storage. Llama-3.2-3B needed two changes from the paper: a per-token RMS
  scale and 3-bit V. This mode stores its own min/max metadata for scoring, so the metadata trick
  doesn't apply to it.
- **Fused Triton decode kernel.** One program per (batch, query head) reads packed K/V tiles
  directly, with no BF16 dequantization buffer, and accumulates with online softmax. The INT4 path
  is checked against an FP32 rebuild of the same packed values.
- **Prefill** runs PyTorch SDPA on the dequantized cache. Only decode uses the sparse kernel.
- **Dispatcher.** `make_quest_persistent_forward` routes `cache.kv_bits ∈ {3, 4, 8}` to the
  matching kernels.

## Tests

```bash
# Non-slow suite: unit, validation-tooling and GPU kernel tests (606 passed, 2026-10-06)
HF_HUB_OFFLINE=1 HF_HUB_CACHE=artifacts/hf-cache PYTHONPATH=src python -m pytest -m 'not slow'

# CPU-only validation tooling
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_phase6_headtohead.py tests/test_quality_validation.py \
    tests/test_validation_stats.py tests/test_validation_ablation.py tests/test_gpu_memory.py \
    tests/test_competitor_backend.py tests/test_competitor_validation.py
```

The non-slow suite needs two local fixtures:

- the `unsloth/Llama-3.2-1B-Instruct` tokenizer and config files (no weights) in the Hugging Face
  cache above. Without them, `tests/test_eval_niah.py` errors out.
- the DuoAttention `full_attention_heads.tsv` pattern for Llama-3.1-8B-Instruct, under
  `vendor/duo-attention/attn_patterns/`. `scripts/vendor_clone.sh` clones the DuoAttention repo
  there.

The 12 `slow` tests load real model weights: AWQ 3B and Llama-3.2-1B-Instruct. All 12 passed on
2026-10-07, in 28.5 s at commit `3fc9097`. CI only runs `tests/test_chat.py` on CPU.

`scripts/run_slow_tests.sh` runs them offline against the pinned model cache listed in
`artifacts/setup/slow-test-models.json`:

```bash
bash scripts/run_slow_tests.sh --check   # make sure the cache is ready; collects tests, doesn't run them
tmux new-session -d -s flashquest-slow-tests -c "$PWD" 'bash scripts/run_slow_tests.sh --run'
```

`--check` still needs a free CUDA GPU, because some test modules set up GPU constants when they're
imported. `--run` refuses to start if other compute jobs are using the GPU, and it keeps the machine
awake while pytest runs. It saves `pytest.log`, `results.xml`, source info and the real exit code in
a new `artifacts/tests/slow-*` folder. Check those files for the result; launching the job doesn't
mean it passed.

## Reproducing the validation

The published results were measured at commit `5e22bf5` with
[`requirements-validation.txt`](requirements-validation.txt) and
[`requirements-vllm.txt`](requirements-vllm.txt). The repo has the JSON records and summaries.
Raw prompts, model answers and memory traces aren't included.

```bash
# Matched dense / all-pages / sparse retrieval run (content-addressed output folder)
python scripts/phase6_run_ruler_4k_int4.py \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --ctx-len 8192 --seeds 0 --retentions 0.20 1.0

# Balanced sparse vs. all-pages timing block; needs a passing quality record for the same context
python scripts/run_validation_ablation.py --contexts 8192 --retention 0.20 \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae --quality "$QUALITY_8K"
```

Competitor setup, pinned versions and runner commands are in
[docs/competitor-validation.md](docs/competitor-validation.md).
[scripts/README.md](scripts/README.md) says which scripts are current, and
[benchmarks/README.md](benchmarks/README.md) maps the evidence folders.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/flashquest/` | Runtime: caches, kernels, Llama patches, evaluation helpers, CLI |
| `tests/` | Unit, kernel and validation-tooling tests |
| `scripts/` | Benchmark, validation and historical experiment scripts ([index](scripts/README.md)) |
| `benchmarks/` | Saved results: `validation/` is the current evidence, `phase*` files are v1.0 history ([index](benchmarks/README.md)) |
| `docs/` | Decision, competitor setup, prior work and process records |
| `data/` | Needle-in-a-haystack filler corpus |

## Documentation

- [Research decision](docs/research-decision.md): the conclusion and the evidence behind each piece.
- [Descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md):
  the full performance, quality and memory report.
- [Competitor validation](docs/competitor-validation.md): llama.cpp and vLLM setup and contracts.
- [Closest prior work](docs/contribution-prior-work.md).
- [Roadmap](roadmap.md): the closed research plan and how each step turned out.
- Process records: [implementation plan](docs/roadmap-implementation-plan.md),
  [execution checkpoint](docs/execution-checkpoint.md) and
  [independent audit](docs/independent-roadmap-audit.md). These are detailed working logs, kept for
  provenance.

## Historical results (v1.0, May 2026)

v1.0 claimed 32k decoding on a 4 GB RTX 3050 Ti Laptop GPU under WSL2 and came with competitor
comparisons. Those numbers don't hold up:

- The 32k runs allocated 5,478 MiB, which is more than the labelled 4 GB card holds.
- The comparison runner hardcoded the hardware info.
- llama.cpp and vLLM were timed the wrong way: llama.cpp's decode ran at empty context, and vLLM
  mixed prefill into its decode number.
- Two decode figures in the old README (8.41 and 9.91 tok/s) don't have a saved measurement
  behind them.

The original files stay under `benchmarks/phase*` for the record. The validation above replaces
them.

## Limitations

- Only validated on the Llama-3.2-3B family with AWQ weights (plus Q4_K_M GGUF for llama.cpp).
  Other models need their own kernel, mapping and quality checks.
- Quality evidence covers synthetic needle-retrieval tasks only; general language quality wasn't
  measured.
- Decode-only sparse kernel, single request, inference only. The kernels don't use
  Hopper/Blackwell-specific features.

## License

Apache License 2.0; see [LICENSE](LICENSE).
