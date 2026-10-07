# flashquest

Experimental sparse-attention decode runtime for Llama-3.2-3B. It combines AWQ INT4 weights, a
packed INT4 (or 3-bit TurboQuant) KV cache, and Quest-style page selection with fused Triton
kernels.

- **Status:** research concluded on 2026-10-06; kept as an engineering reference. In a
  validation on one GPU, FlashQuest did not beat llama.cpp or vLLM on speed, memory
  or retrieval quality, so this direction is no longer being developed.
  See the [research decision](docs/research-decision.md).
- **Run:** `pip install -e ".[bench,dev]"`, then `flashquest --model casperhansen/llama-3.2-3b-instruct-awq --context 8192 -i`
  ([details](#quick-start--cli)).
- **Test:** `python -m pytest -m 'not slow'` (606 tests; needs a CUDA GPU and two small local
  fixtures, see [Tests](#tests)).

## Results at a glance

These measurements come from an RTX 4080 Laptop GPU (12 GB) running Llama-3.2-3B-Instruct.
Each run decodes 127 tokens after the prompt. Values are medians over four input seeds with three
timed runs each. Full report: [descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md).

**Speed and memory**

| Configuration | Decode, 8k prompt | Decode, 32k prompt | Prefill, 32k | KV cache, 32k | Sampled GPU peak, 32k |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlashQuest, sparse INT4 | 37.1 tok/s | 35.0 tok/s | 7.18 s | 991 MiB | 7,006 MiB |
| FlashQuest, all pages INT4 | 38.1 tok/s | 19.2 tok/s | 7.18 s | 991 MiB | 7,006 MiB |
| llama.cpp, Q4_0 KV | 130.9 tok/s | 85.0 tok/s | 7.51 s | 1,016 MiB | 3,368 MiB |
| llama.cpp, FP16 KV | 120.3 tok/s | 66.2 tok/s | 7.19 s | 3,612 MiB | 5,878 MiB |
| vLLM, FP8 KV | 130.7 tok/s | 90.1 tok/s | 6.22 s | ≈5.7 GiB pool | 8,980 MiB |
| vLLM, FP16 KV | 116.7 tok/s | 64.6 tok/s | 6.39 s | ≈5.7 GiB pool | 8,588 MiB |

**Retrieval quality** (hits out of 100 for single / multikey / multivalue needle tasks; same 300
prompts for every configuration)

| Configuration | 8k | 32k |
| --- | --- | --- |
| AWQ model, dense FP16 KV (FlashQuest's reference) | 100 / 98 / 97 | 100 / 91 / 91 |
| FlashQuest, sparse INT4 | 100 / 96 / 89 | 100 / 88 / 89 |
| FlashQuest, all pages INT4 | 100 / 98 / 96 | 100 / 90 / 90 |
| llama.cpp, Q4_0 KV | 100 / 100 / 95 | 100 / 90 / 94 |
| llama.cpp, FP16 KV | 100 / 100 / 97 | 100 / 97 / 96 |
| vLLM, FP8 KV | 100 / 98 / 97 | 100 / 90 / 91 |
| vLLM, FP16 KV | 100 / 98 / 97 | 100 / 91 / 91 |

What this shows:

- **Speed.** FlashQuest decoded 2.4–3.5× slower than llama.cpp and vLLM. Each engine's timer
  covers a slightly different span (synchronized forward passes, native evaluation, scheduler token
  timestamps), so treat the gap as approximate rather than a precise speedup. The measurements also
  don't isolate which parts of FlashQuest cause it.
- **Page selection.** Reading 25% of pages decodes 1.82× faster than reading all of them at 32k,
  across every seed. At 8k there's no gain (0.98×). The all-pages mode still runs page scoring and
  top-k selection, so it's an internal control rather than an optimized dense baseline.
- **Quality.** Before collecting the main quality data, we froze a rule: sparse accuracy must
  stay within 10 points of dense, with every one of nine lower confidence bounds above −10 points.
  The nine endpoints are three tasks at 4k, 8k and 32k, with 100 examples each. Five pass. Multivalue
  at all three contexts and multikey at 32k fail, with observed drops of 2–8 points. So this
  failure means comparable quality isn't established; it doesn't prove a loss larger than 10 points.
- **Memory.** The KV caches are about the same size (991 MiB vs. llama.cpp's 1,016 MiB). The
  difference is in total runtime memory: FlashQuest's 32k prefill alone exceeds 4 GB. vLLM
  preallocates a cache pool sized to its memory budget, so its peak reflects that budget.
- **Metadata reuse.** Scoring pages from the INT4 quantization metadata avoids storing separate
  min/max summaries. That saves 12.5% of the packed INT4 K payload (6.25% of K+V) at similar scoring
  latency, though selections sometimes differ from exact summaries.
- **Novelty.** Prior work already combines sparse page selection with low-bit KV caches and uses
  compressed keys as retrieval indexes ([prior-work map](docs/contribution-prior-work.md)).

Weights differ between engines: llama.cpp uses a Q4_K_M GGUF, while FlashQuest and vLLM use the
same AWQ checkpoint. FlashQuest used retention 0.20 at 8k and 0.25 at 32k.

**4 GB GPUs.** Earlier versions of this README claimed 32k decoding on a 4 GB GPU. That claim
is withdrawn: those runs allocated about 5.5 GB, and the current implementation's 32k peak also
exceeds 4 GB. Testing on 4 GB hardware was not pursued. This is a decision, not a hardware-tested
failure, and the 8k case (3.4 GB sampled peak) remains untested there.

## Install

```bash
pip install -e ".[bench,dev]"
# Validated stack: add -c requirements-validation.txt
```

Tested with Python 3.12, Torch 2.5.1 (CUDA 12.4), Triton 3.1.0, Transformers 4.57.6, AutoAWQ
0.2.9 and Accelerate 1.15.0 on an Ada-generation laptop GPU. v1.0 development targeted an Ampere
laptop GPU under WSL2. The vLLM comparison runs in a separate environment
([`requirements-vllm.txt`](requirements-vllm.txt)).

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

# Lower retention: faster, but lossier on multi-needle retrieval
flashquest --model casperhansen/llama-3.2-3b-instruct-awq --retention 0.10 --context 32768 -i
```

`--kv-bits 4` and `--retention 0.20` are the defaults. The validated 32k setting is retention
0.25, because 0.20 failed the 32k quality screen. `--kv-bits 3` enables TurboQuant K3-V3, whose
packed K/V payload is 25% smaller than INT4 before metadata and staging. With a 32k prompt, the
benchmark harness peaked at about 7 GB of GPU memory.

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

For TurboQuant K3-V3, swap `PersistentInt4KVCache` for `PersistentTurboKVCache` (same constructor
signature, `kv_bits=3`). The dispatcher branches automatically.

## How it works

- **Quest-style top-k page selection.** Each page's channel-wise quantization parameters
  (`K_scale`, `K_mn`) supply the endpoints for the score
  `Q·K_mn + levels·relu(Q)·K_scale`, where `levels` is 255 (INT8) or 15 (INT4). Scoring takes two
  matmuls, with no separate min/max summaries and no key dequantization. Rounded affine ranges can
  differ from the original key extrema, so the score isn't guaranteed to bound the original keys.
  Retention selects a fraction of completed pages; sink pages, recent pages and the unquantized
  tail are always included.
- **Paged INT8/INT4 KV cache.** KIVI-style asymmetric quantization: per-page channel-wise K,
  per-token V. INT4 packs two values per byte. Partial pages stay in BF16 until they fill.
- **TurboQuant K3-V3 (opt-in).** Per-token Walsh-Hadamard rotation, a fixed 8-point Lloyd-Max
  codebook and bit-split storage. Two changes from the paper were needed on Llama-3.2-3B: per-token
  RMS scale and 3-bit V. This mode stores separate min/max metadata for scoring, so metadata reuse
  doesn't apply to it.
- **Fused Triton decode kernel.** One program per (batch, query head) reads packed K/V tiles
  directly, without a BF16 dequantization buffer, and accumulates with online softmax. The INT4
  path is checked against an FP32 reconstruction of the same packed values.
- **Prefill** uses PyTorch SDPA on the dequantized cache; only decode uses the sparse kernel.
- **Dispatcher.** `make_quest_persistent_forward` routes `cache.kv_bits ∈ {3, 4, 8}` to the
  matching kernels.

## Tests

```bash
# Full non-slow suite: unit, validation-tooling and GPU kernel tests (606 pass, 2026-10-06)
HF_HUB_OFFLINE=1 HF_HUB_CACHE=artifacts/hf-cache PYTHONPATH=src python -m pytest -m 'not slow'

# CPU-only validation tooling
CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_phase6_headtohead.py tests/test_quality_validation.py \
    tests/test_validation_stats.py tests/test_validation_ablation.py tests/test_gpu_memory.py \
    tests/test_competitor_backend.py tests/test_competitor_validation.py
```

The non-slow suite expects two local fixtures:

- the `unsloth/Llama-3.2-1B-Instruct` tokenizer and config files (no weights) in the Hugging Face
  cache above. Without them, `tests/test_eval_niah.py` errors.
- the DuoAttention `full_attention_heads.tsv` pattern for Llama-3.1-8B-Instruct, under
  `vendor/duo-attention/attn_patterns/`. `scripts/vendor_clone.sh` clones the DuoAttention
  repository there.

The 12 tests marked `slow` load real model weights (AWQ 3B and Llama-3.2-1B-Instruct). They were
not run in the 2026-10-06 validation. CI runs `tests/test_chat.py` on CPU.

## Reproducing the validation

The published results were measured at commit `5e22bf5`, using
[`requirements-validation.txt`](requirements-validation.txt) and
[`requirements-vllm.txt`](requirements-vllm.txt). The repository holds the JSON records and
summaries. Raw prompts, model answers and memory traces are not included.

```bash
# Matched dense / all-pages / sparse retrieval run (content-addressed output directory)
python scripts/phase6_run_ruler_4k_int4.py \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --ctx-len 8192 --seeds 0 --retentions 0.20 1.0

# Balanced sparse vs. all-pages timing block; needs a passing quality record for the same context
python scripts/run_validation_ablation.py --contexts 8192 --retention 0.20 \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae --quality "$QUALITY_8K"
```

Competitor setup, pinned versions and runner commands are in
[docs/competitor-validation.md](docs/competitor-validation.md). [scripts/README.md](scripts/README.md)
explains which scripts are current, and [benchmarks/README.md](benchmarks/README.md) maps the
evidence folders.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/flashquest/` | Runtime: caches, kernels, Llama patches, evaluation helpers, CLI |
| `tests/` | Unit, kernel and validation-tooling tests |
| `scripts/` | Benchmark, validation and historical experiment scripts ([index](scripts/README.md)) |
| `benchmarks/` | Saved results: `validation/` is current evidence, `phase*` files are v1.0 history ([index](benchmarks/README.md)) |
| `docs/` | Decision, competitor setup, prior work and process records |
| `data/` | Needle-in-a-haystack filler corpus |

## Documentation

- [Research decision](docs/research-decision.md): the conclusion and the evidence behind each part.
- [Descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md):
  the full performance, quality and memory report.
- [Competitor validation](docs/competitor-validation.md): llama.cpp and vLLM setup and contracts.
- [Closest prior work](docs/contribution-prior-work.md).
- [Roadmap](roadmap.md): the closed research plan with outcomes.
- Process records: [implementation plan](docs/roadmap-implementation-plan.md),
  [execution checkpoint](docs/execution-checkpoint.md) and
  [independent audit](docs/independent-roadmap-audit.md). These are detailed working logs, kept
  for provenance.

## Historical results (v1.0, May 2026)

The v1.0 release reported 32k decoding on a 4 GB RTX 3050 Ti Laptop GPU under WSL2, with
competitor comparisons. Those numbers don't hold up:

- The 32k runs allocated 5,478 MiB, more than the labelled 4 GB card.
- The comparison runner hardcoded the hardware metadata.
- llama.cpp and vLLM were timed with mismatched methods: llama.cpp's decode ran at empty context,
  and vLLM mixed prefill into decode throughput.
- Two decode figures quoted in the old README (8.41 and 9.91 tok/s) have no saved measurement.

The original files remain under `benchmarks/phase*` for provenance only. The validation above
replaces them.

## Limitations

- Validated only on the Llama-3.2-3B family, with AWQ weights (and Q4_K_M GGUF for llama.cpp).
  Other models need their own kernel, mapping and quality checks.
- Quality evidence is limited to synthetic needle-retrieval tasks; general language quality wasn't
  measured.
- Decode-only sparse kernel, single request, inference only. FlashQuest kernels don't use
  Hopper/Blackwell-specific features.

## License

Apache License 2.0; see [LICENSE](LICENSE).
