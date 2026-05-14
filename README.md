# flashquest

Sparse attention runtime for 4 GB GPUs — fused INT4/TurboQuant KV + Quest top-k decode.

flashquest lets a 3 B model decode at 32 k context on a laptop GPU that has no business doing so. It combines Quest-style top-k page selection with paged INT4 (or 25 %-smaller TurboQuant K3-V3) KV cache and a fused Triton kernel that reads packed bit planes directly. Validated on RTX 3050 Ti Laptop (4 GB VRAM, sm_86) under WSL2 with Llama-3.2-3B-Instruct-AWQ.

## Why

Long-context inference is bandwidth-bound, and the cheapest bandwidth is the bandwidth you don't spend. Concretely, on a 4 GB card running a 3 B model at 32 k context:

- **vLLM 0.7.3 (AWQ-INT4, FP16 KV)** OOMs above ~4 k.
- **llama.cpp `-ngl 999` (Q4_K_M, FP16 KV)** decodes at 8 k but aborts at 32 k.
- **flashquest (AWQ-INT4 weights + paged INT4 KV + Quest top-k)** decodes at 32 k.

That's a capability gap, not a throughput gap. flashquest exists so people with cheap GPUs can run long contexts at all.

## Install

```bash
pip install -e .
```

Pinned dependencies in `pyproject.toml`. Working stack at the time of writing: Python 3.12, torch 2.5.1+cu121, triton 3.1.0, transformers 4.57.x, autoawq 0.2.9. Built and tested under WSL2 + CUDA 12.5.

## Quick start — CLI

```bash
# Single-shot, default kv-bits=4 (KIVI INT4)
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 \
           "Summarise: <your prompt here>"

# Interactive REPL with context file
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 -i \
           --context-file my_doc.txt

# TurboQuant K3-V3 (smaller cache, slower decode)
flashquest --kv-bits 3 --context 32768 -i

# Tighten retention for single-needle workloads (faster, lossier multi-needle)
flashquest --retention 0.10 --context 32768 -i
```

`--kv-bits 4` is the default (KIVI-INT4, RULER 100/100/95 at default `--retention 0.20`, faster decode). `--kv-bits 3` enables TurboQuant K3-V3 (25 % smaller cache, RULER 100/100/85, slower decode); pick it when storage is the bottleneck.

`--retention 0.20` is the default page budget; opt into `--retention 0.10` for ~1.24× decode on single-needle retrieval workloads (RULER NIAH multivalue regresses to 65 %, so don't ship it as a default for multi-needle tasks).

## Quick start — library

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
# All-retrieval head_pattern; ships with RULER 100/100/95. Replace with a
# learned 70/30 DuoAttention pattern to free up retrieval budget.
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
- **TurboQuant K3-V3 (opt-in).** Per-token Walsh-Hadamard rotation along `head_dim`, fixed 8-codepoint Lloyd-Max codebook, bit-split storage (1-bit MSB plane @ 8/byte + 2-bit LSB plane @ 4/byte). 25 % smaller cache than INT4. Two non-paper adjustments were needed on Llama-3.2-3B: per-token RMS scale (paper uses max-abs) and V at 3-bit (paper's K3-V2 multivalue regressed too far).
- **Fused Triton sparse decode kernel.** One CTA per `(batch, query head)`, decode-only `S_q=1`. Reads packed K/V tiles directly — no BF16 dequant intermediate, no GMEM codebook gather (the TurboQuant codebook is inlined as a `tl.where` chain over compile-time constants). Online-softmax accumulation, same numerics as FlashAttention.
- **Dispatcher.** `make_quest_persistent_forward` branches on `cache.kv_bits ∈ {3, 4, 8}` and routes to the right dequant + sparse kernel. INT8 is the original Phase 3 baseline; INT4 is the v1 default; TurboQuant is the storage opt-in.

## Benchmarks

### Quality — RULER NIAH 4 k subset

Llama-3.2-3B-Instruct-AWQ, all-retrieval head_pattern, n=20 vs vanilla SDPA dense.

| Cache mode | retention | niah_single | niah_multikey | niah_multivalue |
|---|---|---|---|---|
| `--kv-bits 4` (KIVI-INT4) | 0.20 (default) | 20/20 (100 %) | 20/20 (100 %)¹ | 19/20 (95 %) |
| `--kv-bits 4` (KIVI-INT4) | 0.10 (opt-in) | 20/20 (100 %) | — | 13/20 (65 %) |
| `--kv-bits 3` (TurboQuant K3-V3) | 0.25 | 20/20 (100 %) | 20/20 (100 %) | 17/20 (85 %) |

¹ `niah_multikey` at the v1.0 default carried over from Phase 6 measurements (INT4 fused, retention=0.25); single + multivalue re-measured at retention=0.20 (Phase 10 sweep).

### Throughput — single-cell decode at 32 k

| Cache mode | retention | decode tok/s | prefill tok/s | peak VRAM (MiB) |
|---|---|---|---|---|
| `--kv-bits 4` fused | 0.20 (default) | **8.41** | 65.0 | 5478 |
| `--kv-bits 4` fused | 0.10 (opt-in) | 9.91 | 65.0 | 5478 |
| `--kv-bits 3` TurboQuant | 0.25 | 2.62 | 77.6 | 6105 |

(VRAM exceeds nominal 4095 because PyTorch's allocator overcommits via WSL2 swap. The actual GPU residency stays under the limit; the rest is paged.)

### Head-to-head — Llama-3.2-3B, RTX 3050 Ti Laptop, 4 GB

Decode tok/s (32 k checked first):

| Backend | Quant | 8 k | 32 k | 128 k fits? |
|---|---|---|---|---|
| flashquest INT4 fused | AWQ-INT4 + INT4 paged | 4.94² | **8.41** | ✗ |
| flashquest TurboQuant K3-V3 | AWQ-INT4 + K3-V3 paged | 2.05 | 1.93 | ✗ |
| llama.cpp `-ngl 999` | Q4_K_M, FP16 KV | 38.45 | timeout | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | timeout | OOM |

² flashquest 8 k carried over from Phase 6 head-to-head matrix; flashquest 32 k re-measured at v1.0 (Phase 10 default `--retention 0.20`). llama.cpp + vLLM cells unchanged — those backends have not been re-baselined.

flashquest is the only backend that decodes at 32 k on this hardware. At 8 k, llama.cpp wins on raw throughput (its CUDA graph + persistent kernel infrastructure is well-tuned); flashquest is the capability play, not the throughput play.

Reproduce:

```bash
python scripts/bench_flashquest.py --ctx-len 32768 --kv-bits 4 \
    --out benchmarks/decode_int4.json
python scripts/phase6_run_ruler_4k_int4.py        # quality eval
python scripts/phase6_run_headtohead.py           # multi-backend matrix
```

## Non-goals

- Training kernels. Inference only.
- Datacenter GPUs. Hopper/Blackwell-only features (TMA, WGMMA, FP8) are explicitly skipped.
- Beating FlashAttention-3. Not in that league and don't need to be.
- 8 B at 32 k. Llama-3.1-8B AWQ is ~4.5 GiB; doesn't fit alongside any KV cache on a 4 GB card. Revisit on a 12+ GB GPU.

## Caveats

- WSL2 + CUDA 12.5 only (the build hasn't been tested on bare metal Linux or Windows). `torch.cuda.OutOfMemoryError` may show inflated `peak_vram_mib` because the allocator double-counts swapped memory; trust `nvidia-smi` for the on-GPU residency number.
- Decode-only fused kernel. Prefill uses dense BF16 SDPA on the dequant'd cache (with `enable_gqa=True` to keep SDPA on the Flash backend at long ctx). For the bench, pass `logits_to_keep=1` to skip the 7.83 GiB lm_head allocation that the bench discards anyway.
- Quality validated only on Llama-3.2-3B-Instruct-AWQ. Other Llama-family models with the same head_dim (64 or 128) should work; non-Llama architectures need their own dispatcher patch.
