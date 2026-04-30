# References

Compact index of every paper, repo, and code path the spec depends on. Use this as the "before writing any code, read these" list.

## Primary papers (read in this order)

1. **FlashAttention-2** — Tri Dao, 2023. [paper](https://tridao.me/publications/flash2/flash2.pdf) · [repo](https://github.com/Dao-AILab/flash-attention)
   - Foundation. Online softmax, tile-based attention, the structure every other technique extends.
2. **Quest** — Tang et al., ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774) · [repo](https://github.com/mit-han-lab/Quest)
   - Page-level top-k retrieval. Our primary sparsity algorithm.
3. **KIVI** — Liu et al., ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750) · [repo](https://github.com/jy-yuan/KIVI)
   - 2-bit asymmetric KV quant. Per-channel K, per-token V.
4. **DuoAttention** — Xiao et al., ICLR 2025. [arXiv:2410.10819](https://arxiv.org/abs/2410.10819) · [repo](https://github.com/mit-han-lab/duo-attention)
   - Per-head retrieval-vs-streaming classification. Composes with everything else.
5. **StreamingLLM** — Xiao et al., ICLR 2024. [arXiv:2309.17453](https://arxiv.org/abs/2309.17453) · [repo](https://github.com/mit-han-lab/streaming-llm)
   - Attention sinks. Fundamental to making windowed attention work past the training context.
6. **Marlin** — Frantar et al., 2024. [arXiv:2408.11743](https://arxiv.org/abs/2408.11743) · [repo](https://github.com/IST-DASLab/marlin)
   - W4A16 GEMM kernel for sm_80+. Used as-is for weight projections.
7. **EAGLE-2** — Li et al., EMNLP 2024. [paper](https://aclanthology.org/2024.emnlp-main.502/) · [repo](https://github.com/SafeAILab/EAGLE)
   - Speculative decoding. ~2–3× decode speedup, composes with our kernel.

## Secondary references

- **MInference** — Jiang et al., NeurIPS 2024 spotlight. [arXiv:2407.02490](https://arxiv.org/abs/2407.02490) · [repo](https://github.com/microsoft/MInference). Three-pattern taxonomy (A-shape, vertical-slash, block-sparse). Mostly a prefill optimization. We borrow the mental model but skip implementation in v1.
- **Block-Sparse-Attention** — MIT HAN Lab. [repo](https://github.com/mit-han-lab/Block-Sparse-Attention). FA-2 fork supporting mixed per-head patterns. Reference for the per-head dispatch API.
- **PowerInfer / PowerInfer-2** — SJTU IPADS. [arXiv:2312.12456](https://arxiv.org/abs/2312.12456) · [repo](https://github.com/SJTU-IPADS/PowerInfer). Hot/cold neuron-aware CPU/GPU hybrid. Stretch goal for v2.
- **vLLM PagedAttention** — Kwon et al., SOSP 2023. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) · [docs](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html). Block-table layout convention.

## Key code paths to read

### FlashAttention-2 Triton tutorial (canonical reference impl)
- https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
- Read end-to-end. ~500 lines. The autotune configs and online-softmax structure are our scaffolding.

### Quest
- `quest/models/QuestAttention.py` — top-k logic, criticality scoring.
- `quest/csrc/quest_attention.cu` — CUDA kernel; algorithm reference.
- `quest/utils/page_table.py` — page indexing.

### KIVI
- `models/llama_kivi.py` — patched LlamaAttention.
- `quant/new_pack.py` — pack/unpack for asymmetric quant.
- `quant/new_pack.py` + `quant/matmul.py` — Triton pack/unpack + quantized matmul. **Most directly reusable.** (Layout drifted from older spec citation of `triton_quant.py`.)

### DuoAttention
- `duo_attn/patch/llama.py` — patched attention with split-head dispatch.
- `attention_patterns/` — pre-trained head classifications. **Use these directly, don't retrain.**

### StreamingLLM
- `streaming_llm/kv_cache.py` — sink + window cache.
- `streaming_llm/pos_shift/modify_llama.py` — RoPE position shifting (subtle, easy to miss).

### Marlin
- `marlin/kernel.cu` — main mma kernel.
- `marlin/__init__.py` — packing utilities.

### EAGLE
- `eagle/model/ea_model.py` — draft+verify wrapper.
- `eagle/model/cnets.py` — small auxiliary net architecture.

### vLLM
- `csrc/attention/attention_kernels.cu` — CUDA kernel.
- `vllm/attention/backends/flash_attn.py` — block table integration.

### ExLlamaV2 (integration target)
- https://github.com/turboderp-org/exllamav2
- `exllamav2/attn.py` — current attention impl; our backend slots in here.
- `exllamav2_ext/` — C++ extension; our Triton kernel will be invoked from Python and not require C++ extension changes.

### Triton language reference
- https://triton-lang.org/main/python-api/triton.language.html
- Note especially: `tl.dot` operand types (we need int8 support), `tl.load` with masks, `tl.atomic_add` for backward (not v1).

## Evaluation tools

- **RULER** — [github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER). Standard long-context eval. Our quality acceptance bar.
- **LongBench** — [github.com/THUDM/LongBench](https://github.com/THUDM/LongBench). Broader task coverage.
- **lm-eval-harness** — [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Perplexity & general benches.

## Hardware references

- **CUDA Compute Capabilities** — [developer.nvidia.com/cuda/gpus](https://developer.nvidia.com/cuda/gpus). Confirms 3050 Ti = sm_86.
- **NVIDIA RTX 3050 Ti Laptop GPU spec** — [Notebookcheck](https://www.notebookcheck.net/NVIDIA-GeForce-RTX-3050-Ti-Laptop-GPU-Benchmarks-and-Specs.527430.0.html). 2560 cores, 16 SMs, 4GB GDDR6, 192 GB/s, 35–80W TGP.
- **Ampere whitepaper** — for tensor core specifics, shared memory limits, `cp.async` semantics.

## Auxiliary docs read on the project

- DeepGEMM Mega MoE (April 2026 release) — [repo](https://github.com/deepseek-ai/DeepGEMM), [guide](https://antigravity.codes/blog/deepgemm-guide). **Reason this is in references but not used:** Hopper/Blackwell-only. Confirms our hardware target rules out competing in the MoE-kernel space.
- DeepSeek-V4 + SGLang launch — [LMSYS blog Apr 2026](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/). Confirms Mega MoE deployment landscape.
- TMA-Adaptive FP8 Grouped GEMM — [arXiv:2508.16584](https://arxiv.org/html/2508.16584v1). Hopper-only; reference for what we *don't* do.
