# flashquest — Specification

**Last updated:** 2026-04-29
**Status:** design / pre-implementation
**Target hardware:** NVIDIA RTX 3050 Ti Laptop GPU (GA107, sm_86), Intel i9-12900H, 16GB RAM, WSL2

---

## 1. Goal

Make 32k–128k context inference of 3B–8B models *actually usable* on a 4GB-VRAM laptop GPU. Today this is either impossible (OOM) or slow (CPU fallback in llama.cpp). We close that gap with a single fused Triton kernel that composes five published SOTA techniques into one inference path.

The deliverable is a Python package + Triton kernel + thin HF Transformers / ExLlamaV2 integration. Not a paper. Not a production service.

## 2. Hardware constraints (the design driver)

```
GPU: NVIDIA RTX 3050 Ti Laptop GPU
  - Architecture: Ampere GA107, compute capability 8.6 (sm_86)
  - SMs: 16 (vs 80 on H100)
  - CUDA cores: 2560
  - Tensor cores: 80 (3rd gen) — BF16/FP16/INT8/INT4
  - Shared memory: 48KB / SM (vs 228KB on H100)
  - L2 cache: 2MB
  - VRAM: 4GB GDDR6 @ ~192 GB/s
  - TGP: 35–80W (laptop, throttles)
  - Effective usable VRAM: ~3.0–3.3GB (Windows display + browser eats ~700MB-1GB)

CPU: Intel i9-12900H, 6P+8E cores, 20 threads, AVX2 + AVX-VNNI (no AVX-512, no AMX)
RAM: 16GB total, ~10GB available for inference workloads
OS: Windows + WSL2 (Linux 6.6); CUDA + Triton work natively in WSL2
```

What this rules out:
- FP8 / FP4 paths — no hardware support.
- TMA bulk async copies — Hopper-only; use `cp.async` (Ampere variant).
- WGMMA — Hopper-only; use `mma.sync` 16×8×16 BF16 fragments.
- ThunderKittens advanced tile primitives — Hopper-targeted.
- DeepGEMM Mega MoE — Hopper/Blackwell only.
- FlashAttention-3 — Hopper-only.
- Models >8B at any reasonable quant on GPU-only.
- Multi-GPU anything.

What's available and well-supported on sm_86:
- BF16/FP16/INT8 tensor cores (`tl.dot` with int8 operands works).
- `cp.async.bulk` (Ampere version).
- Triton ≥ 2.x with full Ampere tuning paths.
- CUTLASS 2.x (and 3.x with sm_80+ targets).
- FlashAttention-2 (its native target).
- Marlin W4A16 (designed for sm_80+, [Marlin README](https://github.com/IST-DASLab/marlin) calls this out explicitly).
- All KIVI / Quest / DuoAttention / StreamingLLM algorithms (algorithmic, hardware-agnostic).

## 3. VRAM budget

Working budget: **3.0GB**. Concrete fit at 32k context with sparse KV at 10% retention:

| Model | Weight quant | Weights | Sparse KV (INT8) | Activations + scratch | Total | Fits? |
|---|---|---|---|---|---|---|
| Llama-3.2-3B | Q4_K_M | 1.9GB | 70MB | 150MB | ~2.1GB | ✅ comfortable |
| Qwen2.5-3B | Q4_K_M | 1.8GB | 50MB | 150MB | ~2.0GB | ✅ comfortable |
| Llama-3.1-8B | IQ3_XXS | 2.3GB | 100MB | 200MB | ~2.6GB | ✅ tight |
| Llama-3.1-8B | IQ4_XS | 3.4GB | 100MB | 200MB | ~3.7GB | ❌ needs CPU offload |
| Mistral-7B | IQ4_XS | 3.2GB | 80MB | 200MB | ~3.5GB | ❌ needs CPU offload |

Reference numbers for "dense KV" at 32k (GQA Llama-3.1-8B, 8 KV heads, 32 layers, head_dim 128, FP16):
```
KV_bytes_dense = 32_000 × 8 × 32 × 128 × 2 (K+V) × 2 (FP16) ≈ 4.2 GB
KV_bytes_int8  = 32_000 × 8 × 32 × 128 × 2          × 1     ≈ 2.1 GB
KV_bytes_int8_sparse_10% ≈ 210 MB
```

This 20× KV reduction (FP16 dense → INT8 + 10% sparse) is the entire reason this project is feasible.

## 4. The composed kernel — high-level architecture

A single Triton kernel per attention layer that does, per query tile:

```
1. Load Q tile (BLOCK_M queries) from HBM → registers.
2. Compute per-block summary scores: Q · block_summaries → block_scores.
   (block_summaries are min/max per channel per page, precomputed at KV insert time.)
3. Score adjustment: + always-attend (sink tokens, sliding window).
4. Top-k block selection per query head (Quest-style).
5. Per-head pattern dispatch (DuoAttention):
     - retrieval heads → top-k blocks
     - streaming heads → sink + window only
6. Initialize online softmax state (m_i, l_i, acc_i) for each query.
7. For each selected KV block:
     a. cp.async load INT8 K block + scales → shared mem (double-buffered)
     b. Dequant K to BF16 in registers
     c. QK^T via mma.sync (BF16 tensor cores)
     d. Online softmax update (FA-2 style)
     e. cp.async load INT8 V block + scales → shared mem
     f. Dequant V to BF16
     g. softmax · V via mma.sync, accumulate into acc_i
8. Final normalize, write output, write LSE (for logsumexp-aware downstream ops).
```

Tile shapes for sm_86 with 48KB shared mem:
- `BLOCK_M = 64` (queries per kernel block)
- `BLOCK_N = 64` (KV tokens per block — matches Quest's page granularity)
- `BLOCK_K = head_dim` (typically 64, 96, or 128)
- `num_warps = 4`
- Double-buffered KV loads → ~32KB shared mem usage per kernel block, leaves margin.

For comparison: H100-tuned kernels typically use `BLOCK_M=128, BLOCK_N=128`. We're forced smaller by both shared mem and SM count, but Ampere tensor cores accept this without major efficiency loss.

## 5. SOTA technique inventory (composed, with what we crib)

For each technique below: paper one-liner, repo, *specific files to study*, what we copy, what we adapt, what we skip.

### 5.1 Quest — query-aware page retrieval
- **Paper:** [Quest (ICML 2024)](https://arxiv.org/abs/2406.10774) — "Query-Aware Sparsity for Efficient Long-Context LLM Inference"
- **Repo:** https://github.com/mit-han-lab/Quest
- **Files to study:**
  - `quest/models/QuestAttention.py` — top-k selection logic, criticality scoring.
  - `quest/csrc/quest_attention.cu` — their CUDA kernel; we'll write the Triton equivalent but the algorithm is here.
  - `quest/utils/page_table.py` — how pages are organized and indexed.
- **What we crib:**
  - Page-level granularity (default 16-token pages; we'll use 64 for tile alignment).
  - Min/max channel stats per page as the criticality signal.
  - Top-k page selection per query.
- **What we change:**
  - Page size 16 → 64 (matches our `BLOCK_N=64`).
  - Their kernel is FP16 KV; we run on INT8 KV (KIVI-style).
- **What we skip:**
  - Their multi-GPU dispatch.
- **Reported numbers:** up to 7.03× self-attention speedup, 2.23× end-to-end latency reduction, "negligible accuracy loss." Tested on long-dependency tasks.

### 5.2 KIVI — 2-bit asymmetric KV quantization
- **Paper:** [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) — "A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
- **Repo:** https://github.com/jy-yuan/KIVI
- **Files to study:**
  - `models/llama_kivi.py` — the patched LlamaAttention with KIVI's K/V handling.
  - `quant/new_pack.py` — the per-channel-K, per-token-V pack/unpack.
  - `quant/triton_quant.py` — Triton kernels for quant/dequant; **directly relevant** for our pipeline.
- **What we crib:**
  - Per-channel K quantization (channel-wise scales, fits Ampere INT8 tensor cores).
  - Per-token V quantization (per-row scales, append-friendly for autoregressive inference).
  - Asymmetric (zero-point) quant — better than symmetric at 2-bit.
- **What we change:**
  - Start at INT8 (8-bit) for v1 — simpler, Triton mma supports INT8 directly. Move to INT4/INT2 in v2.
  - Their pack format → adapt to our PagedAttention-style block layout.
- **What we skip:**
  - Their FP16 partial residual buffer for the most-recent few tokens — we'll keep KV INT8 throughout for simplicity, accept slight quality cost.
- **Reported numbers:** 2.6× peak memory reduction, 2.35–3.47× throughput, "almost the same quality" on Llama / Mistral / Falcon at 2-bit.

### 5.3 DuoAttention — retrieval/streaming head split
- **Paper:** [DuoAttention (ICLR 2025)](https://arxiv.org/abs/2410.10819) — "Efficient Long-Context LLM Inference with Retrieval and Streaming Heads"
- **Repo:** https://github.com/mit-han-lab/duo-attention
- **Files to study:**
  - `duo_attn/patch/llama.py` — patched LlamaAttention with split-head dispatch.
  - `duo_attn/utils.py` — head-classifier loading and per-head mask construction.
  - `attention_patterns/` — pre-trained per-head retrieval-vs-streaming classifications for popular models.
- **What we crib:**
  - Head-level classification: which heads need full attention (retrieval) vs which can use streaming (sinks + window).
  - Their pre-classified head lists for Llama-3-8B, Mistral-7B etc. — saves us their training step.
- **What we change:**
  - Where they dispatch to two separate kernels, we dispatch *within* one kernel via per-head branching. Less launch overhead.
- **What we skip:**
  - The optimization-based head classification training. Use their pre-trained patterns.
- **Reported numbers:** 2.55× memory reduction (MHA), 1.67× (GQA), 2.18×/1.50× decode speedup, 1.73×/1.63× prefill speedup. Llama-3-8B at 3.3M context on a single A100.

### 5.4 StreamingLLM — attention sinks
- **Paper:** [StreamingLLM (ICLR 2024)](https://arxiv.org/abs/2309.17453) — "Efficient Streaming Language Models with Attention Sinks"
- **Repo:** https://github.com/mit-han-lab/streaming-llm
- **Files to study:**
  - `streaming_llm/kv_cache.py` — sink + window cache management.
  - `streaming_llm/pos_shift/modify_llama.py` — RoPE position shifting for windowed attention (critical detail).
- **What we crib:**
  - First-N-tokens always-attend ("sink tokens", typically N=4).
  - Sliding-window-of-recent-M-tokens always-attend.
  - **The position-shift trick:** when keeping sinks + window, the *positions* you pass to RoPE are not the original token positions — they're remapped to start at 0 for sinks, then continue past the window. Easy to miss.
- **What we change:**
  - Compose with Quest-selected blocks: `attended_set = sinks ∪ window ∪ top_k_quest_blocks`.
- **Reported numbers:** stable performance up to 4M tokens, 22.2× speedup vs sliding-window-recompute baseline.

### 5.5 Marlin — W4A16 GEMM kernel
- **Paper:** [Marlin (2024)](https://arxiv.org/abs/2408.11743) — "Mixed-Precision Auto-Regressive Parallel Inference"
- **Repo:** https://github.com/IST-DASLab/marlin
- **Files to study:**
  - `marlin/kernel.cu` — the main mma kernel; sm_80+ specific.
  - `marlin/__init__.py` — pack/unpack utilities; how 4-bit weights are tiled and interleaved.
  - `marlin/marlin_cuda_kernel.cu:gemm_kernel` — async copy + double-buffered tile loading.
- **What we crib:**
  - Their W4A16 packing layout (groups of 128, group-wise scales).
  - Pipelined async loads + tensor core compute pattern.
- **What we change:**
  - We use Marlin **as-is** for the linear layers (Q/K/V projections, FFN). We do not rewrite it. We just call it.
  - Our novel kernel is the *attention*, not the GEMM. Marlin handles the rest.
- **What we skip:**
  - Sparse-Marlin (2:4 structured sparsity) — orthogonal, optional v2.
- **Reported numbers:** ~3.9× over FP16 on A10 at batch ≤32. Sweet spot for our single-user laptop inference.

### 5.6 FlashAttention-2 (Triton tutorial) — baseline + scaffolding
- **Paper:** [FA-2 (2023)](https://tridao.me/publications/flash2/flash2.pdf)
- **Reference impl:** https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
- **Files to study:**
  - The full file — it's the canonical Triton FA-2 reference, ~500 lines, very readable.
  - Pay attention to: `_attn_fwd` kernel layout, `triton.autotune` configs, online softmax with `m_i`/`l_i` running stats, `tl.dot` accumulator dtype handling.
- **What we crib:**
  - The whole online-softmax structure. Our kernel is FA-2 with a sparse outer loop bolted on.
  - Their autotune config patterns for sm_80+.
  - Their `BLOCK_M`/`BLOCK_N` parameterization style.
- **What we change:**
  - Outer loop iterates over *selected* KV blocks (from Quest), not all blocks.
  - K/V loads include INT8 dequant step in shared mem.
  - Per-head pattern dispatch (DuoAttention).
- **Reference quality:** this is the implementation everyone forks. Tri Dao's own Triton FA reference.

### 5.7 Block-Sparse-Attention — mixed-pattern reference
- **Repo:** https://github.com/mit-han-lab/Block-Sparse-Attention
- **Files to study:**
  - `block_sparse_attn/flash_attn_interface.py` — high-level Python API.
  - `csrc/block_sparse_attn/` — kernel sources, modified from FA 2.4.2 to support mixed patterns per head.
- **What we crib:**
  - Their per-head pattern API design (different heads, different masks).
  - Their block-mask format.
- **What we change:**
  - Their kernel is CUDA C++; we're Triton. Different language but same algorithm.
- **What we skip:**
  - Their build system (FA fork). Triton is much simpler to build.

### 5.8 MInference — pattern catalog (reference, mostly skipped for v1)
- **Paper:** [MInference (NeurIPS 2024 spotlight)](https://arxiv.org/abs/2407.02490)
- **Repo:** https://github.com/microsoft/MInference
- **What we crib:**
  - Their three-pattern taxonomy: A-shape, vertical-slash, block-sparse. Useful mental model.
  - Block-sparse pattern logic (similar to ours).
- **What we skip in v1:**
  - The vertical-slash and A-shape patterns. They require per-head pattern *training* on representative inputs, and the gain over Quest+DuoAttention is small for the hardware we target.
- **Reported numbers:** 10× prefill speedup at 1M context on A100. Largely a *prefill* optimization; we care more about *decode* on a laptop.

### 5.9 EAGLE-2/3 — speculative decoding multiplier
- **Paper:** [EAGLE-2 (EMNLP 2024)](https://arxiv.org/abs/2406.16858), [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/abs/2503.01840)
- **Repo:** https://github.com/SafeAILab/EAGLE
- **Files to study:**
  - `eagle/model/ea_model.py` — the draft+verify wrapper that calls into HF transformers.
  - `eagle/model/cnets.py` — the small auxiliary draft network.
- **What we crib:**
  - Drop-in wrapper around any HF model. Our kernel sits inside, EAGLE wraps outside.
- **What we change:**
  - Nothing in the kernel layer. EAGLE composes cleanly with any attention impl.
- **What we skip in v1:**
  - Pre-training the EAGLE draft network. Use their published checkpoints for Llama-3.
- **Reported numbers:** ~3× decode speedup. Composes multiplicatively with our memory savings.

### 5.10 vLLM PagedAttention — KV layout reference
- **Repo:** https://github.com/vllm-project/vllm
- **Files to study:**
  - `csrc/attention/attention_kernels.cu` — their CUDA kernel layout.
  - `vllm/attention/backends/flash_attn.py` — block table integration.
  - Docs: https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html.
- **What we crib:**
  - Block-table indirection for non-contiguous KV (logical → physical block mapping).
  - This is *the* layout convention for modern LLM serving. Adopting it lets us interoperate with vLLM/SGLang in v3.
- **What we change:**
  - Our block size is 64 (Quest alignment); vLLM defaults to 16. Configurable.
- **What we skip:**
  - Multi-request / multi-tenant scheduling. We're single-user.

### 5.11 PowerInfer — CPU offload (stretch / Phase 4 only)
- **Repo:** https://github.com/SJTU-IPADS/PowerInfer
- **Why it's relevant:** for fitting Mistral-7B IQ4_XS (3.2GB weights, won't fit alone with KV) we need CPU offload of the cold layers.
- **What we crib:**
  - Their hot/cold neuron classification → maps to our hot/cold *layer* offloading.
  - llama.cpp's existing `-ngl` partial offload also works; PowerInfer is the smarter version.
- **Status:** v2 stretch. Not v1.

## 6. Phased build plan

**Total expected duration: 8–12 weeks of focused part-time work. ≤$15 cloud spend.**

### Phase 0 — Setup & baselines (1–2 days)
- Verify CUDA + Triton in WSL2: `nvidia-smi`, `python -c "import torch, triton; print(torch.cuda.is_available(), triton.__version__)"`.
- Build llama.cpp with CUDA. Run Llama-3.2-3B Q4_K_M at 8k context. Record tok/s, VRAM use.
- Install vLLM. Run same model at same context. Record tok/s.
- Install reference impls locally: `mit-han-lab/Quest`, `jy-yuan/KIVI`, `mit-han-lab/duo-attention`, `mit-han-lab/streaming-llm`, `IST-DASLab/marlin`. Just clone them, get them importable.
- Set up `nsys`, `ncu` profilers. Run a baseline FA-2 profile.
- **Deliverable:** README.md with baseline numbers in a table.

### Phase 1 — Eager Python reference (Week 1)
- Pure-PyTorch implementation of Quest-style attention (no Triton yet).
- Wire as `attn_implementation="flashquest_eager"` in HF Transformers via `LlamaAttention` subclass.
- Use Llama-3.2-1B (smallest model — fast iteration).
- Quality harness: RULER 4k subset, Wikitext perplexity.
- **Win condition:** matches dense within 1% perplexity at top-25% retention, within 3% at top-10%.
- **Reference impls to crib from:** `mit-han-lab/Quest/quest/models/QuestAttention.py`.

### Phase 2 — First Triton kernel: dense FA-2 baseline (Weeks 2–3)
- Port the [Triton 06-fused-attention](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py) tutorial onto our 3050 Ti.
- Tune block sizes for sm_86 (smaller than the tutorial defaults).
- Benchmark vs upstream FA-2: target ≥70% of FA-2 perf on the 3050 Ti.
- This phase proves the Triton scaffolding works. Don't add features yet.
- **Win condition:** numerical equivalence with `torch.nn.functional.scaled_dot_product_attention`, perf within 30% of FA-2 on sm_86.

### Phase 3 — Add sparse retrieval + INT8 KV (Weeks 4–5)
- Block-summary precompute kernel (per-channel min/max per 64-token block).
- Top-k selection (start with `torch.topk` outside the kernel; fuse into kernel later if it's a bottleneck).
- Sparse outer loop: kernel iterates over selected blocks only.
- INT8 KV storage with per-channel-K, per-token-V scales. Dequant inside the kernel (in shared mem).
- StreamingLLM sinks + sliding window (always-attended blocks, free in our kernel).
- **Win condition:** Llama-3.2-3B at 32k context, ≥10 tok/s decode, ≥85% of dense RULER 32k.
- **Reference impls to crib from:** `jy-yuan/KIVI/quant/triton_quant.py`, `mit-han-lab/streaming-llm/streaming_llm/kv_cache.py`.

### Phase 4 — DuoAttention head split + bigger model (Weeks 6–7)
- Per-head pattern dispatch inside the kernel.
- Load DuoAttention's pre-trained head classifications for Llama-3-8B from their repo.
- Use `IST-DASLab/marlin` for W4A16 weight projections.
- Get Llama-3.1-8B at IQ3_XXS quant fitting in VRAM (~2.3GB weights + ~100MB sparse KV at 32k).
- **Win condition:** 8B at 32k, ≥4 tok/s, ≥80% RULER.
- **Reference impls to crib from:** `mit-han-lab/duo-attention/duo_attn/patch/llama.py`, `IST-DASLab/marlin`.

### Phase 5 — Integration + speculation (Weeks 8–10)
- ExLlamaV2 attention backend integration. Their API has cleaner extension points than HF Transformers (per their `exllamav2_ext` C++ extension model).
- Optional: EAGLE-2 wrapper for ~2× decode speedup.
- Push KV quant to INT4 (full KIVI). Validate quality.
- **Win condition:** 8B at 64k context, ≥6 tok/s decode with speculation, ≥80% RULER 64k.
- **Reference impls to crib from:** `SafeAILab/EAGLE`, `turboderp-org/exllamav2`.

### Phase 6 — Polish & release (Weeks 11–12)
- README with benchmark tables vs llama.cpp, vLLM, ExLlamaV2 baselines.
- One-command demo: `flashquest chat --model llama-3.2-3b --context 128k`.
- Clean separation: kernel (Triton), Python wrapper (PyTorch), runtime adapter (HF/ExLlama).
- Optional: upstream PRs to ExLlamaV2 or vLLM. Or maintain as independent fork.

## 7. Test & evaluation methodology

### 7.1 Numerical equivalence (correctness)
- Per-kernel: vs eager Python reference, BF16 tolerance (rtol=1e-2, atol=1e-2).
- Per-layer: vs HF dense attention output on identical inputs.
- End-to-end: vs HF dense generation, same prompt, same seed, max-tokens=10. Tokens must match.
- Property-based shape testing via `hypothesis`: random `(B, H, S, D)` within budget.

### 7.2 Quality benchmarks
- **RULER** ([github.com/NVIDIA/RULER](https://github.com/NVIDIA/RULER)) at 4k, 8k, 16k, 32k, 64k. The standard long-context eval.
- **LongBench** (chinese-english, broader task coverage).
- **Perplexity on Wikitext-2 / PG19** at our target context lengths.
- Acceptance threshold per phase win condition.

### 7.3 Performance benchmarks
- Decode tok/s at multiple context lengths (8k, 16k, 32k, 64k, 128k).
- Prefill latency at same.
- Peak VRAM usage (`torch.cuda.max_memory_allocated`).
- All baselines: dense FA-2, llama.cpp CUDA, vLLM, ExLlamaV2.
- Profile critical kernels with `ncu` to validate occupancy and tensor-core usage.

### 7.4 Cross-validation on Colab
- Free T4 (sm_75) — verify the algorithm works on a different gen (Turing). T4 has tensor cores but no `cp.async`; mostly verifies *correctness* not perf.
- Optional: Pro A100 ($10–15) — see how the same kernel scales on more capable hardware. Mostly to validate that our perf isn't accidentally tied to a 3050 Ti quirk.

## 8. Risks & open questions

| Risk | Severity | Mitigation |
|---|---|---|
| Triton on sm_86 lacks features we need | Medium | Phase 0 verification; have a CUTLASS fallback path. |
| INT8 mma quality at 2-bit KV | Medium | Stage at INT8 first (v1), only push to INT4/INT2 in v2 with strict eval gate. |
| 4GB VRAM still too tight even with IQ3 | Medium | Phase 4 is the test; if 8B doesn't fit, drop to 7B Mistral or IQ3_S Llama. |
| Top-k inside kernel is slow | Low | Compute outside kernel first; fuse later only if profile shows it matters. |
| ExLlamaV2 hooks insufficient | Low | Fall back to HF Transformers `attn_implementation` API. |
| Thermal throttling masks perf gains | Low | Run benchmarks multiple times; report sustained, not peak. |
| WSL2 CUDA quirks | Low | Well-supported in 2026; community has solved most issues. |

Open questions to resolve in Phase 0:
1. Does Triton ≥ 3.x on sm_86 fully support `tl.dot` with INT8 operands? (Answer expected: yes, but verify.)
2. Can we fuse INT8 dequant directly into the `mma.sync` operand path, or do we need an explicit BF16 staging buffer? (Profile-dependent.)
3. What's our actual usable VRAM after Windows + Edge + IDE? (Run `nvidia-smi` after a normal session — the spec assumes 3.0–3.3GB.)
4. Do `ncu`/`nsys` work in WSL2 for kernel profiling? (Recent CUDA versions: yes. Verify on this exact machine.)

## 9. Naming, license, distribution

- **Working name:** `flashquest`. Open to rename.
- **License:** Apache 2.0 (matches FlashAttention, Marlin, Quest, KIVI — clean ecosystem fit).
- **Distribution:** GitHub repo + PyPI package. No need for a separate website.
- **Acknowledgments:** must credit Quest, KIVI, DuoAttention, StreamingLLM, Marlin, FlashAttention, EAGLE in README and any paper write-ups.

## 10. Project structure (proposed)

```
flashquest/
├── README.md
├── pyproject.toml
├── docs/
│   ├── SPEC.md                     # this file
│   ├── REFERENCES.md               # paper/repo/code-path index
│   ├── BENCHMARKS.md               # results table, updated per phase
│   └── PHASES/
│       └── phase-N-notes.md        # per-phase journal
├── src/flashquest/
│   ├── __init__.py
│   ├── kernel/
│   │   ├── triton_attn.py          # the main Triton kernel
│   │   ├── triton_quant.py         # KIVI-style quant/dequant
│   │   └── block_summary.py        # block stats precompute
│   ├── cache/
│   │   ├── paged_kv.py             # PagedAttention-style block table
│   │   └── duo_split.py            # head classification + dispatch
│   ├── model/
│   │   ├── llama_patch.py          # HF LlamaAttention subclass
│   │   └── exllama_backend.py      # ExLlamaV2 integration
│   └── runtime/
│       ├── inference.py            # generate() wrapper
│       └── eagle_wrap.py           # optional EAGLE-2 speculation
├── tests/
│   ├── test_correctness.py
│   ├── test_quality.py             # RULER subset, perplexity
│   └── test_perf.py
├── benchmarks/
│   └── run_all.py
└── scripts/
    └── demo_chat.py
```

## 11. What success looks like

A repo that, when cloned on a similar laptop:
1. `pip install -e .`
2. `flashquest chat --model llama-3.2-3b --context 64k`
3. Tokens stream at ~12 tok/s with 64k of context loaded, fitting in 4GB VRAM.
4. README has a benchmark table showing 5–10× capability gain over `llama.cpp -ngl 999` on the same hardware.

That's it. No paper. No Twitter thread. Just a thing that works on a laptop that nobody else's stuff works on.
