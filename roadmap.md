# FlashQuest research roadmap

Updated: 2026-10-03

Establish whether FlashQuest offers a reproducible speed, memory, or context-capacity advantage over dense quantized KV at comparable quality. Keep the current direction while testing the narrower contribution of reusing quantization metadata for page scoring. Research novelty and competitive advantage remain open questions.

## Completed

- [x] Audit the saved research results and review the broad idea against prior work.
- [x] Correct benchmark context depth, separate prefill/decode timing, and advance decode token positions.
- [x] Add warmup, repeated measurements, configurable quantized baselines, and an all-pages INT4 ablation.
- [x] Preserve historical results and correct unsupported README claims.
- [x] Pass 25 benchmark regression checks, 24 GPU kernel/cache checks, and one tiny local Llama smoke test.
- [x] Add matched three-arm quality evidence with immutable identity, atomic resume, safe exports, and one reusable INT4 cache.
- [x] Validate the AWQ loader and run the 4k, retention 0.20 pilot; pass 48 CPU and 26 targeted GPU checks.

The initial full-model quality screen passes; expanded quality and competitor comparisons are still pending. See the [README](README.md) for the new per-example evidence, commands, and historical limitations.

## 1. Prepare full-model experiments

- [x] Install and verify the AWQ loader and kernels for Llama-3.2-3B-Instruct-AWQ.
- [ ] Obtain the matching Llama-3.2-3B-Instruct Q4_K_M GGUF.
- [ ] Prepare an external llama.cpp build supporting context depth, quantized K/V, and Flash Attention.
- [ ] Prepare a pinned current vLLM environment with same-request server timing metrics; verify actual kernels and FP8 KV support on the test GPU.
- [ ] Run one short request with each backend before launching a matrix.
- [ ] Record model revisions, backend versions, GPU, context length, cache settings, and seeds with the results.

AWQ setup and its short-run record are complete. Quality records include source,
model/tokenizer, protocol and environment identity. llama.cpp/vLLM setup and complete
benchmark identity integration remain pending; legacy matrix resume is disabled
when identity cannot be resolved.

Done when every backend produces a valid short-run record with the intended model and cache precision. Unsupported configurations remain explicit failures.

## 2. Validate quality at the chosen retention

- [x] Run single, multikey, and multivalue retrieval at retention 0.20 and 4k context.
- [x] Compare the same examples and seeds against the dense baseline; include all-pages INT4 to separate quantization error from page-selection error.
- [ ] Repeat at 8k and 32k after the 4k check passes.
- [x] Save per-example outcomes and failures, not only aggregate hit rates.
- [ ] Expand beyond the 20-example pilot with additional samples and seeds before making a research claim.

Use the existing quality script's per-task gate of at least 85% of dense hits as an initial screen. Report absolute hit counts too; passing this small retrieval screen does not establish general long-context quality. If retention 0.20 fails, test 0.25 and carry that setting into the performance comparisons.

Completed pilot: dense and all-pages INT4 scored 20/20 on every task; sparse 0.20
scored 20/20 single, 20/20 multikey and 18/20 multivalue. All three sparse tasks
pass the existing screen. The two misses reached the generation limit. The nominal
4k prompts contained 3,839–3,963 input tokens. This does not establish general quality
or a statistically confirmed comparison.

[Pilot evidence](benchmarks/validation/quality/dd8f60de24c2957a2120473d9ff3a194b64b22d85b2adfecca3aab485092f104/quality.json)
contains all 180 paired outcomes; raw answers/prompts are ignored local artifacts.

Reproduce with a fresh identity, or add `--resume` for the identical completed run:

```bash
python scripts/phase6_run_ruler_4k_int4.py \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --retentions 0.20 1.0 --seeds 0
```

## 3. Measure the benefit of page selection

- [ ] Benchmark sparse INT4 and all-pages INT4 at 8k and 32k using the same input IDs, model, and cache layout.
- [ ] Use warmup and at least three repetitions; retain samples, median rates, and run-to-run variation.
- [ ] Record prefill time, decode throughput, end-to-end throughput, and allocated/reserved memory separately.
- [ ] Repeat promising results with additional input seeds.
- [ ] Profile page scoring, selection, packed attention, and partial-page handling if the benefit is small or inconsistent.

Done when the results show whether sparse decode improves on this runtime's all-pages path at a retention that passed the quality screen. Page selection reduces reads; the persistent cache still stores the full context. This ablation shares the sparse kernel and selection machinery, so it also needs independent dense competitors.

```bash
python scripts/phase6_run_headtohead.py --backends flashquest \
    --contexts 8192 32768 --retention 0.20 \
    --output-dir benchmarks/validation-sparse

python scripts/phase6_run_headtohead.py --backends flashquest \
    --contexts 8192 32768 --retention 1.0 \
    --output-dir benchmarks/validation-dense
```

## 4. Compare optimized competitors

- [ ] Run llama.cpp with Q4 K/V and vLLM with FP8 KV at the same input context lengths and decode-step count.
- [ ] Retain FP16 KV runs as additional baselines where feasible.
- [ ] Compare on the same GPU with the same model family and explicitly record differences in weight/cache quantization.
- [ ] Check competitor quality on matching retrieval examples before claiming an advantage at comparable quality.
- [ ] Report timeouts, OOMs, unsupported settings, and missing measurements separately.

Done when a fresh 8k/32k comparison includes valid phase timings and quantized dense baselines. The historical head-to-head files remain provenance, not evidence of a current competitive win.

## 5. Verify memory and capacity

- [ ] Measure physical GPU memory throughout loading, prefill, and decode for every backend.
- [ ] Record GPU usage alongside process/system memory and PyTorch allocator counters; identify any offload or paging.
- [ ] Repeat the relevant quality and performance runs on an actual 4 GB GPU.
- [ ] Document the largest context that completes prefill and decode under the stated memory configuration.

Done when capacity claims have hardware-specific evidence. The available 12 GB RTX 4080 Laptop GPU can validate kernels and comparisons; it cannot establish a GPU-resident 32k result on a 4 GB device. Keep measured values separate from unmeasured fields.

## 6. Test the contribution and decide

- [ ] Compare metadata-based page scoring with separately computed page summaries: score/selection agreement, metadata bytes, and scoring latency.
- [ ] Compare fused packed attention with the reference dequantization path: output agreement, temporary allocations, and latency.
- [ ] Map the precise implementation claim against the closest prior work, recording overlaps and remaining differences.
- [ ] Write a decision supported by the saved quality, performance, and memory results.

Continue research if the benefit is repeatable beyond measurement variation at comparable quality, or if an accurately measured capacity advantage survives quantized dense baselines. A performance win and a novelty claim need separate evidence.

If the full runtime lacks a competitive advantage, consider a narrower kernel or metadata-reuse contribution when its ablations support one. If neither survives the checks, keep FlashQuest as an engineering/reference project and redirect research effort. Prioritize further optimization after these results identify a useful bottleneck.

## Results and next action

Keep new JSON records, logs, configuration/version details, and comparison summaries under `benchmarks/validation*`. Use separate output directories for different experiment configurations and preserve the historical phase files. Capture conclusions and links to their evidence in this roadmap or the README.

The next action is 8k then 32k quality at retention 0.20 where feasible, followed by
the minimal memory sampler and balanced sparse/all-pages performance blocks. Freeze
a confirmatory protocol before fresh seeds 1–5. Competitor setup can proceed separately.
The work is on the local feature branch `refactor/benchmark-validation`. Check off
tasks only when their evidence is saved.
