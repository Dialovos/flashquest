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
- [x] Run the matched 8k and 32k pilots at retention 0.20; preserve the 32k screen failure.
- [x] Prepare FlashQuest device/process memory observation and balanced immutable ablation schedules; pass 76 CPU checks and rerun both tiny-Llama GPU checks.

The 4k/8k screens pass at 0.20. The 32k screen fails at 0.20 and passes its 0.25 fallback. Expanded quality and competitor comparisons remain pending. See the [README](README.md) for per-example evidence, commands, and historical limitations.

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
- [x] Repeat at 8k and 32k after the 4k check passes.
- [x] Screen retention 0.25 at 32k after the 0.20 failure.
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

The [8k pilot](benchmarks/validation/quality/85c9abbb694ffb21ba5c03336188658ca2d0e9dff0790a9a9b62cb6b28c2ef07/quality.json)
passes: dense 20/18/20, all-pages 20/19/18, sparse 20/19/19 (single/multikey/multivalue,
out of 20). The [32k pilot](benchmarks/validation/quality/52b2599c93490bdc948431f43bf4fd2294273a00fac583461fe10bd5b0ba9f93/quality.json)
completed every arm but fails sparse multivalue: dense 20/17/19, all-pages 20/16/20,
sparse 20/17/16. Its multivalue ratio is 16/19 = 84.2%; all four sparse misses reached
the output limit. This is a quality-screen failure, not an execution failure or OOM.

The [32k fallback at 0.25](benchmarks/validation/quality/2baea9556c7ecdb9bb4213e0c02caf7444820468d36d912879d1a63190903f97/quality.json)
passes with sparse 20/17/17. Dense remains 20/17/19 and all-pages 20/16/20 on the
same 60 prompts. One multivalue example recovers, with no paired losses relative
to 0.20; three misses still reach the output limit. Use the screened setting for
each performance context: 0.20 at 8k and 0.25 at 32k. These are tuning pilots,
not independent confirmatory observations.

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
python scripts/run_validation_ablation.py --contexts 8192 --retention 0.20 \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --quality benchmarks/validation/quality/85c9abbb694ffb21ba5c03336188658ca2d0e9dff0790a9a9b62cb6b28c2ef07/quality.json
```

## 4. Compare optimized competitors

- [ ] Run llama.cpp with Q4 K/V and vLLM with FP8 KV at the same input context lengths and decode-step count.
- [ ] Retain FP16 KV runs as additional baselines where feasible.
- [ ] Compare on the same GPU with the same model family and explicitly record differences in weight/cache quantization.
- [ ] Check competitor quality on matching retrieval examples before claiming an advantage at comparable quality.
- [ ] Report timeouts, OOMs, unsupported settings, and missing measurements separately.

Done when a fresh 8k/32k comparison includes valid phase timings and quantized dense baselines. The historical head-to-head files remain provenance, not evidence of a current competitive win.

## 5. Verify memory and capacity

- [x] Prepare physical-device polling, owned process-tree RSS, validated phase windows and safe exports for FlashQuest ablation cells.
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

The next action is balanced sparse/all-pages performance at 8k/0.20 and 32k/0.25
using the prepared memory sampler. Freeze
a confirmatory protocol before fresh seeds 1–5. Competitor setup can proceed separately.
The work is on the local feature branch `refactor/benchmark-validation`. Check off
tasks only when their evidence is saved.
