# FlashQuest research roadmap

Updated: 2026-10-05

Establish whether FlashQuest offers a reproducible speed, memory, or context-capacity advantage over dense quantized KV at comparable quality. Continue bounded validation of affine quantization metadata for page scoring. Broad sparse/low-bit and compressed-index novelty is unsupported by the closest prior work; the narrower implementation's research significance and competitive advantage remain unproven.

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
- [x] Complete balanced 8k/0.20 and 32k/0.25 ablations: four input seeds, three repetitions per arm, all 48 timed samples audited.
- [x] Profile scoring, top-k, packed attention, tail and merge costs on synthetic and actual pinned-model inputs; independently audit the three reports.
- [x] Map the metadata/kernel implementation against closer low-bit sparse and self-indexing prior work.
- [x] Freeze a nine-endpoint fresh-seed confirmation protocol and validate its statistics and evidence contracts.
- [x] Implement pinned competitor adapters, exact-input quality manifests, realized-runtime checks and resumable evidence validation; preserve failed smoke attempts.
- [x] Complete and independently audit all 2,700 fresh-confirmation outcomes and the nine-endpoint simultaneous-bound summary; preserve its failed quality rule.
- [x] Capture and independently audit changed-page IDs and score/error margins in 18 new representative snapshots; disclose their mismatch with earlier Q/K fingerprints.

The full confirmation fails at four of nine endpoints; its pilot screens pass. The historical internal performance screen passes at 32k (1.838× median paired ratio) and fails at 8k (0.995×). Current-environment timing, optimized competitor comparisons and actual 4 GB capacity remain pending. The [changed-page diagnostic](benchmarks/validation/selection-flips/summary.md) explains the new captures; it cannot reconstruct the earlier snapshots. See the [provisional research decision](docs/research-decision.md) and [independent audit](docs/independent-roadmap-audit.md).

## 1. Prepare full-model experiments

- [x] Install and verify the AWQ loader and kernels for Llama-3.2-3B-Instruct-AWQ.
- [x] Obtain the matching Llama-3.2-3B-Instruct Q4_K_M GGUF.
- [x] Prepare an external llama.cpp build supporting context depth, quantized K/V, and Flash Attention.
- [x] Prepare an isolated pinned vLLM environment with same-request metrics and declared compiler/precompiled-kernel dependencies.
- [ ] Verify actual vLLM kernels and FP8/FP16 KV support in successful requests on the test GPU.
- [ ] Run one short request with each backend before launching a matrix.
- [x] Record model revisions, backend versions, GPU, context length, cache settings, and seeds with the results.

AWQ setup and its short-run record are complete. Quality records include source,
model/tokenizer, protocol and environment identity. llama.cpp Q4/FP16 short requests
complete on GPU. vLLM startup failures are preserved, including the terminal sixth attempt.
A matching precompiled FlashInfer package and an immutable AOT-only policy are
being verified after compiler-version and host-header failures. Successful
performance/quality smokes and the full matrix remain pending. See [competitor setup and contracts](docs/competitor-validation.md).

Done when every backend produces a valid short-run record with the intended model and cache precision. Unsupported configurations remain explicit failures.

## 2. Validate quality at the chosen retention

- [x] Run single, multikey, and multivalue retrieval at retention 0.20 and 4k context.
- [x] Compare the same examples and seeds against the dense baseline; include all-pages INT4 to separate quantization error from page-selection error.
- [x] Repeat at 8k and 32k after the 4k check passes.
- [x] Screen retention 0.25 at 32k after the 0.20 failure.
- [x] Save per-example outcomes and failures, not only aggregate hit rates.
- [x] Expand beyond the 20-example pilot with additional samples and seeds before making a research claim.

The [frozen confirmation](benchmarks/validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json)
uses seeds 1–5 and 100 paired examples per task/context, with simultaneous bounds
over all nine endpoints. Fresh 4k is complete under commit `05f6400`: sparse
single/multikey/multivalue score 100/99/96 out of 100, against dense 100/100/99.
The original multivalue lower difference bound is −0.127912, failing the frozen
−0.10 endpoint rule without proving inferiority. Original 8k is also complete:
dense/all-pages/sparse hits are 100/100/100, 98/98/97 and 97/96/86; its multivalue
lower bound is −0.223272. Both original records are independently audited.
The same-seed current-environment 4k re-execution is complete and audited:
100/100/100, 100/100/100 and 99/99/97; multivalue still fails with lower bound
−0.112446. Some generated outputs differ despite identical prompts/settings, so
bitwise greedy reproducibility and kernel causality are not established. The
current 8k re-execution is complete: dense/all-pages/sparse hits are 100/100/100,
98/98/96 and 97/96/89; multivalue fails with lower bound −0.184352. Its independent
terminal audit passed. The 32k context is also complete and audited at unchanged
retention 0.25: dense/all-pages/sparse hits are 100/100/100, 91/90/88 and 91/90/89.
Multikey and multivalue lower bounds are −0.141958 and −0.151236, respectively,
failing the rule. The [complete, independently reconstructed family](benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json)
has five passing and four failing endpoints, with all observed floors passing.
This completes collection while failing to establish non-inferiority. The original
group remains unpooled; no margin, retention or sample budget was changed.

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

- [x] Benchmark sparse INT4 and all-pages INT4 at 8k and 32k using the same input IDs, model, and cache layout.
- [x] Use warmup and at least three repetitions; retain samples, median rates, and run-to-run variation.
- [x] Record prefill time, decode throughput, end-to-end throughput, and allocated/reserved memory separately.
- [x] Repeat promising results with additional input seeds.
- [x] Profile page scoring, selection, packed attention, and partial-page handling if the benefit is small or inconsistent.

The [component reports](benchmarks/validation/contrib/summary.md) contain warmed
operator samples on real model captures. Metadata scoring has similar latency to
stored-summary scoring; selection agreement is qualified. The dequantization/SDPA
reference is separately timed and does not represent an optimized low-bit engine.

Done when the results show whether sparse decode improves on this runtime's all-pages path at a retention that passed the quality screen. Page selection reduces reads; the persistent cache still stores the full context. This ablation shares the sparse kernel and selection machinery, so it also needs independent dense competitors.

The [8k block](benchmarks/validation/ablation/d86279b66c726c5697f408aabfd346170f8f072990bf95f7e0155416643a29a8/summary.json)
has sparse/all-pages medians of 38.84/38.91 tok/s, paired ratio 0.995× (range
0.980–1.015): no practical gain. The [32k block](benchmarks/validation/ablation/c921467929e57dad7293c868610d5dc39cdd0c01275c515a91d9d3d63060d66b/summary.json)
has medians 35.22/19.16 tok/s, paired ratio 1.838× (range 1.770–1.853); every seed
is faster and the proposed pilot screen passes. Prefill is essentially unchanged.
The control still pays scoring/top-k, so these results establish an internal
ablation benefit at 32k, not a competitive or statistically confirmed research win.

Set `QUALITY_8K` to a completed, passing quality record produced with the current
runtime, quality harness, corpus and validation environment. The historical pilot
above remains evidence for its original block; the strengthened semantic gate
rejects it as a prerequisite for a new block at the current source.

```bash
python scripts/run_validation_ablation.py --contexts 8192 --retention 0.20 \
    --revision 272b3bde867b606760447deb9a4d2719fbdfd3ae \
    --quality "$QUALITY_8K"
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
- [x] Measure FlashQuest load/warmup/prefill/decode windows, owned process-tree RSS and system memory at 8k and 32k.
- [ ] Measure physical GPU memory throughout loading, prefill, and decode for every backend.
- [ ] Record GPU usage alongside process/system memory and PyTorch allocator counters; identify any offload or paging.
- [ ] Repeat the relevant quality and performance runs on an actual 4 GB GPU.
- [ ] Document the largest context that completes prefill and decode under the stated memory configuration.

Done when capacity claims have hardware-specific evidence. The available 12 GB RTX 4080 Laptop GPU can validate kernels and comparisons; it cannot establish a GPU-resident 32k result on a 4 GB device. Keep measured values separate from unmeasured fields.

Both arms reached sampled device peaks of 3,442 MiB at 8k and 7,006 MiB at 32k.
Allocated/reserved peaks were 3,001.9/3,186 and 5,477.3/6,750 MiB respectively.
Sampling medians were about 50 ms, with maximum gaps of 110/181.4 ms and no device
or ownership-check dropouts. RSS and MemAvailable remain separate; configured
CUDA placement does not prove absence of OS fallback. The current 32k prefill
needs more than a 4 GB budget on this device. Preserve the target-capacity gate;
evaluate bounded prefill only as separately validated future work.

## 6. Test the contribution and decide

- [x] Compare metadata-based page scoring with separately computed page summaries: score/selection agreement, metadata bytes, and scoring latency.
- [x] Compare fused packed attention with the reference dequantization path: output agreement, temporary allocations, and latency.
- [x] Map the precise implementation claim against the closest prior work, recording overlaps and remaining differences.
- [x] Capture concrete changed-page IDs and score/error margins in new representative snapshots; preserve the mismatch with earlier Q/K fingerprints.
- [ ] Write a decision supported by the saved quality, performance, and memory results.

The [current decision](docs/research-decision.md) is provisional while competitive
measurements are unfinished. Fresh quality collection is complete and fails its
frozen rule. The [selection diagnostic](benchmarks/validation/selection-flips/summary.md) has an independent author
review: 28/216 changed heads at 8k and 102/216 at 32k, without quality causality
or exact replay of the earlier tensors. Metadata reuse avoids separate-summary
bytes but has no established scoring-speed advantage. The closest prior-work map
does not certify first-of-kind novelty.

Continue research if the benefit is repeatable beyond measurement variation at comparable quality, or if an accurately measured capacity advantage survives quantized dense baselines. A performance win and a novelty claim need separate evidence.

If the full runtime lacks a competitive advantage, consider a narrower kernel or metadata-reuse contribution when its ablations support one. If neither survives the checks, keep FlashQuest as an engineering/reference project and redirect research effort. Prioritize further optimization after these results identify a useful bottleneck.

## Results and next action

Keep new JSON records, logs, configuration/version details, and comparison summaries under `benchmarks/validation*`. Use separate output directories for different experiment configurations and preserve the historical phase files. Capture conclusions and links to their evidence in this roadmap or the README.

The user resumed work after supervised confirmation completed. Its independent
terminal audit passed, releasing the source freeze. The complete failed summary
retains source `05f6400` and its exact recorded environment; the original 4k/8k
group remains separate. The reviewed canonical decoder, EOS/cap, token mapping,
compile-worker and attention-backend checks plus the selection diagnostic pass
170 focused CPU tests and were committed as `a0100a5`. Changed-page evidence
and its independent review are complete. The AOT-only native setup adds 160
focused competitor checks; commit its reviewed source, verify native smokes,
and repeat both internal timing
blocks under the current environment before the full optimized comparison.
The earlier usage cutoff remains removed. See
[exact continuation instructions](docs/execution-checkpoint.md).
Actual 4 GB testing remains a separate pending endpoint;
the user authorized deferring unavailable target hardware for collaboration.
The work is on the local feature branch `refactor/benchmark-validation`. Check off
tasks only when their evidence is saved.
