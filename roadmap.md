# FlashQuest research roadmap

Updated: 2026-10-06

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
- [x] Complete and independently audit native setup smokes and fresh `5e22bf5` internal blocks: 48 timings, paired ratios 0.975674 at 8k and 1.824622 at 32k.
- [x] Complete all 32 native performance cells and 96 timings, with strict and independent telemetry/runtime acceptance.
- [x] Collect and independently accept all eight matching native quality cells and 2,400 outcomes.
- [x] Accept the emitted descriptive comparison, then record the final decision.

The full confirmation fails at four of nine endpoints; its pilot screens pass.
Final internal paired medians are 1.824622 at 32k (practical screen passes) and
0.975674 at 8k (fails). Original kernel-34 and source-`76f00fe` groups remain
preserved and unpooled. Native performance and quality collection is terminal.
Final strict and independent audits accept all 2,400 ordered outcomes
(2,322 hits), 250,671 raw memory points and exact manifest/mapping/EOS/scorer/runtime
contracts. The final descriptive report passes independent raw/public reconstruction and
Markdown review; its accepted builder retains measured source `5e22bf5`. See the
[descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md),
[research decision](docs/research-decision.md) and
[independent audit](docs/independent-roadmap-audit.md). Comparable-quality
competitive advantage, broad novelty and actual 4 GB capacity are unestablished.
## 1. Prepare full-model experiments

- [x] Install and verify the AWQ loader and kernels for Llama-3.2-3B-Instruct-AWQ.
- [x] Obtain the matching Llama-3.2-3B-Instruct Q4_K_M GGUF.
- [x] Prepare an external llama.cpp build supporting context depth, quantized K/V, and Flash Attention.
- [x] Prepare an isolated pinned vLLM environment with same-request metrics and declared compiler/precompiled-kernel dependencies.
- [x] Verify actual vLLM kernels and FP8/FP16 KV support in successful requests on the test GPU.
- [x] Run one short request with each backend before launching a matrix.
- [x] Record model revisions, backend versions, GPU, context length, cache settings, and seeds with the results.

AWQ setup and its short-run record are complete. Quality records include source,
model/tokenizer, protocol and environment identity. llama.cpp Q4/FP16 short requests
complete on GPU. All four native settings now pass independently audited
performance and exact-prompt quality smokes, with verified intended kernels,
cache storage/devices, mapping, EOS and telemetry. Failed/interrupted attempts
remain preserved. The matching FlashInfer precompiled/AOT-only path works;
host compilation remains unverified. Full native matrices are now collected
and their accepted evidence is linked below.
See [competitor setup and contracts](docs/competitor-validation.md).

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

The final [8k block](benchmarks/validation/ablation/bd5a7ab68ed2ec69e8e916c27968c57c68783cb3f4514293621fa0c542960f40/summary.json)
has sparse/all-pages medians 37.13/38.11 tok/s, paired ratio 0.975674
(range 0.933246–1.046596): the practical screen fails. The final
[32k block](benchmarks/validation/ablation/efd8bfe82549542f6cc7994414e6af11b0943c09404647a9457cf0e42ca4e559/summary.json)
has medians 34.99/19.18 tok/s, paired ratio 1.824622
(range 1.760771–1.894740): all seeds improve and the practical screen passes.
Both retain measured source `5e22bf5`, with 48 timings and accepted raw telemetry.
The shared scoring/top-k and packed kernel make this an internal ablation;
neither the screen nor the ratio establishes a comparable-quality competitive win.

The [original 8k](benchmarks/validation/ablation/d86279b66c726c5697f408aabfd346170f8f072990bf95f7e0155416643a29a8/summary.json)
and [original 32k](benchmarks/validation/ablation/c921467929e57dad7293c868610d5dc39cdd0c01275c515a91d9d3d63060d66b/summary.json)
groups retain ratios 0.994770/1.8384. Source-`76f00fe` repeats retain
1.014242/1.811117; all older groups remain separate and unpooled.
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

- [x] Run llama.cpp Q4_0 K/V and vLLM FP8 KV at matching 8k/32k input lengths and decode-step count.
- [x] Retain native FP16 controls: llama.cpp FP16 and vLLM auto.
- [x] Record the same GPU/model family and all weight/cache/backend differences.
- [x] Accept matching retrieval quality and its final independent runtime/scorer audit.
- [x] Preserve timeouts, OOMs, unsupported settings, adapter errors and missing measurements separately.
- [x] Emit and independently accept the descriptive comparison with separate timing boundaries.

The [full native performance matrix](benchmarks/validation/competitors/3a5fa9de623a7d2a49b853f6588bc72821ce36a76542b6791a08abba677a0653/schedule.json)
finishes exit 0: 32 cells, 96 timings and 30,280 raw memory points, accepted by
strict and independent telemetry/runtime review. The
[matching native quality matrix](benchmarks/validation/competitors/ee7c318de630e7cdbd3e9b7859134df60a611e761c750bd2fb32fa7cf3133ed9/schedule.json)
finishes exit 0 with eight cells and 2,400 outcomes.
Final strict and independent audits accept all 2,400 ordered outcomes
(2,322 hits), 250,671 raw memory points and exact manifest/mapping/EOS/scorer/runtime
contracts. The final descriptive report passes independent raw/public reconstruction and
Markdown review; its accepted builder retains measured source `5e22bf5`.

The [report](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md) separates synchronized FlashQuest forward
timing, llama.cpp native evaluation/generation, vLLM scheduler token timing and
HTTP wall time. It supports within-backend configuration ratios only. vLLM's
FP8/auto pair also changes FLASHINFER/FLASH_ATTN; FP8 scales are unit/default
uncalibrated scales. llama.cpp Q4_K_M GGUF and the AWQ runtimes have different
weight formats. Quality uses exact fixed manifests and canonical returned-ID
scoring, with 300 requests per cell and no separate excluded warmup. Individual
llama.cpp `/detokenize` response bodies are absent; the performed runtime gate,
static mapping and separately audited saved prompt views have distinct scopes.
An accepted descriptive report does not establish scientific comparability or
upgrade the failed frozen non-inferiority rule.

## 5. Verify memory and capacity

- [x] Measure FlashQuest load/warmup/prefill/decode windows and owned process/system memory at 8k/32k.
- [x] Continuously sample native physical GPU and process/system memory during load, performance warmup and requests.
- [x] Record sampled device peaks, process RSS/MemAvailable, available Torch counters and configured placement separately.
- [ ] Repeat relevant performance and quality on actual 4 GB hardware, deferred for collaboration.
- [x] Document 32k as the largest tested successful context on this 12 GB device, without claiming a maximum.

The final internal blocks reach sampled device peaks 3,442/7,006 MiB at 8k/32k;
allocated/reserved peaks are 3,001.9/3,186 and 5,477.3/6,750 MiB. Selection retains
the full cache and does not reduce those peaks. Native phase markers cover load,
warmup/request windows rather than aligned prefill/decode. vLLM preallocates by
GPU budget, llama.cpp rounds requested context allocation and FlashQuest keeps
a persistent full cache; sampled peaks alone cannot establish a capacity advantage.
Worker physical tensor bytes and estimated token capacities are different
quantities. Aligned native phase peaks and proving absence of OS fallback are
unverified claim limits, not additional execution requirements of this plan. llama.cpp records 29/29 model-layer offload alongside a CPU-mapped
embedding and host buffers. Configured CUDA placement does not prove no OS fallback.

Successful 32k performance and matching-quality requests establish a tested lower
bound on this 12 GB GPU, not a true maximum or actual 4 GB fit. The current
FlashQuest 32k prefill exceeds a 4 GB budget here. Hardware-specific capacity
claims remain withheld; bounded prefill would be separate future work.

## 6. Test the contribution and decide

- [x] Compare affine-metadata scoring with separate summaries: agreement, bytes and latency.
- [x] Compare fused packed attention with exact packed-value and materialized references.
- [x] Map the precise claim against closest prior work and preserve remaining overlaps.
- [x] Capture and independently audit changed-page IDs/margins in 18 new snapshots; preserve the old-Q/K mismatch.
- [x] Record the final evidence-linked research decision after report acceptance.

The final [decision](docs/research-decision.md) retains FlashQuest as an
engineering reference and redirects the current research effort. The completed
confirmation fails to establish non-inferiority; current timings establish an
internal 32k benefit, not a comparable-quality competitive advantage. Broad
novelty is unsupported. Metadata reuse avoids separate-summary bytes but has no
established scoring-speed advantage or research significance. The
[new selection diagnostic](benchmarks/validation/selection-flips/summary.md)
explains 28/216 changed heads at 8k and 102/216 at 32k; no Q/K fingerprints match
the earlier captures, so this is not an old-snapshot replay or quality-causality
result. The narrower kernel/metadata direction is not certified as a novel pivot.

Future research needs a new, precisely bounded question with applicable quality,
matched measurement boundaries and a useful delta against closest prior work.
Missing capacity and general-quality evidence remain unresolved claims rather
than fabricated passes or evidence of universal inferiority.

## Results and next action

The locally executable bounded roadmap work is complete. Final evidence
is linked above and in the [execution checkpoint](docs/execution-checkpoint.md). Original kernel-34,
source-`76f00fe`, failed and interrupted groups remain unchanged and unpooled.
The accepted measurements retain clean source `5e22bf5` and its pinned model and
runtime environments; post-measurement repairs must not relabel them.
Both reviewed post-measurement fixes are applied and validated: 190 focused
tests and 606 full non-slow tests pass; 12 existing slow cases are deselected
and remain unverified. Changed-file Ruff and `git diff --check` pass. These checks
validate later source without relabeling measured `5e22bf5` evidence.

The measured comparison is reproducible from a clean checkout of `5e22bf5`, the
pinned environments/model snapshots, preserved raw artifacts and exact reviewed
checker/builder hashes bound by the report. Public JSON alone cannot recreate
private raw prompts, answers or sampled telemetry. Reproduce against that source
before applying the deferred resume or full-tokenizer-bound fixes; later source
changes do not justify weakening the current-source acceptance gate.

Actual 4 GB testing remains deferred for collaboration as authorized by the user.
Aligned native phase memory, absence of OS fallback and general language quality
remain unverified. Work remains on `refactor/benchmark-validation`; local commits
are authorized, while pushing, PRs and merges still require approval.
