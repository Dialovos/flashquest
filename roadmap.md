# FlashQuest research roadmap

**Closed on 2026-10-06.** Outcome: keep FlashQuest as an engineering reference and stop this
research direction. See the [research decision](docs/research-decision.md) and the
[results summary](README.md#results-at-a-glance).

**Question:** does FlashQuest offer a reproducible speed, memory or context-capacity advantage over
dense quantized KV caches at comparable quality?

**Answer:** not on the tested workload. Page selection speeds up FlashQuest's own decode at 32k,
but llama.cpp and vLLM decode 2.4–3.5× faster, and they match or exceed its retrieval quality. At
32k, llama.cpp also uses less total memory. Comparable quality to dense wasn't established either.

This page records each phase's outcome and links its evidence. The detailed run-by-run history is
in [docs/execution-checkpoint.md](docs/execution-checkpoint.md),
[docs/independent-roadmap-audit.md](docs/independent-roadmap-audit.md) and this file's Git
history.

## 1. Prepare full-model experiments — done

- [x] AWQ loader and kernels for Llama-3.2-3B-Instruct-AWQ, at a pinned model revision.
- [x] Matching Llama-3.2-3B-Instruct Q4_K_M GGUF, and an external llama.cpp build with context
      depth, quantized K/V and Flash Attention.
- [x] An isolated, pinned vLLM environment with per-request timing; FP8 and FP16 KV both verified
      on the test GPU.
- [x] Short smoke requests for every backend before the full runs.
- [x] Model revisions, backend versions, GPU, context, cache settings and seeds recorded with every
      result.

Setup and contracts: [docs/competitor-validation.md](docs/competitor-validation.md).

## 2. Validate quality — done; comparable quality not established

- [x] 20-example pilots: retention 0.20 passed the 85%-of-dense screen at 4k and 8k. At 32k, 0.20
      failed on multivalue (16 vs. 19 dense hits); the 0.25 fallback passed, so 32k uses 0.25.
- [x] Dense, all-pages INT4 and sparse INT4 run on identical examples, which separates quantization
      error from page-selection error.
- [x] Per-example outcomes saved, not only totals.
- [x] Frozen confirmation: seeds 1–5, 100 examples per task and context, nine endpoints, each
      requiring a simultaneous lower bound above −10 points. **5 of 9 pass.** Multivalue at
      4k/8k/32k and multikey at 32k fail, with observed drops of 2–8 points. This doesn't
      establish comparable quality, and it doesn't prove a loss larger than 10 points.

Evidence: [frozen protocol](benchmarks/validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json),
[confirmation summary](benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json).
The 4k and 8k confirmation runs were repeated after an OS update, with the same seeds and settings.
Both groups are preserved and not pooled; see the
[research decision](docs/research-decision.md#confirmation-provenance--completed-history).

## 3. Measure the benefit of page selection — done

- [x] Sparse vs. all-pages INT4 at 8k and 32k: same input IDs, four seeds, one warmup and three
      timed runs per arm, with balanced order.
- [x] Prefill, decode, end-to-end throughput and memory recorded separately.
- [x] Result: **0.98× at 8k** (no benefit) and **1.82× at 32k** (every seed faster).
- [x] Scoring, top-k, packed attention, tail and merge costs profiled on real model captures.

Evidence: [8k block](benchmarks/validation/ablation/bd5a7ab68ed2ec69e8e916c27968c57c68783cb3f4514293621fa0c542960f40/summary.json),
[32k block](benchmarks/validation/ablation/efd8bfe82549542f6cc7994414e6af11b0943c09404647a9457cf0e42ca4e559/summary.json),
[component reports](benchmarks/validation/contrib/summary.md).

## 4. Compare optimized competitors — done

- [x] llama.cpp (Q4_0 and FP16 KV) and vLLM (FP8 and FP16 KV) at matching 8k/32k inputs with
      128 output tokens: 32 performance cells and 96 timings.
- [x] Matching retrieval quality on the same 300 prompts per configuration: 8 cells and 2,400
      outcomes.
- [x] Weight, cache and backend differences recorded. Failures, timeouts and unsupported settings
      preserved separately.
- [x] Result: FlashQuest decodes 2.4–3.5× slower. Each engine's timer covers a slightly different
      span, so this is an approximate gap rather than a precise ratio.

Evidence: [descriptive comparison](benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md).

## 5. Verify memory and capacity — done on 12 GB; 4 GB not pursued

- [x] Device, process and system memory sampled continuously for every backend. FlashQuest also
      has load, warmup, prefill and decode phase windows.
- [x] KV caches are about equal in size at 32k (FlashQuest 991 MiB, llama.cpp Q4_0 1,016 MiB). The
      sampled 32k peaks are 7,006 MiB for FlashQuest and 3,368 MiB for llama.cpp Q4_0.
- [x] 32k is the largest tested context on the 12 GB GPU: a tested lower bound, not a maximum.

Not pursued (decided 2026-10-06):

- **Actual 4 GB hardware.** The current implementation's 32k peak already exceeds a 4 GB budget,
  so a GPU-resident 32k target was dropped for this implementation. This is a decision, not a
  hardware-tested failure. The 8k case (3,442 MiB sampled peak) remains untested on 4 GB hardware.
- **Optional extra validation:** general language quality, phase-aligned native memory peaks and
  the true maximum context.

## 6. Test the contribution and decide — done

- [x] Metadata scoring vs. separate min/max summaries: it saves 12.5% of the packed INT4 K payload
      at similar scoring latency, and selections sometimes differ.
- [x] Fused packed attention vs. exact and materialized references: outputs match closely, with
      much smaller temporary allocations than the materialized reference.
- [x] Prior-work map: sparse plus low-bit KV, and compressed keys as retrieval indexes, already
      exist. Broad novelty is unsupported.
- [x] Selection-flip diagnostic on 18 new captures.
- [x] Final decision recorded.

Evidence: [component reports](benchmarks/validation/contrib/summary.md),
[selection diagnostic](benchmarks/validation/selection-flips/summary.md),
[prior work](docs/contribution-prior-work.md), [decision](docs/research-decision.md).

## Open item

- The 12 `slow` end-to-end tests haven't been run. The Llama-3.2-1B-Instruct weights they need
  aren't cached locally. They cover software reliability, not the research question.
