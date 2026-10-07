# Frozen-source descriptive comparison

2026-10-06. All four measured groups are complete and the descriptive builder record is accepted. Retain FlashQuest as an engineering reference and redirect the current research effort: its internal 32k benefit does not establish a comparable-quality competitive advantage, and broad novelty or a novel metadata/kernel pivot remains unestablished. The fixed retrieval confirmation has five passing and four failing endpoints; failure to establish non-inferiority does not prove inferiority.

The accepted local builder record validates four complete measured groups. Independent final server/worker, mapping/scorer and telemetry review is a separate gate and also passes. Independent raw-report reconstruction also passes; public derivative verification is separate from those measurement and raw-report audits. This is descriptive evidence; `scientific_comparability_established` remains `false`, with no new non-inferiority, general language quality, novelty or capacity claim.

The sibling [public JSON](summary.json) preserves the complete builder record inside a lossless wrapper with a separate public identity. Model-file hash maps use reversible typed path/SHA256 entries; no fingerprint is omitted.

## Source, evidence and audit scope

Measured commit: `5e22bf5c4c97b28bcdb46a030f75b269856ca622`; source content SHA256: `5807705f47e4eccd8e1eb6bc380202de5f05a9b0259037e5bd22798af9124c90`. Hardware: NVIDIA GeForce RTX 4080 Laptop GPU, 12,282 MiB reported device memory, driver `595.91.07`. Original kernel-34, source-`76f00fe`, failed and interrupted groups remain unchanged and unpooled. The fixed confirmation retains its recorded `05f6400` measurement source; later adapter repairs do not relabel it.

| Measured group | Saved cells | Collection | Evidence |
| --- | --- | --- | --- |
| Internal 8k / 0.20 | 8 | 24 timings | [schedule](../../ablation/bd5a7ab68ed2ec69e8e916c27968c57c68783cb3f4514293621fa0c542960f40/schedule.json) |
| Internal 32k / 0.25 | 8 | 24 timings | [schedule](../../ablation/efd8bfe82549542f6cc7994414e6af11b0943c09404647a9457cf0e42ca4e559/schedule.json) |
| Native performance | 32 | 96 timings | [schedule](../../competitors/3a5fa9de623a7d2a49b853f6588bc72821ce36a76542b6791a08abba677a0653/schedule.json) |
| Native matching quality | 8 | 2,400 outcomes | [schedule](../../competitors/ee7c318de630e7cdbd3e9b7859134df60a611e761c750bd2fb32fa7cf3133ed9/schedule.json) |

All four supervisor exits were separately observed as zero: three were read from their original tool sessions, while the quality exit was persisted and read after its session reset. The saved audit inventories leave `terminal_execution_proven` null; their saved-evidence completion/eligibility checks do not reconstruct supervisor exit receipts. Strict four-group review reconstructs 293,706 saved sampler points (4,015 / 8,740 / 30,280 / 250,671 respectively). Independent native quality review reconstructs all 2,400 ordered outcomes, 2,322 hits, 12,548 ownership checks, eight load windows and 2,400 request windows, with no device/ownership dropouts or observed foreign compute. These audit counts are separately verified local evidence, not derived scientific quality claims. Independent raw-report reconstruction passes 3,249 checks with 0 failures and 410 hashed files, including all 12 aggregate/48 seed performance rows, six paired-ratio rows, 51 quality rows and 56 memory/runtime rows.

Builder SHA256: `54050c0b9e710dbf9e2241f985b12d62b11f3bf69d7fb79d7a449829d16d554e`. Checker SHA256: `67cd7432837adbc56b32307667390b86d5fb0abd6373c1701db21d3a19e0e583`. Accepted raw builder identity: `b3a6a24125136414e14345fc20a8ded6f18806b75b14d9445b7cc984ec46e863`; raw-file SHA256: `de91f81ac17356e5d1f8f0acbc59e7059b4aab1713515a7149f3b70b70c538fd`. [Preserved raw builder record](../../../../artifacts/development/comparison/final-comparison-2026-10-06.json) is ignored local provenance and is unavailable from a repository clone alone. The independent terminal quality report is preserved locally at `artifacts/development/evidence-audit/quality-final-8cells-terminal-independent.json`, SHA256 `05f0b09b7f4d2b92dd8ae948d62b800e6175ebe9de4c2810c3e03cbb91db2162`. The accepted raw independent audit is `artifacts/development/comparison/final-raw-independent-audit-2026-10-06-attempt2.json`, SHA256 `dd435f8161314139d895521ccffa63c0668d251aaa7327b48b468792d5e6d361`.

## Timing boundaries and aggregation

| Measurement | Boundary |
| --- | --- |
| FlashQuest | CUDA-synchronized monotonic forward phases |
| llama.cpp | Native prompt evaluation / native generation forwards |
| vLLM | Scheduled-to-first / first-to-last token |
| HTTP | Client request start through complete JSON response |

Performance uses exactly 8,192 or 32,768 input IDs, 128 outputs and 127 later decode forwards, with four seeds and three timed requests per setting/seed after performance warmup. Rates/times are medians of the four per-seed repetition medians. Ratios pair each seed within one backend. Whole-arm order is balanced internally; native settings rotate by seed without randomizing every request. No FlashQuest/native or cross-native speed ratio is supported, and the different phase figures do not establish common application latency.

## Internal performance

| Input tokens | Arm | Prefill seconds | Decode tok/s | Forward request seconds |
| --- | --- | --- | --- | --- |
| 8,192 | FlashQuest all-pages INT4 | 1.237 | 38.11 | 4.568 |
| 8,192 | FlashQuest sparse INT4 | 1.235 | 37.13 | 4.657 |
| 32,768 | FlashQuest all-pages INT4 | 7.176 | 19.18 | 13.799 |
| 32,768 | FlashQuest sparse INT4 | 7.179 | 34.99 | 10.809 |

The all-pages control shares page scoring/top-k and packed attention. It is not an optimized dense competitor. Sparse selection reduces reads while retaining the full persistent cache; its practical screen is an engineering screen rather than statistical significance.

## Native performance

| Input tokens | Configuration | Native prefill seconds | Native decode tok/s | Native request seconds | HTTP wall seconds |
| --- | --- | --- | --- | --- | --- |
| 8,192 | llama.cpp FP16 KV | 0.904 | 120.25 | 1.961 | 1.967 |
| 8,192 | llama.cpp Q4_0 KV | 0.938 | 130.86 | 1.910 | 1.917 |
| 8,192 | vLLM auto (FP16 KV) | 0.973 | 116.66 | 2.062 | 2.069 |
| 8,192 | vLLM FP8 KV | 0.970 | 130.68 | 1.942 | 1.950 |
| 32,768 | llama.cpp FP16 KV | 7.188 | 66.23 | 9.103 | 9.121 |
| 32,768 | llama.cpp Q4_0 KV | 7.508 | 84.99 | 9.004 | 9.020 |
| 32,768 | vLLM auto (FP16 KV) | 6.388 | 64.62 | 8.353 | 8.377 |
| 32,768 | vLLM FP8 KV | 6.222 | 90.12 | 7.631 | 7.651 |

## Within-backend paired decode ratios

| Input tokens | Selected / control | Paired median | Seed range | Internal practical screen |
| --- | --- | --- | --- | --- |
| 8,192 | FlashQuest sparse INT4 / FlashQuest all-pages INT4 | 0.975674 | 0.933246–1.046596 | Fail |
| 32,768 | FlashQuest sparse INT4 / FlashQuest all-pages INT4 | 1.824622 | 1.760771–1.894740 | Pass |
| 8,192 | vLLM FP8 KV / vLLM auto (FP16 KV) | 1.120109 | 1.118667–1.121181 | Not applicable |
| 32,768 | vLLM FP8 KV / vLLM auto (FP16 KV) | 1.394556 | 1.392257–1.399680 | Not applicable |
| 8,192 | llama.cpp Q4_0 KV / llama.cpp FP16 KV | 1.088762 | 1.083331–1.092825 | Not applicable |
| 32,768 | llama.cpp Q4_0 KV / llama.cpp FP16 KV | 1.281486 | 1.277454–1.287190 | Not applicable |

vLLM FP8 uses FLASHINFER with E4M3 interpretation of uint8 cache storage; auto uses FLASH_ATTN and FP16 storage. Both use MarlinLinearKernel / AutoAWQMarlinLinearMethod. FP8 scales are unit/default uncalibrated scales observed after the first worker execution, which may be startup warmup; this is not late per-request calibration evidence. The FP8/auto pair changes attention backend as well as cache dtype and describes configuration effects, not isolated precision causality. llama.cpp uses Q4_K_M GGUF weights and Flash Attention, whereas FlashQuest/vLLM use the pinned AWQ checkpoint with different kernels.

## Matching retrieval quality

| Nominal context | Configuration | Single | Multikey | Multivalue |
| --- | --- | --- | --- | --- |
| 8,192 | AWQ dense FP16 KV | 100/100 | 98/100 | 97/100 |
| 8,192 | AWQ sparse INT4 (0.20) | 100/100 | 96/100 | 89/100 |
| 8,192 | AWQ all-pages INT4 | 100/100 | 98/100 | 96/100 |
| 8,192 | llama.cpp FP16 KV | 100/100 | 100/100 | 97/100 |
| 8,192 | llama.cpp Q4_0 KV | 100/100 | 100/100 | 95/100 |
| 8,192 | vLLM auto (FP16 KV) | 100/100 | 98/100 | 97/100 |
| 8,192 | vLLM FP8 KV | 100/100 | 98/100 | 97/100 |
| 32,768 | AWQ dense FP16 KV | 100/100 | 91/100 | 91/100 |
| 32,768 | AWQ sparse INT4 (0.25) | 100/100 | 88/100 | 89/100 |
| 32,768 | AWQ all-pages INT4 | 100/100 | 90/100 | 90/100 |
| 32,768 | llama.cpp FP16 KV | 100/100 | 97/100 | 96/100 |
| 32,768 | llama.cpp Q4_0 KV | 100/100 | 90/100 | 94/100 |
| 32,768 | vLLM auto (FP16 KV) | 100/100 | 91/100 | 91/100 |
| 32,768 | vLLM FP8 KV | 100/100 | 90/100 | 91/100 |

Quality labels identify frozen nominal contexts. Actual input ranges are 7,935–8,059 at nominal 8k and 32,511–32,635 at nominal 32k; individual counts and rounded cache capacities are recorded and can differ from performance lengths. Each native cell has the same exact 300 fixed examples: 100 per task, seeds 1–5, greedy EOS-or-128 output and the unchanged canonical substring scorer. Quality has no separate excluded warmup. FlashQuest rows come from the frozen confirmation source, with the semantic runtime/model/environment gate verified against the native manifests. These descriptive hits do not establish general language quality or change the failed simultaneous confirmation criterion.

llama.cpp strictly performs all 300 full-prompt `/detokenize` roundtrips per cell and records aggregate mapping coverage. Individual HTTP response bodies were not retained, so independent review does not replay 300 response transcripts. Static full-vocabulary/special/EOS mapping and returned-ID decoding/scoring are independently reconstructible. Each separately audited llama.cpp cell has up to 299 saved prior-request slot views; the last request lacks a subsequent view. Those views are cell-specific and are not HTTP response-body evidence. vLLM uses pinned HF tokenizer artifacts, exact token-ID payloads and server token-count checks.

## Frozen confirmation remains failed

The [fixed confirmation summary](../../confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json) and [frozen protocol](../../protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json) preserve 2,700 arm outcomes and all nine task/context endpoints. Five pass; multivalue at 4k/8k/32k and multikey at 32k fail. All observed dense/sparse floors and pilot screens pass. The criterion requires every simultaneous sparse-minus-dense lower bound above −0.10 and observed dense/sparse accuracy at least 0.80. Observed floors are screens rather than population guarantees. No margin, retention, seed or sample count is retuned; all-pages bounds remain diagnostic. Failure to establish non-inferiority is not proof of inferiority, and native hit counts or timings cannot rescue that failed frozen rule.

## Sampled memory and capacity limits

| Collection | Context label | Configuration | Device peak MiB | Baseline-adjusted peak MiB | Owned RSS peak MiB |
| --- | --- | --- | --- | --- | --- |
| Internal performance | 8,192 | FlashQuest all-pages INT4 | 3,442 | 3,428 | 2,707.4 |
| Internal performance | 8,192 | FlashQuest sparse INT4 | 3,442 | 3,428 | 2,835.7 |
| Internal performance | 32,768 | FlashQuest all-pages INT4 | 7,006 | 6,992 | 2,815.3 |
| Internal performance | 32,768 | FlashQuest sparse INT4 | 7,006 | 6,992 | 2,821.5 |
| Native performance | 8,192 | llama.cpp FP16 KV | 3,166 | 3,152 | 2,779.4 |
| Native performance | 8,192 | llama.cpp Q4_0 KV | 2,502 | 2,488 | 2,773.7 |
| Native performance | 8,192 | vLLM auto (FP16 KV) | 8,588 | 8,574 | 4,310.7 |
| Native performance | 8,192 | vLLM FP8 KV | 8,980 | 8,966 | 4,398.3 |
| Native performance | 32,768 | llama.cpp FP16 KV | 5,878 | 5,864 | 2,769.2 |
| Native performance | 32,768 | llama.cpp Q4_0 KV | 3,368 | 3,354 | 2,780.2 |
| Native performance | 32,768 | vLLM auto (FP16 KV) | 8,588 | 8,574 | 4,228.6 |
| Native performance | 32,768 | vLLM FP8 KV | 8,980 | 8,966 | 4,216.5 |
| Native quality | 8,192 | llama.cpp FP16 KV | 3,138 | 3,124 | 2,023.4 |
| Native quality | 8,192 | llama.cpp Q4_0 KV | 2,494 | 2,480 | 2,923.2 |
| Native quality | 8,192 | vLLM auto (FP16 KV) | 7,946 | 7,932 | 4,323.9 |
| Native quality | 8,192 | vLLM FP8 KV | 8,338 | 8,324 | 4,495.5 |
| Native quality | 32,768 | llama.cpp FP16 KV | 5,850 | 5,836 | 1,984.9 |
| Native quality | 32,768 | llama.cpp Q4_0 KV | 3,360 | 3,346 | 2,512.4 |
| Native quality | 32,768 | vLLM auto (FP16 KV) | 7,946 | 7,932 | 4,486.9 |
| Native quality | 32,768 | vLLM FP8 KV | 8,338 | 8,324 | 4,452.2 |

| Collection | Context label | Configuration | FQ persistent cache bytes | llama.cpp CUDA KV buffer MiB | vLLM physical cache tensor bytes | llama.cpp rounded slot tokens | vLLM reported capacity-token estimate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Internal performance | 8,192 | FlashQuest all-pages INT4 | 268,255,232 | — | — | — | — |
| Internal performance | 8,192 | FlashQuest sparse INT4 | 268,255,232 | — | — | — | — |
| Internal performance | 32,768 | FlashQuest all-pages INT4 | 1,038,958,592 | — | — | — | — |
| Internal performance | 32,768 | FlashQuest sparse INT4 | 1,038,958,592 | — | — | — | — |
| Native performance | 8,192 | llama.cpp FP16 KV | — | 924.00 | — | 8,448 | — |
| Native performance | 8,192 | llama.cpp Q4_0 KV | — | 259.88 | — | 8,448 | — |
| Native performance | 8,192 | vLLM auto (FP16 KV) | — | — | 6,121,586,688 | — | 53,376 |
| Native performance | 8,192 | vLLM FP8 KV | — | — | 6,104,154,112 | — | 106,448 |
| Native performance | 32,768 | llama.cpp FP16 KV | — | 3,612.00 | — | 33,024 | — |
| Native performance | 32,768 | llama.cpp Q4_0 KV | — | 1,015.88 | — | 33,024 | — |
| Native performance | 32,768 | vLLM auto (FP16 KV) | — | — | 6,121,586,688 | — | 53,376 |
| Native performance | 32,768 | vLLM FP8 KV | — | — | 6,104,154,112 | — | 106,448 |
| Native quality | 8,192 | llama.cpp FP16 KV | — | 896.00 | — | 8,192 | — |
| Native quality | 8,192 | llama.cpp Q4_0 KV | — | 252.00 | — | 8,192 | — |
| Native quality | 8,192 | vLLM auto (FP16 KV) | — | — | 5,446,303,744 | — | 47,459 |
| Native quality | 8,192 | vLLM FP8 KV | — | — | 5,428,871,168 | — | 94,614 |
| Native quality | 32,768 | llama.cpp FP16 KV | — | 3,584.00 | — | 32,768 | — |
| Native quality | 32,768 | llama.cpp Q4_0 KV | — | 1,008.00 | — | 32,768 | — |
| Native quality | 32,768 | vLLM auto (FP16 KV) | — | — | 5,446,303,744 | — | 47,480 |
| Native quality | 32,768 | vLLM FP8 KV | — | — | 5,428,871,168 | — | 94,657 |

| Input tokens | Internal arm | Peak Torch allocated MiB | Peak Torch reserved MiB |
| --- | --- | --- | --- |
| 8,192 | FlashQuest all-pages INT4 | 3,001.9 | 3,186.0 |
| 8,192 | FlashQuest sparse INT4 | 3,001.9 | 3,186.0 |
| 32,768 | FlashQuest all-pages INT4 | 5,477.3 | 6,750.0 |
| 32,768 | FlashQuest sparse INT4 | 5,477.3 | 6,750.0 |

Allocation entries are the recorded runtime values; ranges show minimum–maximum across cells rather than a common capacity model. Physical bytes, rounded slots and backend estimates remain separate quantities.

Peaks are sampled observations, not exact allocation peaks. Device samples include idle/display use; owned RSS can double-count shared pages and miss short-lived workers. Available Torch allocation/reservation counters are separate from physical device observations. Sampling requested 50 ms intervals; actual intervals and raw hashes are preserved in the schedules. Final independent quality review finds an 8k FP8 maximum gap of 731.133776 ms that is not wholly inside a validated window but intersects request 200; the 8k Q4 maximum 106.246457 ms crosses requests 67/68. The 32k FP16 maximum 208.033043 ms lies outside observed windows. Thus gaps are not uniformly inter-request gaps, and short peaks within gaps may be missed. Raw earlier audit reports are preserved alongside the corrected terminal supplemental scope.

Internal phase markers cover load/warmup/prefill/decode. Native coverage is load, performance warmup and request windows; aligned native prefill/decode peaks remain unmeasured. Continuous sampling and configured placement complete the bounded memory work; further phase instrumentation is optional future work. Native quality has no separate warmup window.

FlashQuest retains a fixed full persistent INT4 cache in both arms; vLLM reserves a declared GPU-budget pool; llama.cpp rounds the requested context allocation. Reported allocated KV token estimates are group-aware backend request-capacity estimates, not physical cache-byte capacity or tested maximum context. Worker `cache_tensor_bytes` records physical allocation separately. Differences in sampled peaks cannot alone establish a capacity advantage. llama.cpp records 29/29 offloaded model layers plus a 308.23 MiB CPU-mapped embedding and host buffers. Configured CUDA placement/offload settings do not prove absence of OS fallback. Successful 32k performance and matching quality is a tested lower bound on this 12 GB device, not a true maximum. Actual 4 GB testing is explicitly deferred for collaboration and its capacity claim is withheld.

## Decision and reproduction

Retain the engineering reference and redirect the current research effort. The internal 32k benefit, avoided materialized-reference temporaries and summary-byte saving remain useful engineering evidence; a comparable-quality competitive advantage and broad novelty are unestablished. Closest prior work already combines sparse selection and low-bit KV and uses compressed keys as indexes. Metadata scoring has similar latency, rounded affine extrema can differ from original keys, and new selection diagnostics do not replay old Q/K tensors. A narrower kernel/metadata pivot is not certified as novel. See the [research decision](../../../../docs/research-decision.md) and [independent audit](../../../../docs/independent-roadmap-audit.md).

Reproduce this measured comparison from clean source `5e22bf5`, pinned model snapshots/environments, preserved raw evidence and the exact reviewed checker/builder hashes bound above. Build at that source before the post-measurement completed-block resume and full-tokenizer-bound production fixes. Later fixes do not authorize weakening the source gate or relabeling old measurements. A public clone has descriptive JSON and hashes, but cannot recreate unavailable private raw prompts, answers, telemetry, model snapshots or ignored helpers.
