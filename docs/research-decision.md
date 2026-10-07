# FlashQuest research decision

Updated: 2026-10-06. Reviewed final decision after completed measurement audits
and accepted descriptive comparison; post-measurement integration is recorded below.

## In brief

- **Decision:** keep FlashQuest as an engineering reference and stop this research direction.
- **Speed:** page selection makes FlashQuest's own decode 1.82× faster at 32k, with no gain at
  8k. On the same GPU, llama.cpp Q4_0 KV and vLLM FP8 KV record roughly 2.4–3.5× FlashQuest's
  decode rate; including FP16 KV gives roughly 1.8–3.5×. Each engine's
  timer covers a slightly different span, so that gap is approximate. The measurements also don't
  isolate which FlashQuest components cause it.
- **Quality:** the frozen non-inferiority test passed 5 of 9 endpoints. Multivalue at 4k, 8k and
  32k and multikey at 32k failed, with observed drops of 2–8 points. Comparable quality isn't
  established, but a loss larger than 10 points isn't proven either.
- **Memory:** at 32k, FlashQuest's KV cache is about the size of llama.cpp's Q4_0 cache (991 vs.
  1,016 MiB). Its sampled total peak, though, is 7,006 MiB against 3,368 MiB. Testing on 4 GB
  hardware was not pursued.
- **Contribution:** metadata reuse saves 12.5% of the packed INT4 K payload at similar scoring
  latency, and the packed kernel avoids large temporary buffers. Given prior work, neither is a
  novel research result.

The [README](../README.md#results-at-a-glance) has the results tables. The rest of this page is
the detailed decision record.

## Detailed decision

Retain FlashQuest as an engineering reference and redirect the current research
effort. Page selection repeatedly improves its own all-pages path at 32k and
fails the practical-benefit screen at 8k. The fixed retrieval experiment does
not establish non-inferiority, and the completed evidence establishes neither
a comparable-quality competitive advantage nor broad novelty. This is a decision
about the current research case, not proof of inferiority or a universal runtime
verdict. A narrower metadata/kernel pivot is not established as novel.

| Question | Completed evidence | Decision |
| --- | --- | --- |
| Does selection help this runtime? | Final `5e22bf5` paired medians are 1.824622 at 32k and 0.975674 at 8k, practical screens pass/fail. Original kernel-34 and source-`76f00fe` groups remain separate. | Preserve the repeatable internal 32k benefit. The shared scoring/top-k and packed kernel make all-pages an internal control. |
| Is retrieval quality comparable? | Frozen confirmation: 2,700 outcomes, five endpoints pass and four fail. All observed floors and pilot screens pass. Native quality collects eight cells and 2,400 exact-manifest outcomes. | The failed simultaneous non-inferiority rule remains unchanged. Matching native hits do not rescue that rule or establish general language quality. Failure to establish non-inferiority is not proof of inferiority. |
| Is the runtime competitive? | Full native performance: 32 cells, 96 timings, complete strict/independent runtime and telemetry acceptance. Full native quality is terminal; final acceptance is recorded below. | The descriptive timing boundaries differ, so no FlashQuest/native or cross-native speed ratio is supported. An established comparable-quality competitive advantage is absent. |
| Is metadata reuse useful? | An extra BF16 min/max pair is avoided: 12.5% of packed INT4 K payload, or 6.25% of packed K+V. Scoring latency is similar and selections sometimes differ. | Preserve the byte-saving engineering benefit without claiming scoring-speed advantage, exact original-key extrema or research significance. |
| Does the packed kernel avoid materialization? | Exact FP32 packed-value oracle and BF16 reconstruction/SDPA reference, with much smaller temporary increments for packed attention. | This establishes avoided reference-path temporaries in those captures, not superiority over a native quantized engine. |
| Does it fit actual 4 GB hardware? | Available device is 12 GB. Internal sampled peaks are 3,442/7,006 MiB at 8k/32k, with full-cache retention. Native allocation policies differ. | Not pursued (2026-10-06): the current 32k peak already exceeds a 4 GB budget, so this is a decision, not a hardware-tested failure; 8k remains untested on 4 GB hardware. Native aligned phase memory, no OS fallback and true maximum contexts remain unverified; 32k is a successful tested lower bound. |
| Is the contribution novel? | Closest prior work combines sparse/low-bit KV and uses compressed keys as retrieval indexes. New diagnostics explain 28/216 and 102/216 changed heads, but no Q/K fingerprints match older captures. | Broad novelty is unsupported; affine metadata specialization remains an engineering candidate. Byte accounting and an implementation difference do not establish a novel pivot. |

The [final descriptive comparison](../benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md) binds the two
current internal blocks, native performance and matching native quality.
Final strict and independent audits accept all 2,400 ordered outcomes
(2,322 hits), 250,671 raw memory points and exact manifest/mapping/EOS/scorer/runtime
contracts. The final descriptive report passes independent raw/public reconstruction and
Markdown review; its accepted builder retains measured source `5e22bf5`.
Native quality uses fixed 300-example cells, canonical output-ID decoding/EOS and
the same frozen scorer, with no separate excluded warmup. llama.cpp performs
300 strict full-prompt detokenization checks per cell but retains no individual
HTTP response bodies. Static mapping and independently audited cell-specific
saved prompt views are distinct evidence. vLLM uses pinned-tokenizer ID/count
contracts. The report separates forward, native server and HTTP timing; vLLM's
FP8/auto pair changes attention backend as well as dtype and uses unit/default,
uncalibrated FP8 scales. Neither timing nor matching native hits establishes
scientific comparability or changes the fixed confirmation criterion.

The confirmation criterion was frozen before collection: seeds 1–5, 100 paired
examples per task/context, all nine simultaneous sparse-minus-dense lower bounds
above −0.10 and observed dense/sparse accuracy at least 0.80. Its fixed budget is
complete, without changing retention, margin or sample count after seeing results.
Observed floors are screens rather than population guarantees. Both environment
groups and all failed attempts are preserved without pooling or favorable-group
selection. The current family verdict is fail: multivalue at 4k/8k/32k and
multikey at 32k do not pass.

Evidence: [confirmation summary](../benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json),
[results summary](../README.md#results-at-a-glance),
[component/contribution reports](../benchmarks/validation/contrib/summary.md),
[new selection diagnostic](../benchmarks/validation/selection-flips/summary.md),
[closest prior work](contribution-prior-work.md) and
[independent audit](independent-roadmap-audit.md).

Future work should begin only from a separately stated contribution question
with applicable quality and matched cost/measurement evidence against closest
prior work. The completed roadmap does not justify further runtime optimization
as the current research direction or a metadata/kernel novelty claim. The actual
4 GB endpoint and the optional extra validation were dropped on 2026-10-06.
Both reviewed post-measurement fixes are applied and validated: 190 focused
tests and 606 full non-slow tests pass; 12 existing slow cases are deselected
and remain unverified. Changed-file Ruff and `git diff --check` pass. These checks
validate later source without relabeling measured `5e22bf5` evidence. Preserve the `5e22bf5` reproduction recipe and raw artifacts;
later engineering fixes do not relabel measured evidence.

## Confirmation provenance — completed history

The following chronology preserves both original and current-environment groups.
Its reexecution and audit instructions have already been carried out; current
status and the decision are above. No outcomes, failed bounds or source identities
are replaced by the final native measurements.

The fresh 4k run completed at commit
`05f64004dbfb3c33d42d60e68d3f6e5091de7d5d`; its
[record](../benchmarks/validation/quality/0526a7546dbef2cb2e78c919423d65ab0b55e3457ea1aa845edc2cf5502bdb69/quality.json)
contains all 900 arm outcomes. The original
[8k record](../benchmarks/validation/quality/7cec2647f00356ca86a526ff3bb2a16c28a20661d6dcf1fd8ba5d6e7c1427384/quality.json)
also contains all 900 outcomes and has passed independent evidence and bound
reconstruction. Original 32k was not started. Documents remained uncommitted
during confirmation because even a documentation-only commit changes the
recorded source identity. The coherent current-environment group below now
completes the specified nine-endpoint experiment; the original group remains
separate and incomplete.

The independently audited original fresh endpoints are:

| Context / task | Dense / all-pages / sparse hits, out of 100 | Paired gains / losses | Simultaneous-family lower difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| 4k single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| 4k multikey | 100 / 100 / 99 | 0 / 1 | −0.078127 | Pass |
| 4k multivalue | 99 / 99 / 96 | 1 / 4 | −0.127912 | Fail |
| 8k single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| 8k multikey | 98 / 98 / 97 | 0 / 1 | −0.078127 | Pass |
| 8k multivalue | 97 / 96 / 86 | 0 / 11 | −0.223272 | Fail |

All six observed accuracy floors pass. The multivalue observed differences are
−0.03 at 4k and −0.11 at 8k; their conservative lower bounds do not exceed −0.10.
Preserve these original outcomes and rules separately; they fail to establish
non-inferiority rather than proving that sparse accuracy is lower by more than
the margin. The same-seed reexecution does not erase these failures.

The workload pause has ended; the user authorized normal continuation to the
roadmap's end. On resume, the independent audit detected a Linux kernel change
from `7.0.0-34-generic` to `7.0.0-38-generic`. The parent independently checked
that GPU hardware/driver, Python and installed packages remain identical. A new
32k record would therefore differ from the old group's bound environment and
fail the existing same-environment summary contract.

Preserve the original 1800 outcomes and failed endpoints. Reexecute all three
contexts under the current environment, retaining the same clean `05f6400`
source, model, protocol, seeds 1–5, prompt manifests, retention, scorer and fixed
2700-outcome budget. The changed environment creates new immutable run IDs;
neither historical provenance nor measured answers may be relabeled. This is a
same-seed environment reexecution, not independent new statistical confirmation,
post-hoc sample expansion or permission to select a favorable group. Require
the repeated 4k/8k manifests to match the originals exactly, and do not pool
outcomes across groups. Preserve source and current environment through the
coherent full group and independently audit it before a final decision.
That execution and audit are complete; competitor runs and the separately
reviewed adapter fixes may now follow the confirmation group.

The [current-environment 4k reexecution](../benchmarks/validation/quality/7d9921cb0d5b6167c5303932475ccb71402ece5ca9066d7e067f23dab4d9cdef/quality.json)
is complete and independently audited:

| Current 4k task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Simultaneous-family lower difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multivalue | 99 / 99 / 97 | 1 / 3 | −0.112446 | Fail |

The exact original 300 examples are reused, but all-output comparison finds
261 changed texts, 71 changed output counts and 50 changed termination labels;
two sparse misses become hits. Greedy settings therefore do not establish
bitwise output reproducibility, and this comparison does not identify the kernel
change as the cause. Both groups still fail 4k multivalue non-inferiority.
The [current 8k reexecution](../benchmarks/validation/quality/6c938a6b4515ffaf0de9ed8bf833e521584887705e4665e43b51e9e94c3aafb1/quality.json)
is complete and independently audited, with original ordered examples and
current 4k's model/source/environment/runtime:

| Current 8k task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Simultaneous-family lower difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 98 / 98 / 96 | 0 / 2 | −0.096053 | Pass |
| Multivalue | 97 / 96 / 89 | 0 / 8 | −0.184352 | Fail |

All observed floors pass. The current multivalue point difference is −0.08, but
its lower bound fails the frozen non-inferiority rule. Across all 900 old/current
8k pairs, 123 texts, 47 output counts and 23 termination labels change. One sparse
multikey hit becomes a miss and three sparse multivalue misses become hits. This
does not establish a causal kernel effect or bitwise greedy-output reproducibility.

The independently verified [original-group summary](../benchmarks/validation/confirmation/699b3aecbca4be9cf65f52e08c2f3adc49fe6ea4b4a9eeb043109f79b3e7e2c7/summary.json)
remains separately inconclusive at six of nine endpoints, preserving both failed
multivalue endpoints and referencing only original evidence. Its outcomes are
not pooled into the complete current-environment family below.

The [current 32k record](../benchmarks/validation/quality/07ce7aa457aa9070cae386a86c9bcd00fb501e2d29c99aca56410183f74c98c4/quality.json)
now contains all 900 arm outcomes and passes the independent terminal evidence
audit. Its 300 ordered examples, protocol and retention 0.25 are unchanged;
model/source/environment/runtime exactly match current 4k/8k. Capacity 32763
covers the maximum 32635-token prompt plus 128 output tokens. The complete
[coherent current-family summary](../benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json)
has all nine endpoints and 2700 outcomes:

| Context / task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Simultaneous-family lower difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| 4k single | 100 / 100 / 100 | 0 / 0 | −0.057162223505 | Pass |
| 4k multikey | 100 / 100 / 100 | 0 / 0 | −0.057162223505 | Pass |
| 4k multivalue | 99 / 99 / 97 | 1 / 3 | −0.112446092774 | Fail |
| 8k single | 100 / 100 / 100 | 0 / 0 | −0.057162223505 | Pass |
| 8k multikey | 98 / 98 / 96 | 0 / 2 | −0.096053295161 | Pass |
| 8k multivalue | 97 / 96 / 89 | 0 / 8 | −0.184352439700 | Fail |
| 32k single | 100 / 100 / 100 | 0 / 0 | −0.057162223505 | Pass |
| 32k multikey | 91 / 90 / 88 | 2 / 5 | −0.141958404503 | Fail |
| 32k multivalue | 91 / 90 / 89 | 4 / 6 | −0.151236037290 | Fail |

All nine observed dense/sparse accuracy floors pass. The simultaneous family
fails because four endpoint lower bounds do not exceed −0.10. At 32k,
multikey/multivalue have observed differences −0.03/−0.02; conservative lower
bounds −0.141958/−0.151236 fail the criterion. This is failure to establish
non-inferiority, not proof that the true losses exceed the margin.

The independent audit rechecked all current raw/public records and hashes,
manifests/scorers/exact paired inputs, actual source and pinned model files,
common environment/runtime, and the saved summary's identity/evidence. Direct
binomial inversion reproduces all nine primary and eighteen diagnostic paired
bounds within 7e−16. The summary includes only current-environment records;
the original group remains unpooled.

The unchanged 32k pilot screen also passes, with sparse/dense ratios 1.0,
88/91 and 89/91. The planned current-environment internal timing blocks may
proceed under their existing prerequisite, as engineering characterization.
The confirmation source freeze can close after this passed terminal audit;
keep its measured source identity and raw evidence intact. Optimized native
performance/quality and the final research decision
remain to be completed; actual 4 GB testing remains pending collaboration.
