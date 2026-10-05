# FlashQuest research decision

Updated: 2026-10-05. Provisional research decision after completed fresh
confirmation; optimized competitor and remaining contribution checks are pending.

Continue the bounded validation roadmap and retain FlashQuest as an engineering
reference. The completed confirmation fails its specified non-inferiority rule
in four of nine endpoints. Finish the fair comparison as engineering
characterization. The current evidence does not justify a runtime advantage at
established comparable quality, a novelty claim or a pivot to metadata/kernel
research as an established new contribution.

| Question | Current evidence | Decision boundary |
| --- | --- | --- |
| Does page selection help this runtime? | Balanced four-seed internal blocks give a median paired decode ratio of 1.838× at 32k and 0.995× at 8k. | A repeatable internal signal exists at 32k. The all-pages control shares scoring/top-k and the packed kernel, so it is not an optimized dense competitor. |
| Is retrieval quality comparable? | The complete current-environment confirmation has all 2700 outcomes and fails four of nine endpoint rules: 4k/8k multivalue and 32k multikey/multivalue. All observed accuracy floors and pilot-ratio screens pass. The original 4k/8k group remains separately inconclusive with both multivalue failures. | Comparable retrieval quality is not established by the frozen criterion. Preserve both groups without pooling, retuning, relaxing the rule or choosing a favorable group. Failure to establish non-inferiority is not proof of inferiority. Retrieval results do not establish general language quality. |
| Is metadata reuse useful? | Representative captures avoid an additional BF16 min/max pair: 12.5% of packed INT4 K payload, or 6.25% of packed K+V payload. Scoring latency is similar and selections sometimes differ. | The storage accounting supports an engineering benefit. Investigate changed pages and score margins; there is no established scoring-speed or exact-equivalence claim. |
| Does packed attention avoid materialization? | The exact FP32 packed-value oracle and separately timed BF16 reconstruction/SDPA reference are saved, with much smaller temporary increments for the packed path. | This supports avoided temporaries in that reference workload. It does not establish end-to-end superiority over a native quantized engine. |
| Is the runtime competitive? | llama.cpp Q4/FP16 short performance requests work. Saved vLLM smoke cells fail during startup; a subsequent standalone retry has no terminal evidence. | Full 8k/32k repeated performance, realized runtime checks and matching retrieval quality remain pending. Setup failures do not establish engine inferiority. |
| Does it fit an actual 4 GB GPU? | The available device has 12 GB. FlashQuest sampled peaks are 3,442 MiB at 8k and 7,006 MiB at 32k; page selection retains the full persistent cache. | Actual 4 GB prefill, generation and quality tests remain pending collaboration. No physical-target fit or no-OS-fallback claim is supported. |
| Is the contribution novel? | Closest prior work already combines sparse selection and low-bit KV, and compressed key representations already serve as retrieval indexes. | Broad novelty is unsupported. The affine-page-metadata specialization remains a precise engineering candidate; an implementation difference and byte saving alone do not establish research significance. |

Evidence: [matched quality and internal performance](../README.md#benchmarks-and-validation),
[component and contribution reports](../benchmarks/validation/contrib/summary.md),
[fourth short smoke](../benchmarks/validation/competitors/eef23d91caff6fdef91041ba4de25c42d5a37a12caae006fbc46980c3aab03c3/schedule.json),
[closest prior work](contribution-prior-work.md), and
[independent audit checkpoints](independent-roadmap-audit.md).

The completed confirmation used the
[frozen fresh-quality protocol](../benchmarks/validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json):
seeds 1–5, 100 paired examples per task/context, all nine simultaneous lower
bounds above −0.10 and observed dense/sparse accuracy floors of 0.80. Its fixed
budget is complete, without retuning retention, relaxing the margin or expanding
the sample after viewing results. The full current-family verdict is fail.

Complete the optimized comparisons and selection diagnostics to characterize
costs and tradeoffs; their timing results cannot rescue the failed frozen
quality rule. A passed pilot screen permits internal timing but does not
establish a comparable-quality advantage. Narrow to metadata/kernel research
only if its own matched ablations and precise prior-work comparison establish
a useful contribution. If neither research case survives the completed fair
checks, keep the engineering reference and redirect research effort. Remaining
measurements still inform that final research choice.

A research decision may proceed while the actual 4 GB endpoint awaits
collaboration, with that capacity claim withheld.

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
performance/quality, selection diagnostics and the final research decision
remain to be completed; actual 4 GB testing remains pending collaboration.
