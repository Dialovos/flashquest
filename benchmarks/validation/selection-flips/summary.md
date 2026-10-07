# Affine-metadata selection diagnostic

Updated: 2026-10-05. The [complete diagnostic](84923860ca3235323e8882a7f296857e419839cdd5afb6fada0eee7412718a06/report.json)
records concrete score and page-selection differences in 18 new representative
model captures. It does not reproduce the exact tensors of the earlier
contribution ablations: all 18 input fingerprints match, but all 18 query and all
18 key fingerprints differ. Its explanations apply to these new captures.

## Scope and evidence checks

The pinned AWQ model, single-needle seed-0 prompt, layers 0/13/27, decode-forward
steps 0/1/63, 24 query heads, eight KV heads, 128-dimensional heads, 64-token pages,
four sink pages and two window pages are unchanged. Retention is 0.20 at nominal
8k and 0.25 at 32k. Actual inputs contain 7936/32512 tokens; cached captures span
7937–8000/32513–32576 tokens. Each context covers 216 head observations.

A separate CPU calculation, without the producer's explanation functions,
verified the canonical run/protocol/source/model identities, raw-evidence hash,
every raw/public row and score-array hash, all 432 score vectors and actual
selected-ID sets, FP32 retention arithmetic and top-k counts, score ranking,
sink/window unions, boundary thresholds/margins, exchange deltas, tie/error labels,
Jaccard values, endpoint-error records and stored FP32 residual arithmetic. All
recorded source files and all six actual local model files were rehashed; current
CPU package/kernel versions and read-only GPU metadata match the recorded
environment. The diagnostic records clean source `a0100a5`, with kernel
`7.0.0-38-generic` and the
12 GB RTX 4080 Laptop GPU.

The raw score/per-page error arrays support these recalculations. Full Q/K and
affine-metadata tensors are fingerprinted rather than saved, so this audit cannot
independently rerun their GPU score construction from tensor values. The reviewing
agent also authored the producer; this is a separate evidence recalculation,
complementing the earlier external code review, not an independent author review.

A subsequent independent author review by the checkpoint auditor also passed.
It verified all 18 snapshots, 432 head observations, raw/public hashes, 47 scoped
source files and six model files, and independently reconstructed the aggregate
counts, margins, overlaps and residuals above. The tensor-reconstruction and
old-capture linkage limitations remain unchanged.

## Recorded differences

| Observation | 8k / 0.20 | 32k / 0.25 |
| --- | ---: | ---: |
| Heads with different top-k membership | 28 / 216 | 102 / 216 |
| Heads with different effective sink/window selection | 28 / 216 | 102 / 216 |
| Headwise page replacements across nine snapshots | 28 | 115 |
| Mean top-k Jaccard | 0.990028 | 0.991715 |
| Mean effective-selection Jaccard | 0.991385 | 0.991931 |
| Worst head top-k Jaccard | 0.923077 | 0.968992 |
| Maximum absolute page-score difference | 0.862610 | 1.105164 |
| Maximum reconstructed-key-maximum error | 0.109375 | 0.117188 |
| Maximum score residual after recorded endpoint perturbation | 0.000135064 | 0.000149190 |

Page replacements count head/snapshot occurrences, not unique physical cache
pages. For top-k, each replacement removes one summary-selected page and adds
one metadata-selected page. Sink/window inclusion preserves some pages in both
arms: effective metadata-only/summary-only counts total 28/26 at 8k and 114/115
at 32k. It does not remove every membership difference.

No computed boundary is exactly tied. Every changed head has a summary boundary
gap no greater than twice its maximum score perturbation. Gaps among changed
heads range from 0.002518 to 0.326569 at 8k and 0.002258 to 0.326416 at 32k.
The sufficient stability condition, gap strictly greater than twice the maximum
perturbation, holds for 50 unchanged 8k heads and no 32k heads. Failing that
condition does not imply a membership change. No tolerance relabels observed
changes as harmless ties.

Recorded minima agree exactly with separately computed BF16 minima. Reconstructed
maxima differ. The signed endpoint term
`Q_negative × minimum_error + Q_positive × maximum_error`, summed over channels,
accounts for the stored score deltas up to the recorded FP32 evaluation residual.
The FP32 endpoint-range/15 screen finds zero channels at or below 1e−6; this is
not a count of BF16 quantizer clamp decisions. Rounded affine maxima must not be
called exact original-key extrema or strict upper bounds on the original keys.

## Concrete page exchanges

At 8k, layer 0, step 0, query head 1, summary scoring selects page 117 over page
104 by 0.102233887. The metadata score changes are −0.218383789 for page 117 and
+0.599670410 for page 104, reversing that pair by 0.715820313. Their recorded
endpoint terms are −0.218351841 and +0.599618912, leaving residuals of about
−0.000031948 and +0.000051498. Both pages have maximum endpoint errors of
0.046875. Neither page is forced by the sink/window union.

At 32k, layer 0, step 0, query head 0, summary scoring prefers page 266 over page
386 by 0.259521484. Metadata changes their scores by −0.354125977 and
+0.358703613 respectively, preferring page 386 by 0.453308105. The recorded
endpoint terms differ from those score changes by about 0.0000625/0.0000629.
The maximum endpoint errors of those pages are 0.0390625/0.02734375.

A tighter 32k exchange appears at the same layer/step, head 6: page 474 replaces
478. Summary scoring prefers 478 by 0.126556396; the perturbation advantage for
474 is 0.127471924, leaving a metadata advantage and boundary margin of
0.000915527. This is a strict computed ordering, rather than an exact tie. Page
IDs are zero-based; each page covers `[64 × id, 64 × (id + 1))` cached tokens.
All changed IDs, scores and margins remain in the linked report.

## Link to the earlier contribution captures

Comparison with the original [8k captures](../contrib/94de6da12b4ef7a29e499d03099961a2a26d1e762dcc9120b8c101318a52becc/report.json)
and [32k captures](../contrib/c7227ade89ba055612212849d1d433dee80c7b5226b9a4c098b0968a32461124/report.json)
finds 9/9 input matches, 0/9 query matches and 0/9 key matches at each context.
Model contents, runtime/generator/corpus hashes, listed package versions and GPU
metadata match. The environment kernel changed from `7.0.0-34-generic` to
`7.0.0-38-generic`; shared measurement-script fingerprints also differ for
`bench_common.py` and `profile_contribution.py`. The original source snapshot
remains authoritative for its measurements. These differences do not establish
the cause of the tensor mismatches or bitwise greedy-output reproducibility.

This result explains concrete selection disagreements in new representative
captures. It cannot recover the original snapshots' exact changed-page lists,
because their full scores and tensors were not saved. Neither this bounded
diagnostic nor its similar aggregate overlaps establish general quality, a
quality-failure cause, novelty, performance, or physical 4 GB capacity.
