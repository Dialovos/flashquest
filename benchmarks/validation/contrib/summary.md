# INT4 contribution and component pilot

Updated: 2026-10-03. These are isolated operator ablations on the 12 GB RTX 4080
Laptop GPU. They support a narrower implementation analysis; they do not establish
competitive runtime speed, general quality, novelty, or 4 GB capacity.

## Evidence and protocol

- [Actual AWQ 8k/.20 captures](94de6da12b4ef7a29e499d03099961a2a26d1e762dcc9120b8c101318a52becc/report.json).
- [Actual AWQ 32k/.25 captures](c7227ade89ba055612212849d1d433dee80c7b5226b9a4c098b0968a32461124/report.json).
- [Synthetic 8k/32k fixtures](a397b177ba4d64f14a0dc87bcb7499fd82350401f7c3e29fc2baabe69325e9c5/report.json).
- [Measurement script](../../../scripts/profile_contribution.py) and
  [targeted tests](../../../tests/test_profile_contribution.py).

Model: `casperhansen/llama-3.2-3b-instruct-awq`, immutable revision
`272b3bde867b606760447deb9a4d2719fbdfd3ae`. Each actual run uses one single-needle
prompt at seed 0; layers 0, 13, 27; decode forward steps 0, 1, 63; all 24 query
heads and eight KV heads. The loader computes in FP16, then the runtime casts
post-RoPE Q/K to BF16. Capture hooks observe those actual BF16 values.

Each operator has 10 warmups and 32 measured CUDA-event samples. Live allocator
baselines and peak counters reset independently for every measured call.
Capture, correctness checks, profiler dispatch discovery, and JIT are outside
these timing samples. Operators run in fixed order; this small batch does not
control every temperature/cache/order effect. Timings cannot be added to infer
whole-model decode latency. Hooks are restored and captures released one layer
at a time. Every result includes exact source contents, model identity, environment,
protocol, input/Q/K fingerprints, and an ignored raw-evidence reference.

The source commit was `46a319c4463b51717a420484c13f768bcb8a52a3`, with dirty source
contents fingerprinted as
`421edae540f8ff625f21b4d1a8a68a693328969d39a541464040652eb3edd445`.
This content identity is authoritative for these runs; later commits must not
be retroactively substituted for it.

## Metadata reuse

Nominal retrieval budgets reserve generation margin. Actual input lengths are
7,936 and 32,512 tokens. Captures therefore contain 7,937–8,000 and
32,513–32,576 cached tokens. The final sampled step has no partial-page tail;
the earlier steps have one and two tail tokens.

Metadata scores and separately stored BF16 min/max scores use the same optimized
two-matmul GQA algebra. They are compared to true page maxima and attention mass
on the original post-RoPE keys, with tail mass always included.

| Observation | Nominal 8k/.20 | Nominal 32k/.25 |
| --- | ---: | ---: |
| Mean top-k Jaccard across nine layer/step snapshots | 0.98896 | 0.99222 |
| Range of snapshot mean top-k Jaccard | 0.98397–0.99359 | 0.98906–0.99545 |
| Median metadata scoring time, ms | 0.05658 | 0.06746 |
| Median separate-summary scoring time, ms | 0.05987 | 0.07099 |
| Median separate-summary construction time, ms | 0.06267 | 0.38034 |
| Extra stored scoring bytes with metadata reuse | 0 | 0 |
| Separate BF16 min/max bytes per layer at the first sampled step | 507,904 | 2,080,768 |

The byte saving is exact for the measured layout: two additional BF16 summaries
would cost 4/64 bytes per K element, or 12.5% of the packed INT4 **K** payload.
That fraction is not of the entire KV cache, model, or device memory. Shared
quantization metadata still exists and is counted separately in the JSON.

Scoring latency is similar, and metadata scoring is slower in some snapshots;
this batch does not establish a repeatable scoring-latency advantage. Summary
construction is paid when keys/pages change, not once for every decode query.
Rounded affine ranges differ from original extrema and some selections differ.
Several snapshots cross the proposed 0.99 investigation threshold, including
four 8k snapshots. Do not describe this as identical Quest summaries or a strict
upper bound on original extrema. The JSON records extrema errors, clamped
channels, boundary margins, per-head overlap, and retained oracle mass for review.

## Packed attention and components

The exact output/LSE oracle unpacks the same INT4 values into FP32 and promotes
stored BF16 scale/min metadata without an additional BF16 rounding. It matches
selection, GQA, scaling, and the unquantized BF16 tail. It never requantizes INT8.
Reference LSE uses explicit FP32 logits/logsumexp. Fused output and tail merging
return BF16, so output comparisons include that rounding.

The separately timed BF16 comparison reconstructs full cache K/V, rounds to BF16,
repeats GQA heads, and calls PyTorch SDPA with the same selection mask. It is an
optimized-dispatch reference workload, not a native quantized competitor or a
compressed selected-page gather implementation. Observed FP32 SDPA dispatch is
`aten::_scaled_dot_product_attention_math`; BF16 SDPA dispatch is
`aten::_scaled_dot_product_efficient_attention`. These boundaries are explicitly
different, and rerounding error is reported separately.

| Observation | Nominal 8k/.20 | Nominal 32k/.25 |
| --- | ---: | ---: |
| Maximum fused output absolute error versus exact oracle | 0.011434 | 0.013576 |
| Maximum fused LSE absolute error versus exact oracle | 0.000002385 | 0.000001431 |
| Median top-k selection time, ms | 0.07915 | 0.08704 |
| Median packed completed-page attention time, ms | 0.11144 | 0.35933 |
| Median packed attention including tail, ms | 0.22378 | 0.42896 |
| Median tail attention time when present, ms | 0.10152 | 0.11464 |
| Median merge time when present, ms | 0.06244 | 0.07134 |
| Median full dequantization plus BF16 SDPA time, ms | 2.23979 | 11.47077 |
| Maximum temporary allocator increment, packed plus tail, bytes | 70,144 | 70,144 |
| Maximum temporary allocator increment, BF16 reference path, bytes | 163,840,000 | 667,156,480 |

Timings are medians of the nine snapshot medians; tail/merge medians use the six
snapshots where those operations exist. Allocation increments cover the whole
measured operator call, including its output. They are not physical GPU-memory
measurements or full-model peaks. These results show avoided materialization in
this reference workload; they establish no speed ratio against llama.cpp/vLLM.

## Validation and remaining interpretation

Seventeen targeted checks passed: CPU-safe import/oracles, exact affine values,
GQA/masks/tails, clamped scales/ties/bytes, real fused attention at tail lengths
0/1/63, and tiny-model capture/restoration with advancing positions. A further
CPU-only run hides CUDA and skips the four GPU cases. New reports pass finite-value,
privacy/export, identity, and source-hash checks. Raw logs and diagnostics remain
under ignored `artifacts/contrib/`.

The evidence supports reusing existing affine metadata to avoid separate stored
summaries and avoiding large dequantized temporaries in packed attention. The
scoring-latency benefit is modest/unproven, and summary-selection differences
need investigation before a stronger equivalence claim. Use independent runtime
quality/performance comparisons and the prior-work map for the research decision.
