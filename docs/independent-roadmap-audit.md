# Independent roadmap audit

Updated: 2026-10-03. Initial checkpoint: local commit `46a319c`.

This audit was performed independently of implementation and experiment execution.
It covers [the roadmap](../roadmap.md),
[the implementation plan](roadmap-implementation-plan.md), the quality/performance
harnesses, and the saved evidence. Later checkpoints should record new findings
and resolutions here; passing software checks alone does not establish a research
claim.

## Current assessment

Continue the bounded comparison and contribution experiments. The 32k result is a
repeatable internal ablation signal, while 8k shows no practical improvement. There
is currently no demonstrated advantage over an optimized dense competitor, no
confirmed retrieval non-inferiority, no actual 4 GB fit, and no established novelty.
The roadmap correctly keeps these conclusions separate.

## Evidence independently verified

- All four quality records have valid canonical identities, protocol hashes and
  matching local raw-evidence hashes. The manifest hashes, example IDs, input hashes,
  prompt lengths, cache capacities, answer scoring, generated-text hashes and hit
  totals recompute for all 720 saved arm outcomes. These are tuning observations;
  the two 32k retentions reuse the same 60 prompts and are not independent samples.
- The 4k and 8k screens pass at retention 0.20. At 32k, multivalue fails at 0.20
  with sparse 16/20 versus dense 19/20; the 0.25 fallback passes with 17/20 versus
  19/20. Generation-limit misses remain failures.
- Both balanced schedules and their summaries recompute exactly. All 16 arm cells
  and 48 timed samples match their local raw records; result and memory-series
  hashes validate. The 8k median paired decode ratio is 0.994770; 32k is 1.838400.
- Source fingerprints match each recorded commit where the relevant source was
  unchanged. The dirty 4k source snapshot's 76 files match commit `bbc72be`; it can
  be reproduced from that source rather than its earlier recorded base commit.
- The sampled device peaks are 3,442 MiB at 8k and 7,006 MiB at 32k, with equal
  peaks in both arms. The persistent cache retains the entire context. These
  measurements establish behavior on the available device and do not establish
  4 GB capacity or absence of operating-system fallback.

Evidence links: [4k pilot](../benchmarks/validation/quality/dd8f60de24c2957a2120473d9ff3a194b64b22d85b2adfecca3aab485092f104/quality.json),
[8k pilot](../benchmarks/validation/quality/85c9abbb694ffb21ba5c03336188658ca2d0e9dff0790a9a9b62cb6b28c2ef07/quality.json),
[32k failure](../benchmarks/validation/quality/52b2599c93490bdc948431f43bf4fd2294273a00fac583461fe10bd5b0ba9f93/quality.json),
[32k fallback](../benchmarks/validation/quality/2baea9556c7ecdb9bb4213e0c02caf7444820468d36d912879d1a63190903f97/quality.json),
[8k performance](../benchmarks/validation/ablation/d86279b66c726c5697f408aabfd346170f8f072990bf95f7e0155416643a29a8/summary.json),
[32k performance](../benchmarks/validation/ablation/c921467929e57dad7293c868610d5dc39cdd0c01275c515a91d9d3d63060d66b/summary.json).

## Prioritized findings and requirements

### 1. Freeze confirmation before fresh examples

The proposed nine-endpoint design is reasonable: three tasks at 4k/8k/32k,
retentions 0.20/0.20/0.25, seeds 1–5 with 20 examples per task and seed, greedy
generation with EOS or 128 output tokens, and the current all-expected-substrings
scorer. Exclude pilot seed 0. Save the frozen protocol before collecting any fresh
outcomes and preserve the complete sample budget even when intermediate results
look favorable or unfavorable.

For the primary sparse-versus-dense family, exact gain/loss tails with
`alpha = 0.05 / (2 * 9)` and lower difference `L_gain - U_loss` provide conservative
simultaneous coverage of at least 95%. A cell passes the proposed non-inferiority
criterion only when that lower bound exceeds −0.10. Keep all-pages comparisons
diagnostic, outside this primary family. Require complete paired example IDs,
input hashes/counts, binary outcomes, frozen model/settings, and all nine endpoints.

The proposed dense/sparse accuracy floors of 0.80 are observed screening rules;
they do not guarantee that population accuracy exceeds 0.80. Any supported claim
is restricted to this fixed retrieval generator, model and settings. The 32k tuning
multivalue difference is already −0.10, exactly the proposed margin, so confirmation
can reasonably fail. Do not enlarge the sample, loosen the margin or retune retention
after viewing these fresh outcomes and then call the same data confirmatory.

### 2. Tie future performance gates to applicable quality

`quality_prerequisites` currently validates model identity, cell/count consistency
and the saved screen, but does not require the current quantizer/runtime or package
environment to match the screened implementation. A later algorithm change can
therefore inherit an old passing quality record.

Add an explicit semantic fingerprint gate for the active INT4 cache, quantization,
patched attention, scoring/selection and evaluation paths, plus AWQ loading and
Torch/Transformers/Triton/AutoAWQ versions. Relevant current files include
`persistent_int4.py`, `kv_quant.py`, `llama_persistent_patch.py`, `criticality.py`,
`selection.py`, `sparse_int4_fwd.py`, `niah.py`, `runner.py`, `awq_load.py`, and the
quality harness. Handle instrumentation-only differences by an explicit reviewed
allowance; never broadly ignore all source differences. This forward gate gap does
not invalidate the existing independently verified pilot comparisons.

### 3. Optimized competitors are required before a runtime win

The all-pages INT4 control still executes scoring/top-k and the same packed sparse
kernel. Its speed does not represent an optimized dense engine. Complete pinned
llama.cpp Q4 K/V and vLLM FP8 setup, short requests, realized kernel/cache checks,
and matching retrieval quality before accepting a competitive claim. Add native
FP16 KV controls where feasible.

Preserve differences in weight quantization and timing boundaries. vLLM prefill
and generation metrics must come from one request; native scheduling timing and
client end-to-end timing stay separate. llama-bench synthetic workloads are not
seed-paired FlashQuest inputs. Exact-ID quality adapters must verify every used
token and the complete decoded prompts, including special-token/BOS/EOS behavior.

Cover the owned server/worker tree during memory sampling and cleanup. The current
sampler stops the root process group; an adapter that starts a server in a separate
session must explicitly own and stop that additional group rather than assume it
will be cleaned up by the existing root-group logic.

### 4. Use the correct fused-attention oracle

The packed INT4 kernel applies stored affine scales/minima in float32 before QK and
V accumulation. The public dequantization helpers return BF16. Consequently,
BF16 dequantization plus SDPA is a useful optimized comparison, but it is not an
exact-value oracle for the fused kernel.

The exact output/LSE reference must unpack to float32, promote the stored BF16
metadata to float32, use `Q.float()`, identical selections/GQA/tails/scale, and
float32 softmax/logsumexp, casting output at the end. The legacy reference that
requantizes INT4 values to INT8 is also unsuitable. Report these reference
boundaries explicitly when comparing output error, latency and temporaries.

### 5. Bound metadata and memory conclusions

Metadata scoring reconstructs an affine range from rounded stored scale/minimum;
that range need not equal the original K extrema. Scale rounding can move its
upper endpoint below the original maximum. Measure score/selection agreement,
ties and retained attention mass instead of asserting an exact upper bound for
original K without checking it. Compare equivalent optimized two-matmul scoring
implementations, and include separate-summary derivation/storage costs.

The analytic summary saving is 4/64 bytes per K element at page size 64: 12.5% of
packed INT4 K bytes, not 12.5% of total KV or model memory. TurboQuant's separate
raw scoring summaries do not substantiate reuse of its quantizer metadata.

The current 32k prefill exceeds a 4 GB budget on the available card. Naively
splitting prefill is unsafe because `is_causal=True` with unequal query/key lengths
needs correct offsets for cached past tokens. Any bounded-prefill change requires
independent causal/output/quality validation. Actual target-device testing remains
unmeasured until the hardware is available; the research decision can still finish.

### 6. Keep documentation claims aligned with fresh evidence

The patched-attention docstring still cites the historical retention-0.20 quality
claim. When that file is next touched, update it to distinguish historical results
from the current matched screen and context-specific retention choice. Correctness
tests, pilot quality, statistical confirmation, competition, capacity and novelty
remain separate gates.

## Recommended next checkpoints

1. Review the implemented frozen confirmation protocol and statistics before any
   fresh-seed run; verify boundary cases and exact pairing.
2. Audit pinned competitor adapters after their smokes, including realized kernels,
   token equivalence, request counts, timing units and complete worker cleanup.
3. Audit fresh quality/performance and component/contribution evidence against
   their frozen protocols before drawing conclusions.
4. Review the final prior-work map and decision. Mark every roadmap item as measured,
   failed, unsupported or awaiting specific hardware rather than declaring an
   untested scientific gate complete.

No implementation files were edited and no GPU experiment was run for this initial
audit. The only audit artifact is this document.

## Checkpoint 2 — Confirmation and competitor adapters before GPU smoke

The [frozen confirmation protocol](../benchmarks/validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json),
`validation_stats.py`, and quality-harness binding were independently reviewed.
The settings, nine-endpoint multiplicity, strict paired manifest/raw reconstruction,
clean-source requirement and incomplete/weak-baseline handling match the design.
Analytic zero/all-success bounds at sample sizes 1, 20 and 100 pass an independent
check. At 100 pairs with no gains, zero/one/two/three losses produce lower
differences −0.05716/−0.07813/−0.09605/−0.11247: the first three satisfy the margin,
the fourth fails. Require coherent realized runtime fields across contexts as well
as coherent model/source/environment identity.

The first competitor-adapter review found issues to correct before accepting smoke
or comparison evidence:

- **llama.cpp token count:** in build b11382, `timings.predicted_n` counts all
  generated tokens, including the first. The native rate instead uses one fewer
  generation step. Checking `predicted_n == output_tokens - 1` rejects a valid
  response; check it against the output count and derive steps separately. This is
  verified in the pinned [statistics serializer](https://github.com/ggml-org/llama.cpp/blob/b11382/tools/server/server-common.cpp)
  and [generation-step helper](https://github.com/ggml-org/llama.cpp/blob/b11382/tools/server/server-common.h).
- **llama.cpp termination:** use `stop_type`, with explicit EOS/limit handling;
  the new parser checks the older `stopped_limit` field and labels limit responses
  as generic stops. The pinned [native response contract](https://github.com/ggml-org/llama.cpp/blob/b11382/tools/server/README.md)
  specifies the current field. Require requested generated token IDs and validate
  their types/counts for the exact-input quality adapter.
- **Forced timeout cleanup:** normal `Server.__exit__` uses creation-checked worker
  identities, but outer process-group termination can bypass Python finalization.
  An escaped worker in another session may survive. The outer observer must stop
  the known owned process tree on timeout and normal exit, preserving unrelated
  processes. Add a CPU regression with an escaped worker.
- **Realized kernel claims:** a generic occurrence of `marlin` in server logs can
  come from the requested command-line option. It does not establish the selected
  implementation. Require an authoritative dispatch record or keep the field
  unverified. Check effective KV precision/scales, cache capacity, llama.cpp Flash
  Attention and CUDA KV placement before treating a smoke as a valid quantized
  dense comparison.
- **Quality EOS parity:** disabling repository generation configuration in vLLM
  requires explicitly preserving the HF quality harness's full EOS-token set.
  `ignore_eos=False` alone may not preserve additional end-of-turn/message IDs.
  Verify terminal-token/count semantics using the pinned API.

The vLLM same-request timing definitions match its
[pinned metrics documentation](https://docs.vllm.ai/en/v0.30.0/features/per_request_metrics/):
TTFT begins at scheduling, generation spans first to last output, queue time is
separate, and metrics require one generation stream with statistics enabled.
Request-window memory markers are an appropriate addition; they do not establish
isolated native prefill/decode memory windows.

These are pre-experiment review findings, not failures of saved full-model
competitor results; none had been accepted at this checkpoint. Resolution and
successful smokes remain the next audit gate.

A subsequent code review confirms corrections to the llama.cpp output-count and
termination contracts, required generated token IDs, tracked-worker cleanup in the
outer observer, and realized-runtime equality in the confirmation summary. An
escaped-worker timeout regression accompanies the cleanup change. Actual backend
smokes, authoritative kernel/cache verification and EOS-count validation remain
pending; a hardcoded EOS set must be checked against the pinned generation config.

## Checkpoint 3 — Closer prior work for the final novelty decision

An independent current search found prior work beyond Quest/KIVI/TurboQuant that
must constrain the claim:

- [Self-Indexing KVCache](https://arxiv.org/html/2603.14224v1) uses compressed sign
  codes for both retrieval and key reconstruction, with compressed-domain scoring
  and sparse attention kernels. The broad idea of reusing a compressed key
  representation to avoid a separate retrieval index is therefore prior work.
- [Self-Indexing Attention](https://arxiv.org/html/2609.13205v1) shares stored key
  signs and normalization between sparse retrieval and external low-bit KV
  compression. Its exact-budget comparison explicitly accounts for the shared
  representation and absence of extra indexer metadata.
- [LServe](https://arxiv.org/abs/2502.14866) and its
  [OmniServe implementation](https://github.com/mit-han-lab/omniserve) already
  combine low-bit KV compression with query-dependent page selection.

These primary sources do not by themselves establish the same affine INT4/INT8
scale/minimum page formula used here. They do establish substantial overlap with
the broader contribution. The remaining candidate is a specific specialization
that derives Quest-style page scores from existing affine metadata, with measured
storage, selection-quality and latency consequences. A mathematical identity or
an implementation difference alone does not establish research significance or
novelty. The final map must state these overlaps and verify closer implementations;
the search is not an exhaustive novelty certificate.

For fused-attention latency, keep an exact FP32-value reference separate from an
optimized BF16 dequantization/SDPA comparison. A win over FP32 SDPA alone does not
establish a win over an optimized BF16 attention path; disclose rerounding error
and the selected dispatch for each reference.

## Checkpoint 4 — Operator evidence and competitor execution gates

The three [component reports](../benchmarks/validation/contrib/summary.md) were
independently checked against their canonical identities, private raw-reference
SHA256 values and public rows. Every component's 32-sample median and maximum
incremental allocation were recomputed. The two actual AWQ reports contain nine
snapshots each: layers 0/13/27, forward steps 0/1/63, all 24 query heads, one
single-needle prompt at seed 0. The separate synthetic report contains two
fixtures. Their authoritative source-content fingerprint is
`421edae540f8ff625f21b4d1a8a68a693328969d39a541464040652eb3edd445`;
the working source was dirty and the recorded commit remains historical.

The reported component results reproduce exactly. Mean top-k Jaccard is 0.988960
at 8k and 0.992222 at 32k. Four and one snapshot means, respectively, fall below
0.99; worst individual-head overlaps are 0.923077 and 0.968992. Metadata scoring
is slower than separate-summary scoring in two and one snapshots, respectively.
None of these actual snapshots contains epsilon-clamped channels, but maximum
reconstructed-endpoint errors reach 0.109375 and 0.117188. The worst retained
original-key attention-mass changes versus separate summaries are −0.006937 and
−0.004795. These results establish a real rounding/selection difference and do
not support exact summary equivalence. The saved diagnostics do not include
changed-page IDs and their score margins, so the detailed disagreement
investigation remains open.

Maximum fused output errors against the exact FP32 packed-value oracle are
0.011434 and 0.013576; maximum LSE errors are 0.000002385 and 0.000001431.
The separately observed FP32 math and BF16 efficient SDPA dispatches are correctly
distinguished. Allocation increments of 70,144 bytes for packed attention with
tail versus 163,840,000/667,156,480 bytes for the materialized BF16 comparison
support avoided temporaries in that particular workload. Full-cache BF16
reconstruction and repeated GQA heads are part of the reference boundary.
Neither these increments nor the operator timing ratio establish end-to-end
competition, physical-device capacity or quality. The summary observes these
limits correctly.

The next competitor-code review confirms several forward fixes: context keys
are strings before protocol hashing/export, loaded local AWQ/tokenizer artifacts
are compared to the pinned model, child GGUF hashes are compared to the frozen
schedule, actual AutoAWQ kernel-selection logs are parsed, and worker observations
record KV dtype/storage/scales after a request. Initial dependency/CLI failures
are preserved as failed smoke attempts rather than replaced by a successful run.
Actual engine smokes remain a separate evidence gate.

Remaining pre-comparison acceptance requirements identified in this review:

- Preserve outer timeout/nonzero-exit failures even if a child already wrote a
  complete JSON result. A backend result and process-observation status are
  separate facts; a successful result must not erase failed execution.
- Validate child semantic configuration, expected sample completeness, raw
  evidence and realized cache/kernel/residency fields against the scheduled cell.
  On resume, retain failure exit status and require explicit handling of missing
  result files rather than assuming every attempt produced one.
- Strengthen the competitor quality-manifest loader to require coherent public,
  raw and manifest identities, complete unique input IDs/counts and confined raw
  references. Reject an empty quality subset and nonpositive sample limits.
- Reversing backend order between 8k and 32k is alternating context order, not
  balanced backend order within a matched context. Repeat matched-context order
  blocks or disclose that limitation before interpreting competitive differences.
- The performance semantic gate now checks active cache/kernel/runtime source
  and validation package versions. Include the quality harness/scorer and corpus
  in its explicit reviewed coverage, or explain a narrowly justified omission;
  the historical pilot comparisons remain independently verified.

The [prior-work map](contribution-prior-work.md) accurately states the affine
identity, reconstructed-range caveat and analytical 12.5% packed-K/6.25%
packed-K+V summary accounting. Its next revision should include Self-Indexing
Attention as well as Self-Indexing KVCache when discussing compression serving
as an index. The engineering result is presently defensible; a novel or practical
research advantage remains conditional on the fresh quality and competitor gates.

## Checkpoint 5 — Strict competitor validators before fresh runs

On 2026-10-04 the scheduler, native adapters and quality-manifest loader were
reviewed again. An independent run with CUDA hidden passed 92 CPU cases: 82
competitor-validation cases and ten native response-contract cases. The socket
readiness/cleanup case was excluded from this bounded run; its earlier separate
process-cleanup evidence remains applicable. No GPU experiment or implementation
edit was made by the independent auditor.

The previous scheduling and acceptance gaps are now addressed in code:

- Performance freezes the exact synthetic input SHA256 from the verified local
  tokenizer size, context and seed. Complete repetitions must match that hash,
  the scheduled seed/output/repetition budget and the native protocol.
- Child model, backend, source, environment, configuration, protocol, raw hash
  and public/raw reconstruction are checked. Derived rates and medians must
  agree with counts and phases; phases must also agree with the saved backend
  native metrics and their declared timing boundary. Quality outcomes are
  reconstructed from the exact manifest and private decoded answers.
- Timeout/nonzero-exit observations remain separate from backend results.
  Resume validates the schedule identity/protocol, exact cell path/order/count,
  child evidence and derived terminal statuses. A completed prefix or a
  successful status contradicting a timeout cannot pass.
- Public memory summaries are reconstructed from raw baseline/samples/markers
  and compared, with explicit phase-window and ownership coverage requirements.
  Whole-request windows do not establish isolated native prefill/decode memory.
- Default performance uses four backend settings and four input seeds with a
  Latin rotation within each context, balancing each setting's position.
  Different custom budgets need their actual ordering disclosed.
- Quality manifests now require coherent public/raw identities, confined raw
  paths, matching complete example IDs, valid task/seed/index metadata, nonempty
  integer token arrays, hashes and capacity. Nonpositive subset limits fail.
- The FlashQuest performance semantic gate includes the quality harness and
  corpus. The prior-work map now includes Self-Indexing Attention.

Installed vLLM 0.30 source was also inspected directly. Its ordinary `fp8`
cache uses `torch.uint8` storage selected by the cache configuration, so requiring
`torch.float8_e4m3fn` as the observed tensor dtype was incorrect. The realized
validator now checks the pinned uint8 representation, capacity, CUDA parameter
and cache devices, positive finite K/V scales and actual AutoAWQ dispatch. Unit
scales are allowed and labeled; they are not evidence of calibration.

The worker hook can run during vLLM's internal scheduler warmup before an HTTP
request. Its record and protocol now correctly say first worker execution,
possibly internal startup warmup. One final API mismatch was reported and the
observer corrected: CPU offload is read from
`vllm_config.offload_config.uva.cpu_offload_gb`, rather than missing
`CacheConfig` fields followed by a zero default. The observer now also records
the offload backend and async-prefetch group size. Require zero prefetch group
size as well as zero UVA offload before accepting the realized no-offload gate.
These checks still do not establish absence of OS fallback.

After that correction, the next gate is a real short engine smoke, including
cache/Flash Attention placement, complete native counts, EOS token behavior and
owned-worker cleanup. The CPU regressions do not establish that vLLM initializes
or runs successfully. Freeze one clean source/environment before fresh retrieval
confirmation, and keep operator pilots, fresh statistical confirmation,
competitive runtime results and unavailable physical 4 GB testing separate in
the final decision.
