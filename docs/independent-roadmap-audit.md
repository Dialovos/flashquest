# Independent roadmap audit

Updated: 2026-10-04. Initial checkpoint: local commit `46a319c`.

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

## Checkpoint 6 — Evidence integrity and fresh-run source freeze

On 2026-10-04, an independent CPU run passed 174 checks across competitor
validation, native response contracts, confirmation statistics, quality,
internal ablation and contribution tests. CUDA was hidden; four GPU contribution
cases were skipped and the socket readiness/cleanup case was excluded. This
run includes 83 competitor-validation cases and the explicit async-prefetch
offload rejection. It checks the implementation's evidence contracts; it does
not establish backend initialization, retrieval quality or competitive speed.
No implementation file was edited and no GPU experiment was run by this auditor.

All saved competitor cell identities and private raw hashes were independently
rechecked. The fourth
[1024-token smoke schedule](../benchmarks/validation/competitors/eef23d91caff6fdef91041ba4de25c42d5a37a12caae006fbc46980c3aab03c3/schedule.json)
is correctly `complete-with-failures`. Its llama.cpp Q4 and FP16 cells each
complete one eight-output request, report 29/29 layers on GPU, Flash Attention,
CUDA KV buffers and a realized 1280-token cache. The two vLLM cells fail during
startup compilation; their telemetry is explicitly invalid. The private logs
identify incompatible CUDA compiler/toolkit headers, rather than a failed
quality or performance comparison. The compiler packages have subsequently been
pinned to the CUDA 13.0 toolchain. That environment correction alone establishes
no successful vLLM request. The fifth standalone retry was interrupted without
a terminal result, so its outcome remains unknown.

The three contribution reports' canonical identities, protocol hashes and raw
hashes were rechecked. Their interpretation remains the one in checkpoint 4:
exact additional-summary byte accounting and avoided materialization in the
specified reference workload, with selection differences and no established
scoring-latency advantage. Their public tensor hashes were subsequently exported
as digest-first typed `tensor_fingerprints` entries to avoid security-scanner
false positives on raw-key field names. Independent reconstruction of that
allowlist transformation exactly matches all three public reports. Raw files,
hashes, source identities and every measurement remain unchanged. No later commit
should replace their saved dirty-source identity.

The exact scoped producer/runtime sources are now preserved locally under ignored
`artifacts/contrib/source-snapshots/<source-content-sha256>/`, with relative paths
and a source manifest. All 46 file hashes independently match the measured
fingerprint and the source recorded by all three reports, including the original
measurement script and original `bench_common.py`. This preserves the actual
dirty-source experiment independently of the later export-only revision.

The corrected-toolchain
[backend environment record](../benchmarks/validation/environment/5bb41d4781b0b83fdb9d55a010f75c837a293880b3f8ea0f615f6869b26d4480/backends.json)
passes independent canonical-identity, protocol-hash and sanitized-export checks.
The first export had a mismatched run identity because public file-list metadata
was hashed before canonicalization. That metadata-only discrepancy was reported
and corrected; its original invalid export remains an ignored setup artifact for
traceability. No experiment measurement was changed. The accepted record's scope
is package/compiler/binary preparation, rather than a successful GPU smoke or
competitive measurement.

The [implementation plan's current checklist](roadmap-implementation-plan.md#6-current-execution-status-and-decision-framework)
and README correctly keep fresh confirmation, long-context competitor timings,
matching competitor quality and actual 4 GB testing open. The refreshed
[roadmap](../roadmap.md) now aligns its setup/component checkboxes with saved
evidence and leaves the scientific gates open. Its relative evidence links,
and the links in this audit and the decision document, resolve locally. A frozen
protocol and working CPU validators do not complete an experimental endpoint.
Partial fresh observations must remain incomplete and resumable under the same
source/model/environment identity; they cannot support a confirmation verdict.

The [current research decision](research-decision.md) is provisional: continue
bounded validation and retain the implementation as useful engineering work.
The available evidence justifies completing the comparison, but neither certifies
novelty nor establishes a runtime advantage over optimized quantized dense
engines. Failure or absence of confirmation is not proof of inferiority. Physical
4 GB testing remains explicitly pending collaboration, as requested by the user.

### Fresh-run source freeze

The first fresh 4k run has started at local commit
`05f64004dbfb3c33d42d60e68d3f6e5091de7d5d`, with clean measurement-source fingerprint
`8dff99e32861785f33e334b3ccc33d04c410e7eb5817ce450f6e9bd74d0bc30f` and the
saved nine-endpoint protocol. Its
[public record](../benchmarks/validation/quality/0526a7546dbef2cb2e78c919423d65ab0b55e3457ea1aa845edc2cf5502bdb69/quality.json)
is incomplete at this checkpoint. The budget is 300 input examples and 900 arm
outcomes at 4k; the complete three-context protocol requires 2700 arm outcomes.
Starting this run establishes no quality verdict.

Continue the complete fixed budget across all three contexts. Preserve partial
atomic records if interrupted and resume with the identical command/settings and
`--resume`. Keep this Git HEAD and the source, model and both validation
environments unchanged through all three contexts; even a documentation-only
commit changes the recorded source commit. These document edits are intentionally
left uncommitted until the frozen confirmation is finished. If the source or
environment must change, preserve the incomplete run and start a new coherent
confirmation identity.

## Checkpoint 7 — Pinned native quality contracts before competitor runs

An independent read-only review of llama.cpp commit
`11fe02151f79c41d0d4af7da708755d73b9c0da6` confirms that `/detokenize` renders
special/control tokens, including BOS, without an extra request flag: its handler
uses `tokens_to_str`, whose `common_token_to_piece` call defaults to `special=true`.
Pure integer completion prompts bypass tokenization and additional BOS insertion.
The returned generated-token list includes the terminal EOG token, and native
output counts include it; later decode steps still exclude the first output.
Generated response text suppresses control-token spelling under the current
server defaults. These findings follow the pinned
[server handler](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/tools/server/server-context.cpp),
[token conversion](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/tools/server/server-common.cpp), and
[conversion defaults](https://github.com/ggml-org/llama.cpp/blob/11fe02151f79c41d0d4af7da708755d73b9c0da6/common/common.h).

Installed vLLM 0.30 source also retains the stop token in output IDs and completion
usage while omitting it from default text. Its neutral `generation-config=vllm`
sampling defaults do not eliminate model EOS metadata. The explicit quality stop
IDs match the pinned generation config's set `{128001,128008,128009}`; existing
llama.cpp loader logs resolve the same EOG set. Actual quality smokes must still
verify the complete request/response contract on GPU.

Two forward quality gates need strengthening after the confirmation source freeze:

- The pinned HF tokenizer has `clean_up_tokenization_spaces=True`, which the
  FlashQuest answer decoder uses. Native server text is currently scored directly
  by the competitor adapter, so punctuation/contraction cleanup can differ.
  Decode returned generated IDs using the pinned tokenizer and the same HF
  answer-decoding options, keeping the native text as separate private evidence.
  Seven-digit retrieval substrings are insensitive to the observed cleanup
  example, but that does not establish a general decoder-equivalence contract.
- HF `all_special_ids` contains only two registered special-token IDs for this snapshot,
  while 256 added tokens have their `special` flag set. The GGUF mapping gate
  currently checks the former set. Include the latter and all generation EOS IDs
  in the required mapping checks. The earlier full-vocabulary comparison of the
  actual pinned GGUF found no mismatch; this is a forward-validator gap, rather
  than evidence of wrong inputs in that artifact.

Installed FlashInfer's `_get_num_workers` accepts numeric `MAX_JOBS`; `run_ninja`
uses it as `-j`. Setting `MAX_JOBS=2` therefore bounds its compiler worker count
without changing inference settings. Record that build setting with a future
competitor identity. None of these findings requires changing the active fresh
confirmation source or retuning its fixed experimental criteria.

## Checkpoint 8 — Complete fresh 4k retrieval evidence

The [fresh 4k record](../benchmarks/validation/quality/0526a7546dbef2cb2e78c919423d65ab0b55e3457ea1aa845edc2cf5502bdb69/quality.json)
now contains all 45 cells and 900 arm outcomes. An independent reconstruction
validated canonical identity, public/raw equality, raw SHA256, exact protocol,
model and clean source, the 300-example ordered manifest, every input hash and
length, capacity, scorer outcome, generated-text hash and hit count. All three
arms share exactly paired example IDs, input hashes and token counts. Source and
Git HEAD still match the frozen snapshot in checkpoint 6.

Independent Clopper–Pearson inversion used direct binomial coefficient sums,
rather than the implementation's log-tail calculation. Gain lower and loss upper
bounds agree within 1e−12. The bound alpha remains `0.05/(2×9)`, even though only
the first context is complete.

| Task | Dense / all-pages / sparse hits, out of 100 | Sparse gains / losses against dense | Lower sparse-minus-dense difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 100 / 100 / 99 | 0 / 1 | −0.078127 | Pass |
| Multivalue | 99 / 99 / 96 | 1 / 4 | −0.127912 | Fail |

Observed dense and sparse rates exceed 0.80 in every endpoint. The multivalue
endpoint fails because its lower bound does not exceed −0.10, despite an observed
difference of only −0.03. This failure to establish the specified non-inferiority
claim does not prove inferiority. The overall nine-endpoint verdict remains
withheld until 8k and 32k complete. Continue the fixed budget without retuning
retention, expanding the sample or changing the margin after this result.

The user subsequently requested a workload checkpoint after the active 8k run
and its independent audit, with 32k held for continuation. This scheduling pause
does not replace the fixed confirmation budget or authorize a verdict on a
selected subset. Preserve the current Git HEAD, source, environments and complete
records; keep draft documents/results uncommitted until the three-context freeze
ends. Resume 32k with the pinned protocol and unchanged settings. The remaining
competitor/quality-adapter work begins after that confirmation group.

## Checkpoint 9 — Deferred comparison and selection diagnostic review

Ignored development drafts remain separate from the frozen measurement source.
The competitor patch's six original-file hashes still match `05f6400`, and
`git apply --check` passes without application. Independently run draft checks
pass 118 CPU cases, with the loopback cleanup case excluded, and Ruff. Canonical
answers are reconstructed from private returned IDs through the pinned HF
decoder; native text remains separately hashed. An actual CPU comparison finds
all 128,256 GGUF/HF vocabulary pieces equal, including output-only ordinary IDs;
the added-special/EOS union contains 256 IDs. Extracted installed FlashInfer
helpers, with the subprocess mocked, produce `ninja -j 2`. Schedule and child
identities bind that effective compile policy. These checks establish neither
vLLM startup success nor successful native quality requests.

The revised bounded selection diagnostic passes 23 CPU checks and Ruff. Its scoped callback,
actual score and selection implementations, GQA grouping, sink/window union and
signed affine-endpoint perturbation agree with the runtime. Its source manifest
binds the new producer and existing capture/runtime; Q/K, scale/minimum and
prompt fingerprints allow a later comparison to the original contribution
snapshots. Only matching recaptured Q/K establishes that an explanation applies
to those exact earlier snapshots. No recapture has run during the freeze.

The selection revision resolves all three review findings: metadata boundary ties
are distinguished from strict score crossings; selection counts follow the
selector's FP32 retention arithmetic and both masks are checked; epsilon-channel
counts are explicitly an FP32 screen, rather than a BF16 quantizer clamp count.
The fixed 0.20/0.25 default budgets were unaffected by the custom-budget rounding
case. A shared competitor quality termination gate now checks the pinned terminal
EOS, rejects earlier EOS and requires the exact cap for a length stop. Pinned
vLLM and llama.cpp both give EOS precedence when it lands at the cap. Final patch
refresh and focused tests must cover this added gate before application. Apply
and commit either draft only after all three frozen contexts complete;
historical records remain intact.

The execution checkpoint correctly preserves Git HEAD and the source fingerprint
across a pause, resumes the same fixed seeds/settings and keeps both deferred
drafts unapplied. Its 32k command and protocol agree with the frozen experiment.
The actual 4 GB target and optimized competitor comparisons remain pending.

## Checkpoint 10 — Immediate workload pause; 8k audit deferred

The user subsequently requested an immediate Codex workload checkpoint. At that
request, 8k had saved 780 of 900 outcomes; the owned supervised child is allowed
to finish and automatic 32k execution remains held. No terminal 8k evidence audit
or endpoint-bound verdict has been issued by this independent agent. Check its
saved status and independently reconstruct the complete 8k record on resume,
before launching 32k with the unchanged frozen protocol. The prepared read-only
audit requires both 4k and 8k complete and recomputes exact binomial bounds using
direct coefficient sums. Partial saved outcomes establish no endpoint verdict.

This scheduling change supersedes the earlier plan to finish the 8k audit before
pausing. Preserve `05f6400`, the measurement source and both environments; keep
documents/results uncommitted and development drafts unapplied. The 4k audit
remains valid, the nine-endpoint confirmation remains incomplete, and optimized
competitors, actual 4 GB testing and a final research decision remain pending.

## Checkpoint 11 — Complete original 8k audit and coherent environment reexecution

The [original 8k record](../benchmarks/validation/quality/7cec2647f00356ca86a526ff3bb2a16c28a20661d6dcf1fd8ba5d6e7c1427384/quality.json)
is complete with all 45 cells and 900 arm outcomes. Independent reconstruction
validates the canonical identity, public/raw equality and raw SHA256, pinned
protocol/model/source, all 300 ordered manifest examples, every input hash/count,
scorer outcome, output-text hash/count and exact paired inputs across arms.
Original 4k and 8k recorded environments and realized runtimes match. The current
clean source still exactly matches their frozen `05f6400` snapshot. The 8k
inputs span 7935–8059 tokens and the bound cache capacity is 8187 tokens.

Direct binomial-coefficient sums and 100-step bisection independently reproduce
all bounds within 1e−12, retaining alpha `0.05/(2×9)`.

| 8k task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Lower sparse-minus-dense difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 98 / 98 / 97 | 0 / 1 | −0.078127 | Pass |
| Multivalue | 97 / 96 / 86 | 0 / 11 | −0.223272 | Fail |

Every observed accuracy floor passes. Multivalue has an observed difference of
−0.11 and fails the frozen non-inferiority rule; that does not establish
population inferiority beyond the margin. The old six-endpoint summary remains
inconclusive because 32k is absent. Preserve both old failed multivalue endpoints.

The workload pause has ended. Before 32k, an independent current-environment check
detected kernel `7.0.0-34-generic` changing to `7.0.0-38-generic`; ordinary-sandbox
NVML was unavailable. The parent repeated the comparison through its approved
read path and confirmed only the kernel changed, with GPU/driver, Python and
packages identical. Mixing a new 32k record with the old pair would violate the
existing same-environment summary contract. No historical metadata is rewritten.

The resolution is a full, fixed-count same-seed three-context reexecution in the
current environment, with the identical clean source, model, frozen protocol,
seeds, retentions, generation/scorer and per-endpoint sample budget. It creates
new environment-bound run IDs and preserves all original 1800 outcomes. Verify
the new 4k/8k manifests equal the originals, freeze current provenance across all
2700 outcomes, and independently audit before the final decision. This is not
independent new confirmation data or post-hoc expansion. Do not pool groups,
erase the original failures or choose whichever group has favorable outcomes.
Tracked source and Git HEAD remain frozen; both development drafts stay deferred.

## Checkpoint 12 — Final deferred quality patch and repeated input gate

The refreshed competitor patch exactly equals the diff of all six reviewed
baseline/draft files, and its SHA256 matches the saved validation metadata.
Baseline hashes and `git apply --check` pass. Independent CPU checks pass 141
cases, excluding only the loopback cleanup case; the author's saved full run
reports 142. Ruff passes. The shared termination gate is exercised in both the
producer and scheduler, rejects nonterminal/missing EOS and premature length
stops, and preserves performance `ignore_eos` behavior. Its EOS-at-cap precedence
agrees with the pinned native source. The prior decoder, full-vocabulary,
public/private evidence and compile-worker gates remain active. No further
substantive draft defect was found; application and native GPU smokes remain
deferred until the coherent confirmation source freeze ends.

The new [4k environment-reexecution record](../benchmarks/validation/quality/7d9921cb0d5b6167c5303932475ccb71402ece5ca9066d7e067f23dab4d9cdef/quality.json)
binds exactly the original ordered 300 examples, settings, model, source and
protocol. Independently comparing canonical identities finds only the kernel
changed in provenance. Its immutable run ID validates. This is an input/identity
gate for an active run, not a partial-outcome verdict.

## Checkpoint 13 — Complete current 4k audit and active 8k input gate

The [current 4k reexecution](../benchmarks/validation/quality/7d9921cb0d5b6167c5303932475ccb71402ece5ca9066d7e067f23dab4d9cdef/quality.json)
now contains all 45 cells and 900 arm outcomes. Independent reconstruction
validates raw/public equality and hashes, canonical model/source/protocol/run
identity, every ordered manifest/input/scorer/output count and hash, exact paired
arms and current clean source. Environment hash
`ec59cde8cd845e4d60cff4f3a545c614a19ebae3fc28e26487ce108b554527f5`
binds kernel `7.0.0-38-generic`. Direct binomial inversion reproduces the bounds
within 1e−12, retaining alpha `0.05/(2×9)`.

| Current 4k task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Lower sparse-minus-dense difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multivalue | 99 / 99 / 97 | 1 / 3 | −0.112446 | Fail |

Every observed accuracy floor passes; multivalue still fails the specified
non-inferiority rule. The current family remains incomplete pending 8k/32k.
Keep its retention, count and criterion unchanged and preserve the old failures.

An exhaustive comparison of all 900 matched old/new outcomes confirms identical
inputs, expected values, settings, model and source. Generated text/hash changes
in 261 outcomes; output count changes in 71 and termination changes in 50.
Changed text counts are 32 dense, 121 sparse and 108 all-pages. Exactly two hits
change, both sparse miss-to-hit: `multikey:1:18` and `multivalue:5:18`. There are
no hit losses across runs. This reexecution does not establish a causal kernel
effect or bitwise reproducibility of greedy outputs. Neither pooling groups nor
selecting a favorable group is permitted; both retain failed 4k multivalue rules.

The [current 8k record](../benchmarks/validation/quality/6c938a6b4515ffaf0de9ed8bf833e521584887705e4665e43b51e9e94c3aafb1/quality.json)
has started. Its canonical identity and all 300 ordered examples independently
match the original 8k inputs/settings. Model, clean source, environment and
realized runtime exactly match current 4k. Every input hash validates. This is an
input gate only; no incomplete 8k outcome or endpoint verdict is interpreted.

## Checkpoint 14 — Post-freeze comparison gates and deferred runtime fix

The execution order remains coherent: finish and independently audit the fixed
current-environment family; apply the reviewed deferred changes after the source
freeze; verify fresh native performance and quality smokes; repeat the balanced
internal 8k/32k performance blocks with matching current-environment quality;
then complete optimized native comparisons and the final evidence decision.
The internal performance prerequisite still requires the frozen pilot-ratio
screen. A failed 32k screen must leave that current block gated. A passed screen
with failed confirmation non-inferiority permits qualified engineering timings,
but does not establish an advantage at comparable quality. Actual 4 GB hardware
testing remains explicitly deferred to collaboration.

A bounded review found that vLLM runtime verification accepted an empty
attention-backend selection. The installed default-selection logger and
preserved startup logs identify `FLASHINFER` or `FLASH_ATTN`; requested FP8
precision alone does not identify this backend. Only the ignored competitor
draft was revised to reject missing/empty selections, with five new cases for
the missing field, empty list, missing log and two pinned log formats. All 147
focused CPU cases pass, as do Ruff, six-file baseline integrity and patch
applicability. The revised patch SHA256 is
`0a64fcc74b0aae8c0f09b8c1a264c1eac244a86b9a225a8a1d950efe034e1836`;
the preceding `d0376413` patch and validation are preserved privately. The parent
independently reviewed and accepted the one-line guard and five new cases,
verified the patch and all six draft hashes, and confirmed the frozen source is
clean. Application remains deferred until the full family completes. No tracked source,
tests, dependencies, Git HEAD or GPU execution changed during this work.

Final evidence checks must also require every local raw memory series to exist,
match its hash, and reproduce phase windows, peaks and coverage. The internal
`checked_memory` helper checks a present series but returns false when absent;
its consumers ignore that return. This is an evidence-consumption gap, not a
demonstrated invalid measured result. Native memory checks already reconstruct
the raw summary. Report FlashQuest synchronized forward, vLLM native scheduled,
llama.cpp native and HTTP client timing boundaries separately, together with
their different allocated cache capacities and weight formats. Treat diagnostic
recaptures as explanations of original contribution snapshots only when their
Q/K fingerprints match; a new valid snapshot alone does not establish that link.

## Checkpoint 15 — Complete current 8k audit and preserved original summary

The [current 8k reexecution](../benchmarks/validation/quality/6c938a6b4515ffaf0de9ed8bf833e521584887705e4665e43b51e9e94c3aafb1/quality.json)
contains all 45 cells and 900 arm outcomes. Independent reconstruction validates
the raw/public equality and hashes, all 300 ordered manifest examples and input
hashes, substring scorer, output counts/termination labels and generated-text
hashes, exact paired arms, protocol and immutable model/source/run identity.
Current source remains exactly clean `05f6400`. Model, environment and realized
runtime match current 4k, including environment hash `ec59cde8…` and kernel
`7.0.0-38-generic`. Python/kernel/package versions match the actual CPU
environment, and all six actual pinned local model files, including the weights,
were rehashed successfully. Prompt lengths span 7935–8059 tokens with cache
capacity 8187. Direct binomial-coefficient sums and 100-step bisection reproduce
all primary bounds within 1e−12, with unchanged alpha `0.05/(2×9)`.

| Current 8k task | Dense / all-pages / sparse hits, out of 100 | Gains / losses | Lower sparse-minus-dense difference | Frozen endpoint rule |
| --- | ---: | ---: | ---: | --- |
| Single | 100 / 100 / 100 | 0 / 0 | −0.057162 | Pass |
| Multikey | 98 / 98 / 96 | 0 / 2 | −0.096053 | Pass |
| Multivalue | 97 / 96 / 89 | 0 / 8 | −0.184352 | Fail |

All observed accuracy floors pass. Multivalue's observed difference is −0.08;
its conservative lower bound fails the specified non-inferiority rule, without
establishing population inferiority. The current family is incomplete pending
the fixed 32k endpoint budget. Do not change its retention, counts or criterion.

Comparing every one of the 900 original/current 8k outcomes confirms exactly
matching prompts, expected values, settings, model and source. There are 123
changed texts/hashes: 18 dense, 56 sparse and 49 all-pages; output counts change
in 47 cases and termination labels in 23. Four hits change, all sparse:
`multikey:3:5` becomes a miss; `multivalue:1:5`, `multivalue:2:1` and
`multivalue:3:17` become hits. The complete canonical comparison-list hash is
`bc60314c9664a18c6a96d1df9f412fbdf54d0814e57fefe7a73ec1745bccd05c`.
This does not identify a causal kernel effect or establish bitwise greedy-output
reproducibility. Keep both groups, without pooling or selecting favorable runs.

The separately saved [original-group summary](../benchmarks/validation/confirmation/699b3aecbca4be9cf65f52e08c2f3adc49fe6ea4b4a9eeb043109f79b3e7e2c7/summary.json)
also passes independent reconstruction of its canonical hash, original evidence
hashes and exact production summary. Direct binomial inversion verifies all 18
primary/diagnostic paired bounds with maximum numerical difference below 1e−15.
It references only the original 4k/8k records and retains both multivalue
failures; its status remains inconclusive at six of nine endpoints. No reexecution
outcomes are pooled into that original summary.

The active [current 32k record](../benchmarks/validation/quality/07ce7aa457aa9070cae386a86c9bcd00fb501e2d29c99aca56410183f74c98c4/quality.json)
passes its independent input/identity gate. All 300 ordered examples and their
input hashes validate against manifest hash
`6730a158f939256b9433e168bfb6055bce74c5d64dca120ef79c97a974ed2206`.
Its clean source, model, environment and realized runtime exactly match current
4k/8k; the frozen tasks, seeds, 20 examples per task/seed, retention 0.25 and
all-pages arm, scorer/generation budgets and cache settings are unchanged.
Inputs span 32511–32635 tokens, and capacity 32763 covers the actual maximum
plus the 128-token output budget. This verifies its declared input/capacity
contract; no partial 32k outcomes or physical 4 GB capacity verdict is interpreted.

## Checkpoint 16 — Terminal 32k and complete nine-endpoint confirmation

On 2026-10-05, the [current 32k record](../benchmarks/validation/quality/07ce7aa457aa9070cae386a86c9bcd00fb501e2d29c99aca56410183f74c98c4/quality.json)
contains all 45 cells and 900 arm outcomes, with complete status and no error.
The coherent family contains all 2700 fixed-budget outcomes. Independent
reconstruction again checks every current 4k/8k/32k raw/public sample, private
substring scorer, generated-text hash, output count/termination label, paired
example/input ID, manifest hash and ordered 300-example context. It verifies
every source-file hash and immutable model/protocol/run identity. At this audit
snapshot, source remains clean `05f6400` with content hash
`8dff99e32861785f33e334b3ccc33d04c410e7eb5817ce450f6e9bd74d0bc30f`.
All three records have identical model/source/environment/realized runtime;
environment hash remains `ec59cde8…`, kernel `7.0.0-38-generic`. Current CPU
versions match and all six actual pinned model files were rehashed successfully.

The [complete current-family summary](../benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json)
passes its own canonical hash, saved evidence hashes and exact production
reconstruction. It references only the three current-environment records and
does not pool the original group. Direct binomial-coefficient sums with
100-step inversion independently reproduce all nine primary and eighteen
diagnostic paired bounds, with maximum discrepancy below 7e−16. Primary alpha
remains `0.05/(2×9)`; diagnostic alpha remains 0.025.

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

The full family is **complete and fails the frozen non-inferiority rule**:
five endpoints pass and four fail. All observed dense/sparse accuracy floors
pass. Failure to establish non-inferiority does not prove population inferiority
or that the true loss exceeds the 0.10 margin. At 32k the multikey/multivalue
observed differences are −0.03/−0.02, with gains/losses 2/5 and 4/6; their
conservative lower bounds fail the specified rule. Preserve the separately
inconclusive original six-endpoint group and its failures without pooling,
retuning retention, relaxing the margin or adding post-hoc samples.

Independent reconstruction also verifies that the unchanged pilot-ratio screen
passes at every context. At 32k sparse/dense ratios are 1.0, 88/91 and 89/91;
all exceed 0.85. This permits the planned current-environment internal timing
blocks under their existing screen prerequisite. Those measurements remain
engineering characterization, without an established comparable-quality
advantage. The terminal evidence audit passes, so the confirmation source freeze
may close and the separately reviewed deferred work may proceed. No source,
dependencies, GPU work or commits were changed during this audit.
