# Roadmap implementation and test plan

Updated: 2026-10-04. Implements [roadmap.md](../roadmap.md).

The initial 4k milestone, longer retrieval pilots, balanced internal ablations,
component profiling and bounded contribution ablations are complete. They support
a 32k internal page-selection benefit and a metadata-storage saving. Fresh quality
collection is complete and fails its frozen rule. Optimized competitor comparisons
and actual 4 GB capacity remain open. See the [current execution checklist and decision framework](#6-current-execution-status-and-decision-framework)
and the [README evidence](../README.md#benchmarks-and-validation).

**Sections 1–5 preserve the 2026-10-03 audited design snapshot.** Their findings,
candidate versions, planned interfaces and proposed thresholds describe that
starting point, rather than today's completion status. Section 6 records progress
and the subsequently frozen confirmation protocol; that protocol supersedes the
earlier confirmation proposal. Do not retroactively change historical source
identities or reinterpret pilots as confirmatory results.

No approval of publishing or hardware changes is implied.
Follow workspace `AGENTS.md`, preserve existing edits and historical
results, and ask before pushes/PRs as required there. Technical browsing and routine
implementation choices do not acquire additional approval gates here.

## 1. Findings and evidence boundaries

| Finding | Current evidence | Correction |
| --- | --- | --- |
| Quality evidence is incomplete. | The INT4 harness defaults to 0.25, discards per-example samples, runs one seed, and lacks all-pages INT4. | WP2 before the next screen. |
| Replacement caches can overlap. | Patched forward closures retain the cache after `del cache`. | Reuse one reset cache; restore original forwards in `finally`. |
| vLLM timing depends on an old API. | The adapter selects V0 and requires `RequestMetrics`; current server APIs expose per-request metrics. | WP3: a pinned compatible release and same-request timing. |
| A quantization option does not identify the kernel. | vLLM 0.30.0 maps both `awq` and `awq_marlin` to `AutoAWQConfig`. | Inspect and record resolved kernels; remove the blanket slow-AWQ claim. |
| Historical residency is unknown. | Saved 32k allocator usage is 5,478 MiB, but the old hardware label was hardcoded. | Preserve observations; do not label them confirmed spilled or resident 4 GB runs. |
| Small KV storage does not establish capacity. | Selection reduces reads, while the cache stores all tokens; prefill dequantizes full K/V and uses full-context activations. | Measure total behavior; defer chunked prefill without claiming it is impossible or permanently rejected. |
| Resume/export boundaries are incomplete. | Config/schema matching misses source/model/environment changes; logs and raw JSON can expose local paths. | Immutable identity and allowlisted exports, WP1. |
| CPU safety, ordering, and memory attribution need work. | Kernel imports allocate CUDA codebooks; sparse-first order is biased; cumulative child RSS is not per-cell usage. | Injected runtime boundary, balanced matched blocks, and process-tree sampling. |

The previously reported 25 CPU checks, 24 GPU checks, and tiny-model smoke validate
plumbing. They do not establish full-model quality, competitive speed, 4 GB capacity,
or novelty. Re-run affected checks when the implementation changes.

Primary sources verified for this revision:

- [vLLM 0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0):
  candidate pin, subject to local driver/wheel compatibility.
- [Pinned per-request metrics](https://docs.vllm.ai/en/v0.30.0/features/per_request_metrics/):
  server timing fields, units, and missing-metric conditions.
- [Pinned quantization mapping](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/__init__.py)
  and [AWQ implementation](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/layers/quantization/auto_awq.py):
  option names do not establish the executed kernel.
- [llama-bench documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md):
  independent prompt/generation tests and depthful decode; pin the actual build commit.
- [AutoAWQ maintenance notice](https://github.com/casper-hansen/AutoAWQ): deprecated;
  last tested Torch 2.6.0/Transformers 4.51.3 differs from this project's Torch 2.5 pin.
  Compatibility must be demonstrated locally.

## 2. Protocol and bounded gates

Save a versioned protocol before its experiment and include its hash in results.
A local file suffices; a clean Git commit is useful for publication but does not
block a pilot. Thresholds below are proposals unless described as the existing
screen. Changes after viewing confirmatory data require exploratory labeling and
fresh confirmation, not a revised passing verdict on the same examples.

### Quality

- **Initial screen:** seed 0, 20 examples per task (`single`, `multikey`, `multivalue`),
  a nominal 4k budget, identical examples in three arms. Existing rule:
  `sparse_hits >= 0.85 * dense_hits`, with dense hits positive, for every task.
  Dense 20/20 implies 17/20 passes and 16/20 fails. Report absolute/all-pages counts.
  A weak dense baseline makes the screen inconclusive despite a passing ratio;
  inspect prompts, loader, and scoring first.
- **Fallback:** if 0.20 fails, screen 0.25 on the same examples, recording the tuning.
  If both fail, diagnose quantization versus selection using all-pages and stop
  expanded runs. Further optimization is a scope proposal, not an automatic search.
- **Confirmation proposal:** freeze retention, tasks, contexts, scorer, absolute
  quality floor, margin, sample count, and seed list before fresh examples. Initial
  budget: 100 examples per task/context, seeds 1–5 × 20, proposed non-inferiority
  margin 10 percentage points. Pilot/tuning seed 0 is excluded. Retrieval results
  alone cannot establish general long-context language quality.
- Save paired losses (dense hit/sparse miss) and gains (dense miss/sparse hit).
  A conservative lower bound on `sparse_rate - dense_rate` is the one-sided 97.5%
  exact binomial lower bound of gain frequency minus the one-sided 97.5% exact
  upper bound of loss frequency. Joint coverage is at least 95%. A proposed
  non-inferiority gate passes when this lower bound exceeds the negative margin.
  A bootstrap may supplement it; unanimous tiny samples must still have uncertainty.
  Report individual cells unless a multiple-comparison rule/simultaneous bounds
  were pre-specified for a claim across all tasks and contexts.

### Performance

- Batch/concurrency are 1. Fix actual input length, output count, model/tokenizer,
  context capacity, cache format, seed, and warmup. `n_decode=128` means 128 output
  tokens: one from prefill and 127 subsequent forward steps. Sparse/all-pages share
  exact input IDs; llama-bench uses its own workload, which must be disclosed.
- At least three measured repetitions after shape/path warmup. Use four input seeds
  for the initial repeated two-arm comparison to balance order exactly. Save samples,
  medians, spread, schedule, and retries. Never discard slow trials silently.
- **Practical signal proposal:** median of per-seed median throughput ratios >=1.10
  and every per-seed ratio >1.00. This is a repeatable pilot signal, not statistical
  proof. Otherwise report that a repeatable improvement was not established.
- Timing boundaries differ: FlashQuest synchronized forward phases, vLLM server
  scheduling-to-first/first-to-last tokens, llama-bench independent synthetic tests.
  Label them; compare decode cautiously. A common application end-to-end comparison
  needs matched adapter boundaries. Until then, report native request and client
  end-to-end metrics separately and do not ratio them interchangeably.

### Capacity and contribution

A context's successful prefill/generation is hardware-specific. A GPU-resident claim
also requires disabled CPU offload and explicit platform residency evidence. Timeout,
unsupported settings, and OOM are distinct. Sampled device memory is approximate;
report interval/coverage/baseline and keep allocator, device, RSS, and system fields
separate. A peak below total does not itself prove fit or absence of paging.

A smaller measured peak on the available 12 GB GPU estimates feasibility on 4 GB;
it proves neither target fit nor capacity parity. A capacity advantage requires
success and a reproducible, reasonably configured quantized dense competitor's
capacity failure on the same actual target device, with quality differences visible.

Metadata-scoring agreement is a correctness diagnostic, not a novelty/speed gate.
A proposed top-k Jaccard target of 0.99 triggers investigation; rounding, scale
clamping, and near-ties can change selection and must be reported.

## 3. Work packages

New flags/scripts below are planned interfaces unless already present. Implement
and inspect `--help` before using their example commands. Keep changes bounded.

### WP0 — Preserve and verify the starting point

Inspect the existing feature branch and dirty tree; preserve user edits without
switching/cleaning to facilitate this plan. Re-run affected benchmark regressions
and tiny GPU checks. Confirm shared hooks and commit identity if committing.

During later implementation, align roadmap/README vLLM, commands, result layout,
and status with actual behavior. Historical throughput remains non-comparable and
residency unverified. Remove unsupported patch docstring claims when touching it.
A reviewable local diff suffices; clean commits are not pilot prerequisites. Keep
code and evidence distinguishable, and follow workspace permission rules for PRs.

### WP1 — Identity, provenance, and safe exports

Extend `scripts/bench_common.py` with standard-library helpers; import-time behavior
must be GPU-free. Backend code passes CUDA metadata after runtime imports; the
common helper must not import torch/kernels merely to inspect package versions.

**Identity, resolved before reuse:**

- Schema/measurement versions and semantic config: retention/precision, context,
  output/EOS settings, warmup/reps/seed, generator/input hash, timing and memory budget.
- Code commit plus content fingerprint of relevant source/evaluation scripts,
  protocol, and dependency manifests. Include dirty and relevant untracked files;
  a dirty flag alone cannot distinguish edits. Store relative filenames/hashes only.
- Immutable model/tokenizer identity: HF revision plus config/tokenizer hashes;
  hash local weight artifacts when a revision cannot establish their contents.
  Add revision selection to loaders/adapters if needed. GGUF: name, size, SHA256,
  source revision, original model family. Local paths are operational config only.
- Backend version/build and binary hash, child interpreter package versions,
  Python/OS/CUDA, GPU model/compute capability/memory, driver, realized kernels,
  and fixed engine settings. Exclude hostnames, account names, serials, credentials.

Canonical JSON yields a run-identity hash for immutable result directories.
`--skip-existing` requires matching identity and valid complete samples; reject old
schemas, unknown required identity, non-finite metrics, errors, and incomplete
counts. Resolve model/backend identity before reuse; disable reuse if unresolved.
Telemetry/timestamps are provenance, not keys. Do not overwrite another identity.
Partial quality output uses atomic writes and explicit incomplete status; resume
preserves matched example IDs and cannot duplicate/reassign completed observations.

**Exports:** raw logs, tracebacks, package dumps, original llama.cpp JSON, prompts,
and sampler series stay in ignored `artifacts/`. Export allowlisted numeric/enum
fields, normalized model IDs/basenames, and reviewed summaries to `benchmarks/validation/`.
Normalize the whole record, including nested config/raw fields; prefer structured
error category/code over tracebacks. Prefix replacement alone misses temporary,
third-party, and Windows paths. Do not copy raw llama.cpp JSON into tracked output.
Final export validation rejects private identifiers/local POSIX/Windows paths and
unsupported fields, retaining public URLs/relative links. Ambiguous free text stays
local until reviewed. Retain a hash/relative reference to raw evidence.

CPU tests: missing packages/GPU tools; dirty/untracked source, model, backend, timing
changes; malformed/partial records; atomic/resume writes; nested absolute paths,
local model config and llama.cpp `model_filename`; exports retain numeric evidence.

### WP2 — Three-arm quality harness

Extend `scripts/phase6_run_ruler_4k_int4.py`: `--retentions` (default `0.20 1.0`),
`--seeds` (default `0`), tasks/count/output and identity/resume support. Keep old
single-value aliases if practical and reject conflicting aliases. Dense is the same
AWQ-weight model with native FP16 KV, not an FP16-weight reference.

- Load once and run unpatched dense first. Save each attention module's original
  bound forward before any patch; never save a patched closure as the original.
- Allocate one INT4 cache, reuse it for all patched retentions, and re-patch with
  that same object. Before every prompt reset seen-token counts and HF request
  state as needed. Views expose seen tokens; prefill overwrites their contents.
- In `finally`, restore originals, remove hooks/callback/default-argument references
  and last outputs, then release cache/model references. If allocation settings
  change, detach/release the old cache before allocating another. Rebinding forwards
  after allocation does not prevent the temporary double-cache peak.
- Generate a manifest once and run every arm on its exact token IDs. A small optional
  prompt-input adapter to `run_niah` may be needed; preserve existing callers.
  Save task/seed/index, input hash, actual prompt tokens, expected values, generated
  text, hit, output count/termination and arm settings; bulk IDs stay local.
- Verify `actual_input_tokens + max_new_tokens <= model/cache capacity` for every
  example, especially extra multikey/multivalue needles. Size cache from actual
  manifest lengths, not only nominal budget. Current NIAH leaves generation margin;
  nominal 4k/8k/32k quality labels do not imply exact-depth throughput inputs.
- Save after each task/arm; distinguish screen pass/fail, inconclusive baseline,
  incomplete, and execution error. Exceptions return nonzero. Optional
  `--require-screen-pass` can return nonzero for a screen failure without calling
  it an execution exception.
- Defer cache/patch/quantization/model imports until production execution; guard
  cleanup at runtime. Inject loader, cache factory, patcher, evaluator and cleanup
  so CPU tests neither import CUDA codebooks nor construct CUDA caches. Global
  codebook refactoring is outside this package unless necessary for this boundary.

CPU tests with GPU hidden: identical examples/three arms; aliases/default; dense=0
and 17/20 boundaries; capacity assertions; partial/resume alignment; cleanup on
failures; original forwards restored; weak references show old cache released
before replacement allocation. No models or AWQ kernels required.

Tiny-Llama GPU tests use `model.generate`, not only manual forwards: consecutive
prompts restart at position 0, decode positions advance from actual prompt length,
and retention repatching does not grow live allocations. Expanded statistics can
land later in `scripts/validation_stats.py`; test zero/all successes, non-degenerate
uncertainty for identical arms, opposite outcomes, and frozen margins.

### WP3 — Compatible environments and competitor adapters

**AWQ first:** inspect `.venv`, resolve bench dependencies under explicit constraints
and inspect the graph before installation. Do not force incompatible wheels with
`--no-deps` or silently replace validated Torch/Triton. If no coherent stack resolves,
report the dependency conflict and evaluate a bounded loader/version fix. Intentional
stack changes require updated constraints/provenance and relevant GPU revalidation.
Load a pinned AWQ model/tokenizer, inspect realized kernels and selected-device
placement, and generate briefly. Preserve native HF attention/cache for dense quality.

**llama.cpp:** build outside this repository in an allowed workspace location; never
run `vendor_clone.sh`. Set `LLAMA_BIN` and local `MODEL`, excluding paths from exports.
Pin build commit; check help for depth, cache formats, and Flash Attention syntax;
fail explicitly if unsupported. Record matching-model GGUF weight/tokenizer differences.
Smoke bench and later server independently; retain depthful decode/127-step convention.

**vLLM:** isolated interpreter under `.venv/backends/vllm/` (already ignored), with
independent dependencies. Candidate pin `vllm==0.30.0`; verify driver, CUDA wheel,
Python, GPU architecture and model support first. Do not upgrade system drivers as
routine setup. If incompatible, document a recent compatible pin and its API/source
contract, or mark unavailable. Do not revert to obsolete V0 just to preserve the adapter.

Add runner `--vllm-python` and configurable memory budget. The adapter manages a
loopback-only server and standard-library HTTP client inside its cell process group:
free local port, bounded readiness wait, complete server/worker cleanup on every exit.
Use one prompt/stream (`n=1`), exact IDs, no chat wrapping, prefix caching disabled,
deterministic sampling, fixed performance output count, no speculation, CPU offload
zero. Validate actual cache dtype/scales and resolved attention/weight kernels during
smoke; choose supported optimized AWQ behavior and record requested/resolved methods.

**Same-request timing:** enable `--enable-per-request-metrics`, keeping statistics
logging enabled. Validate units, finite positive durations and counts from the
pinned `/v1/completions` response:

```text
prefill_s = time_to_first_token_ms / 1000    # server scheduled-to-first token
decode_s = generation_time_ms / 1000       # same request's first-to-last token
prefill_tok_s = input_tokens / prefill_s
decode_tok_s = (output_tokens - 1) / decode_s
native_request_s = prefill_s + decode_s
native_output_tok_s = output_tokens / native_request_s
client_request_s = client_receive_complete - client_send_start
```

Store queue time separately, raw numeric fields and adapter/boundary versions.
Missing metrics mean unsupported phase measurement; one output token has no decode
interval. Never subtract a separate one-token request: its prefill/scheduling noise
does not reliably cancel. Upstream `tokens_per_second` is not pure decode throughput.

Smokes: quality at nominal 1024 with two samples/task; performance at 1024 exact
input tokens, eight output tokens and one repetition/backend. Save to ignored
`artifacts/smoke/`. Only AWQ quality smoke is needed before WP4. CPU tests cover
interpreter/command selection, counts/units/null timings, config identity, readiness,
and fake multi-process cleanup on timeout. Confirm FP8 scales/quality before treating
FP8 as a useful competitor; unsupported configurations remain explicit failures.

### WP4 — Quality screen and gated expansion

After WP1/WP2 and AWQ smoke, this is a **planned updated-harness** command:

```bash
.venv/bin/python scripts/phase6_run_ruler_4k_int4.py \
    --ctx-len 4096 --n-samples 20 --seeds 0 --retentions 0.20 1.0 \
    --out 'benchmarks/validation/quality/<run-identity>/screen-4096.json'
```

Replace `<run-identity>` with WP1's computed identity hash before execution; it is
a placeholder, not a literal directory name. An existing mismatched-identity output
must fail without overwrite; same-identity partial files follow the explicit resume policy.

Save protocol/identity without replacing another run. Inspect baseline absolute
quality, failed examples, actual lengths, positions and all arms. Apply the 0.25
fallback if needed. Failing all-pages calls for quantization/correctness diagnosis.

If useful, repeat screens at 8k then 32k where feasible. Freeze the confirmatory
protocol before fresh seeds 1–5, collect chosen sparse/all-pages/dense arms, and
report uncertainty. OOM is a capacity observation, not an incorrect-answer score.
Link evidence and limitations in the roadmap; do not claim general quality.

### WP5 — Sparse versus all-pages INT4 performance

Use a small standard-library schedule wrapper or saved one-cell schedule. Hold
context and seed fixed within a block; do not run all sparse contexts first:

```text
8192:  seed 0 sparse→all-pages, 1 all-pages→sparse,
       seed 2 sparse→all-pages, 3 all-pages→sparse
32768: seed 0 all-pages→sparse, 1 sparse→all-pages,
       seed 2 all-pages→sparse, 3 sparse→all-pages
```

Each arm warms then measures three repetitions on the same input IDs. Save order,
block/schedule, retries and telemetry. Whole-arm runs are the order unit; do not
claim repetitions were interleaved. Extend with reproducible balanced/randomized
schedules; competitor blocks use balanced backend permutations. No concurrent GPU jobs.

`scripts/summarize_validation.py` links records, computes per-seed medians/ratios and
spread, applies proposed practical gates, validates identity/exports/completion,
and separates missing/error categories. Incompatible records never join silently.

If gains are small/inconsistent, profile scoring, top-k, packed attention, tail and
merge costs. Retention 1.0 still pays scoring/top-k; label the primary ablation.
A selection-bypass all-pages arm is an optional follow-up with independent identity
and correctness checks, not a silent replacement of the original control.

Tests: per-context balance, resume retaining schedule/history, ratios/inconsistent
seeds, missing cells, identity mismatch and export validation.

### WP6 — Competitors and matching quality

Run fresh 8k/32k comparisons on the same GPU, one request at a time, matched output
counts and comparable warmup/repetition protocol. Use llama.cpp quantized K/V and
vLLM FP8; add FP16 where feasible. Record prefill chunking, engine settings, allocation
policy, weight differences and native boundaries. llama-bench repetitions are its
own workload, with no invented seed pairing to FlashQuest. Add a common request
adapter only if an application end-to-end claim is needed.

Export WP2's exact-ID manifest locally with generator/corpus/tokenizer hashes,
expected values, actual lengths and quality EOS policy. Require GGUF token-ID mapping
agreement for every token used in the manifest, including special-token and BOS/EOS
behavior, plus full decoded-prompt or tokenizer round-trip agreement for each input.
Representative checks and equal token counts alone do not establish equivalent prompts.
Mark incompatible inputs unsupported instead of silently retokenizing or adding BOS/chat tokens.

Add `scripts/competitor_niah.py` for pinned vLLM completions and llama-server token-array
completions. Match examples, answer scorer, deterministic settings, no prompt reuse,
output limit/EOS policy and token counts. Quality may stop at EOS; throughput forces
its count. Use native FP16/all-pages controls to isolate each backend's KV quality.
CPU tests cover payloads, identity/count mismatches, malformed responses, EOS/output
handling, matched scoring and cleanup. Matching quality accompanies competitive claims.

### WP7 — Memory, residency, and actual capacity

Implement the minimal sampler before WP5/WP6, not before the first quality screen.
`scripts/gpu_memory.py` polls the selected device with `nvidia-smi` (e.g. 50 ms),
recording timestamp, memory.used, clock and temperature. Match physical device to
backend visibility/remapping; record actual interval/dropouts/support. Refuse another
compute workload on that device without terminating it. Include graphics/idle usage;
report absolute and baseline-adjusted sampled peaks.

Cover complete cells, initialization and workers. Raw series stay ignored. Mark
observable load/warmup/measured windows using validated backend markers and a common
clock/documented conversion; otherwise keep phase peaks unmeasured. vLLM metric
durations do not locate phase timestamps in the sampler; use overall request windows.
Each llama-bench invocation includes its own load, warmup, and depth/KV preparation;
the current runner has no internal markers. Label its sampled peaks as invocation-level
peaks. Isolated llama.cpp load/prefill/decode peaks remain unmeasured unless validated
backend markers are added; never infer them from whole invocations. Report available
phase, invocation-level, and overall peaks, and missing fields, separately.

Sample the owned process tree's RSS, tracking PID identity/creation time, including
vLLM workers. Label its maximum simultaneous sum `process_tree_rss_sampled_peak_mib`:
shared pages may double-count and short-lived children may be missed. Record coverage
and MemAvailable before/after. Cumulative `RUSAGE_CHILDREN.ru_maxrss` is not per-cell;
direct-child `wait4` alone is not simultaneous worker-tree usage. RSS growth alone
does not prove paging/offload.

Verify configured residency: model parameters/buffers and persistent cache on selected
CUDA device; llama.cpp actual full GPU placement; vLLM no CPU KV/weight offload.
These establish configured placement, not absence of OS fallback. Record platform
policy and observability limits.

**12 GB feasibility measurement:** FlashQuest INT4 and llama.cpp quantized KV at 32k
with telemetry and actual weight/cache accounting. Lower sampled peaks suggest 4 GB
feasibility; they establish no actual 4 GB fit/parity conclusion. A fixed vLLM budget
on the bigger card also does not emulate target hardware or OS behavior.

**Actual 4 GB tests, when hardware is available:** record device/driver/OS, usable idle
memory, supported backends and no-offload/fallback policy. For Windows/WSL2, arrange
a documented no-fallback policy with the user; do not change system policy or run
oversized-allocation probes as routine validation. Unknown residency means withholding
a GPU-resident claim.

Try {4k, 8k, 16k, 24k, 32k} with bounded timeouts. Report largest tested success, not
an exact capacity limit. A success completes prefill and 128-token generation plus
a short matching quality check. vLLM uses a declared fixed budget/utilization with
realized cache capacity recorded: admission-budget failure differs from runtime OOM.
Allow equivalent reasonable target tuning, recording settings/retries; no capacity
conclusion from an avoidably oversized reservation or unsupported dtype.

Tests: device selection, synthetic phase windows/dropouts, idle rejection, per-cell
RSS after a larger previous run, worker accounting and cleanup on every exit.

### WP8 — Narrow contribution and research decision

Pursue ablations when early results justify them. INT4/INT8 reuse affine metadata;
TurboQuant stores separate raw scoring metadata, so exclude it from this claim.

- Scoring: capture representative post-RoPE BF16 K/Q by layer/head/step, releasing
  captures between layers. Compare metadata scoring to separately stored Quest
  summaries using the same optimized two-matmul identity, and true page maxima/oracle
  attention mass. Report errors, top-k overlap/tie margins, retained mass, shared
  versus incremental bytes, and warmed CUDA-event timing distributions. Two BF16
  min/max arrays at page size 64 cost 4/64 bytes per K element: 12.5% of packed INT4
  **K**, not of total KV/model memory. Include scale-clamping/rounding and summary
  derivation/storage costs.
- Fused attention: identical dequantized quantized-cache values, selected pages,
  GQA/scale/mask/tail behavior. Original full-precision K/V is a separate quantization
  comparison. SDPA supplies optimized output/latency, not LSE; use explicit float32
  logsumexp on a small/chunked reference for LSE. Disclose those different boundaries.
  The current `sparse_int4_fwd._flash_attn_sparse_int4_fwd_reference` requantizes INT4
  dequantized values to INT8, so it cannot be the exact dequantized-value output/LSE oracle.
  Measure temporaries above synchronized baselines with independent resets; large
  reference shapes run only if they fit.
- Prior-work map: primary papers/code, precise overlaps and differences for Quest,
  KIVI, TurboQuant and closer hybrids. Technical browsing is part of research; no
  private identifiers in requests. A literature search narrows confidence, not an
  exhaustive novelty certificate.
- Decision: weigh competitive quality/performance/capacity separately from novelty.
  Continue the runtime, narrow to a supported kernel/metadata contribution, or retain
  an engineering reference and redirect research. A pilot gate is not publishability.

Tests: exact/tied/rounded scoring, GQA/tails, small output/LSE agreement, independent
allocation reset and analytic versus actual bytes.

## 4. Sequence, checks, and open decisions

```text
WP0 + WP1 identity/export + WP2 + WP3 AWQ → WP4 4k screen
                                            ↓ if useful
WP3 competitors + WP7 sampler → longer quality screens + WP5 → WP6
fresh confirmatory quality/performance + available capacity evidence → WP8 decision
```

The pilot does not require commits, full competitor setup, expanded statistics, or
4 GB hardware. A research decision can proceed while a 4 GB capacity claim remains
pending target hardware. Keep roadmap/README commands aligned with implemented
behavior and record completion only when its evidence is saved.

| Check | When |
| --- | --- |
| Existing/new affected CPU tests with CUDA hidden | After relevant changes, before smoke. |
| Lint on changed files | Before implementation handoff. |
| Relevant tiny-model/cache GPU checks | After runtime/environment changes, before full-model experiments. |
| Short smoke under each pinned interpreter | After adapter/environment/model revisions. |
| Identity, completeness, inputs, timing definitions and export audit | Before accepting/publishing evidence. |
| Free VRAM and project-local logs | Before batches; long jobs use tmux/nohup and `systemd-inhibit --what=sleep:idle:handle-lid-switch`. |

Open scientific/resource decisions: absolute quality floor, margin, sample budget,
multiplicity rule, compatible vLLM pin if the candidate fails, and actual 4 GB hardware
with residency policy. They do not block the 4k pilot. External build locations and
ignored artifact retention are routine workspace choices. System policy changes,
purchases, pushes and PRs require their applicable permission; no prior approval is
asserted here.

## 5. Output layout

```text
benchmarks/validation/
  protocols/                         versioned gates/measurement contracts
  environment/                       sanitized backend identities
  quality/<run-identity>/             pilot/confirmatory records
  perf/<protocol>/<context>/<seed>/<run-identity>/   immutable arm records
  competitors/<run-identity>/        timings and matching quality
  memory/<run-identity>/             estimates or target-device observations
  contrib/<run-identity>/            contribution ablations
artifacts/
  smoke/ logs/ prompts/ memory/       ignored raw evidence/diagnostics
```

Save units, observed/estimated/unmeasured status, hashes, source links and limitations.
Check roadmap boxes only after complete evidence is saved and linked. Preserve
historical files and all user work.

## 6. Current execution status and decision framework

Status at this final documentation checkpoint, 2026-10-06. Sections 1–5 preserve
the original plan; this section records completed execution and remaining claim
limits. Final measurements retain clean source `5e22bf5`, fingerprint
`5807705f47e4eccd8e1eb6bc380202de5f05a9b0259037e5bd22798af9124c90`.
The frozen confirmation retains its separate `05f6400` source. Historical
groups, dirty-source fingerprints and failed attempts remain unchanged/unpooled.

| Work package | Completed evidence | Remaining or qualified endpoint |
| --- | --- | --- |
| WP0–WP2 | Pinned AWQ/model records, immutable matched quality/resume, cache restoration, pilots and affected-path regressions. | Preserve recorded identities; final post-measurement integration is reported separately below. |
| WP3 | Pinned AWQ/GGUF, llama.cpp b11382 and isolated vLLM 0.30.0; all four settings pass independently reviewed performance and quality setup smokes and full realized-runtime checks. | Precompiled/AOT-only vLLM execution is verified; host compilation is unverified. |
| WP4 | All 2,700 fixed-confirmation outcomes, nine endpoints and independent simultaneous-bound reconstruction. | Five endpoints pass and four fail; no non-inferiority or general-language-quality claim. |
| WP5 | Final four-seed internal blocks: 48 timings, paired medians 0.975674 at 8k and 1.824622 at 32k. Component and representative metadata/packed-kernel ablations complete. | Internal 32k benefit and avoided materialization do not establish native superiority. |
| WP6 | Full native performance: 32 cells, 96 timings and 30,280 accepted memory points. Full matching quality: eight cells, 2,400 outcomes, terminal exit 0. | Final independent quality/runtime/scorer and comparison acceptance are recorded below; timing boundaries remain separate. |
| WP7 | Internal aligned phases and continuous native load/performance-warmup/request device/process sampling; configured placement and source-specific memory preserved. | Aligned native prefill/decode memory, absence of OS fallback and actual 4 GB capacity remain unverified. |
| WP8 | Closest-prior-work map, representative contribution ablations and independent review of all 18 new diagnostics. | Retain engineering reference and redirect current research; broad novelty and a novel metadata/kernel pivot remain unsupported. |

Evidence: [current internal and native results](../README.md#benchmarks-and-validation),
[descriptive comparison](../benchmarks/validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md),
[contribution reports](../benchmarks/validation/contrib/summary.md),
[competitor contracts](competitor-validation.md),
[prior-work map](contribution-prior-work.md) and
[independent audit](independent-roadmap-audit.md).
Final strict and independent audits accept all 2,400 ordered outcomes
(2,322 hits), 250,671 raw memory points and exact manifest/mapping/EOS/scorer/runtime
contracts. The final descriptive report passes independent raw/public reconstruction and
Markdown review; its accepted builder retains measured source `5e22bf5`.

### Frozen fresh quality confirmation

The [saved protocol](../benchmarks/validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json)
selects 100 examples per task/context: seeds 1–5 × 20, excluding tuning seed 0;
retention 0.20 at nominal 4k/8k and 0.25 at 32k. Dense, all-pages and sparse use
identical examples, the pinned model, greedy EOS-or-128 generation and the frozen
substring scorer. Actual input lengths and input-plus-output capacity remain
checked independently of nominal labels.

All nine simultaneous sparse-minus-dense lower bounds must exceed −0.10, with
observed dense/sparse rates at least 0.80 in each endpoint. Exact gain/loss bounds
use `alpha = 0.05/(2×9)` per bound; all-pages comparisons are diagnostics outside
that family. Observed floors are screens, not population guarantees. One clean
source snapshot, fixed sample counts and complete paired records are required.
The [complete current-family summary](../benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json)
passes independent evidence/bound reconstruction and fails four endpoint rules:
multivalue at 4k/8k/32k and multikey at 32k. All observed floors/pilot screens pass.
The original earlier-environment group remains separate; neither native hits nor
timings change this result. Failure to establish non-inferiority does not prove
inferiority. Retuning after viewing it would require fresh confirmation.

### Contribution and comparison limits

Representative captures cover one seed-0 single-needle prompt, layers 0/13/27,
steps 0/1/63 and all query/KV heads. Avoiding an extra BF16 min/max pair saves
12.5% of packed INT4 K payload, not total KV/model/device memory. Scoring latency
is similar and some metadata snapshots are slower. The new diagnostic explains
changed IDs/margins in 18 independently reviewed captures; inputs match older
captures, but none of the Q/K fingerprints do. It cannot recover the old flips
or establish quality/kernel causality. Rounded affine endpoints are not exact
original-key extrema or strict original-key score bounds.

The fused output agrees within the recorded BF16-output/LSE errors with exact
FP32 reconstruction of the same packed values. Smaller temporary allocation
than full BF16 dequantization/SDPA establishes avoided reference materialization,
not a native competitive win. Closest prior work already combines sparse and
low-bit KV and uses compressed keys as indexes; a precise implementation
difference and byte saving do not certify research novelty.

Native and FlashQuest phase boundaries remain separate, with within-backend
ratios only. vLLM FP8/auto also changes attention backend and uses unit/default
uncalibrated FP8 scales. Native quality has no separate excluded warmup; llama.cpp
runtime detokenization gates, absent individual HTTP transcripts and independently
audited cell-specific slot views have distinct scopes. vLLM uses exact pinned
ID/count contracts. Backend token estimates are not physical cache bytes.
Different allocation policies, CPU embedding/host buffers and sampled peaks
prevent a no-OS-fallback or actual-target capacity claim. Native aligned phase
peaks and absence of OS fallback remain unverified claim limits; continuous
sampling and configured-placement reporting satisfy the bounded native memory
work package. Further phase instrumentation would be optional new work. Successful 32k here
is a tested lower bound on the 12 GB GPU, not a maximum.

### Completed checklist and deferred claims

- [x] Accept complete fixed confirmation and preserve its failed criterion.
- [x] Complete independently reviewed setup smokes and final repeated native performance.
- [x] Independently accept all eight native quality cells, 2,400 outcomes and final raw/runtime/scorer review.
- [x] Accept the emitted descriptive comparison.
- [x] Explain selection disagreements in new tensors without claiming old-tensor replay.
- [x] Record the reviewed evidence-linked decision after comparison acceptance.
- [ ] Perform actual 4 GB prefill, 128-output generation and matching quality with a declared offload/fallback policy; deferred for collaboration.

The locally executable bounded roadmap work is complete. The decision is to
retain FlashQuest as an engineering reference and redirect
current research. An internal 32k ratio alone does not meet the runtime research
gate; the narrower metadata/kernel specialization is not established as a novel
pivot. Future research needs its own bounded useful delta, applicable quality
and closest-prior-work comparison. Missing target hardware does not prevent this
decision, provided the capacity claim stays withheld.
Both reviewed post-measurement fixes are applied and validated: 190 focused
tests and 606 full non-slow tests pass; 12 existing slow cases are deselected
and remain unverified. Changed-file Ruff and `git diff --check` pass. These checks
validate later source without relabeling measured `5e22bf5` evidence.

Reproduce the measured report from clean source `5e22bf5`, pinned environments
and snapshots, preserved raw evidence and exact reviewed checker/builder hashes
bound by the report. Build against that source before either post-measurement
production fix; later source changes do not authorize a weaker gate or relabeled
measurements. Public JSON alone cannot recreate private raw evidence. Local
commits are authorized; publishing, pushes, PRs and merges require approval.
