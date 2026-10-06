# Roadmap execution checkpoint

Updated: 2026-10-06. The user explicitly resumed normal roadmap work. The GPU
was free at the resume check; source and measured core environment match the
saved checkpoint. The earlier usage-based cutoff remains removed.

## Previous pause and new resume

The reviewed implementation, dependency pins and selection evidence are locally
committed as `3ee8192`. Matching FlashInfer precompiled FP16/E4M3 prefill modules
pass CPU load and wheel-record hashes; the isolated 199-package environment passes
dependency checks. This does not establish successful native vLLM requests.

[Attempt 7](../benchmarks/validation/competitors/7254f9440ffd1190d7c1d4ce173953d5dc36c2b20976ed87b1dd3ac87e56641f/schedule.json)
was refused by the foreign-GPU preflight before any backend launched. Its saved
last-progress status remains `running` with zero cells; the terminal rejection is
preserved in `artifacts/smoke/competitors-attempt7.log`. Do not interpret that saved
status as an active job or overwrite its evidence.

[Attempt 8](../benchmarks/validation/competitors/07bf3ec87d814404ca73b61a28dba977fd1d758061d849a46c2530f28069ce43/schedule.json)
ended incomplete. llama.cpp Q4 completed with validated telemetry; vLLM FP8 stopped
on a new foreign GPU process before any completed request, with invalid/incomplete
telemetry. Independent review reconstructed both raw series and confirmed the
contamination stop. This supplies no vLLM/AOT failure or performance verdict.
Owned batch sessions finished with exit 1; foreign processes were left untouched.

The ignored strict local evidence checker and its CPU regression fixtures are
preserved under `artifacts/development/evidence-audit/`. Its author reconstructed
all 16 historical ablation cells, 20,131 raw memory points and 48 timed samples,
and passed 45 focused CPU cases plus Ruff after the quality-reference corrections.
Independent review accepted its bounded checks, including original/current
ablations and successful native smokes. It reconstructs
saved telemetry; actual kernel/cache placement and missing terminal evidence
remain separate gates.

The fresh ninth smoke loaded vLLM FP8 and recorded its intended GPU cache, but
its healthy HTTP 200 `/health` response has no body. The JSON-only adapter kept
polling. The root stopped its owned batch, preserving its one-cell last-progress
schedule and interruption log. It supplies no completed vLLM request or timing.
The independent review confirms the pinned vLLM contract and the bounded fix:
health requires HTTP 200, while completion/detokenize still require valid JSON.
Focused tests pass. The required full non-slow project suite finished exit 0:
562 passed, 12 existing slow cases deselected. Its first collection failed on
missing declared Hypothesis; only
Hypothesis 6.168.5 and sortedcontainers 2.4.0 were installed, leaving the core
AWQ and isolated vLLM runtime packages unchanged. The next attempt failed on
missing public DuoAttention pattern data and an uncached 1B tokenizer. Restoring
pinned upstream test fixtures resolved those failures without changing assertions.
Both failed logs, fixture hashes and the final successful log remain under
`artifacts/setup/`. The slow end-to-end cases were not verified by this check.

The health correction is committed as `76f00fe`. The independently audited
[performance smoke 10](../benchmarks/validation/competitors/81fc388e18a8943f09c53067b95ff3e0164b72d846219d76d4c780606cabb1c3/schedule.json)
and [quality smoke 11](../benchmarks/validation/competitors/e7157add0ff588ca0c2e2b81726d30a1d06dbbc068085b5e627a4135be5615d2/schedule.json)
complete every setting. Performance has exact 1024 inputs and eight outputs;
quality has 24/24 hits on six matched setup examples. Intended native cache
types, realized kernels, full GGUF vocabulary/prompt mapping, EOS, same-request
timings and 5951 combined memory points pass reconstruction. These short setup
checks do not establish general quality or a competitive advantage.

Quality attempt 10 failed before loading backends because its legacy smoke
export used filename-to-hash maps rather than typed file entries. Preserve its
[failed schedule](../benchmarks/validation/competitors/7b78dae942e30efcb83ae24577ea4ec23f3459c9b266430e56b7dd253e137587/schedule.json).
A fresh current-format AWQ smoke uses exactly the same six ordered examples;
dense/all-pages score 6/6 and sparse 5/6, preserving the multivalue screen failure.
Its ignored export is `artifacts/smoke/awq-ruler-1024-current.json`. The local
checker corrections admit only intended smoke/canonical namespaces, reject path
escapes and bind the frozen quality reference model/run plus pinned tokenizer.
Earlier checker path failures remain separately preserved.


## Completed quality and released source freeze

The [current-environment confirmation](../benchmarks/validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json)
contains all 2700 outcomes and all nine endpoints. Independent reconstruction of
raw/public records, manifests, source/model/environment, scorer and all 27
primary/diagnostic bounds passed. Five endpoints pass; multivalue at every
context and multikey at 32k fail. All observed accuracy floors and pilot-ratio
screens pass. Failure to establish non-inferiority does not prove inferiority.

These measurements retain source commit
`05f64004dbfb3c33d42d60e68d3f6e5091de7d5d`, content fingerprint
`8dff99e32861785f33e334b3ccc33d04c410e7eb5817ce450f6e9bd74d0bc30f`
and environment `ec59cde8cd845e4d60cff4f3a545c614a19ebae3fc28e26487ce108b554527f5`
(kernel `7.0.0-38-generic`). Their source freeze is now released. Do not relabel
historical identities, retune this completed confirmation, or pool repeated
same-seed groups. The original kernel-34 group remains
[separately incomplete](../benchmarks/validation/confirmation/699b3aecbca4be9cf65f52e08c2f3adc49fe6ea4b4a9eeb043109f79b3e7e2c7/summary.json).
Old/new answers differ despite identical prompts/settings; neither kernel
causality nor bitwise greedy reproducibility is established.

Current quality references:

- [4k](../benchmarks/validation/quality/7d9921cb0d5b6167c5303932475ccb71402ece5ca9066d7e067f23dab4d9cdef/quality.json)
- [8k](../benchmarks/validation/quality/6c938a6b4515ffaf0de9ed8bf833e521584887705e4665e43b51e9e94c3aafb1/quality.json)
- [32k](../benchmarks/validation/quality/07ce7aa457aa9070cae386a86c9bcd00fb501e2d29c99aca56410183f74c98c4/quality.json)

## Reviewed implementation ready for new measurements

The accepted competitor patch
`0a64fcc74b0aae8c0f09b8c1a264c1eac244a86b9a225a8a1d950efe034e1836`
has been applied after the freeze. It binds canonical pinned-HF decoding,
private generated IDs/native text, full GGUF vocabulary/special/EOS mapping,
first-EOS-or-exact-cap validation, `MAX_JOBS=2` for vLLM compilation and nonempty
realized attention-backend selection. Its six-file predecessor and validation
remain ignored development provenance.

The independently reviewed selection diagnostic and tests are installed at
`scripts/diagnose_selection_flips.py` and `tests/test_selection_flips.py`.
Together the real-path changes pass 170 focused CPU checks, Ruff and diff checks.
This source was committed through the shared hooks as `a0100a5`.
The measured runtime, quantizer, kernels, retrieval generator and scorer are
unchanged; fresh matching-environment quality can satisfy the internal timing
runner's semantic prerequisite.

## New completed diagnostic and native setup repair

The [selection diagnostic](../benchmarks/validation/selection-flips/summary.md)
and its independent author review cover all 18 new snapshots and 432 query-head
observations. Rounded affine maxima explain concrete ranked-score perturbations.
All input fingerprints match the older captures, but no Q/K fingerprints match;
these explanations do not recover the exact old flips or establish quality/kernel
causality. Full Q/K/metadata values were not saved, so the audit reconstructs the
stored score/error evidence rather than independently recomputing GPU scores.

The terminal sixth native smoke preserves successful llama.cpp Q4/FP16 requests
and both vLLM startup failures. NVVM 13.4 generated PTX unsupported by the 13.0
assembler; the coherent NVVM 13.0.88 pin then exposed a glibc 2.43 header conflict.
The matching official FlashInfer 0.6.18.post1+cu130 precompiled package is
installed; both FP16/E4M3 prefill modules pass CPU load and wheel-record hashes. The reviewed adapters bind AOT-only execution and enabled version
checking, with 160 focused competitor CPU checks passing. Missing precompiled
coverage remains failure; compilation and successful vLLM execution are not yet
verified. Keep all setup/failed-attempt evidence.

## Preserved internal timing and failed native collection

The independently reviewed current-source blocks are complete:

- [8k / 0.20](../benchmarks/validation/ablation/f22ba06c8c74dd6aa294fc69707404ead02002f9c3958ab4b0015af32acaf7cd/summary.json): median paired decode ratio 1.014242,
  range 0.955149–1.059477; practical screen fails.
- [32k / 0.25](../benchmarks/validation/ablation/0b19179a04ffe455f53d821d4cb827c38958c6d2375e578db1c8f51ce5a78087/summary.json): median 1.811117,
  range 1.620548–1.915303; practical screen passes.

All 16 cells, 48 timings, 12,919 memory points and 128 observed phase windows
reconstruct, with no dropouts or foreign compute. Sampled device peaks remain
3442/7006 MiB. Runtime/model and existing quality prerequisites match; failed
confirmation limits the timings to engineering characterization. The original
32k supervisor tool session disappeared on resume: its outer exit is unavailable,
while every saved child exit is zero and terminal schedule/log plus independent
data checks are verified. Do not claim that outer exit was read.

The [full native performance attempt 1](../benchmarks/validation/competitors/a7fd1cdea28626b00fd9002b5fe62aa8e2351da8d6295d3767bf811742298c4f/schedule.json)
finished with exit 1 and `complete-with-failures` from clean source `76f00fe`.
All 32 cells retain 96 timings and 32,021 memory points. Independent review
reconstructs all 160 telemetry windows without foreign compute or dropouts.
Cells 21 and 23, llama.cpp F16/Q4 at 32k seed 1, failed strict UTF-8 decoding of
generated token bytes in the diagnostic log after saving three requests each.
Their runtime records remain absent and the matrix remains ineligible. The
other 30 realized runtimes pass independent log/worker checks. Its preserved
log is `artifacts/smoke/competitors-full-performance-attempt1.log`; this is an
adapter error, without an inference or OOM verdict.

The independently reviewed repair reads native diagnostic logs with reversible
UTF-8 `surrogateescape`, preserving every raw byte. HTTP and worker JSON and
runtime precision/kernel/device checks stay strict. Candidate checks pass all
194 competitor CPU tests, including 27 new regressions. Root integration checks
finish exit 0 with 589 non-slow tests passed and 12 existing slow cases deselected;
Ruff passes. After a clean commit, repeat both complete internal timing blocks and both native
matrices under the new source identity. Preserve the older groups separately;
do not repair, relabel or pool their observations.

Independent review found that the internal runner's completed-block `--resume`
rewrites its status to `incomplete`. Do not resume either completed block; preserve
them and fix/test that behavior after the frozen measurement blocks. The ignored
comparison builder passes independent review and 61 CPU cases, including real
export compatibility. Final report emission remains unverified until all four
new schedules pass the strict raw-evidence and realized-runtime gates.

## Remaining execution sequence

1. Commit the reviewed log repair after required checks. Repeat both internal
   blocks, then collect the optimized 8k/32k performance matrix (32 cells, 96
   samples) and matching native quality matrix (8 cells, 2400 outcomes), with
   periodic independent audits and a fixed source/environment throughout.
2. Require every local raw memory series to exist, match its hash and reproduce
   windows, peaks and coverage before accepting the final comparison. Keep
   FlashQuest forward, native server and HTTP client timing boundaries separate.
   Record weight formats, allocated cache capacities, offload configuration and
   unmeasured OS fallback explicitly.
3. Save the comparison and evidence-linked research decision, update roadmap
   checkboxes only for completed evidence.
4. Correct completed-block resume status, finish the appropriate project checks,
   then commit the reviewed result through shared hooks.

Check actual free VRAM and compute ownership before each GPU batch. Run one GPU
job at a time, under `systemd-inhibit --what=sleep:idle:handle-lid-switch` plus
`nohup` or `tmux`, with project-local logs. Keep source and environments fixed
through each measured block; inspect owned workers before resuming an interrupted
job. Keep raw evidence, model snapshots and environments.

Actual 4 GB hardware testing is deferred for collaboration as authorized by the
user. The available 12 GB device cannot establish target fit or a maximum capacity
limit. Local commits are authorized; pushing, PRs and merges still require approval.
See the [roadmap](../roadmap.md), [research decision](research-decision.md) and
[independent audit](independent-roadmap-audit.md).
