# Phase 11 — TurboQuant Approach C: per-layer calibrated codebook (quality recovery at K3-V3) Design

**Status:** drafted 2026-05-14; pending user review.
**Plan:** to be written by `superpowers:writing-plans` after spec approval.
**Refs:**
- TurboQuant paper: arXiv 2504.19874 (ICLR 2026 spotlight); Google blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Prior phase: Phase 7 (`docs/superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md`) — TurboQuant K3-V3 with paper's fixed Lloyd-Max codebook; `docs/PHASES/phase-7-notes.md`
- Roadmap memory: `~/.claude/projects/-home-hoang-code-personal-active-flashquest/memory/project_post_v1_kernel_research.md` (Track B = TurboQuant calibrated codebook, ranked #1 post-v1.0 candidate)
- Profile-first discipline: `~/.claude/projects/-home-hoang-code-personal-active-flashquest/memory/feedback_profile_before_speedup_specs.md` (4 prior kills); v1 ceiling memory: `project_v1_ceiling.md`

## Goal

Replace TurboQuant K3-V3's paper-default 8-codepoint Lloyd-Max codebook with **28 per-layer codebooks (K + V each)** calibrated against Llama-3.2-3B-AWQ's actual KV distribution on PaulGrahamEssays. Quality target: close the multivalue gap (Phase 7's 17/20 = 85 % → ≥19/20 = 95 %) so K3-V3 can replace `--kv-bits 4` as the v1.2 default while keeping the 25 % cache shrink (980 → 736 MiB at 32 k full cache).

The hypothesis: TurboQuant's data-oblivious codebook is fit to a theoretical Gaussian, but post-WHT Llama-3.2-3B K/V are not perfectly Gaussian — they exhibit per-layer skew that calibration on real activations can recover.

## Non-goals

- **Per-head or per-channel codebook granularity.** Per-layer is the chosen granularity; escalation is a separate phase if the per-layer codebook still misses the gate.
- **V bit-width revisit (K3-V2 attempt).** Capability extension is a different target than the quality recovery picked at brainstorm Q2.
- **Calibration of the per-token RMS scale itself.** Phase 7's scale formula stays exactly as-is.
- **Online / runtime / model-load calibration.** Codebook is shipped as a precomputed artifact, not learned at load time.
- **Calibration for other models.** v1.2 ships Llama-3.2-3B-AWQ calibration only; the loader falls back to Phase 7's paper codebook for everything else (Phase 7 behavior preserved).
- **Throughput recovery vs INT4.** K3-V3's 2.62 tok/s @ 32 k speed gap vs INT4's 8.41 is not addressed here — only quality. K3-V3 becomes a quality-parity storage opt-in; users who care about throughput stay on INT4.

## Architecture

### Components touched

| Component | Change |
|---|---|
| `src/flashquest/turbo/codebook.py` (new) | `load_codebook(model_id)` returns `(num_layers, 2, 8)` fp32 tensor or raises `KeyError`. |
| `src/flashquest/turbo/codebook_llama_3_2_3b.pt` (new artifact) | Per-layer K+V codebooks for `casperhansen/llama-3.2-3b-instruct-awq`. ~2 KB. |
| `src/flashquest/kernel/sparse_turbo_fwd.py` | Kernel accepts 16 new `tl.constexpr` codepoint arguments (`K_CP0..K_CP7`, `V_CP0..V_CP7`); Python wrapper unpacks per-layer codebook into the constexpr tuple. |
| `src/flashquest/cache/persistent_turbo.py` | `PersistentTurboKVCache.__init__` accepts `codebook: Tensor | None = None`. None = autoload via `load_codebook(model._name_or_path)`; falls back to `codebook="paper"` with warning if unavailable. |
| `src/flashquest/eager/llama_persistent_patch.py` | Dispatcher closure passes `cache.codebook_k[layer_idx]` and `cache.codebook_v[layer_idx]` into the wrapper per layer. Layer index already in scope; no signature changes beyond `make_quest_persistent_forward`. |
| `src/flashquest/runtime/chat.py` | New flag `--codebook {calibrated,paper}` (default `calibrated`); CLI plumbing only. |

### Why per-layer is the right granularity here

Phase 7's K3-V3 kernel inlines the codebook as `tl.constexpr`. The natural axis to vary on at constexpr granularity is per-launch — and per-launch is per-(layer, head). Per-head was rejected at brainstorm Q3 (overfitting risk at calibration corpus size <100 k tokens; 224 codebooks for Llama-3.2-3B is too many to validate). Per-layer (28 codebooks) is the next granularity down. It captures the dominant K/V variation axis (depth-dependent distribution shift, which is well-attested in transformer activations) while keeping each codebook's calibration sample large (≈ 50 k–100 k tokens × all heads merged per layer).

### Codebook bit-split encoding constraints

Phase 7 uses bit-split storage: K stores codepoint index `idx ∈ {0..7}` as (1-bit MSB, 2-bit LSB), packed across `head_dim`. The kernel reads MSB and LSB planes separately, recombines as `(msb << 2) | lsb`, then looks up `K_codebook[idx]`. This encoding is monotonicity-sensitive: the kernel assumes `K_codebook` is sorted ascending so that ordinal `idx` correlates with codepoint magnitude. The calibration script MUST enforce sorted output (and the kernel parity test pins this assumption with a random-codebook stress test).

## Data flow

### Offline calibration (`scripts/phase11_calibrate_codebook.py`)

1. Load `casperhansen/llama-3.2-3b-instruct-awq` (autoawq + transformers).
2. Install forward hooks on every layer's `LlamaAttention.{k_proj, v_proj}`. Hooks record the post-projection tensor and return immediately; the model continues to forward normally so memory stays bounded by the longest single prefill.
3. Iterate over ≥10 distinct PG essays from `data/PaulGrahamEssays.json`, prefilling each at 8 k context. Target ≥50 k cached K + V tokens total per layer (after dedup of `<bos>` + short prefixes).
4. For each layer, take K and V separately. Apply Phase 7's transform: per-token RMS scale (`x / s` where `s = sqrt(mean(x²))`) followed by Walsh-Hadamard rotation along `head_dim`. Stack and flatten to 1-D float32 sample sets.
5. Fit Lloyd-Max codebook per `(layer, {K|V})`: k-means (`sklearn.cluster.KMeans` with `n_clusters=8`, `init="k-means++"`, `n_init=1`). Warm-start from paper's codepoints — converges in <50 iters vs random init's ~200.
6. Enforce sorted-ascending codepoints. Stack to `(num_layers, 2, 8)` fp32 tensor.
7. Write the artifact + a sidecar JSON with metadata: source commit, model id, calibration corpus, n_tokens_per_layer, per-layer fit residual RMS.

### Online dispatch (decode path, unchanged kernel shape)

1. CLI / library constructs `PersistentTurboKVCache(model, codebook=None)`. The cache constructor calls `load_codebook(model.config._name_or_path)`; on success stores `self.codebook_k: (num_layers, 8)`, `self.codebook_v: (num_layers, 8)`.
2. `patch_llama_for_quest_persistent` builds the per-layer forward closure. The closure captures `layer_idx` and the cache reference.
3. Per decode step, per layer, the closure calls `sparse_turbo_fwd(Q_rot, K_msb, K_lsb, V_packed, ..., codebook_k=cache.codebook_k[layer_idx], codebook_v=cache.codebook_v[layer_idx])`.
4. Wrapper unpacks the 8-vector codebooks into 16 fp32 constants and launches the kernel with those as `tl.constexpr`. Triton's autotune cache keys on the constexpr tuple, so each layer-specific codebook compiles a separate kernel variant on first use. 28 variants × 2-5 s compile each = ~1-2 min first-run latency hit; cached thereafter.

## Entry-gate probe (profile-first)

Before any kernel work — single afternoon's worth of measurement:

1. **Capture**: capture Llama-3.2-3B-AWQ K + V tensors per layer at 8 k prefill on three PG essays (~24 k tokens calibration set; subset of the full calibration run to keep the probe under 30 min wall).
2. **Transform**: apply per-token RMS scale + WHT.
3. **Fit (two granularities)**: per-layer Lloyd-Max codebook (k-means with k=8, warm-start from paper) AND per-head Lloyd-Max codebook (k=8, warm-start from paper) on the same data. The per-head fit is for the granularity side-check only — it is not loaded by the kernel at this phase.
4. **Compute three numbers**:
   - **Codepoint divergence (per-layer vs paper)** — `max_{layer, kv, codepoint} |calibrated_layer - paper| / |paper|`. If max relative shift <5 %, the WHT-Gaussianization assumption holds tightly and calibration won't materially shift quality. **Kill the phase.**
   - **Quality-simulator delta** — run one RULER NIAH multivalue sample (n=1, ctx=4 k) twice via `_flash_attn_sparse_turbo_fwd_reference` (Phase 7's non-fused Python reference): once with the calibrated per-layer codebook, once with paper's. If the generated token sequences match exactly, the codebook is not the bottleneck for this sample. Repeat on 3 samples; if all 3 match identically across runs, **kill the phase**.
   - **Granularity pre-signal** (per-head vs per-layer divergence ratio) — `R = max_{layer, head, kv, codepoint} |calibrated_head[layer,head] - calibrated_layer[layer]| / max(|calibrated_layer - paper|, ε)`. Interpretation: per-head fits show structure beyond what per-layer captures iff R > ~1. If R > 2 across multiple layers, granularity is likely the dominant axis of K/V variation; per-layer calibration may still help but per-head (Phase 11b) is the better target. Reported as advisory, **not a kill condition** — Phase 11 proceeds at per-layer regardless; the number is recorded so Phase 11b can be scoped immediately if the post-implementation RULER gate misses.
5. The probe lives in `scripts/phase11_calibrate_probe.py` and writes a one-page `benchmarks/phase11/probe.md` with the three numbers + a recommendation line + the per-layer codepoint-divergence table + the per-head/per-layer ratio R summary.

This gate is non-negotiable per the profile-first discipline (4 prior kills documented in `feedback_profile_before_speedup_specs.md`). If both checks pass, proceed to implementation. If either fails, document the kill in `docs/PHASES/phase-11-killed-by-probe.md` and reconsider scope.

## Success criteria

| Axis | Bar | Kill if |
|---|---|---|
| RULER NIAH 4 k (n=20, all-retrieval head_pattern, retention=0.20) | single ≥95 %, multikey ≥95 %, **multivalue ≥95 %** | Multivalue stays <95 % after calibration — the 17/20 ceiling holds; codebook calibration was not the lever. Document and ship K3-V3 as-was. |
| Decode tok/s @ 32 k | within ±5 % of current K3-V3 baseline (2.62 tok/s, so ≥2.49) | >5 % regression — per-layer constexpr dispatch is more expensive than expected. Investigate; do not ship as default. |
| Cache footprint @ 32 k | unchanged from K3-V3 (736 MiB) | (n/a — codebook ships as a 2 KB artifact; cache layout untouched) |
| Calibration cost | one-time offline ≤30 min wall on the dev box | (n/a — offline, doesn't gate runtime experience) |
| First-run kernel compile latency | ≤120 s total for 28 layer variants | >120 s — investigate whether constexpr explosion is excessive; consider passing codebook as a runtime tensor with `tl.load` instead |

If quality and speed gates both pass, **K3-V3 becomes the v1.2 default**; `--kv-bits 4` stays available as the throughput-priority opt-in. CHANGELOG and README updated accordingly.

## Testing strategy

| Test | What it checks | Where | Marker |
|---|---|---|---|
| Fused-vs-reference parity (3 random codebooks) | Triton kernel correctly threads per-layer codebook through the `constexpr` chain; sorted-codebook assumption holds | `tests/test_sparse_turbo_calibrated.py` | (no marker; runs in `pytest -m "not slow"`) |
| Codebook load + fallback | `load_codebook` returns shape `(28,2,8)` for Llama-3.2-3B; raises clean `KeyError` for unknown model id; cache falls back to paper with warning on `KeyError` | `tests/test_turbo_codebook_loader.py` | (no marker) |
| Monotonicity invariant | Calibration script's output enforces sorted codepoints per `(layer, kv)`; assertion stays loud | `tests/test_calibrate_codebook_monotonic.py` | (no marker) |
| Persistent cache wiring smoke (1 B fallback) | `PersistentTurboKVCache(codebook=None)` runs end-to-end on Llama-3.2-1B (no calibration artifact ships → falls back to paper with warning) | `tests/test_persistent_turbo_calibrated_smoke.py` | `slow` |
| Entry-gate probe (manual gate, not unit) | Documents kill-or-proceed verdict; gates spec → plan transition | `scripts/phase11_calibrate_probe.py` + `benchmarks/phase11/probe.md` | manual |
| RULER NIAH 4 k @ K3-V3 calibrated | Quality gate; manual; n=20 each task on Llama-3.2-3B-AWQ at retention=0.20 | `scripts/phase11_run_ruler_4k_calibrated.py` | manual |
| Decode tok/s @ 32 k @ K3-V3 calibrated | Speed gate; manual; ±5 % of Phase 7 baseline | `scripts/phase11_bench_decode_32k.py` | manual |
| Compile-time budget check | First-run wall of the smoke test must be ≤120 s; measured from a clean Triton cache (`rm -rf ~/.triton/cache` once before the `pytest -k turbo_calibrated -m slow` run) | reading of `pytest --durations` output for the smoke test | manual |

CI runs `pytest -m "not slow"` only, per the existing project pattern. Slow tests and manual gates live behind the markers and are executed during the manual quality + speed gate run.

## Open questions

(None blocking — all addressed in brainstorm Q1-Q4 or pinned above.)

## Out of scope for v1.2 — natural follow-ups

- **Per-head or per-channel codebook granularity.** Triggered only if per-layer misses the gate; would be Phase 11b.
- **Calibration for other models** (Llama-3.1-8B, Mistral, Gemma). Same script, different model id; ships per-model artifacts incrementally.
- **K3-V2 with calibration** (capability-extension target from brainstorm Q2). Re-evaluate only if the user revisits the constraint set toward longer context.
- **Per-token scale calibration.** Phase 7's RMS scale stays; revisiting would interact with the codebook calibration so should not happen in the same phase.
