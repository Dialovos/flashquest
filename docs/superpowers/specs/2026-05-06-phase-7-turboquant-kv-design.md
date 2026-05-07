# Phase 7 — TurboQuant KV (K=3-bit + V=2-bit) Design

**Status:** drafted 2026-05-06; pending user review.
**Plan:** to be written by `superpowers:writing-plans` after spec approval.
**Refs:**
- TurboQuant paper: arXiv 2504.19874 (ICLR 2026 spotlight); Google blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- Reference implementations: https://github.com/OnlyTerp/turboquant (Python ref), https://github.com/AmesianX/TurboQuant (llama.cpp port)
- Prior phase: Phase 6 task 6 (`docs/superpowers/plans/2026-05-03-phase-6-int4-fused-kernel.md`) — fused INT4 kernel + dual-stat dispatcher pattern that this builds on
- Memory: `~/.claude/projects/-home-hoang-code-personal-active-flashquest/memory/project_post_v1_kernel_research.md`

## Goal

Replace KIVI-INT4 KV with **TurboQuant K3-V2** as a new opt-in cache mode (`--kv-bits 3`), shrinking the KV footprint by ~34% per layer (980 MiB → 644 MiB at full 32k cache, 28-layer Llama-3.2-3B) while holding the RULER NIAH 4k quality gate. Ship a new `@triton.jit` fused kernel that reads bit-split-packed K (1+2 bit planes) and INT2-packed V directly, with the Walsh-Hadamard rotation handled via the wrapper-side identity `Q · K^T = (W·Q) · (W·K)^T` for orthogonal `W`.

## Non-goals

- Training, fine-tuning, or model-specific calibration. TurboQuant's data-oblivious codebook (Rayleigh-Lloyd-Max) is the algorithm under test; if it regresses, fall back to `--kv-bits 4` rather than calibrate.
- Replacing INT4. INT4 stays as the v1 default; TurboQuant is opt-in until Phase 7 ships its own ≥5× verdict.
- Prefill kernel. The dense-prefill path keeps the existing BF16 SDPA on dequant'd cache (only the dequant function changes to handle TurboQuant).
- 1-bit QJL residual on K (deferred — re-evaluate if RULER multivalue regresses below 90%).

## Architecture

TurboQuant's claim: applying a per-block Walsh-Hadamard transform along `head_dim` Gaussianizes the value distribution, allowing a single fixed scalar codebook (placed at Rayleigh quantiles of the unit-variance Gaussian) to quantize K to 3 bits and V to 2 bits with near-zero PPL gap. Because WHT is orthogonal, `Q · K^T ≡ (W·Q) · (W·K)^T`, so the cache can store rotated K (and V) and the kernel rotates Q at decode-time.

Key compatibility requirement: Quest's `page_scores_int4_fast` uses per-page channel-wise `K_scale` and `K_mn` in the **un-rotated** basis. Rotation mixes channels, breaking the per-channel envelope identity. We resolve this by storing two sets of statistics:

- `K_scale_turbo`, `V_scale_turbo` per-token — used by the fused kernel for dequant.
- `K_scale_raw`, `K_mn_raw` per-page channel-wise — used by `page_scores_int4_fast` only. Computed from raw (un-rotated) K.

The dual-stat overhead is ~2 MiB per layer (vs ~1 MiB for INT4's single stat set) — negligible against the K+V packing savings.

## Data flow

### Write (`PersistentTurboKVCache.update_quantized`)

Given raw `K_in: (B, H_kv, S_new, D)` BF16, raw `V_in: (B, H_kv, S_new, D)` BF16:

1. **K rotation:** `K_rot = wht_along_head_dim(K_in)`. WHT block size = `D` (single per-token rotation; 7 butterfly stages at D=128, ~1k FP ops per token-head).
2. **K per-token scale:** `s_K = max(|K_rot|, axis=-1, keepdim=True) / c_K_max` where `c_K_max ≈ 1.749` is the largest codepoint magnitude. Shape `(B, H_kv, S_new, 1)` BF16.
3. **K quantization:** `K_idx = nearest_codepoint_index(K_rot / s_K, K_codebook)`. Codebook is 8 fp32 constants (Rayleigh-Lloyd-Max for `N(0,1)`); indices are uint8 in 0..7.
4. **K bit-split packing:**
   - `K_msb_plane[..., k] = (K_idx[..., k] >> 2) & 0x1` packed 8 values per uint8 byte → shape `(B, H_kv, S_new, D/8)`.
   - `K_lsb_plane[..., k] = K_idx[..., k] & 0x3` packed 4 values per uint8 byte → shape `(B, H_kv, S_new, D/4)`.
5. **K raw stats** (criticality only): `K_scale_raw, K_mn_raw = _scale_mn_per_page_channel(K_in, page_size)` from raw K — same code path KIVI-INT8 already uses, no change.
6. **V analogous:** WHT along `head_dim`, per-token scale, 2-bit Lloyd-Max codebook (4 codepoints), packed `(B, H_kv, S_new, D/4)` uint8.
7. Partial-page staging unchanged: `K_partial`, `V_partial` keep BF16 raw values until a page completes; identical machinery to `PersistentInt4KVCache`.

### Decode (fused kernel via `flash_attn_sparse_turbo_fwd`)

Given `Q: (B, H_q, 1, D)` BF16:

1. **Wrapper rotates Q:** `Q_rot = wht_along_head_dim(Q)` — single output vector per head, trivial.
2. **Page selection (raw basis):** `scores = page_scores_int4_fast(Q_in_raw, K_scale_raw, K_mn_raw)`; `sel = select_pages_vectorized(scores, retention=...)`. Note this uses the un-rotated Q on un-rotated stats — entirely separate from the kernel's rotated dot product.
3. **Fused kernel** (one CTA per `(batch, query head)`, decode-only `S_q=1`):
   - For each selected page:
     - Load `K_msb` tile: shape `(PAGE_SIZE, D/8)` uint8. Expand 1-bit-per-value via 8 successive shifts + masks → `(PAGE_SIZE, D)` uint8 of 0/1 values. Use `tl.join` + `tl.reshape` chained 3× (or `tl.reshape((PAGE_SIZE, D/8, 8))`) to broadcast.
     - Load `K_lsb` tile: shape `(PAGE_SIZE, D/4)` uint8. Expand 2-bit-per-value via 4-way `tl.join` → `(PAGE_SIZE, D)` uint8 of 0..3 values.
     - Combine: `k_idx = (k_msb << 2) | k_lsb` → uint8 in 0..7.
     - Codebook lookup: `k_rot = K_CODEBOOK[k_idx] * s_K[token]`. Codebook passed as `tl.constexpr` tuple of 8 fp32 constants; lookup via `tl.where` chain or small `tl.gather` over a constant.
     - Dot `Q_rot · k_rot` → attention scores. Math is identical to `Q · K^T` because WHT is orthogonal.
     - Online-softmax accumulation (mirrors INT4 fused kernel exactly).
     - V tile: load `V_packed` `(PAGE_SIZE, D/4)` uint8 → 2-bit unpack → 4-way codebook → `* s_V[token]` → V_rot tile.
     - Accumulate `softmax · v_rot`.
   - Output `acc` is in rotated basis (V was rotated).
4. **Wrapper inverse-rotates output:** `O = wht_along_head_dim(acc)`. WHT is its own inverse for the normalized Hadamard convention, so it's the same butterfly call.

### Prefill (existing dense BF16 SDPA path, no kernel change)

The prefill dispatcher (`llama_persistent_patch.py`, `S_q > 1` branch) calls `_dequant_k(views[K_msb], views[K_lsb], ...)` and `_dequant_v(views[V_packed], ...)` to produce raw BF16 K, V:

1. Load all bit planes; reconstruct `K_idx` / `V_idx`.
2. Codebook gather; multiply by `s_K` / `s_V` → `K_rot`, `V_rot` (BF16, full shape).
3. **Inverse WHT** along `head_dim` → raw K, V.
4. Existing `scaled_dot_product_attention(Q, K, V, is_causal=True, enable_gqa=True)` call — unchanged.

Step 3's inverse WHT is one extra rotation per dequant pass, applied to a `(B, H_kv, S_kv, D)` BF16 tensor. At 32k it's ~700M FP ops — ~0.5ms on the 3050 Ti, negligible vs the SDPA itself.

## Storage layout

Per-layer at Llama-3.2-3B, 32k cache (B=1, H_kv=8, S=32768, D=128, page_size=64 → 512 pages):

| Tensor | Shape | dtype | bytes |
|---|---|---|---|
| `K_msb` | (1, 8, 32768, 16) | uint8 | 4 MiB |
| `K_lsb` | (1, 8, 32768, 32) | uint8 | 8 MiB |
| `K_scale_turbo` | (1, 8, 32768, 1) | bf16 | 0.5 MiB |
| `K_scale_raw` | (1, 8, 512, 128) | bf16 | 1 MiB |
| `K_mn_raw` | (1, 8, 512, 128) | bf16 | 1 MiB |
| `V_packed` | (1, 8, 32768, 32) | uint8 | 8 MiB |
| `V_scale_turbo` | (1, 8, 32768, 1) | bf16 | 0.5 MiB |
| `K_partial`, `V_partial` | (1, 8, 64, 128) × 2 | bf16 | 0.13 MiB |
| **Per layer** |  |  | **~23 MiB** |
| **Full cache (28 layers)** |  |  | **~644 MiB** |

vs current INT4 cache: 35 MiB/layer × 28 = ~980 MiB. **Shrink: 34%.**

## Codebook constants

Hardcoded in the kernel module as `tl.constexpr` tuples (precise values to be confirmed against the reference `OnlyTerp/turboquant` impl during plan execution):

- **K (3-bit, 8 codepoints, symmetric Rayleigh-Lloyd-Max for `N(0,1)`):**
  Approximate values: `(-1.749, -1.050, -0.501, -0.155, +0.155, +0.501, +1.050, +1.749)`.
- **V (2-bit, 4 codepoints):**
  Approximate values: `(-1.510, -0.453, +0.453, +1.510)`.

`c_K_max = 1.749`, `c_V_max = 1.510` are used by the per-token scaling step (`s_K = max(|K_rot|) / c_K_max`).

## Bit-plane unpack (kernel detail)

3-bit packing has three options; we picked **bit-split (option iii)**:

- (i) 8 values per 3 bytes (24-bit aligned groups). Awkward Triton tile loads — requires `PAGE_SIZE × D/8 × 3` byte loads per page with non-byte-aligned bit math.
- (ii) 8 values per 4 bytes with 8 padding bits. 25% wasted storage per K tile.
- (iii) **Bit-split into 1-bit MSB plane + 2-bit LSB plane.** Each plane is byte-aligned; packed bits per byte (8 for MSB, 4 for LSB). Reconstruction is `idx = (msb << 2) | lsb` — a single bitwise op. **Picked.**

For the 1-bit MSB expansion in Triton:
```python
# k_msb_byte: (PAGE_SIZE, D/8) uint8
bit_offsets = tl.arange(0, 8)  # constexpr
expanded = (k_msb_byte[:, :, None] >> bit_offsets[None, None, :]) & 0x1  # (PAGE_SIZE, D/8, 8)
k_msb = tl.reshape(expanded, (PAGE_SIZE, HEAD_DIM))
```

For the 2-bit LSB expansion:
```python
# k_lsb_byte: (PAGE_SIZE, D/4) uint8 — each byte holds 4 × 2-bit values
b0 = k_lsb_byte & 0x3
b1 = (k_lsb_byte >> 2) & 0x3
b2 = (k_lsb_byte >> 4) & 0x3
b3 = (k_lsb_byte >> 6) & 0x3
# tl.join chained: (PAGE_SIZE, D/4, 4) → reshape (PAGE_SIZE, D)
stacked = tl.join(tl.join(b0, b1), tl.join(b2, b3))  # (PAGE_SIZE, D/4, 4)
k_lsb = tl.reshape(stacked, (PAGE_SIZE, HEAD_DIM))
```

(V uses the LSB expansion pattern only — V has no MSB plane.)

## Module + file inventory

**New files:**
- `src/flashquest/kernel/wht.py` — pure-PyTorch `wht_along_head_dim(x)` (vectorized over leading dims; expects last dim a power of 2). Forward = inverse for normalized Hadamard convention (`H · H^T / N = I`).
- `src/flashquest/kernel/sparse_turbo_fwd.py` — `@triton.jit _sparse_attn_fwd_kernel_turbo` + Python wrapper `flash_attn_sparse_turbo_fwd`. Mirrors `sparse_int4_fwd.py` structurally; only the K/V tile-load + dequant pipeline differs.
- `src/flashquest/cache/persistent_turbo.py` — `PersistentTurboKVCache(Cache)` with `kv_bits = 3`. Mirrors `PersistentInt4KVCache` shape; storage is the seven tensors listed in the layout table.
- `tests/test_sparse_turbo.py` — unit tests (see Test strategy).
- `tests/test_kv_quant_turbo.py` — quant primitive tests.
- `tests/test_persistent_turbo.py` — cache class tests + slow dispatcher smoke.
- `scripts/phase7_run_ruler_4k_turbo.py` — RULER 4k runner with `kv_bits=3`.
- `benchmarks/phase7_decode_turbo_32k.json` — single-cell 32k bench artifact.
- `docs/PHASES/phase-7-notes.md` — phase notes (status + results).

**Modified files:**
- `src/flashquest/kernel/kv_quant.py` — add `quantize_k_turbo(K, page_size)`, `quantize_v_turbo(V)`, `dequantize_k_turbo`, `dequantize_v_turbo`, `_pack_bit_split`, `_unpack_bit_split`, K/V codebook constants. Existing INT4/INT8 functions unchanged.
- `src/flashquest/eager/llama_persistent_patch.py` — extend dispatcher to handle `kv_bits=3`. Branches:
  - `K_view_key = "K_msb"`, second view `K_lsb` for the bit-split.
  - `_dequant_k`, `_dequant_v` route to TurboQuant inverses.
  - `_quest_duo_fused_with_lse` extends to `kv_bits == 3` → `flash_attn_sparse_turbo_fwd`.
- `src/flashquest/eager/criticality.py` — no change. `page_scores_int4_fast` works as-is on `K_scale_raw` / `K_mn_raw` (15 vs 255 constant — same as INT4 case because raw stats are KIVI-style asymmetric INT4-equivalent statistics, just over the un-rotated K).
- `src/flashquest/runtime/chat.py` — `--kv-bits` accepts `{4, 8, 3}`; new help text mentions TurboQuant.
- `scripts/bench_flashquest.py` — `--kv-bits` accepts `{4, 8, 3}`; passes to cache constructor.

## Test strategy

**Unit tests** (mirror `tests/test_sparse_int4.py` structure):

- `tests/test_kv_quant_turbo.py::test_wht_inverse` — `wht(wht(x)) ≈ x` to 5e-3 rtol on random BF16 inputs at D ∈ {64, 128}.
- `tests/test_kv_quant_turbo.py::test_wht_orthogonal` — `wht(x) @ wht(y).T ≈ x @ y.T` to 5e-3 rtol. Confirms the orthogonality identity that the kernel relies on.
- `tests/test_kv_quant_turbo.py::test_quantize_k_turbo_roundtrip` — `dequantize_k_turbo(quantize_k_turbo(K))` reconstructs raw K within ~0.15 abs error (3-bit quant noise band on Gaussian K).
- `tests/test_kv_quant_turbo.py::test_quantize_v_turbo_roundtrip` — V analog within ~0.30 abs error (2-bit).
- `tests/test_kv_quant_turbo.py::test_pack_bit_split_roundtrip` — `_unpack_bit_split(_pack_bit_split(idx)) == idx` for random uint8 in 0..7.
- `tests/test_persistent_turbo.py::test_cache_roundtrip` — write a small fixture, read back via `get_views`, dequant, compare to original within 0.2 abs.
- `tests/test_persistent_turbo.py::test_packed_shape` — assert `K_msb` is `(B, H, S, D/8)` etc.
- `tests/test_persistent_turbo.py::test_int_dispatcher_smoke_llama_1b` (slow) — patch Llama-3.2-1B, run a single forward at ctx=64, assert finite logits.
- `tests/test_sparse_turbo.py::test_sparse_turbo_matches_dense` — fixture `(B=1, H_q=4, H_kv=2, S_q=1, S_kv=256, D=64, page_size=64)`, retention=1.0. Compare to `scaled_dot_product_attention` on dequant'd K, V. **Tolerance: 1e-1 abs** (looser than INT4's 5e-2 because of K=3-bit + V=2-bit quant noise; tighten if it passes empirically).
- `tests/test_sparse_turbo.py::test_fused_matches_reference_dequant` — Python reference path `(dequant + inverse_WHT + dense SDPA)` matched against the fused kernel within rtol=5e-2 on tiny fixture.
- `tests/test_sparse_turbo.py::test_no_kv_shaped_bf16_intermediate` — same monkeypatch pattern as Phase 6 task 6: assert no BF16 allocation matches `B*H_kv*S_kv*D*2` bytes during `flash_attn_sparse_turbo_fwd`.

**Quality gate** — RULER NIAH 4k @ TurboQuant:
- New script `scripts/phase7_run_ruler_4k_turbo.py` parametrized like `phase6_run_ruler_4k_int4.py` but with `kv_bits=3` plumbed through.
- **Pass: ≥85% per category** (niah_single, niah_multikey, niah_multivalue) — existing SPEC §6 task 5 gate.
- **Stretch: 100/100/100** — matches INT4 fused; the paper claims near-zero PPL gap, so the realistic expectation.
- **Failure handling:** if multivalue regresses below 85% but single/multikey pass, document the regression and queue Approach C (calibrated codebook on Llama-3.2-3B) as Phase 7.5. Do not back out unless all categories regress.

**Throughput gate** — single-cell decode bench at 32k:
- New script invocation: `python scripts/bench_flashquest.py --ctx-len 32768 --n-decode 32 --kv-bits 3 --out benchmarks/phase7_decode_turbo_32k.json`.
- **Pass: ≥6 tok/s decode at 32k** (1.55× over INT4 fused's 3.88).
- **Stretch: ≥8 tok/s** — closes half the throughput gap to llama.cpp's ~40 tok/s @ 8k.
- **Floor: ≥3.88 tok/s** — anything slower means the WHT + codebook lookups cost more than the storage shrink saves; rollback signal.

**Regression gate** — fast suite:
- Run `pytest tests/ -m "not slow"`. Existing 204 tests + ~10 new TurboQuant tests must all pass.

**Head-to-head re-run** — `scripts/phase6_run_headtohead.py` extended to accept `--kv-bits {4, 8, 3}`. Three new flashquest cells written to `benchmarks/phase7_cells_turbo/`, aggregated as `benchmarks/phase7_headtohead_turbo.{json,md}`. Compare flashquest TurboQuant vs flashquest INT4 fused (current canonical) vs llama.cpp vs vLLM 0.7.3.

## Acceptance gates (Phase 7 ship criteria)

1. RULER NIAH 4k @ TurboQuant clears ≥85% per category. Stretch: 100/100/100.
2. 32k decode @ TurboQuant ≥ 6 tok/s.
3. Fast suite (204 + new TurboQuant tests) all green.
4. Head-to-head shows TurboQuant numbers against llama.cpp / vLLM at all three contexts (8k, 32k, 128k).
5. Tag `phase-7`. Update DOC.md, README.md, docs/SPEC.md (new §6 task 7 entry).

## Out of scope (deferred)

- **1-bit QJL residual on K** — would push K to effective 4-bit (3 main + 1 residual) with extra storage. Re-evaluate if RULER multivalue regresses below 90%.
- **Calibrated codebook (Approach C)** — fallback for if the data-oblivious Rayleigh codebook regresses on Llama-3.2-3B. Not built unless gate fails.
- **TransMLA (GQA → MLA conversion)** — separate phase; needs ~6B-token fine-tune. Composes with TurboQuant in principle.
- **xKV (cross-layer KV sharing)** — orthogonal compression axis; ICLR 2025 method. Composes with TurboQuant.
- **Prefill TurboQuant kernel** — current kernel is decode-only; prefill stays through the dense BF16 SDPA path with inverse-WHT during dequant.
- **TurboQuant on Llama-3.1-8B** — hardware-blocked (8B AWQ alone is ~4.5 GiB on 4 GiB VRAM); revisit on a larger card.

## Rollback plan

If quality gate fails (any category < 85% on RULER):
1. Document the regression in `docs/PHASES/phase-7-notes.md`.
2. Try Approach C (calibrated codebook) as Phase 7.5 first — if the data-oblivious assumption is the issue, calibration on RULER prompts may close the gap.
3. If Approach C also fails: drop `--kv-bits 3` from the CLI defaults, archive the kernel under `_flash_attn_sparse_turbo_fwd_v1` for reference, and ship the flag as `--kv-bits 3 --experimental` only.
4. INT4 (`--kv-bits 4`, current default) stays canonical.

If throughput gate fails (decode < 3.88 tok/s at 32k):
1. Profile the kernel — `nsys nvprof` or Triton profiler — to find the hot spot.
2. Most likely culprits: 1-bit MSB unpack (8 successive shifts is more work than the INT4 nibble unpack), codebook lookup pattern (`tl.where` chain may not autovectorize), or per-token scale broadcast.
3. Try `tl.gather` over a constant codebook tensor instead of `tl.where`. Try smaller `BLOCK_D` (sub-token blocks).
4. If still slow: drop the kernel; ship TurboQuant as a quality-only path that runs through dense SDPA on dequant'd K, V. Storage shrink stands; throughput parity with INT4 is the floor.

## Open questions (resolve during plan execution)

- Exact codebook values: derive from `OnlyTerp/turboquant` reference, not reinvented. Spec assumes the symmetric 8-codepoint K table and 4-codepoint V table from the paper.
- WHT normalization convention: paper uses `H · H^T = N · I` (un-normalized) or `H · H^T / N = I` (normalized) — pick the latter to keep `wht(wht(x)) = x` exactly without a `/N` scale at decode. Confirm during impl.
- Whether to compute `s_K` from `max(|K_rot|)` or a softer quantile (e.g. q99) to suppress outliers. Default to `max` for v1; revisit if quality regresses.

## Self-review notes

- **Placeholder scan:** No "TBD" / "TODO" / "fill in later". Open questions are scoped + answered with defaults. ✓
- **Internal consistency:** Storage table matches the file inventory and kernel signature. Bit-plane shapes (D/8 MSB, D/4 LSB, D/4 V) are consistent across §Storage, §Bit-plane unpack, and §Module inventory. ✓
- **Scope:** Single subsystem (KV codec + kernel + cache class + dispatcher branch). Fits one implementation plan. ✓
- **Ambiguity:** Codebook values are listed as "approximate" — the spec is explicit that exact values come from the reference impl, not invented during planning. Tolerance numbers (1e-1 for sparse_turbo_matches_dense, 0.15 abs for K dequant) are starting points; spec calls them out as "tighten if it passes". ✓
