# Phase 8a — Compact-List Sparse Kernel + Sync Cleanup + Triton Fused Projections — Design (revision 3)

**Author:** flashquest core
**Date:** 2026-05-07
**Revision:** 3 (Phase 8 split into 8a + 8b after codex review on r2 surfaced graph-viability issues that require cache redesign. This spec covers 8a only — kernel + sync + fusion foundation. Phase 8b graph capture deferred to a separate future brainstorm.)
**Status:** Approved post-split
**Phase 7 baseline:** [`docs/PHASES/phase-7-notes.md`](../../PHASES/phase-7-notes.md) — 3.88 tok/s @ 32k INT4 fused, 2.62 tok/s @ 32k TurboQuant K3-V3.
**Roadmap position:** First of an OOM cumulative push. **Phase 8a (this spec)** → 8b (CUDA Graphs + cache redesign, future brainstorm) → 9 (specdec) → 10 (DuoAttention/CATS) → 11 (lookahead/MTP). 8a alone targets ~1.3-1.5×.

---

## 1. Goal

Take Llama-3.2-3B AWQ-INT4 decode throughput @ 32k on RTX 3050 Ti Laptop from **3.88 → ≥5.0 tok/s (1.29× floor) / ≥5.8 tok/s stretch (1.50×)**, without changing the model, KV bit-width, or quality.

Phase 8a is **foundation work**:
- Compact-list sparse kernel rewrite — replaces the current full-mask iteration that wastes ~75% of loop overhead at retention=0.25.
- `.item()` removal in selection — preparatory for Phase 8b CUDA Graphs.
- Triton fused QKV + gate+up GEMM — reduces 5 layer-GEMMs to 2.
- RoPE-post cache audit.

No CUDA Graphs in 8a. No Marlin. No async streams. No bucket dispatcher. Those move to Phase 8b (graphs + cache redesign) and Phase 9 (Marlin + specdec).

## 2. Background — why the split

Codex review on r2 surfaced 12 findings, of which 3 were kernel correctness bugs (easy fixes for r3) and 3 were graph-viability issues that pointed to a deeper truth:

> **CUDA Graph capture under flashquest's current cache design requires a cache redesign — fixed-max-size views + length-mask tensors instead of Python-int-dependent slices. That's its own substantial work.**

Specifically, codex flagged on r2:
- (#7) `cache.get_views()` returns slices with Python-int shapes (`K_packed[:completed_pages]`, `V_partial[:partial_len]`). Graphs need fixed shapes.
- (#8) Predicated partial-merge needs fixed page_size tail buffer + LSE=−inf for empty tail; not just "always run both branches".
- (#9) QKV/cache-update/top-k stay outside the captured graph; the "~250 launches → 1 launch" claim from r2 was overclaimed (closer to ~100 launches saved).

Trying to bundle all of this into one phase risks a third codex round-trip and a fragile implementation. Splitting:
- **Phase 8a (this spec):** compact kernel + sync removal + Triton fusion. Honest ~1.3-1.5× ceiling, all r2 kernel-correctness bugs fixed, no graph engineering. Ships cleanly.
- **Phase 8b (future brainstorm):** cache-view redesign + bucketed CUDA Graphs. Standalone ~1.2-1.4× on top of 8a. Brainstorm starts only after 8a lands and we have measurements.

This split makes Phase 8a a small, well-bounded change and lets 8b's brainstorm benefit from real 8a numbers when designing the graph layer.

## 3. Design

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 8a decode step (per layer × 28 layers)                       │
│                                                                     │
│  input ─┐                                                           │
│         ├── Triton fused QKV GEMM (AWQ-INT4 direct, BF16 in/out)    │
│         │   slice → Q, K, V                                         │
│         │   RoPE on Q, K                                            │
│         │                                                           │
│         ├── cache.update_quantized(K, V)                            │
│         │                                                           │
│         ├── Quest criticality (existing per H_q)                    │
│         │   k_max = static, precomputed from retention × P_max      │
│         │   topk → selected_page_ids: int32 (B, H_q, BUCKET_MAX)    │
│         │   pad with sentinel -1 for unused slots                   │
│         │                                                           │
│         ├── flash_attn_sparse_int4_fwd_compact(                     │
│         │       q, K_packed, K_scale, K_mn,                         │
│         │       V_packed, V_scale, V_mn,                            │
│         │       selected_page_ids,                                  │
│         │       BUCKET_MAX,  # constexpr at JIT                     │
│         │       page_size, return_lse=True)                         │
│         │                                                           │
│         ├── partial-page tail merge (existing path; unchanged)      │
│         │                                                           │
│         ├── Triton fused gate+up GEMM (AWQ-INT4 direct)             │
│         │   slice → gate, up; SwiGLU                                │
│         │   AWQ down_proj (unchanged)                               │
│         │                                                           │
│         └── output (next-token hidden state)                        │
└─────────────────────────────────────────────────────────────────────┘
```

Differences from current Phase 7:
- Kernel iterates only `BUCKET_MAX` (= retention × P_max ≈ 128 at 32k) instead of `NUM_PAGES` (= 512 at 32k). ~4× fewer loop iterations.
- Selection uses precomputed static `k_max`, no `.item()` per step.
- 5 separate AWQ GEMMs (Q, K, V, gate, up) → 2 fused Triton GEMMs (qkv, gate+up).
- Partial-merge path unchanged (lives outside Phase 8a's reshape work; will be redesigned in 8b for graphs).

## 4. Components

### 4.1 Compact-list sparse kernel rewrite

**Surface (new files):**
- `src/flashquest/kernel/sparse_int4_fwd_compact.py`
- `src/flashquest/kernel/sparse_int8_fwd_compact.py` (Phase 3 INT8 path)
- `src/flashquest/kernel/sparse_turbo_fwd_compact.py` (Phase 7 TurboQuant path)

**API:**
```python
def flash_attn_sparse_int4_fwd_compact(
    q,                              # (B, H_q, 1, D) bf16
    K_packed, K_scale, K_mn,
    V_packed, V_scale, V_mn,
    selected_page_ids: torch.Tensor,  # int32 (B, H_q, BUCKET_MAX)
    page_size: int,
    return_lse: bool,
    BUCKET_MAX: int = ...,            # passed as constexpr at JIT compile
) -> tuple[torch.Tensor, torch.Tensor]:   # (O, lse)
```

**Kernel ABI shape: `(B, H_q, BUCKET_MAX)`.** Selection in `selection.py:81-117` produces a per-`H_q` mask `(B, H_q, S_q, P)`. Compact form is per-`H_q` to preserve quality; each Q head can request a different page set. Codex review (r2) flagged that an `H_kv` ABI would lose distinct Q-head top-k unless heads were unioned — Phase 8a takes the per-`H_q` route to avoid that quality risk.

**Kernel inner loop (replaces the current `for p in range(0, NUM_PAGES)` body):**
```python
for i in range(0, BUCKET_MAX):
    # Load page index from compact list
    sel_off = b * stride_selb + h_q * stride_selh + i * stride_seli
    p = tl.load(selected_page_ids_ptr + sel_off)  # int32
    page_valid = p >= 0
    p_safe = tl.where(page_valid, p, 0)  # avoid negative addresses

    page_start = p_safe * PAGE_SIZE
    n_idx = page_start + offs_n
    # combine per-page validity with per-token validity
    valid_kv = (n_idx < S_kv) & page_valid

    # ALL K/V loads use the combined mask (codex r2 finding #2)
    k_byte = tl.load(k_byte_ptrs, mask=valid_kv[:, None], other=0)
    # ... unpack INT4 nibbles, scale ...
    qk = tl.sum(q[None, :].to(tl.float32) * k, axis=1)
    # padded slots and out-of-range tokens contribute -inf to softmax (no effect)
    qk = tl.where(valid_kv, qk, NEG_INF)
    # ... rest of online-softmax body unchanged ...
```

**Sentinel padding (codex r2 #2, #3, #4):**
- Caller pads `selected_page_ids` with `-1` for unused slots.
- Kernel computes `page_valid = p >= 0`, replaces `p` with `p_safe = tl.where(page_valid, p, 0)` BEFORE computing `n_idx`. Address arithmetic uses `p_safe`; load mask uses `page_valid & valid_kv`.
- After load, `qk = tl.where(valid_kv, qk, NEG_INF)`. Padded slots end up at `-inf` which contributes `exp(-inf) = 0` to softmax. Numerator and denominator both unchanged → result identical to running with only the real selected pages.
- (Initial "pad with first page repeated" idea was rejected — algebra showed `(2 num_X + num_Y) / (2 den_X + den_Y) ≠ (num_X + num_Y) / (den_X + den_Y)` when other pages are present. Sentinel is correct.)

**Fallback (current bool-mask kernel) stays in place** as `flash_attn_sparse_int4_fwd` for regression-test reference. Both paths must produce identical output (parity test §7.1).

### 4.2 Sync elimination in selection

**Surface modified:** `src/flashquest/eager/selection.py`.

**Current:**
```python
k_max = int(k_per_h.max().item())   # CPU sync per decode step
if k_max > 0:
    topk_idx = scores.topk(k_max, dim=-1).indices
```

**Phase 8a:**
```python
def select_pages_vectorized(scores, retention, num_sinks, window_pages,
                            k_max_static: int):
    # k_max_static precomputed at module init from retention × P_max
    if k_max_static > 0:
        topk_idx = scores.topk(k_max_static, dim=-1).indices
    # ... rest unchanged
```

`k_max_static` is computed once at the top of `make_quest_persistent_forward`:
```python
P_max = max_context_len // page_size  # known at model load
if isinstance(retention, float):
    k_max_static = math.ceil(retention * P_max)
else:
    k_max_static = math.ceil(float(retention.max().item()) * P_max)
```

This is a one-time `.item()` at module init, not a per-step sync.

**Note:** Phase 8a doesn't yet use `k_max_static` to drive `BUCKET_MAX` for the kernel — kernel sees `BUCKET_MAX = k_max_static + num_sinks + window_pages` (slack for sink/window pages). Phase 8b will tighten this.

**Note:** the `partial_len`, `completed_len` Python-int reads in `llama_persistent_patch.py:177-201` are **not** removed in 8a. They're not blocking anything in 8a (no graphs). They become Phase 8b's concern (graph-friendly cache views).

### 4.3 Triton fused QKV + gate+up

**Surface (new files):**
- `src/flashquest/kernel/fused_proj.py` — `fused_qkv_proj`, `fused_gate_up_proj` Triton kernels
- AWQ packing audit utility — `src/flashquest/quant/awq_layout.py` — confirms `qweight`/`qzeros`/`scales` shape and packing axis used by AutoAWQ-loaded checkpoints

**Pre-design audit (Task 2):** before writing the fused kernel, verify the actual AWQ packed-weight layout used in our model (codex r2 #11). AutoAWQ stores:
- `qweight: (in // pack_factor, out)` int32 packed along the K-axis (8 INT4 values per int32 along K)
- `qzeros: (in_groups, out // pack_factor)` int32 packed along the N-axis
- `scales: (in_groups, out)` BF16

Different AWQ implementations (vLLM, AutoAWQ, MLX) sometimes differ on which axis is packed. Phase 8a Task 2 prints the actual shapes from a real loaded Llama-3.2-3B-AWQ checkpoint and locks the kernel against those shapes.

**Fused QKV kernel:**
- Stack `q_proj.qweight`, `k_proj.qweight`, `v_proj.qweight` along the N axis into `qweight_qkv: (in // 8, N_q + N_k + N_v) = (3072//8, 5120)`.
- Same for scales and zeros.
- One Triton GEMM: `O[1, N_q+N_k+N_v]`.
- Slice into Q (N_q), K (N_k), V (N_v) views.
- Apply RoPE on Q and K views (existing path).

**Fused gate+up kernel:**
- Stack `gate_proj.qweight`, `up_proj.qweight` along N: `qweight_gu: (3072//8, 8192*2)`.
- Same fusion pattern. Slice into gate (8192), up (8192).
- Apply SwiGLU: `output = silu(gate) * up`.

**Down projection (`down_proj`) stays as separate AWQ Linear** — only one GEMM per layer, no fusion benefit.

**Why custom Triton, not Marlin:** Phase 6 notes (`docs/PHASES/phase-6-notes.md:148`) document "Marlin at M=1 ≈ AWQ. Marlin's design point is M=16-32." At our decode batch=1, Marlin standalone gain ≈ 0×. Marlin migration deferred to Phase 9 where speculative decoding raises effective M to 8-16.

**Estimated standalone gain:** 1.05-1.10× from launch reduction (5 GEMMs → 2 GEMMs) + larger N tile fitting Ampere tensor-core shape.

### 4.4 RoPE-post cache verification

**Audit:** confirm `K` is stored *post-RoPE* in `PersistentInt8KVCache`, `PersistentInt4KVCache`, `PersistentTurboKVCache`. Reading `llama_persistent_patch.py:155-167`:
```python
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# ... bf16 casts ...
cache.update_quantized(k, v, layer_idx=self.layer_idx)
```
appears post-RoPE. **Document this in cache module docstrings** so future authors don't accidentally pass pre-RoPE K.

If pre-RoPE somewhere, fix so decode reads use cached K directly. Likely already correct — small task, no code change expected.

## 5. Compatibility matrix

| `--kv-bits` | Compact kernel | Fused proj | Status after Phase 8a |
|---|---|---|---|
| 4 (Phase 6 INT4) | ✅ rewritten | ✅ | Primary target |
| 8 (Phase 3 INT8) | ✅ rewritten | ✅ | Maintained |
| 3 (Phase 7 Turbo) | ✅ rewritten | ✅ | Maintained |
| any | OFF (`--no-compact-kernel`) | OFF (`--no-fused-proj`) | Phase 7 baseline path (regression-test reference) |

Both flags default ON post-validation. Defaults OFF during task progression until each is validated independently.

## 6. Estimated 32k decode flow numbers

At 32k context, page_size=64, retention=0.25:
- `num_pages = 512`
- `k_per_h = ceil(0.25 × 512) = 128`
- `num_sinks = 4`, `window_pages = 2` → effective request ≈ 130-134 unique pages per H_q (some sinks/window may overlap with top-k)
- `BUCKET_MAX = 128 + 4 + 2 = 134` (loose upper bound; precomputed at module init)

Compact kernel runs **~134 loop iterations per (B, H_q)**, vs current **512 iterations**. 3.8× fewer iterations.

## 7. Tests

### 7.1 Parity tests (fast suite)

`tests/test_sparse_int4_fwd_compact_parity.py`:
- Random K/V cache (B=1, H_kv=8, S=2048 (= 32 pages), D=128). Random selection of 8 of 32 pages.
- Full-mask kernel output vs compact kernel output (real_count=8, BUCKET_MAX=8). Max abs err < 1e-2 BF16.
- Repeat for all `kv_bits ∈ {3, 4, 8}`.

`tests/test_sparse_int4_fwd_compact_padding.py`:
- Selection of 8 pages, BUCKET_MAX=16 (padded with 8× sentinel `-1`). Output identical to BUCKET_MAX=8 unpadded version.
- Edge case: all sentinel (`real_count=0`, BUCKET_MAX=16, all -1). Output should be all-zeros (no contribution to softmax → degenerate case; document expected behavior).

`tests/test_sparse_compact_kernel_address_safety.py`:
- Unit test that confirms negative-page-id `p=-1` does NOT cause out-of-bounds memory access. Use a small cache; verify CUDA doesn't crash and output is zero-contribution for the padded slot. Run under compute-sanitizer if available.

`tests/test_select_pages_static_kmax.py`:
- Verify `select_pages_vectorized` with `k_max_static` param produces same output as current `.item()`-based version. Trace via PyTorch profiler, confirm zero `aten::_local_scalar_dense` (=`.item()`) calls in hot path.

`tests/test_fused_qkv_triton.py`:
- Random AWQ-INT4 weights for q/k/v with audited layout. Compare separate AWQ-Linear outputs to fused Triton output. Max abs err < 5e-2 BF16.

`tests/test_fused_gate_up_triton.py`:
- Same pattern for gate+up + SwiGLU.

`tests/test_awq_layout_audit.py`:
- Loads a real Llama-3.2-3B-AWQ checkpoint (one layer), prints `qweight.shape`, `scales.shape`, `qzeros.shape`. Asserts the layout matches the design assumption (`(in // 8, out)` packed along K).

### 7.2 Quality gate (slow suite)

`tests/test_phase8a_ruler_4k.py` (slow):
- RULER NIAH 4k @ Llama-3.2-3B AWQ-INT4 with `--compact-kernel --fused-proj --kv-bits 4`.
- All 3 categories ≥85% (Phase 7 baseline).
- Stretch: 100/100/100 (Phase 6 INT4 baseline).

### 7.3 Performance benches (slow)

`benchmarks/phase8a_decode_8k.py`:
- 8k decode tok/s. Expect ≥6 tok/s (vs Phase 6 4.94).

`benchmarks/phase8a_decode_32k.py`:
- 32k decode tok/s. Expect ≥5.0 tok/s floor (vs Phase 6 3.88), stretch ≥5.8.

`benchmarks/phase8a_ablation.py`:
- 3 cells:
  - Phase 7 baseline (no Phase 8a)
  - Compact kernel only (no fused proj)
  - Compact + fused proj (full Phase 8a)
- Confirms each component's standalone contribution.

## 8. Acceptance gates

| Gate | Floor | Stretch | Test/Bench |
|---|---|---|---|
| Compact kernel parity (vs current) | max abs err < 1e-2 BF16, all kv_bits | 0 errors | `test_sparse_*_compact_parity.py` |
| Compact kernel address safety | no OOB crashes, sentinel pads contribute 0 | — | `test_sparse_compact_kernel_address_safety.py` |
| RULER NIAH 4k all 3 cats | ≥85% (no regression vs Phase 7) | 100/100/100 | `test_phase8a_ruler_4k.py` |
| 32k decode tok/s (kv-bits 4) | ≥5.0 (1.29× current 3.88) | ≥5.8 (1.50×) | `phase8a_decode_32k.py` |
| 8k decode tok/s | ≥6 (1.21× current 4.94) | ≥7 (1.42×) | `phase8a_decode_8k.py` |
| Per-feature ablation | each component non-negative | each ≥1.05× | `phase8a_ablation.py` |
| Peak VRAM @ 32k | ≤ Phase 7 + 100 MiB | ≤ Phase 7 | bench output |
| Fast suite | 227+ pass, 0 regressions | — | `pytest tests/ -m "not slow"` |
| Compatibility | `--kv-bits {3,4,8}` all still pass | — | bench at each kv-bits |

VRAM overhead is modest (no graph pools, no static buffers, no Marlin workspaces): just kernel scratch + the precomputed static k_max which is a Python int. +100 MiB is generous slack.

**On VRAM and the 4 GB budget (codex r2 #12):** RTX 3050 Ti Laptop has 4 GB dedicated VRAM. Phase 7 reports 6105 MiB peak — this exceeds the dedicated GPU memory and uses WSL2 shared-memory overcommit (host RAM via UMA bridge). This is documented behavior on Ampere+WSL2; performance penalty is modest because access is mostly sequential. Phase 8a does not change this regime — same overcommit, +100 MiB slack on top.

## 9. Risks & mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Compact kernel doesn't beat full-mask kernel | Low | Phase 8a ceiling drops to fusion-only (~1.1×) | Microbench in Task 4 (compact vs current at 32k shapes) before integration. |
| Sentinel `-1` causes Triton compile error or UB | Medium | Kernel doesn't compile or crashes | Address-safety test in Task 4. Use `tl.where(p_valid, p, 0)` BEFORE address arithmetic, mask all loads. Documented in §4.1. |
| AWQ packing axis assumption wrong | Medium | Fused proj kernel produces wrong output | Audit task (Task 2) prints real layout from loaded checkpoint before kernel write. |
| Triton fused QKV/gate+up doesn't beat AWQ | Medium | Component drops, ceiling −5-10% | Microbench in Task 9. Don't ship a regression. |
| BF16 dequant in fused Triton has accuracy issues | Low | Quality gate fail | RULER 4k catches it. |
| Per-`H_q` compact list memory grows | Low | Selected_page_ids tensor at (1, 24, 134) = 12 KiB; trivial | No mitigation needed. |

## 10. Non-goals (deferred)

- **CUDA Graphs (any flavor)** — Phase 8b. Requires cache-view redesign first. Future brainstorm.
- **Bucketed dispatcher** — Phase 8b.
- **Fixed-size predicated partial-merge** — Phase 8b.
- **Marlin INT4 GEMM** — Phase 9 (post-specdec, when M ≥ 8).
- **Async two-stream layer pipeline** — DROPPED (broken dependency model).
- **Speculative decoding** — Phase 9.
- **DuoAttention head split** — Phase 10 (also needs head-pattern training first per Phase 6 notes).
- **Activation sparsity (CATS)** — Phase 10.
- **Lookahead / Jacobi / prompt-lookup** — Phase 11.
- **`partial_len` / `completed_len` Python-int branch removal** — Phase 8b (graph-friendly cache views).
- **Prefill optimization** — separate axis.
- **Multi-batch / continuous batching** — single-user laptop runtime; not in scope.

## 11. Surface — files to create / modify

### New files
- `src/flashquest/kernel/sparse_int4_fwd_compact.py`
- `src/flashquest/kernel/sparse_int8_fwd_compact.py`
- `src/flashquest/kernel/sparse_turbo_fwd_compact.py`
- `src/flashquest/kernel/fused_proj.py`
- `src/flashquest/quant/__init__.py` (if not present)
- `src/flashquest/quant/awq_layout.py` — packing-axis audit utility
- `tests/test_sparse_int4_fwd_compact_parity.py`
- `tests/test_sparse_int8_fwd_compact_parity.py`
- `tests/test_sparse_turbo_fwd_compact_parity.py`
- `tests/test_sparse_int4_fwd_compact_padding.py`
- `tests/test_sparse_compact_kernel_address_safety.py`
- `tests/test_select_pages_static_kmax.py`
- `tests/test_fused_qkv_triton.py`
- `tests/test_fused_gate_up_triton.py`
- `tests/test_awq_layout_audit.py`
- `tests/test_phase8a_ruler_4k.py` (slow)
- `benchmarks/phase8a_decode_8k.py`
- `benchmarks/phase8a_decode_32k.py`
- `benchmarks/phase8a_ablation.py`
- `docs/PHASES/phase-8a-notes.md` (post-execution writeup)

### Modified files
- `src/flashquest/eager/selection.py` — accept `k_max_static` param; eliminate `.item()` in hot path
- `src/flashquest/eager/llama_persistent_patch.py` — wire compact kernel + fused proj; precompute `k_max_static` and `BUCKET_MAX` once at module init
- `src/flashquest/cache/persistent_int4.py`, `persistent_int8.py`, `persistent_turbo.py` — docstring update confirming post-RoPE storage
- `scripts/bench_flashquest.py` — `--compact-kernel / --no-compact-kernel`, `--fused-proj / --no-fused-proj` flags
- `src/flashquest/cli.py` — same flags

## 12. Implementation order (preview — full plan in writing-plans)

Rough sequencing (12 tasks):
1. **AWQ layout audit** — print real `qweight`/`qzeros`/`scales` shapes from loaded Llama-3.2-3B-AWQ checkpoint. Lock kernel design against actual shapes.
2. **`k_max_static` precompute** — one-shot refactor in `selection.py` + `make_quest_persistent_forward` to eliminate per-step `.item()`. Parity test against current.
3. **Compact INT4 kernel** — write `sparse_int4_fwd_compact` with sentinel padding + load-mask correctness. Parity + address-safety tests.
4. **Compact INT4 microbench** — 32k decode kernel-only vs current. Gate: ≥1.2× kernel-wall, otherwise drop kernel rewrite.
5. **Compact INT8 + Turbo kernels** — same pattern, parity tests.
6. **Wire compact kernel into Llama patch** (eager mode, current cache views unchanged). Quality gate: RULER 4k holds.
7. **Per-feature ablation 1**: compact-only tok/s vs Phase 7. Validate kernel win in full pipeline.
8. **Triton fused QKV kernel** — design against audited AWQ layout. Parity + microbench gate.
9. **Triton fused gate+up kernel** — same pattern.
10. **Wire fused proj into Llama patch**. Compatibility check: all `kv_bits {3,4,8}` modes still pass.
11. **Per-feature ablation 2**: compact + fused tok/s. Validate fused proj win.
12. **Final RULER + bench + writeup**. Phase 8a tag, push to master.

## 13. References

- **Quest paper:** "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference" — Tang et al., ICML 2024. [arXiv:2406.10774](https://arxiv.org/abs/2406.10774).
- **Triton tutorials — Fused matrix multiplication:** [triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
- **AutoAWQ GEMM kernel reference:** [github.com/casper-hansen/AutoAWQ/blob/main/awq/modules/linear/gemm.py](https://github.com/casper-hansen/AutoAWQ).
- **Phase 6 notes (Marlin batch=1 finding):** `docs/PHASES/phase-6-notes.md:148`.
- **Phase 7 spec:** `docs/superpowers/specs/2026-05-06-phase-7-turboquant-kv-design.md`.
- **Phase 7 notes:** `docs/PHASES/phase-7-notes.md`.
- **Codex review of r1, r2:** internal — see commit log.

---

## Appendix A — Revision history

### r1 (2026-05-07) — initial spec
Marlin INT4 + bucketed CUDA Graphs + async two-stream pipeline. Codex review found 12 issues: 4 dealbreakers (bucketed bool-mask doesn't reduce work, bucket sizes calibrated for 4k not 32k, Marlin at M=1 ≈ AWQ per Phase 6 notes, async pipeline has same-layer write/read dependency) + 8 corrections.

### r2 (2026-05-07) — full rewrite
Dropped Marlin and async pipeline. Added compact-list kernel, sync elimination, real-32k bucket calibration, sentinel padding (with corrected algebra). Codex review found 12 more issues:
- 3 kernel correctness bugs: ABI shape (per H_q not H_kv), sentinel needs load-time masking, dispatcher pseudocode (searchsorted index vs bucket size confusion).
- 3 graph-viability issues: dynamic cache views graph-hostile, predicated partial-merge under-specified, capture boundary inconsistent (~250 launches → 1 launch claim overstated).
- 6 misc.

### r3 (2026-05-07, this spec) — split Phase 8 → 8a + 8b
- **Phase 8a (this):** compact kernel + sync removal + Triton fusion only. Honest 1.3-1.5× ceiling. All r2 kernel-correctness bugs fixed. No CUDA Graphs.
- **Phase 8b (future):** cache-view redesign + bucketed graphs. Future brainstorm starts after 8a lands and we have measurements. Estimated standalone +1.2-1.4× on top of 8a.

The split avoids piling cache-redesign + graph engineering + kernel rewrite into one phase. Each phase is small, well-bounded, and ships independently.

**Process note:** I should have re-read Phase 6 notes (Marlin M=1 finding) AND examined the actual cache view structure before writing r1. Codex caught both gaps. Adding "re-read prior phase findings + examine actual cache/kernel surface before drafting" as personal pre-spec habit.
