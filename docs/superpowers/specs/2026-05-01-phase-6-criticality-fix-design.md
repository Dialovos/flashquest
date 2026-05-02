# Phase 6 task 1 — Criticality + top-k fix (no kernel rewrite)

**Date:** 2026-05-01
**Status:** Design approved; ready for implementation plan
**Supersedes (in scope):** SPEC §6 task 1 ("kernel-fused criticality + top-k") — the *gate* (≥4 tok/s decode at 32k) stands; the *means* changes from a Triton kernel rewrite to a vectorised PyTorch fix.

## Problem

Phase 5 ships persistent INT8 KV + fused DuoAttention dispatch but decodes at **0.10 tok/s at 32k** vs SPEC target ≥4 tok/s. The original Phase 6 plan called for porting Quest's CUDA criticality kernel into Triton with shared-mem bitonic top-k — a 1–2 week effort.

## Profile evidence

`scripts/phase6_profile_decode.py` (cuda-synchronised manual timing per layer 0 sub-step at 32k context):

| op | ms / layer | % |
|---|---|---|
| `dequantize_k` | 155.2 | 46.0 |
| `compute_page_summary` | 133.4 | 39.5 |
| `repeat_interleave_K` | 38.5 | 11.4 |
| `select_pages` | 4.8 | 1.4 |
| `page_scores` | 3.8 | 1.1 |
| `flash_attn_sparse_fwd` | 1.8 | 0.5 |

× 28 layers = 9.45 s / token → **0.10 tok/s** (matches end-to-end measurement).

The CPU profiler showed `aten::_local_scalar_dense` (`.item()`) at 37.8 s across 4 decode steps = 9.45 s / step. This was the per-head Python loop in `select_pages` *blocking* on the in-flight dequant + summary CUDA work — not the bottleneck itself, just the visible wait.

## Algebraic identity

`src/flashquest/kernel/kv_quant.py` defines:

```python
mn = K.min(dim=page).values        # per-page per-channel min
mx = K.max(dim=page).values        # per-page per-channel max
scale = (mx - mn) / 255 .clamp_min(eps)
K_uint8 = round((K - mn) / scale)
```

So:

- `K_mn ≡ page_min` (exact)
- `K_mn + 255 * K_scale ≡ page_max` (exact, modulo the eps clamp on constant channels — same eps the dequant path rounds through)

Quest's criticality `score(p, q) = sum_d max(Q[d] * page_min[p, d], Q[d] * page_max[p, d])` can be computed directly from `K_scale` and `K_mn`. **No dequantization, no `repeat_interleave_K` of the full S×D cache** — broadcast happens on the (P, D) summary only.

## Design

Two new pure-PyTorch functions, no Triton.

### 1. `page_scores_int8(Q, K_scale, K_mn) → scores`

Replaces the chain `dequantize_k → compute_page_summary → page_scores`.

```python
def page_scores_int8(
    Q: torch.Tensor,        # (B, H_q, S_q, D) bf16
    K_scale: torch.Tensor,  # (B, H_kv, P, D) bf16
    K_mn: torch.Tensor,     # (B, H_kv, P, D) bf16
) -> torch.Tensor:          # (B, H_q, S_q, P) fp32
    B, H_q, S_q, D = Q.shape
    H_kv = K_scale.shape[1]
    n_rep = H_q // H_kv
    Q_e = Q.unsqueeze(3).float()                          # (B, H_q, S_q, 1, D)
    Kmn = K_mn.float().repeat_interleave(n_rep, dim=1)    # (B, H_q, P, D) — but (P, D) is small
    Kmx = (K_mn.float() + 255.0 * K_scale.float()).repeat_interleave(n_rep, dim=1)
    Kmn_e = Kmn.unsqueeze(2)                              # (B, H_q, 1, P, D)
    Kmx_e = Kmx.unsqueeze(2)
    cand_mx = Q_e * Kmx_e
    cand_mn = Q_e * Kmn_e
    return torch.maximum(cand_mx, cand_mn).sum(dim=-1)    # (B, H_q, S_q, P)
```

Memory: `K_scale + K_mn = 2 × (8 × 512 × 128 × 2 B) = 2 MB` per layer at 32k. Multiplied by `n_rep` after broadcast = 6 MB per layer. **vs ~256 MB for the dequant'd cache + repeat_interleave today.**

Edge: GQA broadcast happens *after* loading K_scale/K_mn (small tensors), not on the cache.

### 2. `select_pages_vectorized(scores, retention_per_h, num_sinks, window_pages) → mask`

Replaces the per-head Python loop with `.item()` syncs.

```python
def select_pages_vectorized(
    scores: torch.Tensor,        # (B, H, S_q, P) fp32
    retention: float | torch.Tensor,  # scalar or (H,)
    num_sinks: int,
    window_pages: int,
) -> torch.Tensor:               # (B, H, S_q, P) bool
    B, H, S_q, P = scores.shape
    if isinstance(retention, torch.Tensor):
        retention_per_h = retention.to(scores.device).float()
    else:
        retention_per_h = torch.full((H,), float(retention), device=scores.device)

    k_per_h = (retention_per_h * P).ceil().long().clamp(min=0, max=P)  # (H,)
    k_max = int(k_per_h.max().item())  # ONE host sync per layer, not per head

    mask = torch.zeros_like(scores, dtype=torch.bool)

    if k_max > 0:
        topk_idx = scores.topk(k_max, dim=-1).indices               # (B, H, S_q, k_max)
        ranks = torch.arange(k_max, device=scores.device).view(1, 1, 1, k_max)
        keep = ranks < k_per_h.view(1, H, 1, 1)                     # (1, H, 1, k_max)
        src = keep.expand_as(topk_idx)                              # (B, H, S_q, k_max)
        # scatter_ writes src value at topk_idx positions. For ranks beyond
        # k_per_h[h], src is False, so we'd write False to a False mask cell —
        # no-op. For ranks < k_per_h[h], we write True at the top-k indices.
        mask.scatter_(-1, topk_idx, src)

    if num_sinks > 0:
        n = min(num_sinks, P)
        mask[..., :n] = True
    if window_pages > 0:
        w = min(window_pages, P)
        mask[..., P - w:] = True
    return mask
```

Why `mask.scatter_(-1, topk_idx, src)` is safe: `topk_idx` is shape `(B, H, S_q, k_max)`. For invalid positions (where `keep` is False), `src` is False, so the scatter writes False to the existing False location — no harm. Where `src` is True we write True at the top-k positions.

One `.item()` per layer (for `k_max`) instead of `H` per layer. Could be eliminated by always using `k_max = ceil(max(retention) * P)` precomputed at patch time; defer that micro-optimisation.

### 3. Wire into `llama_persistent_patch.py`

Replace the decode-path chain in `_quest_duo_fused_with_lse`:

```python
# Before (Phase 5):
K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
scores = page_scores(Q.float(), page_min, page_max)
sel = select_pages(scores, retention=retention_per_q, num_sinks=num_sinks, window_pages=window_pages)

# After (Phase 6):
scores = page_scores_int8(Q, K_scale, K_mn)
sel = select_pages_vectorized(scores, retention=retention_per_q, num_sinks=num_sinks, window_pages=window_pages)
```

Prefill path (`S_q > 1`) stays unchanged — it does dense SDPA over the dequant'd cache, doesn't hit criticality.

### 4. Tests

- `tests/test_page_scores_int8.py`:
  - **EQ18**: scores match the Phase 5 path (`dequantize_k → compute_page_summary → page_scores`) within rtol=2e-2 on a synthetic case (B=1, H_kv=2, P=4, D=64, n_rep=2). Eps-clamp tolerance.
  - **EQ19**: GQA broadcast — `H_q ≠ H_kv` produces correct per-head scores (each q-head sees its corresponding kv-head's K_scale/K_mn).
  - **EQ20**: shape and dtype contract.

- `tests/test_select_pages_vectorized.py`:
  - **EQ21**: equivalence to `select_pages` (Phase 5) for scalar retention, range of (P, k) sizes.
  - **EQ22**: per-head retention tensor — same selections as the loop version.
  - **EQ23**: retention=0 head produces empty top-k mask (sinks/window may still set bits).
  - **EQ24**: retention=1 head selects all pages.
  - **EQ25**: sinks + window edge — sinks > P, window > P clamp.

- `tests/test_persistent_e2e.py` (extend existing): re-run with the new functions wired in. `S_q=1` decode logits ≡ Phase 5 logits (rtol=5e-2). Already covers the integration.

### 5. Validation gates

| Gate | Target | Source |
|---|---|---|
| Decode tok/s at 32k | ≥4 tok/s | re-run `scripts/phase5_bench_decode_32k.py` |
| 32k passkey | 6/6 across 3 depths × 2 trials | re-run `scripts/phase5_run_passkey_32k.py` |
| Logit equivalence | rtol=5e-2 vs Phase 5 path | extend `tests/test_persistent_e2e.py` |
| All Phase 5 unit tests | green | `pytest tests/` |

If decode tok/s lands ≥4 tok/s, SPEC §6 task 1 is closed and we move to task 2 (RULER 4k subset eval).

## Why we still defer the Triton kernel

The kernel approach was justified by the assumption that "Python wrapper overhead dominates kernel time" — and it does, but the overhead is **memory-bound CUDA work** (full cache dequant + reduce), not Python. The fix is "don't do that work," which is a math change, not a kernel change. The Triton kernel would solve the same problem more elegantly but at 10× the cost.

The kernel revisits when:
- 64k / 128k context: at 128k the (K_scale, K_mn) tensors are 24 MB, still tiny, but the per-token criticality compute scales linearly. At ~2 ms/layer × 28 = 56 ms/step we're at 17 tok/s; if 128k pushes that past 100 ms/step, kernelise.
- INT4 KV (Phase 6 task 5) needs new dequant kernels anyway; folding criticality in alongside is natural at that point.

Until then, this PyTorch fix lands the SPEC win condition, unlocks RULER 4k, the CLI demo, and the benchmark table — the rest of Phase 6.

## Out of scope

- Triton sparse_fused.py (deferred).
- INT4 KV (Phase 6 task 5).
- Marlin (Phase 6 task 7).
- EAGLE-2 (Phase 6 task 6).
- 128k context retest (Phase 6 task 4 benchmark; will pick up if it lands as bonus).

## File touch list

```
src/flashquest/eager/
├── criticality.py              # add page_scores_int8 alongside page_scores
└── selection.py                # add select_pages_vectorized alongside select_pages

src/flashquest/eager/
└── llama_persistent_patch.py   # swap chain in _quest_duo_fused_with_lse

tests/
├── test_page_scores_int8.py    # NEW — EQ18-EQ20
├── test_select_pages_vectorized.py  # NEW — EQ21-EQ25
└── test_persistent_e2e.py      # extend equivalence assertion

docs/PHASES/
└── phase-6-notes.md            # NEW

docs/superpowers/plans/
└── 2026-05-01-phase-6-criticality-fix.md  # NEW (next step)

DOC.md                          # add Phase 6 section after results land
README.md                       # update Phase 5 decode row + add Phase 6 row
```

`compute_page_summary` and `page_scores` and `select_pages` (the Phase 5 implementations) stay — used by `tests/test_persistent_e2e.py` and the eager-mode reference path. They're correct, just not on the hot path.

## Open questions

None. The profile is unambiguous, the algebraic identity is exact, and the vectorised top-k is a standard pattern.
