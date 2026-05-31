# Phase 12 — EAGLE-3 Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add lossless-by-contract EAGLE-3 chain speculative decoding to the Quest sparse paged INT4 runtime, breaking the 8.41 tok/s @ 32k decode ceiling.

**Architecture:** A published EAGLE-3 draft head proposes a depth-N token chain; the target verifies all N in one `S_q>1` forward pass over the sparse INT4 cache (per-page K/V tile loaded once, reused across the N query rows = the bandwidth amortization); the longest greedily-matching prefix is committed. **A profile-first entry gate (Part A) measures verify cost and acceptance on the real target+workload before any build (Part B).**

**Tech Stack:** torch 2.5.1+cu121 · triton 3.1.0 · transformers 4.57.x · autoawq 0.2.9 · pytest · CUDA 12.5 · WSL2. Target `casperhansen/llama-3.2-3b-instruct-awq`; draft `thoughtworks/Llama-3.2-3B-Instruct-Eagle3`.

**Spec:** `docs/superpowers/specs/2026-05-31-phase-12-eagle3-speculative-decoding-design.md` (committed `adb1c2c`). Reuses the twice-codex-reviewed-but-unbuilt designs in the Phase 9 spec `docs/superpowers/specs/2026-05-08-phase-9-pld-greedy-design.md` §4.2–4.5.

---

> ## ⛔ HARD GATE
> **Do NOT start Part B (Task 4+) until Part A's gate verdict (after Task 3) is PROCEED.** Part A is a kill-capable entry gate, consistent with the project's profile-first discipline (Phases 8b / 9 / 10-CATS / 11-per-head all died at this kind of gate — `feedback_profile_before_speedup_specs.md`). If either sub-gate fails, write `docs/PHASES/phase-12-killed-by-gate.md` and STOP (pivot candidate: Lookahead decoding ~1.8×, draft-free).

## File Structure

| File | Responsibility | Part |
|---|---|---|
| `src/flashquest/kernel/sparse_int4_fwd_compact.py` | **Modify** — add `_sparse_attn_fwd_kernel_int4_compact_sq_gt_1` (`tl.dot`, `SQ_MAX=16`) + `flash_attn_sparse_int4_fwd_compact_sq` wrapper alongside the unchanged `S_q=1` kernel/wrapper. | A (kernel) / B (wrapper polish) |
| `benchmarks/phase12/task1a_verify_economics.py` | **Create** — `T_verify(q∈{2,4,8})` vs `T_decode(q=1)` on a 32k INT4 cache; writes `benchmarks/phase12/gate.md`. | A |
| `src/flashquest/specdec/__init__.py` | **Create** — package init. | A |
| `src/flashquest/specdec/eagle_draft.py` | **Create** — load EAGLE-3 head; `propose_chain(...)`. | A (loader) / B (chain) |
| `benchmarks/phase12/task1b_acceptance.py` | **Create** — mean accepted tokens/iter on real 32k traces; appends to `gate.md`. | A |
| `src/flashquest/cache/persistent_int4.py` | **Modify** — sandbox + two-phase commit API. | B |
| `src/flashquest/eager/selection.py` | **Modify** — `build_compact_union_selection`. | B |
| `src/flashquest/eager/llama_persistent_patch.py` | **Modify** — verify-mode forward arm + hidden-state tap for the drafter. | B |
| `src/flashquest/specdec/dispatcher.py` | **Create** — draft→verify→accept generation loop. | B |
| `src/flashquest/runtime/chat.py` | **Modify** — `--speculative`, `--n-draft`, `--draft-model`. | B |
| `tests/test_sparse_int4_compact_sq_gt_1.py`, `test_union_selection.py`, `test_persistent_int4_sandbox.py`, `test_eagle_draft.py`, `test_specdec_dispatcher.py`, `test_specdec_equivalence.py`, `test_chat_speculative_smoke.py` | **Create** — per-component tests. | A/B |
| `scripts/phase12_run_ruler_4k.py`, `scripts/phase12_bench_decode_32k.py` | **Create** — manual quality + speed gates. | B |

---

# PART A — ENTRY GATE (execute now)

## Task 1: Minimal `S_q>1` sparse INT4 verify kernel + parity test

This kernel is the real first slice (reused unchanged in Part B). It computes, for each of `S_q` query rows, full (non-causal) attention over the selected **completed** pages of an INT4 cache. Causality among draft positions and the partial-page tail is handled later by the existing `_merge_two_attentions` path, so this kernel has **no causal mask** between queries and completed pages.

**Files:**
- Modify: `src/flashquest/kernel/sparse_int4_fwd_compact.py`
- Test: `tests/test_sparse_int4_compact_sq_gt_1.py`

- [ ] **Step 1: Write the failing parity test.** The reference dequantizes INT4 the same way the existing `_flash_attn_sparse_int4_fwd_reference` does and softmaxes each query over all valid positions in the selected pages.

```python
# tests/test_sparse_int4_compact_sq_gt_1.py
import math
import torch
import pytest
from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4, dequantize_k_int4, dequantize_v_int4
from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact_sq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def _ref_sq(Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, sel, page_size):
    # Q: (B,Hq,Sq,D). sel: (B,Hq,BUCKET) int32, -1 sentinel. Non-causal over selected pages.
    B, Hq, Sq, D = Q.shape
    Kd = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)  # (B,Hkv,Scompleted,D)
    Vd = dequantize_v_int4(V_packed, V_scale, V_mn)
    Hkv = Kd.shape[1]; n_rep = Hq // Hkv
    Kd = Kd.repeat_interleave(n_rep, dim=1); Vd = Vd.repeat_interleave(n_rep, dim=1)
    sm = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q.float())
    for b in range(B):
        for h in range(Hq):
            pages = [int(p) for p in sel[b, h].tolist() if p >= 0]
            idx = torch.cat([torch.arange(p*page_size, (p+1)*page_size) for p in pages]) if pages else torch.empty(0, dtype=torch.long)
            idx = idx[idx < Kd.shape[2]]
            if idx.numel() == 0:
                continue
            k = Kd[b, h, idx].float(); v = Vd[b, h, idx].float()       # (n,D)
            qk = (Q[b, h].float() @ k.T) * sm                           # (Sq,n)
            p = torch.softmax(qk, dim=-1)
            O[b, h] = p @ v
    return O.to(Q.dtype)

@pytest.mark.parametrize("Sq", [2, 4, 8])
def test_compact_sq_gt_1_matches_reference(Sq):
    torch.manual_seed(0)
    B, Hq, Hkv, D, page_size, n_pages = 1, 8, 2, 128, 64, 6
    Scompleted = n_pages * page_size
    Q = torch.randn(B, Hq, Sq, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, Hkv, Scompleted, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, Hkv, Scompleted, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    # select pages {0,2,4} for every (b,h); BUCKET=4 with one -1 sentinel
    sel = torch.full((B, Hq, 4), -1, dtype=torch.int32, device="cuda")
    sel[..., 0] = 0; sel[..., 1] = 2; sel[..., 2] = 4
    O, lse = flash_attn_sparse_int4_fwd_compact_sq(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selected_page_ids=sel, page_size=page_size, return_lse=True,
    )
    ref = _ref_sq(Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, sel, page_size)
    assert O.shape == (B, Hq, Sq, D)
    assert torch.allclose(O.float(), ref.float(), atol=2e-2, rtol=2e-2), (O.float()-ref.float()).abs().max()
```

- [ ] **Step 2: Run it; verify it fails** with `ImportError`/`AttributeError` (wrapper not defined).
Run: `pytest tests/test_sparse_int4_compact_sq_gt_1.py -q`
Expected: FAIL (cannot import `flash_attn_sparse_int4_fwd_compact_sq`).

- [ ] **Step 3: Add the kernel** to `sparse_int4_fwd_compact.py` (after the existing `S_q=1` kernel; do not touch it). Derived from the existing compact kernel + spec §4 / Phase 9 §4.4. The parity test is the source of truth — iterate until it passes.

```python
@triton.jit
def _sparse_attn_fwd_kernel_int4_compact_sq_gt_1(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr, sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_seli,
    H_q, H_kv, S_kv, S_q,
    HEAD_DIM: tl.constexpr, HEAD_DIM_PACKED: tl.constexpr, PAGE_SIZE: tl.constexpr,
    BUCKET_MAX: tl.constexpr, SQ_MAX: tl.constexpr, WRITE_LSE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q; h_q = pid_bh % H_q
    n_rep = H_q // H_kv; h_kv = h_q // n_rep
    offs_n = tl.arange(0, PAGE_SIZE); offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED); offs_sq = tl.arange(0, SQ_MAX)
    sq_mask = offs_sq < S_q
    NEG_INF: tl.constexpr = float("-inf")
    q_ptrs = (Q_ptr + b*stride_qb + h_q*stride_qh
              + offs_sq[:, None]*stride_qs + offs_d[None, :]*stride_qd)
    q = tl.load(q_ptrs, mask=sq_mask[:, None], other=0.0)              # (SQ_MAX, HEAD_DIM)
    m_i = tl.full((SQ_MAX,), NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((SQ_MAX,), dtype=tl.float32)
    acc = tl.zeros((SQ_MAX, HEAD_DIM), dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504
    for i in range(0, BUCKET_MAX):
        sel_off = b*stride_selb + h_q*stride_selh + i*stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0; p_safe = tl.where(page_valid, p, 0)
        n_idx = p_safe*PAGE_SIZE + offs_n
        valid_kv = (n_idx < S_kv) & page_valid
        k_byte = tl.load(K_packed_ptr + b*stride_kb + h_kv*stride_kh
                         + n_idx[:, None]*stride_ks + offs_dp[None, :]*stride_kdp,
                         mask=valid_kv[:, None], other=0)
        k_int = tl.reshape(tl.join((k_byte & 0xF).to(tl.uint8), ((k_byte >> 4) & 0xF).to(tl.uint8)),
                           (PAGE_SIZE, HEAD_DIM))
        k_scale = tl.load(K_scale_ptr + b*stride_ksb + h_kv*stride_ksh + p_safe*stride_ksp + offs_d*stride_ksd).to(tl.float32)
        k_mn = tl.load(K_mn_ptr + b*stride_kmb + h_kv*stride_kmh + p_safe*stride_kmp + offs_d*stride_kmd).to(tl.float32)
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]    # (PAGE_SIZE, HEAD_DIM)
        qk = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee")  # (SQ_MAX, PAGE_SIZE)
        qk = tl.where(sq_mask[:, None] & valid_kv[None, :], qk, NEG_INF)
        qk_max = tl.max(qk * qk_scale, axis=1)
        m_ij = tl.maximum(m_i, qk_max)
        m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
        p_sm = tl.math.exp2(qk * qk_scale - m_ij_safe[:, None])
        p_sm = tl.where(qk == NEG_INF, 0.0, p_sm)
        alpha = tl.where(m_i == NEG_INF, 0.0, tl.math.exp2(m_i - m_ij_safe))
        l_i = l_i * alpha + tl.sum(p_sm, axis=1)
        acc = acc * alpha[:, None]
        v_byte = tl.load(V_packed_ptr + b*stride_vb + h_kv*stride_vh
                         + n_idx[:, None]*stride_vs + offs_dp[None, :]*stride_vdp,
                         mask=valid_kv[:, None], other=0)
        v_int = tl.reshape(tl.join((v_byte & 0xF).to(tl.uint8), ((v_byte >> 4) & 0xF).to(tl.uint8)),
                           (PAGE_SIZE, HEAD_DIM))
        v_scale = tl.load(V_scale_ptr + b*stride_vsb + h_kv*stride_vsh + n_idx*stride_vss, mask=valid_kv, other=0.0).to(tl.float32)
        v_mn = tl.load(V_mn_ptr + b*stride_vmb + h_kv*stride_vmh + n_idx*stride_vms, mask=valid_kv, other=0.0).to(tl.float32)
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]
        acc += tl.dot(p_sm.to(tl.float32), v, input_precision="ieee")  # (SQ_MAX, HEAD_DIM)
        m_i = m_ij
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]
    o_ptrs = O_ptr + b*stride_ob + h_q*stride_oh + offs_sq[:, None]*stride_os + offs_d[None, :]*stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=sq_mask[:, None])
    if WRITE_LSE:
        lse_val = tl.where(l_i == 0.0, NEG_INF, (m_i + tl.math.log2(safe_l)) * 0.69314718)
        tl.store(L_ptr + b*stride_lb + h_q*stride_lh + offs_sq*stride_ls, lse_val, mask=sq_mask)
```

- [ ] **Step 4: Add the wrapper** (same file). `SQ_MAX=16` is forced (Triton 3.1 `tl.dot` requires each operand dim ≥16; mask the unused rows).

```python
def flash_attn_sparse_int4_fwd_compact_sq(
    Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
    *, selected_page_ids, page_size=64, sm_scale=None, return_lse=True,
):
    """S_q>1 verify variant of flash_attn_sparse_int4_fwd_compact. Non-causal
    over selected completed pages. selected_page_ids: (B, H_q, BUCKET_MAX) int32."""
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_packed.dtype == torch.uint8 and V_packed.dtype == torch.uint8
    assert selected_page_ids.dtype == torch.int32 and selected_page_ids.dim() == 3
    B, H_q, S_q, D = Q.shape
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")
    SQ_MAX = 16
    if S_q > SQ_MAX:
        raise NotImplementedError(f"S_q={S_q} > SQ_MAX={SQ_MAX}")
    Bk, H_kv, S_kv, Dp = K_packed.shape
    assert B == Bk and Dp == D // 2 and H_q % H_kv == 0
    _, _, BUCKET_MAX = selected_page_ids.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    O = torch.zeros_like(Q)
    L = torch.empty(B, H_q, S_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    sl_b, sl_h, sl_s = (L.stride() if L is not None else (0, 0, 0))
    sel = selected_page_ids.contiguous()
    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_int4_compact_sq_gt_1[grid](
        Q, K_packed, V_packed, O, L_ptr, K_scale, K_mn, V_scale, V_mn, sel, sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sl_b, sl_h, sl_s,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel.stride(0), sel.stride(1), sel.stride(2),
        H_q, H_kv, S_kv, S_q,
        HEAD_DIM=D, HEAD_DIM_PACKED=D // 2, PAGE_SIZE=page_size,
        BUCKET_MAX=BUCKET_MAX, SQ_MAX=SQ_MAX, WRITE_LSE=bool(return_lse),
        num_warps=4, num_stages=2,
    )
    return O, L
```

- [ ] **Step 5: Run the test to PASS.** Run: `pytest tests/test_sparse_int4_compact_sq_gt_1.py -q` — Expected: 3 passed. If a `tl.dot`/register-spill compile error appears, that is itself a **Gate 1a signal** — record it in `gate.md` and raise it before forcing a workaround.
- [ ] **Step 6: Commit.** `git add src/flashquest/kernel/sparse_int4_fwd_compact.py tests/test_sparse_int4_compact_sq_gt_1.py && git commit -m "phase 12 task 1: S_q>1 sparse INT4 verify kernel + parity test (gate slice)"`

## Task 2: Gate 1a — verifier economics microbench

**Files:** Create `benchmarks/phase12/task1a_verify_economics.py`

- [ ] **Step 1: Write the bench.** Builds a 32k INT4 cache (random), selects a realistic page bucket, times the existing `S_q=1` compact kernel vs the new `S_q>1` kernel across all heads/layers-worth of work for one layer (×28 is linear), using cuda.Event. No model load needed.

```python
# benchmarks/phase12/task1a_verify_economics.py
import math, json, statistics, torch
from pathlib import Path
from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
from flashquest.kernel.sparse_int4_fwd_compact import (
    flash_attn_sparse_int4_fwd_compact, flash_attn_sparse_int4_fwd_compact_sq)

def _time(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    ts = []
    for _ in range(iters):
        s.record(); fn(); e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e))
    return statistics.median(ts)

def main():
    torch.manual_seed(0); dev = "cuda"
    B, Hq, Hkv, D, page_size = 1, 24, 8, 128, 64          # Llama-3.2-3B shape
    ctx = 32768; n_pages = ctx // page_size
    retention = 0.20
    k_pages = math.ceil(retention * n_pages) + 4 + 2       # +sinks +window
    K = torch.randn(B, Hkv, ctx, D, dtype=torch.bfloat16, device=dev)
    V = torch.randn(B, Hkv, ctx, D, dtype=torch.bfloat16, device=dev)
    Kp, Ks, Kmn = quantize_k_int4(K, page_size=page_size); Vp, Vs, Vmn = quantize_v_int4(V)
    sel1 = torch.full((B, Hq, k_pages), -1, dtype=torch.int32, device=dev)
    sel1[..., :k_pages] = torch.arange(k_pages, device=dev).int()
    Q1 = torch.randn(B, Hq, 1, D, dtype=torch.bfloat16, device=dev)
    t1 = _time(lambda: flash_attn_sparse_int4_fwd_compact(
        Q1, Kp, Ks, Kmn, Vp, Vs, Vmn, selected_page_ids=sel1, page_size=page_size))
    out = {"T_decode_q1_ms": t1, "ctx": ctx, "retention": retention, "k_pages": k_pages}
    for q in (2, 4, 8):
        Qn = torch.randn(B, Hq, q, D, dtype=torch.bfloat16, device=dev)
        tn = _time(lambda: flash_attn_sparse_int4_fwd_compact_sq(
            Qn, Kp, Ks, Kmn, Vp, Vs, Vmn, selected_page_ids=sel1, page_size=page_size))
        out[f"T_verify_q{q}_ms"] = tn; out[f"ratio_q{q}"] = tn / t1
    out["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
    out["PASS_1a"] = (out["ratio_q4"] <= 1.6) or (out["ratio_q8"] <= 2.4)
    Path("benchmarks/phase12").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/phase12/task1a.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it** (one GPU job, `nice -n 19`, no parallel torch — `feedback_avoid_wsl_lag`).
Run: `nice -n 19 python benchmarks/phase12/task1a_verify_economics.py`
Expected: prints JSON with `ratio_q4`, `ratio_q8`, `PASS_1a`.

- [ ] **Step 3: Record the verdict** in `benchmarks/phase12/gate.md` (create it): the ratios, peak VRAM, and PASS/FAIL. **Gate 1a passes iff `ratio_q4 ≤ 1.6` OR `ratio_q8 ≤ 2.4`.** If it fails (verify scales ~linearly → likely register spill at `SQ_MAX=16`), STOP per the HARD GATE.
- [ ] **Step 4: Commit.** `git add benchmarks/phase12/ && git commit -m "phase 12 task 2: gate 1a verifier-economics microbench + verdict"`

## Task 3: Gate 1b — EAGLE-3 head load + real-trace acceptance

**Files:** Create `src/flashquest/specdec/__init__.py`, `src/flashquest/specdec/eagle_draft.py`, `benchmarks/phase12/task1b_acceptance.py`

- [ ] **Step 1: Task 0 read (no code).** Open the `thoughtworks/Llama-3.2-3B-Instruct-Eagle3` model card + `config.json`: record the draft architecture (single decoder layer), the **3 target layers** whose hidden states it fuses, the `d2t`/`t2d` vocab-map tensors, and dtype. Note them as comments at the top of `eagle_draft.py`. (Crib the forward from `vendor/eagle` `cnets.py` if vendored; else `scripts/vendor_clone.sh`.)
- [ ] **Step 2: Write the loader** (`eagle_draft.py`): `load_eagle3_draft(model_id, device, dtype) -> EagleDraft` that loads the safetensors and exposes `forward(fused_hidden, input_ids) -> logits_over_full_vocab` (apply `t2d`/`d2t` mapping). Keep `propose_chain` as a thin greedy loop (full chain logic finalized in Task 8; here a depth-N greedy chain over the head's own dense KV is enough to measure acceptance).
- [ ] **Step 3: Write the acceptance probe** (`task1b_acceptance.py`): load `casperhansen/llama-3.2-3b-instruct-awq` + patch with the existing persistent INT4 path; load the EAGLE-3 head; on PG-summarize and RULER-NIAH-single/multivalue prompts at 32k, run plain greedy and, at each step, ask the draft for a depth-4 chain, then count how many chain tokens equal the target's greedy continuation (no kernel changes — verification here is plain sequential target decode). Report mean accepted/iter + histogram + peak VRAM with the head resident.

```python
# core of task1b_acceptance.py
acc_counts = []
for prompt in real_traces:                 # PG-summarize + RULER single/multivalue @ 32k
    ids = prefill(model, tok, prompt)
    for _ in range(128):                    # 128 verify iterations
        chain = draft.propose_chain(fused_hidden, last_token, n_draft=4)   # (4,)
        m = 0
        for i in range(4):
            tgt = greedy_next(model)        # one target decode step (advances cache)
            if i < 4 and tgt.item() == chain[i].item(): m += 1
            else: break
        acc_counts.append(1 + m)            # free token + accepted drafts
mean_acc = sum(acc_counts) / len(acc_counts)
out = {"mean_accepted": mean_acc, "PASS_1b": mean_acc >= 2.2,
       "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
       "fits": torch.cuda.max_memory_allocated()/2**20 <= 6144}
```

- [ ] **Step 4: Run it** (single GPU job, `nice -n 19`, run-in-background + Monitor for the long 32k traces).
Run: `nice -n 19 python benchmarks/phase12/task1b_acceptance.py`
Expected: JSON with `mean_accepted`, `PASS_1b`, `peak_mib`, `fits`.
- [ ] **Step 5: Append the verdict** to `gate.md`. **Gate 1b passes iff `mean_accepted ≥ 2.2` AND it fits (≤ ~6.0 GiB; else INT8-quantize the head and re-measure).**
- [ ] **Step 6: Commit.** `git add src/flashquest/specdec/ benchmarks/phase12/ && git commit -m "phase 12 task 3: gate 1b EAGLE-3 head load + real-trace acceptance"`

## ⛔ GATE DECISION (after Task 3)

Compute projected net speedup `≈ mean_accepted / ratio(chosen q)`.
- **PROCEED to Part B** iff Gate 1a passed **and** Gate 1b passed **and** projected ≥ **1.3×**.
- Else: write `docs/PHASES/phase-12-killed-by-gate.md` (numbers + which sub-gate failed + the ~1.41× margin analysis from spec §6) and STOP. Pivot candidate: Lookahead decoding (~1.8×, draft-free, exact).

---

# PART B — BUILD (execute only on PROCEED)

## Task 4: Cache sandbox + two-phase commit API

**Files:** Modify `src/flashquest/cache/persistent_int4.py`; Test `tests/test_persistent_int4_sandbox.py`

- [ ] **Step 1: Failing test** — `add_draft` stages without advancing `_seen_tokens`; `commit_draft_all_layers(M+1)` advances by exactly `M+1`; the page-boundary guard raises on a crossing commit.

```python
def test_sandbox_commit_advances_by_accept_count():
    cache = _fresh_int4_cache(num_layers=1, max_seq_len=4096)   # helper builds a 1-layer cache
    cache.update_quantized(_rand_kv(64), *_rand_v(64)[1:], layer_idx=0)  # seen=64 (one full page)
    k, v = _rand_kv(4)                                           # 4 draft positions
    cache.add_draft(k, v, layer_idx=0)
    assert cache._seen_tokens[0] == 64                           # not advanced
    cache.commit_draft_all_layers(3)
    assert cache._seen_tokens[0] == 67
def test_page_boundary_guard_raises():
    cache = _fresh_int4_cache(num_layers=1, max_seq_len=4096)
    cache.update_quantized(*_rand_kvv(62), layer_idx=0)          # seen=62, page_size=64
    cache.add_draft(*_rand_kv4(), layer_idx=0)
    with pytest.raises(RuntimeError, match="page boundary"):
        cache.commit_draft_all_layers(4)                         # 62->66 crosses 64
```

- [ ] **Step 2: Run → FAIL** (`add_draft` undefined). `pytest tests/test_persistent_int4_sandbox.py -q`
- [ ] **Step 3: Implement** (add to `PersistentInt4KVCache`, lifting Phase 9 §4.2, using the existing `update_quantized` for commit):

```python
MAX_DRAFT = 8
def _ensure_sandbox(self):
    if hasattr(self, "K_sandbox"): return
    shp = (self.num_layers, self.batch_size, self.num_kv_heads, self.MAX_DRAFT, self.head_dim)
    self.K_sandbox = torch.zeros(shp, dtype=torch.bfloat16, device=self.K_partial.device)
    self.V_sandbox = torch.zeros(shp, dtype=torch.bfloat16, device=self.K_partial.device)
    self._sandbox_count = [0] * self.num_layers
def add_draft(self, K_new, V_new, layer_idx):
    self._ensure_sandbox()
    S = K_new.shape[2]
    if S > self.MAX_DRAFT: raise ValueError(f"S_new={S} > MAX_DRAFT={self.MAX_DRAFT}")
    self.K_sandbox[layer_idx, :, :, :S, :] = K_new
    self.V_sandbox[layer_idx, :, :, :S, :] = V_new
    self._sandbox_count[layer_idx] = S
def get_views_with_sandbox(self, layer_idx):
    v = self.get_views(layer_idx); s = self._sandbox_count[layer_idx]
    v["K_sandbox"] = self.K_sandbox[layer_idx, :, :, :s, :]
    v["V_sandbox"] = self.V_sandbox[layer_idx, :, :, :s, :]
    v["sandbox_count"] = s
    return v
def preflight_commit(self, accept_count, layer_idx):
    if accept_count < 0: raise ValueError("negative accept_count")
    if accept_count > self._sandbox_count[layer_idx]: raise ValueError("accept_count > sandbox_count")
    seen = self._seen_tokens[layer_idx]
    if accept_count > 0 and (seen + accept_count) // self.page_size != seen // self.page_size:
        raise RuntimeError(f"commit_draft crosses page boundary (seen={seen})")
def commit_draft(self, accept_count, layer_idx):
    if accept_count == 0: self._sandbox_count[layer_idx] = 0; return
    self.update_quantized(self.K_sandbox[layer_idx, :, :, :accept_count, :],
                          self.V_sandbox[layer_idx, :, :, :accept_count, :], layer_idx)
    self._sandbox_count[layer_idx] = 0
def commit_draft_all_layers(self, accept_count):
    for li in range(self.num_layers): self.preflight_commit(accept_count, li)
    for li in range(self.num_layers): self.commit_draft(accept_count, li)
```

- [ ] **Step 4: Run → PASS.** **Step 5: Commit** `phase 12 task 4: persistent INT4 sandbox + two-phase commit API`.

## Task 5: Score-prioritized UNION selection

**Files:** Modify `src/flashquest/eager/selection.py`; Test `tests/test_union_selection.py`

- [ ] **Step 1: Write the failing test.**

```python
# tests/test_union_selection.py
import torch
from flashquest.eager.selection import build_compact_union_selection

def test_union_forces_sinks_and_window_and_includes_members():
    B, Hq, Sq, P = 1, 1, 2, 10
    sel = torch.zeros(B, Hq, Sq, P, dtype=torch.bool)
    sel[0, 0, 0, 5] = True; sel[0, 0, 1, 6] = True          # union {5, 6}
    scores = torch.zeros(B, Hq, Sq, P)
    scores[0, 0, 0, 5] = 9.0; scores[0, 0, 1, 6] = 8.0
    out = build_compact_union_selection(
        sel, scores, num_sinks=1, window_pages=1,
        completed_len=P * 64, page_size=64, BUCKET_MAX_UNION=4)
    got = {int(x) for x in out[0, 0].tolist() if x >= 0}
    assert 0 in got and (P - 1) in got and 5 in got and 6 in got
    assert out.shape == (B, Hq, 4) and out.dtype == torch.int32

def test_union_overflow_keeps_highest_score():
    B, Hq, Sq, P = 1, 1, 1, 10
    sel = torch.zeros(B, Hq, Sq, P, dtype=torch.bool)
    sel[0, 0, 0, [2, 3, 4, 7]] = True
    scores = torch.zeros(B, Hq, Sq, P)
    scores[0, 0, 0, [2, 3, 4, 7]] = torch.tensor([1.0, 5.0, 2.0, 9.0])
    out = build_compact_union_selection(
        sel, scores, num_sinks=0, window_pages=0,
        completed_len=P * 64, page_size=64, BUCKET_MAX_UNION=2)
    assert {int(x) for x in out[0, 0].tolist() if x >= 0} == {3, 7}
```

- [ ] **Step 2: Run → FAIL** (`build_compact_union_selection` undefined). `pytest tests/test_union_selection.py -q`
- [ ] **Step 3: Implement** in `selection.py` (spec / Phase 9 §4.4; `finfo` sentinels instead of `±inf` for safe `topk`):

```python
def build_compact_union_selection(
    sel_per_q, scores, *, num_sinks, window_pages, completed_len, page_size, BUCKET_MAX_UNION,
):
    """Score-prioritized UNION over the S_q axis, sinks+window force-included.
    sel_per_q: bool (B,H_q,S_q,P); scores: float (B,H_q,S_q,P).
    Returns int32 (B,H_q,BUCKET_MAX_UNION) with -1 sentinels. GPU-resident, no .item()."""
    if sel_per_q.dtype != torch.bool:
        raise ValueError(f"sel_per_q must be bool, got {sel_per_q.dtype}")
    union = sel_per_q.any(dim=2)                       # (B,H_q,P)
    max_scores = scores.amax(dim=2).float()            # (B,H_q,P)
    P = union.shape[-1]
    n_pages = completed_len // page_size
    forced = torch.zeros_like(union)
    ns = min(num_sinks, P)
    if ns > 0:
        forced[..., :ns] = True
    if n_pages > window_pages:
        forced[..., n_pages - window_pages:n_pages] = True
    elif n_pages > 0:
        forced[..., :n_pages] = True
    NEG = torch.finfo(torch.float32).min
    POS = torch.finfo(torch.float32).max
    priority = torch.where(union | forced, max_scores, torch.full_like(max_scores, NEG))
    priority = torch.where(forced, torch.full_like(priority, POS), priority)
    k = min(BUCKET_MAX_UNION, P)
    top = priority.topk(k, dim=-1).indices             # (B,H_q,k)
    top_pri = priority.gather(-1, top)
    out_k = torch.where(top_pri <= NEG, torch.full_like(top, -1), top).to(torch.int32)
    if k == BUCKET_MAX_UNION:
        return out_k.contiguous()
    out = torch.full((*out_k.shape[:-1], BUCKET_MAX_UNION), -1,
                     dtype=torch.int32, device=sel_per_q.device)
    out[..., :k] = out_k
    return out.contiguous()
```

- [ ] **Step 4: Run → PASS.** `pytest tests/test_union_selection.py -q`
- [ ] **Step 5: Commit** `phase 12 task 5: score-prioritized UNION page selection`.

## Task 6: Verify-mode forward branch + hidden-state tap

**Files:** Modify `src/flashquest/eager/llama_persistent_patch.py`; Test covered by Task 9 dispatcher + Task 10 equivalence.

- [ ] **Step 1:** Add `_verify_active` attribute read in `forward`. New third arm: `if S_q > 1 and getattr(self, "_verify_active", False):` → `cache.add_draft(k, v, self.layer_idx)`; `views = cache.get_views_with_sandbox(...)`; compute per-Q `scores`/`sel` (reuse the decode-branch code), `sel_union = build_compact_union_selection(...)`, `O_sparse, lse_sparse = flash_attn_sparse_int4_fwd_compact_sq(q, views..., selected_page_ids=sel_union)`; dense over `cat([K_partial, K_sandbox])` with offset-causal mask + per-query LSE; merge via an `S_q`-generalized `_merge_two_attentions` (broadcast over the `S_q` axis — verify the existing one already broadcasts since it operates on `(B,H_q,·)` shapes; if not, add `_merge_two_attentions_sq`). The existing `S_q>1` prefill arm stays as the `not _verify_active` case.
- [ ] **Step 2:** Add `bf16_dense_attn_offset_causal_with_lse(Q, K, V, q_offset)` (spec / Phase 9 §4.3) for the partial-tail‖sandbox region.
- [ ] **Step 3:** Add module helper `set_verify_active(model, flag)` iterating patched `LlamaAttention`.
- [ ] **Step 4: Commit** `phase 12 task 6: verify-mode forward arm + offset-causal tail`.

## Task 7: EAGLE-3 `propose_chain` + draft KV

**Files:** Modify `src/flashquest/specdec/eagle_draft.py`; Test `tests/test_eagle_draft.py`

- [ ] **Step 1: Failing test** — `propose_chain(fused_hidden, last_token, n_draft=4)` returns a `(4,)` int64 chain; draft KV resets between sequences; `d2t`/`t2d` round-trip identity.
- [ ] **Step 2: Run → FAIL. Step 3: Implement** the autoregressive depth-N chain over the draft layer's own dense KV (HF `DynamicCache` for the single layer), mapping draft-vocab logits → full vocab via `d2t`. **Step 4: Run → PASS. Step 5: Commit** `phase 12 task 7: EAGLE-3 propose_chain + draft KV`.

## Task 8: Spec-decode dispatcher (draft→verify→accept)

**Files:** Create `src/flashquest/specdec/dispatcher.py`; Test `tests/test_specdec_dispatcher.py`

- [ ] **Step 1: Failing test** — feed synthetic target logits + a known chain; assert the walk-and-accept count, the free-token hold, the `commit_draft_all_layers(M+1)` call, and `committed_history` bookkeeping (Phase 9 §4.5 semantics).
- [ ] **Step 2: Run → FAIL. Step 3: Implement** `make_quest_specdec(model, cache, draft, *, n_draft=4, page_size=64)` returning `init(prompt_ids)` / `step()`. Per step: get fused hidden from the last target pass → `draft.propose_chain` → admissibility (page-boundary guard) → `set_verify_active(True)`; `model(verify_input)`; `set_verify_active(False)` in `finally` → walk-and-accept → `commit_draft_all_layers(M+1)` → update state. Fall back to single decode when inadmissible. **Step 4: Run → PASS. Step 5: Commit** `phase 12 task 8: spec-decode dispatcher`.

## Task 9: End-to-end losslessness equivalence

**Files:** Test `tests/test_specdec_equivalence.py` (marker `slow`)

- [ ] **Step 1:** On Llama-3.2-3B-AWQ at a small fixed ctx + seed, assert the `--speculative` token stream matches the non-spec greedy stream with ≥99% argmax agreement over ≥200 tokens (spec §5 contract; not bit-identical — UNION selection). **Step 2: Run (slow).** **Step 3: Commit** `phase 12 task 9: losslessness equivalence test`.

## Task 10: `flashquest chat --speculative`

**Files:** Modify `src/flashquest/runtime/chat.py`; Test `tests/test_chat_speculative_smoke.py` (`slow`)

- [ ] **Step 1:** Add `--speculative` (store_true), `--n-draft` (int, default 4), `--draft-model` (default `thoughtworks/Llama-3.2-3B-Instruct-Eagle3`). When set: build the draft via `load_eagle3_draft`, wrap the generate loop with `make_quest_specdec` instead of HF `generate`. Keep `--no-patch` and all existing flags working. **Step 2:** Smoke test streams coherent output. **Step 3: Commit** `phase 12 task 10: chat --speculative flag`.

## Task 11: Manual quality + speed gates

**Files:** Create `scripts/phase12_run_ruler_4k.py`, `scripts/phase12_bench_decode_32k.py`

- [ ] **Step 1:** `phase12_run_ruler_4k.py` = `scripts/phase7_run_ruler_4k_turbo.py` template with `--speculative` on; assert single 100 / multikey 100 / multivalue ≥95 (n=20, ret 0.20). `phase12_bench_decode_32k.py` measures clean 32k decode tok/s with `--speculative` vs baseline 8.41.
- [ ] **Step 2: Run both** (single GPU job each, `nice -n 19`, run-in-background + Monitor).
- [ ] **Step 3:** Record results in `docs/PHASES/phase-12-notes.md`: decode ×, RULER, peak VRAM, mean accepted, the §7 verdict (≥1.5× → default; 1.2–1.5× → opt-in; <1.2× → keep but off by default). Update `DOC.md` (new phase row) + `CHANGELOG.md`. Tag `phase-12` if it clears.
- [ ] **Step 4: Commit** `phase 12 task 11: quality + speed gates + DOC/CHANGELOG`.

---

## Self-Review notes

- **Spec coverage:** every spec §4 component maps to a task (kernel→T1, sandbox→T4, UNION→T5, forward arm→T6, draft loader/chain→T3/T7, dispatcher→T8, chat→T10); both §6 gates→T2/T3 + GATE DECISION; §7 bands→T11; §9 tests all present.
- **One task without a standalone unit test:** Task 6 (verify-mode forward arm) is integration glue over a patched model + cache; it is covered end-to-end by Task 8 (dispatcher) and Task 9 (equivalence) rather than a brittle isolated harness — a deliberate choice.
- **Part B code altitude:** Tasks 1, 4, 5 paste complete code (the novel kernel + cache + selection primitives). Tasks 6–8 give precise per-step wiring and reference the committed spec §4.3/§4.5 (and Phase 9's reviewed designs) for the forward-arm / dispatcher bodies — those are gated behind the entry verdict and intentionally not over-specified before the gate may kill them.
- **Type consistency verified:** `flash_attn_sparse_int4_fwd_compact_sq`, `build_compact_union_selection`, `add_draft`/`commit_draft_all_layers`, `propose_chain`, `make_quest_specdec`, `set_verify_active`, `SQ_MAX=16` — consistent across definer and caller tasks. Task 2's microbench calls match the real `flash_attn_sparse_int4_fwd_compact` signature in current code.
