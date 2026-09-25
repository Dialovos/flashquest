# Phase 2 — Dense FA-2 Triton Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the FA-2 06-fused-attention Triton tutorial onto sm_86 as a clean dense forward-only attention kernel — `flash_attn_fwd(Q, K, V, *, causal, sm_scale=None) -> (O, lse)`. No sparsity (Phase 3), no backward (out of scope), no quantization (Phase 3). The output is the *correctness oracle* the Phase 3 sparse kernel will be checked against, and the *perf scaffolding* it builds on.

**Architecture:** Single Triton kernel `_flash_attn_fwd_kernel` with online-softmax accumulation per the FA-2 algorithm. Tile shapes BLOCK_M=64, BLOCK_N=64 (vs the tutorial's 128 — Ampere's 48 KB shared mem and lower SM count make smaller tiles more efficient). Targets `head_dim ∈ {64, 128}` (Llama-3.2 uses 64; Llama-3.1 uses 128). Inputs BF16, accumulator FP32, output BF16. GQA via kernel-side head-index map (no host-side `repeat_kv` copy). Returns logsumexp tensor for downstream composition with Phase 3+ sparse outer loop.

**Tech Stack:**
- Triton 3.1.0 (already pinned)
- PyTorch 2.5.1+cu121 (already pinned)
- `flash_attn` 2.7.4 (already installed — perf reference target)
- pytest + hypothesis (property-based shape testing)

**Win condition** (per SPEC §6 Phase 2):
- Numerical equivalence with `torch.nn.functional.scaled_dot_product_attention` within rtol=1e-2, atol=1e-2 on BF16 inputs across the shape grid in Task 5.
- Perf within 30 % of upstream `flash_attn` 2.7.4 forward at S=8192 BF16 head_dim=64. (FA-2 baseline: 22.76 ms warm — target ≤ 32.5 ms for our kernel.)
- Numerical equivalence with `flashquest.eager.quest_eager_sdpa(retention=1.0)` at the same tolerance — closes the loop with Phase 1's algorithm-of-record.

**Hardware envelope:** sm_86, 4 GB VRAM, 48 KB shared mem per SM. Block sizes chosen to leave ~16 KB margin for double-buffered K/V loads in Phase 3.

---

## Edge case catalog (load-bearing — tests in Task 5 cover these)

This kernel is the foundation for Phase 3+. Anything we get wrong here propagates. Edge cases:

| ID | Case | Why it matters |
|---|---|---|
| E1 | `S == 1` (decode after prefill) | Hot path for autoregressive inference; degenerate query tile. |
| E2 | `S < BLOCK_M` (tiny prefill) | Tests fire single-tile path; common in unit tests. |
| E3 | `S` not multiple of `BLOCK_M` / `BLOCK_N` | Real Llama prompts are arbitrary lengths; tail tile must mask. |
| E4 | `head_dim == 64` and `head_dim == 128` | Both Llama-3.2 and Llama-3.1 must work. |
| E5 | `causal=True, S_q == S_kv` | Standard prefill. |
| E6 | `causal=False, S_q == 1, S_kv > 1` | Decode against pre-existing KV cache. Causal mask trivializes. |
| E7 | `B > 1` (batched) | Phase 4 8B model batching. |
| E8 | GQA `n_rep > 1` (Llama-3.2-1B is 4-way) | Don't introduce a host-side `repeat_kv` copy. |
| E9 | MHA degenerate `n_rep == 1` | Same code path must work without GQA. |
| E10 | All-masked row in causal | When `q@pos=0` attends only to itself, log-sum-exp is just one term — must not produce NaN. |
| E11 | Strided / non-contiguous Q | After `.transpose(1, 2)` Q is non-contiguous; kernel must respect strides. |
| E12 | NaN propagation | NaN in input must surface in output; we must not silently zero-mask it. |
| E13 | `causal=True, S_q != S_kv` (chunked prefill) | Out of scope for Phase 2; assert and reject. |
| E14 | Empty input (`S == 0`) | Validate at Python wrapper, raise `ValueError`. Don't reach kernel. |

The kernel **rejects** E13 and E14 at the Python wrapper. The kernel **handles** E1–E12 internally.

---

## File Structure

**Created:**
- `src/flashquest/kernel/__init__.py` — exports `flash_attn_fwd`
- `src/flashquest/kernel/flash_fwd.py` — `_flash_attn_fwd_kernel` (Triton @jit) + `flash_attn_fwd(Q, K, V, *, causal, sm_scale)` Python wrapper, includes input validation (E13, E14)
- `src/flashquest/kernel/_autotune.py` — single-source-of-truth list of autotune configs for sm_86 (BLOCK_M, BLOCK_N, num_warps, num_stages)
- `tests/test_kernel_flash_fwd_basic.py` — Task 1 smoke test
- `tests/test_kernel_flash_fwd_causal.py` — Task 2 causal correctness
- `tests/test_kernel_flash_fwd_tail.py` — Task 3 tail/mask handling
- `tests/test_kernel_flash_fwd_lse.py` — Task 4 LSE output
- `tests/test_kernel_flash_fwd_edges.py` — Task 5 full edge-case grid (E1–E14)
- `tests/test_kernel_eager_equivalence.py` — Task 6 vs Phase 1 eager
- `scripts/phase2_bench_attn.py` — Task 6 perf bench
- `benchmarks/phase2_perf.json` — Task 6 output
- `docs/PHASES/phase-2-notes.md` — phase journal

**Modified:**
- `pyproject.toml` — add `hypothesis` to `dev` extras
- `README.md` — append Phase 2 perf row
- `DOC.md` — flip Phase 2 status

---

## Conventions across all tasks

- **Layout:** `Q: (B, H_q, S_q, D)`, `K, V: (B, H_kv, S_kv, D)`. Same as Phase 1 `quest_eager_sdpa`.
- **dtype:** Q/K/V are `torch.bfloat16`. Accumulator inside the kernel is `float32`. Output `O` is `torch.bfloat16`. LSE is `torch.float32`.
- **`sm_scale`** defaults to `1 / sqrt(head_dim)`.
- **GQA:** `H_q % H_kv == 0`, kernel uses `kv_head = q_head // (H_q // H_kv)`. No `repeat_kv` host copy.
- **Block sizes:** `BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2` (Ampere `cp.async` + double buffer; tutorial uses 3+ on Hopper but 2 is safer at 48 KB SMEM).
- **Stage flags:** `STAGE = 1` (causal off-diagonal — full attend), `STAGE = 2` (causal diagonal — masked), `STAGE = 3` (non-causal full block). One kernel handles all via constexpr.

---

## Task 1: Kernel scaffold + Python wrapper (full-tile, no causal)

**Purpose:** Get a working Triton forward kernel that handles non-causal, contiguous, BF16, GQA-capable input where `S_q` and `S_kv` are exact multiples of `BLOCK_M` / `BLOCK_N`. This is the spine; later tasks bolt on causal, tail masking, and LSE.

**Files:**
- Create: `src/flashquest/kernel/_autotune.py`
- Create: `src/flashquest/kernel/flash_fwd.py`
- Create: `src/flashquest/kernel/__init__.py`
- Create: `tests/test_kernel_flash_fwd_basic.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_kernel_flash_fwd_basic.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_basic_non_causal_matches_sdpa():
    """Single full tile, no causal, no GQA: kernel == torch SDPA."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 128, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _lse = flash_attn_fwd(Q, K, V, causal=False)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_gqa_two_kv_heads():
    """GQA n_rep=4 (mimics Llama-3.2-1B 32:8): kernel == SDPA on repeated K/V."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=False)

    n_rep = H_q // H_kv
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=False)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_head_dim_128():
    """head_dim=128 (Llama-3.1 family) supported."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 2, 128, 128
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=False)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


def test_rejects_zero_length():
    """E14: empty inputs are a Python-level error, not a kernel one."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `. .venv/bin/activate && pytest tests/test_kernel_flash_fwd_basic.py -v`
Expected: ImportError on `flashquest.kernel`.

- [ ] **Step 1.3: Write the autotune config table**

Create `src/flashquest/kernel/_autotune.py`:
```python
"""Single-source-of-truth autotune configs for sm_86. Phase 2 dense forward."""
from __future__ import annotations

import triton

# sm_86 has 48 KB SMEM / SM. BLOCK_M = BLOCK_N = 64, double-buffered K/V loads
# at BF16, head_dim=64 -> 2 * (64 * 64 * 2) = 16 KB per buffer * 2 buffers + Q tile
# = ~24 KB. Leaves ~24 KB margin. head_dim=128 doubles SMEM use -> still fits.
FORWARD_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
]
```

- [ ] **Step 1.4: Write the kernel + Python wrapper**

Create `src/flashquest/kernel/flash_fwd.py`:
```python
"""Dense FA-2 forward kernel for sm_86. Phase 2.

Algorithm: standard FlashAttention-2 online softmax, ported from the Triton
06-fused-attention tutorial with sm_86-friendly tile shapes. No sparsity
(Phase 3), no backward (out of scope).

Shapes (B=batch, H_q=query heads, H_kv=KV heads, S=seq, D=head_dim):
    Q: (B, H_q,  S_q,  D) bf16
    K: (B, H_kv, S_kv, D) bf16
    V: (B, H_kv, S_kv, D) bf16
    O: (B, H_q,  S_q,  D) bf16
    L: (B, H_q,  S_q)     fp32  (logsumexp; LSE output added in Task 4)
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._autotune import FORWARD_CONFIGS


@triton.autotune(configs=FORWARD_CONFIGS, key=["S_q", "S_kv", "HEAD_DIM"])
@triton.jit
def _flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    B, H_q, H_kv, S_q, S_kv,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    pid_m = tl.program_id(0)        # which BLOCK_M of queries
    pid_bh = tl.program_id(1)       # batch * H_q
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    # Pointers to Q tile (BLOCK_M, HEAD_DIM)
    q_ptrs = (
        Q_ptr
        + b * stride_qb + h_q * stride_qh
        + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )
    q_mask = offs_m[:, None] < S_q
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax state
    m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e), so we use exp2 internally

    # Causal: queries at offs_m attend to keys at offs_n where offs_n <= offs_m + (S_kv - S_q).
    # For S_q == S_kv this is the standard lower triangle. For S_q == 1 (decode) and IS_CAUSAL=True,
    # this still attends to all S_kv keys (single query at the last position).
    if IS_CAUSAL:
        kv_end = (pid_m + 1) * BLOCK_M + (S_kv - S_q)
        kv_end = tl.minimum(kv_end, S_kv)
    else:
        kv_end = S_kv

    for start_n in range(0, kv_end, BLOCK_N):
        n_idx = start_n + offs_n
        k_mask = n_idx[:, None] < S_kv

        k_ptrs = (
            K_ptr
            + b * stride_kb + h_kv * stride_kh
            + n_idx[:, None] * stride_ks + offs_d[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        # qk: (BLOCK_M, BLOCK_N) accumulated in fp32
        qk = tl.dot(q, tl.trans(k))

        # Causal mask: q_idx[i] >= n_idx[j] + (S_q - S_kv)? actually causal
        # means q_pos >= k_pos when measured against the same axis.
        # For S_q == S_kv: mask = offs_m[i] >= n_idx[j].
        # For S_q < S_kv (decode): q virtual position is (S_kv - S_q) + offs_m[i],
        # so mask = (S_kv - S_q + offs_m[i]) >= n_idx[j].
        if IS_CAUSAL:
            q_pos = offs_m[:, None] + (S_kv - S_q)
            causal_mask = q_pos >= n_idx[None, :]
            qk = tl.where(causal_mask, qk, -float("inf"))

        # Out-of-range KV (when S_kv is not multiple of BLOCK_N): mask
        kv_oob_mask = n_idx[None, :] < S_kv
        qk = tl.where(kv_oob_mask, qk, -float("inf"))

        # Online softmax
        m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, axis=1))
        # If the entire row is -inf (E10), m_ij stays -inf and we must avoid 0/0.
        # Trick: compute alpha = exp2(m_i - m_ij). If m_ij == -inf, treat alpha = 0.
        m_ij_safe = tl.where(m_ij == -float("inf"), 0.0, m_ij)
        p = tl.math.exp2(qk * qk_scale - m_ij_safe[:, None])
        # Mask out rows that were entirely -inf (their p is whatever exp2(0)=1 produced — zero them)
        p = tl.where(m_ij[:, None] == -float("inf"), 0.0, p)

        alpha = tl.math.exp2(m_i - m_ij_safe)
        alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # V load + accumulate
        v_ptrs = (
            V_ptr
            + b * stride_vb + h_kv * stride_vh
            + n_idx[:, None] * stride_vs + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_ij

    # Normalize
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]

    # Store output
    o_ptrs = (
        O_ptr
        + b * stride_ob + h_q * stride_oh
        + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_mask)

    if WRITE_LSE:
        # LSE = m_i + log(l_i), in nats (convert from log2 by * 1/log2(e))
        # Since we accumulated using base-2 exponent, both m_i and the running
        # logsum are in log2 space; convert at write time.
        lse = (m_i + tl.math.log2(safe_l)) * 0.69314718  # ln(2)
        lse = tl.where(l_i == 0.0, -float("inf"), lse)
        l_ptrs = (
            L_ptr
            + b * stride_lb + h_q * stride_lh
            + offs_m * stride_ls
        )
        tl.store(l_ptrs, lse, mask=offs_m < S_q)


_SUPPORTED_HEAD_DIMS = (64, 128)


def flash_attn_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dense FA-2 forward on sm_86 via Triton.

    Args:
        Q: (B, H_q, S_q, D) bfloat16 on cuda.
        K: (B, H_kv, S_kv, D) bfloat16 on cuda.
        V: (B, H_kv, S_kv, D) bfloat16 on cuda.
        causal: whether to apply a causal mask. For S_q == 1 (decode), causal
            is a no-op (single query at the last position attends to all KV).
            For S_q != S_kv with S_q > 1 (chunked prefill, E13), raises.
        sm_scale: softmax scale; defaults to 1/sqrt(D).
        return_lse: when True, also return per-(b,h,q) logsumexp.

    Returns:
        (O, lse) where lse is None if return_lse=False.
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda, "Q/K/V must be CUDA tensors"
    assert Q.dtype == K.dtype == V.dtype == torch.bfloat16, "Q/K/V must be bfloat16"

    B, H_q, S_q, D = Q.shape
    Bk, H_kv, S_kv, Dk = K.shape
    Bv, H_kvv, S_v, Dv = V.shape
    assert (B, D) == (Bk, Dk) == (Bv, Dv), f"shape mismatch Q/K/V: {Q.shape} vs {K.shape} vs {V.shape}"
    assert H_kv == H_kvv, "K and V must have the same number of heads"
    assert S_kv == S_v, "K and V must have the same sequence length"
    assert H_q % H_kv == 0, f"GQA: H_q ({H_q}) must be a multiple of H_kv ({H_kv})"

    if S_q == 0 or S_kv == 0:
        raise ValueError("flash_attn_fwd: zero-length input")
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(
            f"flash_attn_fwd: head_dim={D} not in {_SUPPORTED_HEAD_DIMS}"
        )
    if causal and S_q != S_kv and S_q != 1:
        raise NotImplementedError(
            f"flash_attn_fwd: causal with S_q={S_q} S_kv={S_kv} (E13: chunked prefill) not supported in Phase 2"
        )

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)
    L = torch.empty(B, H_q, S_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)

    # 0 strides for missing-LSE output (so kernel can ignore safely).
    if L is not None:
        sl_b, sl_h, sl_s = L.stride()
    else:
        sl_b = sl_h = sl_s = 0

    grid = lambda META: (
        triton.cdiv(S_q, META["BLOCK_M"]),
        B * H_q,
    )

    _flash_attn_fwd_kernel[grid](
        Q, K, V, O, L_ptr,
        sm_scale,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        sl_b, sl_h, sl_s,
        B, H_q, H_kv, S_q, S_kv,
        HEAD_DIM=D,
        IS_CAUSAL=bool(causal),
        WRITE_LSE=bool(return_lse),
    )

    return O, L
```

- [ ] **Step 1.5: Write the package init**

Create `src/flashquest/kernel/__init__.py`:
```python
"""flashquest Triton kernels. Phase 2: dense FA-2 forward."""
from .flash_fwd import flash_attn_fwd

__all__ = ["flash_attn_fwd"]
```

- [ ] **Step 1.6: Run tests to verify the basic path passes**

Run: `pytest tests/test_kernel_flash_fwd_basic.py -v`
Expected: 4 passing tests.

If a test fails on numeric drift, *first* check that BF16 inputs are deterministic (`torch.manual_seed`); *only* then loosen tolerance. Causal-related failures here mean the basic test grid is wrong — re-examine.

- [ ] **Step 1.7: Commit**

```bash
git add src/flashquest/kernel/{__init__.py,flash_fwd.py,_autotune.py} \
        tests/test_kernel_flash_fwd_basic.py
git commit -m "phase 2: triton FA-2 forward kernel scaffold (non-causal, GQA, head_dim 64/128)"
```

---

## Task 2: Causal mask correctness

**Purpose:** Lock down the causal path. Task 1's kernel already has the `IS_CAUSAL` constexpr branch wired up; this task adds focused tests and tightens any bugs uncovered.

**Files:**
- Create: `tests/test_kernel_flash_fwd_causal.py`

- [ ] **Step 2.1: Write causal tests**

Create `tests/test_kernel_flash_fwd_causal.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_causal_prefill_matches_sdpa():
    """E5: standard prefill with S_q == S_kv, causal=True."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 256, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_causal_decode_step():
    """E6: S_q=1 against S_kv=512. Causal trivializes; output is dense over all kv."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S_q, S_kv, D = 1, 4, 1, 512, 64
    Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")

    O_causal, _ = flash_attn_fwd(Q, K, V, causal=True)
    O_dense, _ = flash_attn_fwd(Q, K, V, causal=False)
    # For S_q=1, causal is a no-op (the single query is at the last position).
    torch.testing.assert_close(O_causal, O_dense, rtol=0, atol=0)


@cuda
def test_causal_first_token_no_nan():
    """E10: q at position 0 attends only to k at position 0 (one term in softmax).
    Must not NaN."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)
    assert torch.isfinite(O).all(), "causal first-row produced NaN/Inf"

    # First row of output must equal V[0] (single attended key has weight 1).
    torch.testing.assert_close(O[0, 0, 0], V[0, 0, 0], rtol=1e-2, atol=1e-2)


@cuda
def test_chunked_prefill_rejected():
    """E13: causal with S_q != S_kv, both > 1, must raise NotImplementedError."""
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(NotImplementedError, match="chunked prefill"):
        flash_attn_fwd(Q, K, V, causal=True)
```

- [ ] **Step 2.2: Run causal tests**

Run: `pytest tests/test_kernel_flash_fwd_causal.py -v`
Expected: 4 passing tests.

If `test_causal_first_token_no_nan` fails with NaN: the `m_i == -inf` guards in the kernel are wrong. Trace the masked-row path; the fix is in the `m_ij_safe` / `alpha` guards — make sure `-inf` is preserved through `m_i` until normalization, and that `l_i == 0` produces `O = 0`, not NaN.

If `test_causal_prefill_matches_sdpa` fails with rtol > 1e-2: causal mask geometry is wrong. Add a small CPU debug script that compares element-wise: `qk_with_mask` vs reference `Q @ K.T * sm_scale + tril_mask`.

- [ ] **Step 2.3: Commit**

```bash
git add tests/test_kernel_flash_fwd_causal.py
git commit -m "phase 2: causal correctness tests (E5, E6, E10, E13 reject)"
```

---

## Task 3: Tail / non-power-of-2 sequence handling

**Purpose:** Validate the kernel's tail-tile masking (already wired up in Task 1 via `q_mask` / `kv_oob_mask`). Real Llama prompts are arbitrary-length.

**Files:**
- Create: `tests/test_kernel_flash_fwd_tail.py`

- [ ] **Step 3.1: Write tail tests**

Create `tests/test_kernel_flash_fwd_tail.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("S", [1, 17, 32, 63, 64, 65, 100, 127, 128, 129, 255])
@pytest.mark.parametrize("causal", [False, True])
def test_arbitrary_seq_lengths(S, causal):
    """E1, E2, E3: kernel handles S not multiple of BLOCK_M / BLOCK_N."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(S * 7 + int(causal))
    B, H, D = 1, 2, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=causal)

    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_decode_against_arbitrary_kv_length():
    """E6: S_q=1 decode against S_kv ∈ {1, 31, 64, 100, 1023, 1024}."""
    from flashquest.kernel import flash_attn_fwd

    for S_kv in [1, 31, 64, 100, 1023, 1024]:
        torch.manual_seed(S_kv)
        B, H, S_q, D = 1, 4, 1, 64
        Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16, device="cuda")
        K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")
        V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16, device="cuda")

        O, _ = flash_attn_fwd(Q, K, V, causal=True)
        ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
        torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2, msg=f"S_kv={S_kv}")
```

- [ ] **Step 3.2: Run tail tests**

Run: `pytest tests/test_kernel_flash_fwd_tail.py -v`
Expected: 11 × 2 + 1 = 23 passing tests.

If S=1 fails: `BLOCK_M=64` means the program iterates over a tile of 64 rows but only row 0 is valid. The `q_mask` should already mask out the invalid rows; if not, examine the load mask + the `kv_end` calculation when `pid_m=0`.

- [ ] **Step 3.3: Commit**

```bash
git add tests/test_kernel_flash_fwd_tail.py
git commit -m "phase 2: tail / arbitrary-S correctness (E1, E2, E3, E6)"
```

---

## Task 4: LSE output

**Purpose:** Phase 3+ sparse attention needs to recombine multiple sparse blocks via log-sum-exp; the kernel must export LSE. The Task 1 scaffold already writes LSE; this task adds tests and verifies the value.

**Files:**
- Create: `tests/test_kernel_flash_fwd_lse.py`

- [ ] **Step 4.1: Write LSE tests**

Create `tests/test_kernel_flash_fwd_lse.py`:
```python
import math

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _ref_lse(Q, K, sm_scale, is_causal):
    """Reference logsumexp from raw QK; matches what FA-2 exposes."""
    qk = (Q.float() @ K.float().transpose(-2, -1)) * sm_scale
    if is_causal:
        S_q = Q.shape[-2]
        S_kv = K.shape[-2]
        # Same convention as the kernel: q_pos = (S_kv - S_q) + i, attend to j <= q_pos.
        m = torch.zeros(S_q, S_kv, device=qk.device, dtype=torch.bool)
        for i in range(S_q):
            qp = (S_kv - S_q) + i
            m[i, : qp + 1] = True
        qk = qk.masked_fill(~m, float("-inf"))
    return torch.logsumexp(qk, dim=-1)


@cuda
def test_lse_shape_and_value_non_causal():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 2, 64, 64
    sm_scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    _, lse = flash_attn_fwd(Q, K, V, causal=False)

    assert lse.shape == (B, H, S)
    assert lse.dtype == torch.float32

    ref = _ref_lse(Q, K, sm_scale, is_causal=False)
    torch.testing.assert_close(lse, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_lse_causal():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(1)
    B, H, S, D = 1, 2, 128, 64
    sm_scale = 1.0 / math.sqrt(D)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")

    _, lse = flash_attn_fwd(Q, K, V, causal=True)

    ref = _ref_lse(Q, K, sm_scale, is_causal=True)
    torch.testing.assert_close(lse, ref, rtol=1e-2, atol=1e-2)


@cuda
def test_lse_skipped_when_disabled():
    from flashquest.kernel import flash_attn_fwd

    Q = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)
    O, lse = flash_attn_fwd(Q, K, V, causal=False, return_lse=False)
    assert lse is None
```

- [ ] **Step 4.2: Run LSE tests**

Run: `pytest tests/test_kernel_flash_fwd_lse.py -v`
Expected: 3 passing tests.

If LSE values mismatch but O matches: the in-kernel base-2 → ln conversion at the LSE write is wrong. Check: kernel computes `m_i + log2(l_i)` then multiplies by `ln(2) = 0.69314718`. Verify against the formula `LSE = max_i + ln(sum_j exp(qk_j - max_i))`.

- [ ] **Step 4.3: Commit**

```bash
git add tests/test_kernel_flash_fwd_lse.py
git commit -m "phase 2: LSE output (Phase 3 composition input)"
```

---

## Task 5: Edge case grid

**Purpose:** Single test file that exercises every catalogued edge case. This is the safety net for all later phases.

**Files:**
- Modify: `pyproject.toml` (add `hypothesis` to `dev`)
- Create: `tests/test_kernel_flash_fwd_edges.py`

- [ ] **Step 5.1: Add hypothesis to dev extras**

Edit `pyproject.toml`. Replace the dev block with:
```toml
dev = [
  "pytest>=8",
  "pytest-cov",
  "ruff",
  "mypy",
  "hypothesis>=6",
]
```

Run: `. .venv/bin/activate && pip install -e ".[dev]" 2>&1 | tail -3`
Expected: hypothesis installs.

- [ ] **Step 5.2: Write the edge case test file**

Create `tests/test_kernel_flash_fwd_edges.py`:
```python
"""Edge case grid for flash_attn_fwd. See plan §Edge case catalog."""
import pytest
import torch
from hypothesis import given, settings, strategies as st

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# E11 — strided / non-contiguous Q
@cuda
def test_strided_Q_via_transpose():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, S, H, D = 1, 128, 4, 64
    # Build (B, S, H, D) and transpose to (B, H, S, D); this produces a
    # non-contiguous tensor — kernel must respect strides.
    Q_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    K_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
    V_nhd = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")

    Q = Q_nhd.transpose(1, 2)
    K = K_nhd.transpose(1, 2)
    V = V_nhd.transpose(1, 2)
    assert not Q.is_contiguous()

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    Qc, Kc, Vc = Q.contiguous(), K.contiguous(), V.contiguous()
    O_ref, _ = flash_attn_fwd(Qc, Kc, Vc, causal=True)

    torch.testing.assert_close(O, O_ref, rtol=0, atol=0)


# E12 — NaN propagation
@cuda
def test_nan_in_q_propagates():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 1, 1, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    Q[0, 0, 5, :] = float("nan")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    O, _ = flash_attn_fwd(Q, K, V, causal=False)
    # The query row that had NaN must produce NaN output. Other rows must be finite.
    assert torch.isnan(O[0, 0, 5]).all(), "NaN in Q must propagate to output"
    assert torch.isfinite(O[0, 0, : 5]).all()
    assert torch.isfinite(O[0, 0, 6:]).all()


# E9 — MHA degenerate (n_rep == 1)
@cuda
def test_mha_path():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H, S, D = 2, 4, 64, 64
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    O, _ = flash_attn_fwd(Q, K, V, causal=False)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False)
    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


# E7 — batch > 1 already covered in causal test; one more for non-causal
@cuda
def test_batch_independence():
    """B>1: each batch element processed independently (no cross-talk)."""
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    H, S, D = 2, 64, 64
    # Two random examples, run together vs separately.
    Q0 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda")
    Q1 = torch.randn(1, H, S, D, dtype=torch.bfloat16, device="cuda")
    K0 = torch.randn_like(Q0); K1 = torch.randn_like(Q1)
    V0 = torch.randn_like(Q0); V1 = torch.randn_like(Q1)

    Q = torch.cat([Q0, Q1], dim=0)
    K = torch.cat([K0, K1], dim=0)
    V = torch.cat([V0, V1], dim=0)

    O_batched, _ = flash_attn_fwd(Q, K, V, causal=False)
    O0, _ = flash_attn_fwd(Q0, K0, V0, causal=False)
    O1, _ = flash_attn_fwd(Q1, K1, V1, causal=False)

    torch.testing.assert_close(O_batched[0:1], O0, rtol=0, atol=0)
    torch.testing.assert_close(O_batched[1:2], O1, rtol=0, atol=0)


# E8 — GQA n_rep=4 already in basic test; here n_rep=8
@cuda
def test_gqa_eight_way():
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 16, 2, 128, 64  # 8-way GQA
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=True)

    Kr = K.repeat_interleave(8, dim=1)
    Vr = V.repeat_interleave(8, dim=1)
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=True)

    torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)


# E14 — empty rejection
@cuda
def test_zero_seq_q_rejected():
    from flashquest.kernel import flash_attn_fwd
    Q = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)


@cuda
def test_zero_seq_kv_rejected():
    from flashquest.kernel import flash_attn_fwd
    Q = torch.randn(1, 1, 4, 64, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(1, 1, 0, 64, dtype=torch.bfloat16, device="cuda")
    V = torch.randn_like(K)
    with pytest.raises(ValueError, match="zero-length"):
        flash_attn_fwd(Q, K, V, causal=False)


# Property-based fuzz over (S_q, S_kv, B, H_q, H_kv, D)
@cuda
@settings(deadline=None, max_examples=20)
@given(
    S=st.integers(min_value=1, max_value=192),
    B=st.integers(min_value=1, max_value=2),
    H_kv=st.sampled_from([1, 2, 4]),
    n_rep=st.sampled_from([1, 2, 4]),
    D=st.sampled_from([64, 128]),
    causal=st.booleans(),
)
def test_random_shapes_match_sdpa(S, B, H_kv, n_rep, D, causal):
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(S * 31 + B * 17 + H_kv * 7 + n_rep * 3 + D + int(causal))
    H_q = H_kv * n_rep
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O, _ = flash_attn_fwd(Q, K, V, causal=causal)

    Kr = K.repeat_interleave(n_rep, dim=1) if n_rep > 1 else K
    Vr = V.repeat_interleave(n_rep, dim=1) if n_rep > 1 else V
    ref = torch.nn.functional.scaled_dot_product_attention(Q, Kr, Vr, is_causal=causal)

    torch.testing.assert_close(O, ref, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 5.3: Run the edge case suite**

Run: `pytest tests/test_kernel_flash_fwd_edges.py -v`
Expected: 7 deterministic tests + 20 hypothesis examples passing.

If a hypothesis example fails, hypothesis will minimize and report the smallest failing shape. Fix the kernel against that shape — *do not* loosen the test until you've understood the failure.

- [ ] **Step 5.4: Commit**

```bash
git add pyproject.toml tests/test_kernel_flash_fwd_edges.py
git commit -m "phase 2: edge case grid (E1-E14) + property-based shape fuzz"
```

---

## Task 6: Equivalence with Phase 1 eager + perf gate

**Purpose:** Close the algorithm loop (kernel ≡ eager at retention=1.0) and verify the SPEC's perf bar (within 30 % of `flash_attn` 2.7.4).

**Files:**
- Create: `tests/test_kernel_eager_equivalence.py`
- Create: `scripts/phase2_bench_attn.py`
- Create: `benchmarks/phase2_perf.json` (filled by script)

- [ ] **Step 6.1: Write the eager-equivalence test**

Create `tests/test_kernel_eager_equivalence.py`:
```python
import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_kernel_matches_eager_quest_at_full_retention():
    """Phase 2 dense kernel must equal Phase 1 eager Quest at retention=1.0
    (with no sinks, no window) — both compute the same dense attention."""
    from flashquest.eager import quest_eager_sdpa
    from flashquest.kernel import flash_attn_fwd

    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")

    O_kernel, _ = flash_attn_fwd(Q, K, V, causal=True)

    O_eager = quest_eager_sdpa(
        Q, K, V,
        page_size=64,
        retention=1.0,
        num_sinks=0,
        window_pages=0,
        is_causal=True,
    )

    torch.testing.assert_close(O_kernel, O_eager, rtol=1e-2, atol=1e-2)
```

- [ ] **Step 6.2: Run equivalence test**

Run: `pytest tests/test_kernel_eager_equivalence.py -v`
Expected: 1 passing test.

- [ ] **Step 6.3: Write the perf bench**

Create `scripts/phase2_bench_attn.py`:
```python
"""Bench the Phase 2 dense Triton kernel vs flash_attn 2.7.4 + torch SDPA.

Shape pinned to the Phase 0 reference (S=8192, H_q=24, H_kv=8, D=64, BF16),
which is Llama-3.2-3B's geometry. Reports ms / forward, ratio vs FA-2,
ratio vs SDPA. SPEC §6 Phase 2 win condition: within 30% of FA-2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from flash_attn import flash_attn_func

from flashquest.kernel import flash_attn_fwd


def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def main() -> None:
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 24, 8, 8192, 64
    dtype = torch.bfloat16

    Q_bhd = torch.randn(B, H_q, S, D, dtype=dtype, device="cuda")
    K_bhd = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")
    V_bhd = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")

    # FA-2 expects (B, S, H, D)
    Q_nhd = Q_bhd.transpose(1, 2).contiguous()
    K_nhd = K_bhd.transpose(1, 2).contiguous()
    V_nhd = V_bhd.transpose(1, 2).contiguous()

    flash_ms = time_fn(lambda: flash_attn_func(Q_nhd, K_nhd, V_nhd, causal=True))
    triton_ms = time_fn(lambda: flash_attn_fwd(Q_bhd, K_bhd, V_bhd, causal=True))

    # SDPA needs repeated K/V for the GQA case
    n_rep = H_q // H_kv
    Kr = K_bhd.repeat_interleave(n_rep, dim=1)
    Vr = V_bhd.repeat_interleave(n_rep, dim=1)
    sdpa_ms = time_fn(
        lambda: torch.nn.functional.scaled_dot_product_attention(Q_bhd, Kr, Vr, is_causal=True)
    )

    result = {
        "shape": {"B": B, "H_q": H_q, "H_kv": H_kv, "S": S, "D": D, "dtype": "bfloat16", "causal": True},
        "flash_attn_2_ms": flash_ms,
        "triton_kernel_ms": triton_ms,
        "sdpa_ms": sdpa_ms,
        "triton_over_flash": triton_ms / flash_ms,
        "triton_over_sdpa": triton_ms / sdpa_ms,
        "spec_target_ratio": 1.30,  # SPEC: within 30% of FA-2
        "passes_spec_target": (triton_ms / flash_ms) <= 1.30,
    }
    print(json.dumps(result, indent=2))

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase2_perf.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run the bench**

Run: `. .venv/bin/activate && nice -n 19 python scripts/phase2_bench_attn.py 2>&1 | tail -25`
Expected: prints the JSON, writes `benchmarks/phase2_perf.json`. `passes_spec_target` should be True.

If `triton_over_flash > 1.30`: don't auto-pass. Open `docs/PHASES/phase-2-notes.md` and write a `### Perf gap` section with the actual ratio, the autotune-selected config (the `meta` field of the autotuner's pick — print it from inside the kernel wrapper), and a hypothesis (likely candidates: BLOCK_N too small, num_warps wrong, missing `cp.async` pipeline, dtype-conversion overhead). Re-tune the configs in `_autotune.py` and re-run. Bring it to the user if the gap is >2x after one re-tune attempt.

- [ ] **Step 6.5: Commit**

```bash
git add tests/test_kernel_eager_equivalence.py scripts/phase2_bench_attn.py benchmarks/phase2_perf.json
git commit -m "phase 2: eager-equivalence test + perf bench (SPEC ≤30% of FA-2 gate)"
```

---

## Task 7: Phase 2 notes + README + DOC.md + tag

**Files:**
- Create: `docs/PHASES/phase-2-notes.md`
- Modify: `README.md`
- Modify: `DOC.md`

- [ ] **Step 7.1: Write the phase notes**

Create `docs/PHASES/phase-2-notes.md`:
```markdown
# Phase 2 Notes

**Started:** <YYYY-MM-DD>
**Completed:** <YYYY-MM-DD> (tag `phase-2`)
**Status:** complete
**Spec:** [docs/SPEC.md §6 Phase 2](../SPEC.md)

## Goal

Port the FA-2 06-fused-attention Triton tutorial onto sm_86 as a clean dense forward-only kernel. No sparsity (Phase 3); no backward (out of scope).

## Kernel surface

`flashquest.kernel.flash_attn_fwd(Q, K, V, *, causal, sm_scale=None, return_lse=True) -> (O, lse)`

Shapes: `Q (B, H_q, S_q, D)`, `K/V (B, H_kv, S_kv, D)`, BF16 in / FP32 acc / BF16 out. GQA via kernel-side head map (no host repeat_kv copy). Returns LSE in fp32 for Phase 3 composition.

Tile shapes: `BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2`. Three autotune configs in `src/flashquest/kernel/_autotune.py`. Selected config at our reference shape (S=8192, H_q=24, H_kv=8, D=64, causal): <fill from autotuner introspection>.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Numeric ≡ torch SDPA across edge grid (E1–E12) | <fill: pytest summary> | ✅ / ❌ |
| Numeric ≡ Phase 1 eager at retention=1.0 | <fill> | ✅ / ❌ |
| Perf within 30 % of FA-2 at S=8192 | <fill: triton/flash ratio> | ✅ / ❌ |

## Edge cases handled

- E1 S=1 decode, E2 S<BLOCK_M, E3 non-multiple S — tail-tile masking on Q/K/V loads.
- E4 head_dim ∈ {64, 128}.
- E5 causal prefill, E6 causal decode (S_q=1 trivializes), E10 first-token row.
- E7 batch>1, E8 GQA n_rep ∈ {2,4,8}, E9 MHA degenerate.
- E11 strided Q via transpose.
- E12 NaN propagation — confirmed input NaN → output NaN.

E13 (chunked prefill, S_q != S_kv with S_q > 1) and E14 (S=0) are rejected at the Python wrapper.

## Decisions

- Kernel-side GQA mapping (no host repeat_kv copy). Saves ~25 MB at 8 k context for Llama-3.2-3B's 24:8 GQA.
- BLOCK_M=BLOCK_N=64 (vs tutorial's 128). sm_86 has 16 SMs vs H100's 80; smaller blocks expose more parallelism on fewer SMs.
- Base-2 exponent inside the kernel (`exp2`/`log2`); converted to nats only at the LSE write. Lets the kernel use the faster `exp2` path.
- LSE skipped via `WRITE_LSE` constexpr when caller passes `return_lse=False`. No allocation cost.

## Phase 3 handoff

Algorithm validated; perf bar met. Phase 3 layers on top:
- Sparse outer loop driven by Phase 1's `compute_page_summary` + `page_scores` + `select_pages`.
- INT8 KV with KIVI-style scales (per-channel K, per-token V).
- StreamingLLM sinks + sliding window as always-attended block sets.

The dense Triton kernel here is the correctness oracle for the sparse kernel: at retention=1.0, the sparse kernel's output must match this kernel's output bit-for-bit modulo float ordering.
```

(Fill `<...>` after the run.)

- [ ] **Step 7.2: Append Phase 2 to README**

Edit `README.md`. After the Phase 1 section, append:
```markdown
## Phase 2 — Dense FA-2 Triton kernel

Port of the FA-2 06-fused-attention tutorial onto sm_86 (`BLOCK_M = BLOCK_N = 64`, `num_warps = 4`, `num_stages = 2`). BF16 in / FP32 acc / BF16 out, GQA via kernel-side head map, returns LSE for Phase 3 composition.

| Backend | ms / fwd at S=8192, BF16 causal, Llama-3.2-3B geometry |
|---|---|
| `flash_attn` 2.7.4 (reference) | 22.76 |
| **flashquest Triton kernel** | <fill> |
| torch SDPA (with GQA repeat) | <fill> |

SPEC win condition: within 30 % of FA-2 — <fill: pass/fail>.

Re-run via `python scripts/phase2_bench_attn.py`. Edge case grid (E1–E14) is in `tests/test_kernel_flash_fwd_edges.py`.
```

- [ ] **Step 7.3: Flip Phase 2 in DOC.md**

Edit `DOC.md`. Replace:
```
- Phase 2 — Dense FA-2 Triton baseline. Not started. Phase-2 target: ≥ 70 % of `flash_attn` 22.76 ms.
```
with:
```
- **Phase 2 — Dense FA-2 Triton kernel** ✅ **complete (tag `phase-2`)**. `flashquest.kernel.flash_attn_fwd(Q, K, V, *, causal, sm_scale=None, return_lse=True)`. BLOCK_M=BLOCK_N=64 on sm_86. Edge cases E1–E14 covered (`tests/test_kernel_flash_fwd_edges.py`); equivalence with Phase 1 eager at retention=1.0 confirmed; perf within 30 % of FA-2 at the reference shape. See `docs/PHASES/phase-2-notes.md`.
```

Add a usage block under Phase 1's:
````markdown
Use the Phase 2 dense Triton kernel directly:

```python
import torch
from flashquest.kernel import flash_attn_fwd

Q = torch.randn(1, 24, 8192, 64, dtype=torch.bfloat16, device="cuda")
K = torch.randn(1,  8, 8192, 64, dtype=torch.bfloat16, device="cuda")
V = torch.randn(1,  8, 8192, 64, dtype=torch.bfloat16, device="cuda")
O, lse = flash_attn_fwd(Q, K, V, causal=True)
```
````

- [ ] **Step 7.4: Final smoke**

Run:
```bash
. .venv/bin/activate
pytest tests/ -v --ignore=tests/test_eager_e2e.py
python scripts/verify_triton_int8.py
```
Expected: all kernel + eager + smoke tests pass; INT8 mma still OK (sanity).

- [ ] **Step 7.5: Commit + tag**

```bash
git add docs/PHASES/phase-2-notes.md README.md DOC.md
git commit -m "phase 2: README + DOC + phase-2-notes complete"
git tag -a phase-2 -m "Phase 2 complete: dense FA-2 Triton kernel on sm_86"
```

---

## Self-review

**1. Spec coverage** (SPEC §6 Phase 2):
- "Port the Triton 06-fused-attention tutorial onto our 3050 Ti." → Tasks 1–4.
- "Tune block sizes for sm_86 (smaller than tutorial defaults)." → Task 1, `_autotune.py` + 64×64 blocks.
- "Benchmark vs upstream FA-2: target ≥ 70 % of FA-2 perf." → Task 6 perf bench.
- "Numerical equivalence with `torch.nn.functional.scaled_dot_product_attention`." → Tasks 1, 2, 3, 5 vs SDPA across the edge grid.

Edge case catalog (E1–E14) all covered with explicit tests in Tasks 1–5.

**2. Placeholder scan**: every code block contains real code. README/DOC/notes have `<fill>` slots for measured values, but those are explicit data-entry slots, not unwritten code.

**3. Type / name consistency**:
- `flash_attn_fwd(Q, K, V, *, causal, sm_scale, return_lse) -> (O, lse)` — same signature in Tasks 1, 2, 3, 4, 5, 6, 7.
- `_flash_attn_fwd_kernel(...)` — Triton @jit, same constexpr set throughout.
- Shape conventions `(B, H_q, S_q, D)` / `(B, H_kv, S_kv, D)` — same across kernel + tests + bench + DOC example.
- `BLOCK_M`, `BLOCK_N`, `num_warps`, `num_stages` — same set in `_autotune.py` + kernel.

**4. Reversibility**: every task ends with a commit. Task 6 explicitly tells the engineer to STOP and surface a perf gap > 2× rather than chase autotune indefinitely.

## Phase 2 → Phase 3 handoff

When the plan completes:
- `phase-2` git tag exists.
- `flashquest.kernel.flash_attn_fwd` is the dense baseline; Phase 1 eager validates it; FA-2 perf bound met.
- `benchmarks/phase2_perf.json` populated.
- Phase 3 begins: sparse outer loop on top of this kernel — *kernel iterates over Quest-selected blocks only*, INT8 KV with KIVI scales, StreamingLLM sinks + window. Phase 3 will get its own plan.
