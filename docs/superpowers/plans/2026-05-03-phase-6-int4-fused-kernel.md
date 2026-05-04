# Phase 6 Task 6 — Kernel-Fused Inline INT4 Unpack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the reference path in `src/flashquest/kernel/sparse_int4_fwd.py` (currently dequantizes INT4 → BF16 → re-quantizes to INT8 → calls existing INT8 kernel) with a real Triton kernel that reads packed `uint8` K/V directly and unpacks lo/hi nibbles inline during the tile load — eliminating the BF16 dequant intermediate that OOMs at 32 k.

**Architecture:** Mirror the INT8 sparse Triton kernel (`src/flashquest/kernel/sparse_fwd.py`), changing only the K/V tile loads. Read `(PAGE_SIZE, HEAD_DIM/2)` `uint8` tiles for K_packed and V_packed; extract lo/hi nibbles via bitwise ops; interleave into `(PAGE_SIZE, HEAD_DIM)` INT tiles via `tl.join` + reshape; dequant via the existing per-page channel-wise scale + mn (K) or per-token scale + mn (V) into FP32. Same online-softmax math, same selection-mask path, same return contract.

**Tech Stack:** Triton, PyTorch, existing `flashquest.kernel.sparse_fwd` as the structural reference.

**Reference:** Task 5 spec at `docs/superpowers/specs/2026-05-03-phase-6-int4-kv-design.md`. The kernel sketch preserved in `src/flashquest/kernel/sparse_int4_fwd.py` docstring is now the implementation target. Reference Triton kernel: `src/flashquest/kernel/sparse_fwd.py:_sparse_attn_fwd_kernel` (INT8 path).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/flashquest/kernel/sparse_int4_fwd.py` | Drop reference path; ship real `@triton.jit` kernel + Python wrapper with the same shape contracts | Rewrite |
| `tests/test_sparse_int4.py` | Keep dense-equivalence test; add fused-vs-reference and no-BF16-allocation tests | Modify |
| `benchmarks/phase6_decode_int4_fused.json` | Single-cell decode bench at 32 k under the fused kernel | Create at run end |
| `docs/PHASES/phase-6-notes.md` | Append task 6 section | Modify at end |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 6; update §11.4 with measured throughput | Modify at end |

---

## Task 1: Rename current reference function so the fused kernel can take its name

**Files:**
- Modify: `src/flashquest/kernel/sparse_int4_fwd.py`

The dispatcher in `eager/llama_persistent_patch.py` and the eager wrapper in `eager/sparse_int4.py` both call `flash_attn_sparse_int4_fwd`. We want the *fused* kernel under that name and the reference path preserved as `_flash_attn_sparse_int4_fwd_reference` for the bit-equality test.

- [ ] **Step 1: Inspect the current file**

Run:
```
grep -n "def flash_attn_sparse_int4_fwd" /home/hoang/code/personal/active/flashquest/src/flashquest/kernel/sparse_int4_fwd.py
```
Expected: a single match around line 33.

- [ ] **Step 2: Edit the function name**

In `src/flashquest/kernel/sparse_int4_fwd.py`, find:

```python
def flash_attn_sparse_int4_fwd(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
```

Replace the function name with `_flash_attn_sparse_int4_fwd_reference`. Keep the rest of the function unchanged.

- [ ] **Step 3: Verify the rename does not break anything yet (the public name will fail)**

Run:
```
source .venv/bin/activate && python -c "from flashquest.kernel.sparse_int4_fwd import _flash_attn_sparse_int4_fwd_reference; print('reference rename OK')"
```
Expected: `reference rename OK`.

Also run:
```
source .venv/bin/activate && python -c "from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd"
```
Expected: `ImportError: cannot import name 'flash_attn_sparse_int4_fwd'`. (We re-add the public name in Task 3.)

- [ ] **Step 4: Do not commit yet — Task 2 adds the failing tests; Task 3 adds the fused kernel**

Skip — leave staged for Task 3's commit.

---

## Task 2: Add the failing fused-vs-reference equivalence test

**Files:**
- Modify: `tests/test_sparse_int4.py`

- [ ] **Step 1: Append the equivalence test**

Append to `tests/test_sparse_int4.py`:

```python
def test_fused_matches_reference():
    """Fused Triton kernel ≡ reference path on small fixtures (FP rtol=1e-3).

    Fixture is intentionally tiny so that any divergence surfaces at the per-element
    level, not buried under attention-scale variance. retention=1.0 (all pages
    selected) so we test every dequant path, not the page-selection logic.
    """
    from flashquest.kernel.sparse_int4_fwd import (
        flash_attn_sparse_int4_fwd,
        _flash_attn_sparse_int4_fwd_reference,
    )
    torch.manual_seed(17)
    B, H_q, H_kv, S_q, S_kv, D, page_size = 1, 2, 1, 1, 64, 64, 64
    P = S_kv // page_size

    Q = torch.randn(B, H_q, S_q, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")

    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)

    sel = torch.ones(B, H_q, S_q, P, dtype=torch.bool, device="cuda")
    kw = dict(
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    O_ref, lse_ref = _flash_attn_sparse_int4_fwd_reference(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, **kw,
    )
    O_fused, lse_fused = flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn, **kw,
    )

    err_O = (O_fused.float() - O_ref.float()).abs().max()
    err_lse = (lse_fused.float() - lse_ref.float()).abs().max()
    assert err_O < 1e-2, f"fused vs reference O max abs err {err_O}"
    assert err_lse < 1e-2, f"fused vs reference lse max abs err {err_lse}"
```

- [ ] **Step 2: Run the new test — both names import, but fused is missing**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_int4.py::test_fused_matches_reference -v
```
Expected: `ImportError: cannot import name 'flash_attn_sparse_int4_fwd' from 'flashquest.kernel.sparse_int4_fwd'`.

- [ ] **Step 3: Do not commit yet**

Skip — Task 3 adds the implementation and runs the test green.

---

## Task 3: Implement the fused Triton kernel + Python wrapper

**Files:**
- Modify: `src/flashquest/kernel/sparse_int4_fwd.py`

- [ ] **Step 1: Replace the entire module with the fused implementation**

Open `src/flashquest/kernel/sparse_int4_fwd.py`. Keep the renamed `_flash_attn_sparse_int4_fwd_reference` function from Task 1 at the end of the file (or move it after the public wrapper). Insert the new `@triton.jit` kernel and the public wrapper above it.

The full file content after this step is:

```python
"""Phase 6 task 6 — sparse-attention forward with INT4 KV (fused Triton kernel).

The kernel reads packed uint8 K_packed/V_packed directly (head_dim/2 trailing
axis), unpacks lo/hi nibbles inline during the tile load, dequants per-page
channel-wise (K) or per-token (V), and runs the same online-softmax math as
the INT8 reference kernel. No BF16 dequant intermediate, no INT8 re-quant
round-trip.

The reference path (`_flash_attn_sparse_int4_fwd_reference`) is preserved
for bit-equivalence testing.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

_SUPPORTED_HEAD_DIMS = (64, 128)


@triton.jit
def _sparse_attn_fwd_kernel_int4(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    sel_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_selp,
    H_q, H_kv, S_kv, NUM_PAGES,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PACKED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
):
    """Decode-only sparse forward with INT4 KV. One CTA per (batch, query head).

    HEAD_DIM_PACKED == HEAD_DIM // 2. K_packed/V_packed have trailing axis
    HEAD_DIM_PACKED; each byte holds two 4-bit values (low nibble at d=2k,
    high nibble at d=2k+1).
    """
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED)

    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh + offs_d * stride_qd
    )
    q = tl.load(q_ptrs)

    NEG_INF: tl.constexpr = float("-inf")
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504  # log2(e)

    for p in range(0, NUM_PAGES):
        sel_ptr_p = sel_ptr + b * stride_selb + h_q * stride_selh + p * stride_selp
        is_sel = tl.load(sel_ptr_p)

        if is_sel:
            page_start = p * PAGE_SIZE
            n_idx = page_start + offs_n
            valid_kv = n_idx < S_kv

            # === Load K: packed uint8 tile (PAGE_SIZE, HEAD_DIM/2), unpack to (PAGE_SIZE, HEAD_DIM). ===
            k_byte_ptrs = (
                K_packed_ptr + b * stride_kb + h_kv * stride_kh
                + n_idx[:, None] * stride_ks + offs_dp[None, :] * stride_kdp
            )
            k_byte = tl.load(k_byte_ptrs, mask=valid_kv[:, None], other=0)
            k_lo = (k_byte & 0xF).to(tl.uint8)
            k_hi = ((k_byte >> 4) & 0xF).to(tl.uint8)
            # Interleave: at d=2k take lo[k]; at d=2k+1 take hi[k].
            # tl.join stacks along a new last axis: (PAGE_SIZE, HEAD_DIM/2, 2).
            # tl.reshape flattens to (PAGE_SIZE, HEAD_DIM) row-major:
            #   [lo[0,0], hi[0,0], lo[0,1], hi[0,1], ...] — exactly the packing layout.
            k_int_2 = tl.join(k_lo, k_hi)  # (PAGE_SIZE, HEAD_DIM/2, 2)
            k_int = tl.reshape(k_int_2, (PAGE_SIZE, HEAD_DIM))

            ks_ptrs = (
                K_scale_ptr + b * stride_ksb + h_kv * stride_ksh
                + p * stride_ksp + offs_d * stride_ksd
            )
            km_ptrs = (
                K_mn_ptr + b * stride_kmb + h_kv * stride_kmh
                + p * stride_kmp + offs_d * stride_kmd
            )
            k_scale = tl.load(ks_ptrs).to(tl.float32)
            k_mn = tl.load(km_ptrs).to(tl.float32)
            k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]

            qk = tl.sum(q[None, :].to(tl.float32) * k, axis=1)
            qk = tl.where(valid_kv, qk, NEG_INF)

            qk_max = tl.max(qk * qk_scale, axis=0)
            m_ij = tl.maximum(m_i, qk_max)
            m_ij_safe = tl.where(m_ij == NEG_INF, 0.0, m_ij)
            p_softmax = tl.math.exp2(qk * qk_scale - m_ij_safe)
            row_all_neg_inf = m_ij == NEG_INF
            p_softmax = tl.where(row_all_neg_inf, 0.0, p_softmax)

            alpha = tl.math.exp2(m_i - m_ij_safe)
            if m_i == NEG_INF:
                alpha = 0.0

            l_i = l_i * alpha + tl.sum(p_softmax, axis=0)
            acc = acc * alpha

            # === Load V: packed uint8 tile (PAGE_SIZE, HEAD_DIM/2), unpack to (PAGE_SIZE, HEAD_DIM). ===
            v_byte_ptrs = (
                V_packed_ptr + b * stride_vb + h_kv * stride_vh
                + n_idx[:, None] * stride_vs + offs_dp[None, :] * stride_vdp
            )
            v_byte = tl.load(v_byte_ptrs, mask=valid_kv[:, None], other=0)
            v_lo = (v_byte & 0xF).to(tl.uint8)
            v_hi = ((v_byte >> 4) & 0xF).to(tl.uint8)
            v_int_2 = tl.join(v_lo, v_hi)
            v_int = tl.reshape(v_int_2, (PAGE_SIZE, HEAD_DIM))

            vs_ptrs = V_scale_ptr + b * stride_vsb + h_kv * stride_vsh + n_idx * stride_vss
            vm_ptrs = V_mn_ptr + b * stride_vmb + h_kv * stride_vmh + n_idx * stride_vms
            v_scale = tl.load(vs_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
            v_mn = tl.load(vm_ptrs, mask=valid_kv, other=0.0).to(tl.float32)
            v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]

            acc += tl.sum(p_softmax[:, None] * v, axis=0)

            m_i = m_ij

    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l

    o_ptrs = O_ptr + b * stride_ob + h_q * stride_oh + offs_d * stride_od
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty))

    if WRITE_LSE:
        lse_val = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse_val = tl.where(l_i == 0.0, NEG_INF, lse_val)
        l_ptr_bh = L_ptr + b * stride_lb + h_q * stride_lh
        tl.store(l_ptr_bh, lse_val)


def flash_attn_sparse_int4_fwd(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Decode-only sparse forward with INT4 KV (fused Triton kernel — no BF16 intermediate).

    Args:
        Q: (B, H_q, 1, D) bf16 cuda.
        K_packed: (B, H_kv, S_kv, D/2) uint8 cuda.
        K_scale, K_mn: (B, H_kv, num_pages, D) bf16 cuda.
        V_packed: (B, H_kv, S_kv, D/2) uint8 cuda.
        V_scale, V_mn: (B, H_kv, S_kv, 1) bf16 cuda.
        selection_mask: (B, H_q, 1, num_pages) bool cuda — True = attend.

    Returns:
        (O (B, H_q, 1, D) bf16, lse (B, H_q, 1) fp32 or None).
    """
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_packed.dtype == torch.uint8 and V_packed.dtype == torch.uint8

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(
            f"flash_attn_sparse_int4_fwd: decode-only (S_q={S_q})"
        )
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")

    Bk, H_kv, S_kv, Dp = K_packed.shape
    assert B == Bk
    assert Dp == D // 2, f"K_packed last axis {Dp} != D/2={D//2}"
    assert H_q % H_kv == 0
    num_pages = K_scale.shape[2]
    assert selection_mask.shape == (B, H_q, 1, num_pages)
    assert selection_mask.dtype == torch.bool

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    Q_2d = Q.squeeze(2)
    O_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    if L is not None:
        sl_b, sl_h = L.stride()
    else:
        sl_b = sl_h = 0

    sel_2d = selection_mask.squeeze(2)

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_int4[grid](
        Q_2d, K_packed, V_packed, O_2d, L_ptr,
        K_scale, K_mn, V_scale, V_mn,
        sel_2d,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_packed.stride(0), K_packed.stride(1), K_packed.stride(2), K_packed.stride(3),
        V_packed.stride(0), V_packed.stride(1), V_packed.stride(2), V_packed.stride(3),
        O_2d.stride(0), O_2d.stride(1), O_2d.stride(2),
        sl_b, sl_h,
        K_scale.stride(0), K_scale.stride(1), K_scale.stride(2), K_scale.stride(3),
        K_mn.stride(0), K_mn.stride(1), K_mn.stride(2), K_mn.stride(3),
        V_scale.stride(0), V_scale.stride(1), V_scale.stride(2),
        V_mn.stride(0), V_mn.stride(1), V_mn.stride(2),
        sel_2d.stride(0), sel_2d.stride(1), sel_2d.stride(2),
        H_q, H_kv, S_kv, num_pages,
        HEAD_DIM=D,
        HEAD_DIM_PACKED=D // 2,
        PAGE_SIZE=page_size,
        WRITE_LSE=bool(return_lse),
        num_warps=4,
        num_stages=2,
    )

    O = O_2d.unsqueeze(2)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out


def _flash_attn_sparse_int4_fwd_reference(
    Q: torch.Tensor,
    K_packed: torch.Tensor,
    K_scale: torch.Tensor,
    K_mn: torch.Tensor,
    V_packed: torch.Tensor,
    V_scale: torch.Tensor,
    V_mn: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: float | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Reference path — kept for bit-equivalence testing.

    Dequants INT4 → BF16 → re-quantizes to INT8 → calls the existing INT8
    sparse kernel. Materializes a full-precision K/V intermediate; OOMs at
    32 k+ on 4 GB. The fused kernel above is the production path.
    """
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")
    if K_packed.dtype != torch.uint8 or V_packed.dtype != torch.uint8:
        raise ValueError(
            f"K_packed/V_packed must be uint8; got K={K_packed.dtype} V={V_packed.dtype}"
        )

    from flashquest.kernel.kv_quant import (
        dequantize_k_int4, dequantize_v_int4, quantize_k, quantize_v,
    )
    from flashquest.kernel.sparse_fwd import flash_attn_sparse_fwd

    K_bf16 = dequantize_k_int4(K_packed, K_scale, K_mn, page_size=page_size)
    V_bf16 = dequantize_v_int4(V_packed, V_scale, V_mn)
    K_int8, K_scale8, K_mn8 = quantize_k(K_bf16, page_size=page_size)
    V_int8, V_scale8, V_mn8 = quantize_v(V_bf16)

    return flash_attn_sparse_fwd(
        Q, K_int8, K_scale8, K_mn8,
        V_int8, V_scale8, V_mn8,
        selection_mask=selection_mask,
        page_size=page_size, sm_scale=sm_scale, return_lse=return_lse,
    )
```

- [ ] **Step 2: Run the equivalence test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_int4.py::test_fused_matches_reference -v
```
Expected: PASS. If it fails with a Triton compile error on `tl.join` / `tl.reshape`, check the Triton version (the codebase pins triton>=3.1; both ops are supported there).

Common failure modes and fixes:
- *`tl.join` not found*: the codebase already uses Triton 3.1; if it's missing, replace with `tl.cat([k_lo[:, :, None], k_hi[:, :, None]], dim=2)`.
- *Numerical mismatch* (`max abs err > 1e-2`): inspect whether the reshape order matches the pack order. The pack rule is `lo at d=2k, hi at d=2k+1`. If the test fails by a permutation pattern, the reshape is producing `[lo[0], lo[1], …, hi[0], hi[1], …]` instead of interleaved — switch to `tl.interleave(k_lo, k_hi)` or use explicit gathers via `offs_d // 2` with a `tl.where(offs_d % 2 == 0, lo, hi)` pattern.

- [ ] **Step 3: Run the existing dense-equivalence test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_int4.py::test_sparse_int4_matches_dense -v
```
Expected: PASS at rtol=5e-2 (existing gate).

- [ ] **Step 4: Commit**

```bash
git add src/flashquest/kernel/sparse_int4_fwd.py tests/test_sparse_int4.py
git commit -m "phase 6 task 6: fused INT4 Triton kernel — no BF16 intermediate"
```

---

## Task 4: No-BF16-intermediate test

**Files:**
- Modify: `tests/test_sparse_int4.py`

The reference path's failure mode at 32 k is allocating a `(B, H_kv, S_kv, D)` BF16 tensor (~7 GiB at 32 k). The fused kernel must allocate at most the output `(B, H_q, 1, D)` BF16 plus the LSE FP32 — nothing K/V-shaped.

- [ ] **Step 1: Write the test**

Append to `tests/test_sparse_int4.py`:

```python
def test_no_kv_shaped_bf16_intermediate(monkeypatch):
    """Fused kernel must not allocate any BF16 tensor with K/V-shape (B, H_kv, S_kv, *).

    Tracks every torch.empty / torch.zeros call during flash_attn_sparse_int4_fwd;
    if any such allocation has size matching B*H_kv*S_kv*head_dim*2 bytes (BF16
    K/V intermediate), we've regressed to the reference path.
    """
    from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd
    torch.manual_seed(19)
    B, H_q, H_kv, S_kv, D, page_size = 1, 4, 2, 1024, 64, 64
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_kv, D, dtype=torch.bfloat16, device="cuda")
    K_packed, K_scale, K_mn = quantize_k_int4(K, page_size=page_size)
    V_packed, V_scale, V_mn = quantize_v_int4(V)
    sel = torch.ones(B, H_q, 1, S_kv // page_size, dtype=torch.bool, device="cuda")

    kv_bytes_threshold = B * H_kv * S_kv * D * 2  # BF16 K (or V) intermediate.
    allocations: list[tuple[tuple[int, ...], torch.dtype, int]] = []
    orig_empty = torch.empty
    orig_zeros = torch.zeros

    def _track(name, fn):
        def inner(*args, **kwargs):
            t = fn(*args, **kwargs)
            try:
                nbytes = t.numel() * t.element_size()
                allocations.append((tuple(t.shape), t.dtype, nbytes))
            except Exception:
                pass
            return t
        inner.__wrapped_name__ = name
        return inner

    monkeypatch.setattr(torch, "empty", _track("empty", orig_empty))
    monkeypatch.setattr(torch, "zeros", _track("zeros", orig_zeros))

    flash_attn_sparse_int4_fwd(
        Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
        selection_mask=sel, page_size=page_size,
        sm_scale=D ** -0.5, return_lse=True,
    )

    bf16_kv_intermediates = [
        (shape, dt, nbytes)
        for (shape, dt, nbytes) in allocations
        if dt == torch.bfloat16 and nbytes >= kv_bytes_threshold
    ]
    assert not bf16_kv_intermediates, (
        f"fused kernel allocated K/V-shaped BF16 intermediates "
        f"(threshold {kv_bytes_threshold} bytes): {bf16_kv_intermediates}"
    )
```

- [ ] **Step 2: Run the new test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_int4.py::test_no_kv_shaped_bf16_intermediate -v
```
Expected: PASS — no allocation matches the threshold.

If it fails, the fused kernel is silently falling through to the reference path. Verify the public function name in `sparse_int4_fwd.py` points to the Triton-backed implementation, not `_flash_attn_sparse_int4_fwd_reference`.

- [ ] **Step 3: Run the full test_sparse_int4.py file**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_sparse_int4.py -v
```
Expected: 3 PASS (`test_sparse_int4_matches_dense`, `test_fused_matches_reference`, `test_no_kv_shaped_bf16_intermediate`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_sparse_int4.py
git commit -m "phase 6 task 6: assert fused kernel allocates no BF16 K/V intermediate"
```

---

## Task 5: Phase 1-6 fast-suite regression

**Files:** none (run-only)

- [ ] **Step 1: Run the full fast suite**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -m "not slow"
```
Expected: 200+ passed (199 from end of task 5 + 2 new INT4 fused tests). No regressions in any phase.

If a regression appears in `test_persistent_int4.py::test_int4_dispatcher_smoke_llama_1b` (the slow test that the dispatcher routes through the fused kernel), re-run it explicitly:

```
source .venv/bin/activate && python -m pytest tests/test_persistent_int4.py::test_int4_dispatcher_smoke_llama_1b -v
```
Expected: PASS in <60 s. If logits go non-finite, the dequant path mis-orders nibbles — go back to Task 3 step 2's failure-mode notes.

- [ ] **Step 2: No commit** — this task is a gate, not a code change.

Skip.

---

## Task 6: Single-cell decode bench at 32 k under the fused kernel

**Files:**
- Create: `benchmarks/phase6_decode_int4_fused.json`

- [ ] **Step 1: Run the bench**

Run via Bash `run_in_background` (this is a single-cell, foreground-bounded run; ~5–15 min wall):
```
source .venv/bin/activate && python -u scripts/bench_flashquest.py \
    --ctx-len 32768 --n-decode 32 --kv-bits 4 \
    --out benchmarks/phase6_decode_int4_fused.json
```

Monitor via Monitor on the background process:
```
tail -F /tmp/phase6_decode_int4_fused.log 2>/dev/null | grep --line-buffered -E "decode_tok_s|oom|wall_s|prefill|Traceback|OutOfMemoryError"
```

(If the bench prints to stdout and not a log file, capture with `2>&1 | tee /tmp/phase6_decode_int4_fused.log` in the launch command.)

- [ ] **Step 2: Inspect the JSON**

Run:
```
cat benchmarks/phase6_decode_int4_fused.json | python -m json.tool
```
Expected: a single record with non-null `decode_tok_s`, `peak_vram_mib`, `oom: false`. Compare against the prior INT4 reference-path 32 k cell (which OOM'd):

```
diff <(cat benchmarks/phase6_cells_int4/flashquest_32768.json) benchmarks/phase6_decode_int4_fused.json | head -30
```

The reference-path cell had `oom: true, error: "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 7.83 GiB"`. The fused cell should report a real `decode_tok_s` and a much smaller `peak_vram_mib` (no 7 GiB BF16 intermediate).

- [ ] **Step 3: Commit**

```bash
git add benchmarks/phase6_decode_int4_fused.json
git commit -m "phase 6 task 6: 32k decode bench under fused INT4 kernel"
```

If the bench OOMs or times out: the fused kernel has a regression that survived the unit tests. Likely root causes: (a) Triton compilation produces an oversized scratch buffer; (b) the `tl.join` + `tl.reshape` path materializes an FP32 intermediate per page that scales with `S_kv`. Diagnose by re-running at `--ctx-len 8192` and `--ctx-len 16384` to find where it tips over; look at `peak_vram_mib` to see whether VRAM grows with ctx (reference-path symptom) or stays flat (fused-kernel symptom).

---

## Task 7: RULER NIAH 4k @ INT4 re-test

**Files:** none (run-only; results overwrite `benchmarks/phase6_ruler_4k_int4.json`)

- [ ] **Step 1: Sanity check before launching the long run**

Before committing 50+ minutes of wall, confirm Tasks 5 + 6 are green. Quality must not regress relative to the reference path.

- [ ] **Step 2: Run the full RULER eval**

Run via Bash `run_in_background` (~50 min wall):
```
source .venv/bin/activate && python -u scripts/phase6_run_ruler_4k_int4.py 2>&1 | tee /tmp/phase6_ruler_4k_int4_fused.log
```

Monitor via Monitor on the log:
```
tail -F /tmp/phase6_ruler_4k_int4_fused.log 2>/dev/null | grep --line-buffered -E "===|hits=|All-pass|Wrote|PASS|FAIL|Error|Traceback|OOM"
```

- [ ] **Step 3: Verify the gate still clears**

Run:
```
cat benchmarks/phase6_ruler_4k_int4.json | python -m json.tool | head -30
```
Expected: `"all_pass": true` and per-task ratios ≥0.85 (the existing gate). Numerical drift from the kernel-fused dequant path may shift hit counts by ±1, which is within sampling variance.

If the gate fails (any task <85%): the fused kernel introduces a quality regression beyond the rounding-error budget. Roll back: remove the fused kernel, restore the reference path as the public symbol. The prior task 5 result (100/100/100) stands as the canonical INT4 quality baseline.

- [ ] **Step 4: Commit the updated JSON (if all_pass)**

```bash
git add benchmarks/phase6_ruler_4k_int4.json
git commit -m "phase 6 task 6: RULER 4k INT4 re-test under fused kernel — gate holds"
```

---

## Task 8: Head-to-head re-test under fused kernel

**Files:** none (run-only; results overwrite `benchmarks/phase6_headtohead_int4.{json,md}` and `benchmarks/phase6_cells_int4/`)

- [ ] **Step 1: Archive the prior reference-path INT4 head-to-head results**

Run:
```
mv benchmarks/phase6_headtohead_int4.json benchmarks/phase6_headtohead_int4_reference.json
mv benchmarks/phase6_headtohead_int4.md   benchmarks/phase6_headtohead_int4_reference.md
mv benchmarks/phase6_cells_int4 benchmarks/phase6_cells_int4_reference
```

- [ ] **Step 2: Rename canonical INT8 head-to-head out of the way (orchestrator writes to phase6_headtohead.{json,md})**

```
mv benchmarks/phase6_headtohead.json benchmarks/phase6_headtohead_int8.json
mv benchmarks/phase6_headtohead.md   benchmarks/phase6_headtohead_int8.md
mv benchmarks/phase6_cells benchmarks/phase6_cells_int8
```

- [ ] **Step 3: Run the head-to-head with fused INT4**

Run via Bash `run_in_background` (~1.5 hr wall):
```
source .venv/bin/activate && python -u scripts/phase6_run_headtohead.py 2>&1 | tee /tmp/phase6_headtohead_int4_fused.log
```

Monitor via Monitor on the log (use the same regex as task 5's monitor).

- [ ] **Step 4: Rename the new outputs to the INT4 slot; restore INT8 to canonical**

```
mv benchmarks/phase6_headtohead.json benchmarks/phase6_headtohead_int4.json
mv benchmarks/phase6_headtohead.md   benchmarks/phase6_headtohead_int4.md
mv benchmarks/phase6_cells benchmarks/phase6_cells_int4
mv benchmarks/phase6_headtohead_int8.json benchmarks/phase6_headtohead.json
mv benchmarks/phase6_headtohead_int8.md   benchmarks/phase6_headtohead.md
mv benchmarks/phase6_cells_int8 benchmarks/phase6_cells
```

- [ ] **Step 5: Inspect the diff vs reference-path INT4**

Run:
```
diff benchmarks/phase6_headtohead_int4_reference.md benchmarks/phase6_headtohead_int4.md
```

Expected: flashquest cells improve. The 32 k cell flips from `OOM (BF16 dequant intermediate)` to a real tok/s number; the 8 k cell either matches or improves on the reference path's 1.81 tok/s (the reference path's INT4→BF16→INT8 round-trip overhead is gone).

- [ ] **Step 6: Commit**

```bash
git add benchmarks/phase6_headtohead_int4.json benchmarks/phase6_headtohead_int4.md \
        benchmarks/phase6_cells_int4/ \
        benchmarks/phase6_headtohead_int4_reference.json benchmarks/phase6_headtohead_int4_reference.md \
        benchmarks/phase6_cells_int4_reference/
git commit -m "phase 6 task 6: head-to-head re-run under fused INT4 kernel — SPEC §11.4 re-test"
```

---

## Task 9: Update notes + DOC + README + SPEC; tag

**Files:**
- Modify: `docs/PHASES/phase-6-notes.md`
- Modify: `DOC.md`
- Modify: `README.md`
- Modify: `docs/SPEC.md`

- [ ] **Step 1: Read the measured numbers once**

Run:
```
cat benchmarks/phase6_decode_int4_fused.json
echo '---'
cat benchmarks/phase6_headtohead_int4.md
echo '---'
cat benchmarks/phase6_ruler_4k_int4.json | python -m json.tool | head -40
```
Note the actual numbers — Task 9's docs use them verbatim.

- [ ] **Step 2: Append task 6 section to `docs/PHASES/phase-6-notes.md`**

Append at the end of the file. Replace the placeholders with measured numbers from Step 1:

```markdown

---

# Phase 6 Task 6 Notes

**Started:** 2026-05-03
**Status:** **complete (tag `phase-6-task-6`); fused INT4 Triton kernel ships; SPEC §11.4 ≥5× verdict: <CLEARED / STILL GATED — fill from measured tok/s>.**
**Plan:** [../superpowers/plans/2026-05-03-phase-6-int4-fused-kernel.md](../superpowers/plans/2026-05-03-phase-6-int4-fused-kernel.md)

## Summary

Replaces the reference path in `flashquest.kernel.sparse_int4_fwd` (which
dequantized INT4 → BF16 → re-quantized to INT8 → called the existing
INT8 kernel) with a real Triton kernel that reads packed `uint8`
K/V tiles directly and unpacks lo/hi nibbles inline during the tile load.
Eliminates the BF16 `(B, H_kv, S_kv, D)` intermediate that OOM'd the
reference path at 32 k (~7 GiB allocation).

The unpack uses `tl.join(k_lo, k_hi)` + `tl.reshape` to interleave the
two nibble tiles into a `(PAGE_SIZE, HEAD_DIM)` INT vector matching the
pack layout (lo at `d=2k`, hi at `d=2k+1`). Same online-softmax math,
same selection-mask path, same shape contracts — only the K/V load
+ dequant changes from the INT8 kernel.

## Result — quality (RULER NIAH 4k @ INT4)

`benchmarks/phase6_ruler_4k_int4.json`:

| task | dense | patched (INT4 fused) | ratio | gate ≥85 % |
|---|---|---|---|---|
| niah_single | <fill> | <fill> | <fill> | <fill> |
| niah_multikey | <fill> | <fill> | <fill> | <fill> |
| niah_multivalue | <fill> | <fill> | <fill> | <fill> |

(Replace placeholders with measured values.)

## Result — single-cell decode bench at 32 k

`benchmarks/phase6_decode_int4_fused.json`:

- `decode_tok_s = <fill>`
- `prefill_tok_s = <fill>`
- `peak_vram_mib = <fill>`
- `wall_s = <fill>`
- Compare to reference path (`benchmarks/phase6_cells_int4_reference/flashquest_32768.json`):
  reference-path cell OOM'd at `peak_vram_mib` ≈ 7 GiB allocation attempt.

## Result — head-to-head re-test (SPEC §11.4)

`benchmarks/phase6_headtohead_int4.json`:

| backend | quant + KV | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest INT4 (fused) | AWQ-INT4 + INT4 paged | <fill> | <fill> | <fill> |
| flashquest INT4 (ref, prior) | AWQ-INT4 + INT4 paged via INT8 round-trip | 1.81 | OOM | ✗ |
| flashquest INT8 (prior) | AWQ-INT4 + INT8 paged | 2.29 | timeout (>30 min) | ✗ |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 40.43 | timeout | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | OOM | OOM |

**Verdict:** <CLEARED — flashquest 32 k tok/s = X, ratio vs llama.cpp 8 k = Y× /
STILL GATED — closing axis is TurboQuant per `memory/project_post_v1_kernel_research.md`>.

## v2 follow-ups

- **TurboQuant** (Walsh-Hadamard rotation + Lloyd-Max codebook) — separate spec; further ~3× KV shrink at near-zero quality cost. Composes with the paged INT4 layout.
- **DuoAttention pattern training** for Llama-3.2-3B — replaces the all-retrieval head_pattern with a learned 70/30 split.
- **Prefill INT4 kernel** — current kernel is decode-only (S_q=1). Prefill still goes through the dense BF16 path (cheap at 32 k anyway).
- **Larger block sizes / num_warps autotune** — the fused kernel uses `num_warps=4, num_stages=2` mirroring the INT8 kernel; benchmark sweeps may unlock another 10-30 % decode speedup.
```

- [ ] **Step 3: Append task 6 row to `DOC.md`**

Find:
```markdown
- Phase 6 task 6+ — kernel-fused INT4 unpack, TurboQuant, EAGLE-2, Marlin, ExLlamaV2 (post-§11 follow-up).
```
Replace with:
```markdown
- **Phase 6 task 6 — kernel-fused INT4 unpack** ✅ **complete (tag `phase-6-task-6`); SPEC §11.4 verdict: <CLEARED / STILL GATED>**. `flashquest.kernel.sparse_int4_fwd._sparse_attn_fwd_kernel_int4` is a real Triton kernel that reads packed `uint8` K/V tiles directly and unpacks lo/hi nibbles inline during the tile load. Eliminates the reference path's BF16 `(B, H_kv, S_kv, D)` intermediate that OOM'd at 32 k. Reference path preserved as `_flash_attn_sparse_int4_fwd_reference` for the bit-equivalence test. Decode at 32 k under fused INT4: `<fill>` tok/s, `<fill>` MiB peak. RULER 4k @ INT4 re-test: <fill>/<fill>/<fill>. See `docs/PHASES/phase-6-notes.md`.
- Phase 6 task 7+ — TurboQuant, EAGLE-2, Marlin, ExLlamaV2.
```

(Fill placeholders with measured numbers.)

- [ ] **Step 4: Update `README.md`**

Find the "## Phase 6 task 5 — INT4 KV cache" section. Insert immediately after its last paragraph (before "## Non-goals"):

```markdown
## Phase 6 task 6 — fused INT4 Triton kernel

Replaces the task 5 reference path (INT4 → BF16 → INT8 → existing kernel)
with a real Triton kernel that reads packed `uint8` K/V tiles directly and
unpacks lo/hi nibbles inline. Eliminates the BF16 `(B, H_kv, S_kv, D)`
intermediate that OOM'd the reference path at 32 k.

### Decode at 32 k under the fused kernel

`benchmarks/phase6_decode_int4_fused.json`:

- decode_tok_s = `<fill>`
- peak_vram_mib = `<fill>` (vs reference-path attempt: ~7 GiB OOM)
- wall_s = `<fill>`

### Throughput re-test — `benchmarks/phase6_headtohead_int4.json`

| backend | quant + KV | 8 k tok/s | 32 k tok/s | 128 k fits? |
|---|---|---|---|---|
| flashquest INT4 (fused) | AWQ-INT4 + INT4 paged | **<fill>** | **<fill>** | <fill> |
| flashquest INT4 (ref, prior) | AWQ-INT4 + INT4 via INT8 round-trip | 1.81 | OOM | ✗ |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 40.43 | timeout | ✗ |
| vLLM 0.7.3 | AWQ-INT4, FP16 KV | OOM | OOM | OOM |

(Fill placeholders with measured numbers.)

**SPEC §11.4 ≥5× verdict:** <CLEARED — capability ratio at 32 k is X× over
llama.cpp, which times out / STILL GATED — closing axis is TurboQuant
per `memory/project_post_v1_kernel_research.md`>.

Re-run via:
```bash
python scripts/bench_flashquest.py --ctx-len 32768 --kv-bits 4 \
    --out benchmarks/phase6_decode_int4_fused.json
python scripts/phase6_run_ruler_4k_int4.py        # quality re-test
python scripts/phase6_run_headtohead.py           # head-to-head re-test
```
```

- [ ] **Step 5: Update `docs/SPEC.md`**

Find the §6 task 5 line ("**INT4 KV.** ✅ DONE …"). Insert a new task 6 line immediately after it:

```markdown
6. **Kernel-fused INT4 unpack.** ✅ **DONE (2026-05-03, tag `phase-6-task-6`); SPEC §11.4 verdict: <CLEARED / STILL GATED>.** `flashquest.kernel.sparse_int4_fwd._sparse_attn_fwd_kernel_int4` is a real Triton kernel that reads packed `uint8` K/V tiles directly and unpacks lo/hi nibbles inline during the tile load. Eliminates the reference path's BF16 `(B, H_kv, S_kv, D)` intermediate (~7 GiB at 32 k) that OOM'd on 4 GB. Reference path preserved as `_flash_attn_sparse_int4_fwd_reference` for the bit-equivalence test. Decode at 32 k under fused INT4: `<fill>` tok/s, `<fill>` MiB peak. RULER 4k @ INT4 holds at <fill>/<fill>/<fill>. See `docs/PHASES/phase-6-notes.md`.
```

Then re-number tasks 6→7, 7→8, 8→9 (EAGLE-2, Marlin, ExLlamaV2 — currently §6 task 6 / 7 / 8) so the priority list stays correctly numbered.

Also update the §11 acceptance bullet 4 status text. Find:
```markdown
4. README has a benchmark table showing ≥5× capability gain over `llama.cpp -ngl 999` on the same hardware. **Status (2026-05-03):**
```

Replace its body (after `**Status (2026-05-03):**`) with:

```markdown
INT8 table at `benchmarks/phase6_headtohead.{json,md}`; INT4 reference-path
re-test at `benchmarks/phase6_headtohead_int4_reference.{json,md}`;
INT4 fused-kernel re-test at `benchmarks/phase6_headtohead_int4.{json,md}`.
**Verdict (post-task-6):** <CLEARED — flashquest INT4 fused at 32 k = X tok/s
vs llama.cpp timeout = ∞× capability ratio / STILL GATED — closing axis
is TurboQuant per `memory/project_post_v1_kernel_research.md`>. At 8 k,
flashquest INT4 fused = <fill> tok/s vs llama.cpp 40.43 tok/s; capability
frame: flashquest > vLLM (vLLM OOMs at ≤4 k); flashquest > llama.cpp on
max-fit ctx (32 k decodes vs llama.cpp times out).
```

- [ ] **Step 6: Commit and tag**

```bash
git add docs/PHASES/phase-6-notes.md DOC.md README.md docs/SPEC.md
git commit -m "phase 6 task 6: complete — DOC + README + SPEC + notes (verdict: <CLEARED|STILL GATED>)"
git tag phase-6-task-6
git log --oneline phase-6-task-5..HEAD
```

---

## Self-review (post-write)

**1. Spec coverage** — every requirement in the task brief maps to a task:

- §Goal: replace reference path with a fused Triton kernel → Tasks 1-3.
- §Architecture: tile load + lo/hi unpack via `tl.join` + `tl.reshape`, online-softmax unchanged → Task 3 step 1.
- §Tests: dense-equivalence (existing), fused-vs-reference (Task 2), no-BF16-intermediate (Task 4) → Tasks 2 + 4.
- §Validation gates: unit tests (Tasks 2-4), regression suite (Task 5), 32 k decode bench (Task 6), RULER re-test (Task 7), head-to-head re-test (Task 8) → Tasks 5-8.
- §Rollback: if numerical equivalence fails → Task 3 step 2 documents the fallback decision.

**2. Placeholder scan** — Task 9 contains `<fill>`, `<CLEARED / STILL GATED>` placeholders. These are deliberate: the actual numbers come from Tasks 6-8's measured runs. Same pattern as Phase 6 tasks 4 and 5 used. Engineer instructions are explicit about which JSON to copy from. No other placeholder patterns ("TBD", "implement later", etc.) appear.

**3. Type consistency** — `_sparse_attn_fwd_kernel_int4` (kernel) and `flash_attn_sparse_int4_fwd` (wrapper) names are consistent across the spec, Task 3 implementation, and the test imports in Tasks 2 + 4. The reference function is consistently named `_flash_attn_sparse_int4_fwd_reference` (Task 1 rename, Task 2 import, Task 3 module body, Task 8 archive name). Stride / shape names match the INT8 reference kernel verbatim except `stride_kdp` / `stride_vdp` (packed last-axis stride) which appear consistently in the kernel signature, the Python launcher, and the docstring. `K_packed.shape[-1] == D / 2` is asserted in the wrapper and checked by the existing `test_persistent_int4.py::test_packed_shape`.

No issues found.
