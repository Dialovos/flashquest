# Phase 6 task 5 — INT4 KV cache

**Date:** 2026-05-03
**Status:** Design approved; ready for implementation plan
**SPEC reference:** `docs/SPEC.md` §6 task 5 ("INT4 KV. Phase 6's biggest kernel change. After (1) and (2) ship, re-validate quality at INT4 vs INT8 + dense.") + §11.4 acceptance bullet (the head-to-head ≥5× gate that task 4 measured as not-cleared and annotated as gated on this work).

## Problem

After Phase 6 task 4 the SPEC §11.4 ≥5× gate is unmet on raw tok/s and on max-fit context. The 4 GB VRAM ceiling caps all three benchmarked backends at 32 k+ on Llama-3.2-3B. The clean axis to push next is KV quantization: halving K and V from INT8 (1 byte/element) to INT4 (½ byte/element) drops KV memory traffic 2× per decode step and KV-resident VRAM 2×, opening headroom for either longer contexts or faster decode.

The Phase 0/6 INT8 path is already modular: `kv_quant.py` (quant), `criticality.py` (page_scores_int8_fast), `selection.py` (top-k), `sparse_fwd.py` (Triton kernel), `persistent_int8.py` (Cache). INT4 is a parallel sibling in every module, gated by `cache.kv_bits`.

Two SPEC-level constraints frame the work:

1. **Quality must hold.** RULER NIAH 4k stays the gate (≥85% vs dense per task). INT4 quality may drop; if it does we keep INT8 default and ship INT4 behind a flag.
2. **Hardware target unchanged.** RTX 3050 Ti Laptop, sm_86, 4 GB VRAM, WSL2.

## Decisions

| Question | Decision |
|---|---|
| Scope | **INT4 KV only** (Q1 = A). Kernel-fused criticality and TurboQuant are separate phases per the post-§11 research notes. |
| Quant scheme | **KIVI-style asymmetric INT4** (Q2 = A). Per-page channel-wise K, per-token V. Range 0-15. Identity: `page_max ≡ page_min + 15 × scale`. One-constant change from the existing INT8 algebraic fast path. |
| Storage | **Packed: two 4-bit values per uint8 byte, packed along `head_dim`** (Q3 = A). 2× VRAM shrink vs INT8. Triton inline-unpacks during kernel tile load. |
| Quality rollback | **Soft: both INT4 and INT8 modules co-exist; `--kv-bits {4,8}` flag** (Q4 = B). INT8 stays default until INT4 clears RULER ≥85%; if INT4 clears, default flips to 4. If not, INT4 lives behind the flag. |

## Architecture

```
src/flashquest/kernel/
├── kv_quant.py              # MODIFY — add INT4 quant primitives + pack/unpack
└── sparse_int4_fwd.py       # NEW — Triton sparse kernel with inline INT4 unpack

src/flashquest/eager/
├── criticality.py           # MODIFY — add page_scores_int4_fast
├── sparse_int4.py           # NEW — eager-path wrapper around sparse_int4_fwd
└── llama_persistent_patch.py # MODIFY — branch on cache.kv_bits

src/flashquest/cache/
└── persistent_int4.py       # NEW — PersistentInt4KVCache(Cache), packed storage

src/flashquest/runtime/
└── chat.py                  # MODIFY — --kv-bits {4,8} flag

scripts/
├── bench_flashquest.py      # MODIFY — --kv-bits flag
└── phase6_run_ruler_4k_int4.py  # NEW — RULER NIAH 4k re-run at INT4

tests/
├── test_kv_quant_int4.py    # NEW
├── test_persistent_int4.py  # NEW
├── test_sparse_int4.py      # NEW
├── test_eager_criticality.py # MODIFY — INT4 fast-path tests
└── test_chat.py             # MODIFY — --kv-bits flag dispatch
```

INT8 modules remain untouched. INT4 modules are parallel siblings. Dispatcher lives in `llama_persistent_patch.py` (reads `cache.kv_bits`, picks the matching `page_scores_int{4,8}_fast` and `flash_attn_sparse_int{4,8}_fwd`).

## Components

### `kernel/kv_quant.py` additions

```python
def _pack_int4(x: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values per byte along the last axis.
    x: uint8 tensor with values in 0..15, last-axis even.
    out: uint8 tensor with shape (..., last_axis // 2).
    """
    lo = x[..., 0::2]
    hi = x[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of _pack_int4. Returns uint8 with values in 0..15, last-axis 2× input."""
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2,
        dtype=torch.uint8, device=packed.device,
    )
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def quantize_k_int4(K, page_size):
    """Asymmetric per-page channel-wise INT4 quant.
    Returns (K_packed (B, H, S, D//2) uint8, scale (B, H, P, D) bf16, mn (B, H, P, D) bf16).
    Identity: K ≈ unpack(K_packed) * scale + mn; page_max ≡ mn + 15*scale exactly.
    """
    scale, mn = _scale_mn_per_page_channel_int4(K, page_size)
    # scale = (max - min) / 15.0
    K_q4 = ((K_norm).round().clamp(0, 15)).to(torch.uint8)
    return _pack_int4(K_q4), scale.to(torch.bfloat16), mn.to(torch.bfloat16)


def dequantize_k_int4(K_packed, scale_per_token, mn_per_token):
    K_q4 = _unpack_int4(K_packed)
    return K_q4.to(torch.float32) * scale_per_token.float() + mn_per_token.float()


# Per-token V (range 0..15, packed along head_dim).
def quantize_v_int4(V): ...
def dequantize_v_int4(V_packed, scale, mn): ...
```

### `eager/criticality.py` addition

```python
def page_scores_int4_fast(Q, K_scale, K_mn):
    """Identical to page_scores_int8_fast except term2 *= 15 instead of 255.

    Algebraic identity: max(Q·K_mn, Q·K_mx) = Q·K_mn + 15·relu(Q)·K_scale, since K_scale ≥ 0
    and page_max ≡ K_mn + 15*K_scale.
    """
    # Same shape contracts: Q (B, H_q, S_q, D); K_scale, K_mn (B, H_kv, P, D);
    # GQA grouped via view, not repeat_interleave.
    ...
    term2 = torch.matmul(Q_pos_g, Kscale_f.transpose(-1, -2)) * 15.0
    ...
```

### `kernel/sparse_int4_fwd.py`

Triton kernel mirroring `sparse_fwd.py`. Tile load reads `K_packed` (uint8) and `V_packed` (uint8) with `head_dim/2` extent in the head-dim axis; inline unpack:

```python
# Inside the kernel tile:
k_byte = tl.load(K_packed_ptr + offsets)              # uint8
k_lo   = k_byte & 0xF                                  # 0..15
k_hi   = (k_byte >> 4) & 0xF
# Interleave into a head_dim-extent FP register:
k_int  = tl.where(d_axis & 1 == 0, k_lo, k_hi)
k_f32  = k_int.to(tl.float32) * scale[d] + mn[d]      # FP32 accumulate
```

Same online-softmax math as the INT8 kernel; only the load + dequant changes. The selection mask path is unchanged.

### `cache/persistent_int4.py`

`PersistentInt4KVCache(Cache)` with class-level `kv_bits = 4`. Storage shapes (vs INT8 in parens):

- `K_packed`: `(L, B, H_kv, max_seq_len, head_dim/2)` uint8 (was `head_dim` uint8)
- `V_packed`: `(L, B, H_kv, max_seq_len, head_dim/2)` uint8
- `K_scale`, `K_mn`: `(L, B, H_kv, num_pages, head_dim)` BF16 — unchanged
- `V_scale`, `V_mn`: `(L, B, H_kv, max_seq_len, 1)` BF16 — unchanged
- `partial_K_bf16`: `(L, B, H_kv, page_size, head_dim)` BF16 — unchanged (partial-page staging stays full-precision)

Public API mirrors `PersistentInt8KVCache`:
- `update_quantized(layer_idx, K_full, V_full)` — same signature; internally uses `quantize_k_int4` / `quantize_v_int4`.
- `get_views(layer_idx)` — same dict shape, with `K_packed` substituted for `K_uint8`.
- `get_seq_length(layer_idx)`, `get_max_length()` — unchanged.
- Class attribute `kv_bits = 4`.

`PersistentInt8KVCache` exposes `kv_bits = 8` (one-line addition; behavior unchanged).

### `eager/llama_persistent_patch.py` dispatcher

```python
if cache.kv_bits == 4:
    from flashquest.eager.criticality import page_scores_int4_fast as _scores
    from flashquest.eager.sparse_int4 import quest_sparse_int4_fwd as _sparse_fwd
elif cache.kv_bits == 8:
    from flashquest.eager.criticality import page_scores_int8_fast as _scores
    from flashquest.eager.sparse_int8 import quest_sparse_int8_fwd as _sparse_fwd
else:
    raise ValueError(f"unsupported cache.kv_bits={cache.kv_bits}")
```

### `runtime/chat.py` flag

```python
p.add_argument("--kv-bits", type=int, choices=[4, 8], default=8)
...
if args.kv_bits == 4:
    cache = PersistentInt4KVCache(...)
else:
    cache = PersistentInt8KVCache(...)
```

Same in `bench_flashquest.py`.

## Data flow

```
Prefill                                    Decode
-------                                    ------
K (B,H,S,D) BF16                           K_packed (B,H,S,D/2) uint8
  │                                          │
  ▼                                          ▼
quantize_k_int4 (round + clip + pack)      page_scores_int4_fast(Q, K_scale, K_mn)
  │                                          │  no unpack: identity uses BF16 statistics
  ▼                                          ▼
K_packed, K_scale, K_mn                    page_scores → select_pages_vectorized
                                             │
                                             ▼
                                           flash_attn_sparse_int4_fwd(
                                             Q, K_packed, K_scale, K_mn,
                                             V_packed, V_scale, V_mn,
                                             selection_mask, page_size, sm_scale)
                                             │  Triton kernel:
                                             │    1. read selected pages of K/V_packed (uint8)
                                             │    2. inline unpack lo/hi 4-bit → uint8
                                             │    3. dequant to FP32 via scale + mn
                                             │    4. attention (existing online-softmax)
                                             ▼
                                           attn_out (B,H,S_q,D) BF16
```

The page-scoring path operates on BF16 statistics (`K_scale`, `K_mn`) and never touches the packed uint8 storage — same trick that makes the INT8 fast path 22× faster than dequant. Only the sparse kernel unpacks INT4, and only for selected pages (~25% retention), with unpack bandwidth-amortized inside the tile load.

## Edge cases / tests

| ID | Case | Test |
|---|---|---|
| ER1 | INT4 round-trip within quant error | `test_kv_quant_int4.py::test_roundtrip` |
| ER2 | Algebraic identity: `K_mn + 15 × K_scale` ≡ page_max | `test_kv_quant_int4.py::test_page_max_identity` |
| ER3 | `page_scores_int4_fast` ≡ slow reference at rtol=1e-3 | `test_eager_criticality.py::test_int4_fast_matches_slow` |
| ER4 | `sparse_int4_fwd` ≡ dense reference at rtol=5e-2 | `test_sparse_int4.py::test_matches_dense` |
| ER5 | Cache packed-storage shape: `K_packed.shape[-1] == head_dim // 2` | `test_persistent_int4.py::test_packed_shape` |
| ER6 | INT4 + INT8 caches coexist in same process; dispatcher picks right path | `test_persistent_int4.py::test_int4_int8_coexist` |
| ER7 | CLI `--kv-bits 4` constructs INT4 cache; default 8 stays INT8 | `test_chat.py::test_kv_bits_flag_dispatches` |
| ER8 | Odd `head_dim` rejected at construction with clear error | `test_persistent_int4.py::test_odd_head_dim_rejected` |

## Validation gates

| Gate | Target |
|---|---|
| Unit tests for quant + criticality + sparse INT4 | All ER1-ER8 green |
| Smoke: Llama-3.2-1B at ctx=512, INT4 KV, single decode | <60 s, no NaN, no OOM |
| RULER NIAH 4k @ INT4 vs dense | **≥85% per task** — promotion criterion |
| Phase 1-6 fast suite | No regressions; 178+ tests still green |
| Re-run head-to-head bench at INT4 (8 k / 32 k / 128 k) | Capture in `benchmarks/phase6_headtohead_int4.{json,md}` |

**Quality-gate decision:**

```
RULER NIAH 4k @ INT4
       │
       ├── all 3 tasks ≥85% → INT4 promoted to default (chat default flips to --kv-bits 4)
       │                       README + SPEC §11.4 re-test;
       │                       INT8 remains available via --kv-bits 8.
       │
       └── any task <85% → INT4 stays opt-in via --kv-bits 4
                            INT8 default; SPEC §11.4 stays gated on TurboQuant
                            or kernel-fused criticality. Failure documented honestly.
```

## Non-goals

- Kernel-fused criticality + top-k (separate phase per post-§11 research).
- TurboQuant (Walsh-Hadamard rotation + Lloyd-Max codebook; separate spec).
- Symmetric INT4 / AWQ-group-quant for KV (eliminated by Q2).
- Hybrid per-layer INT4/INT8 mix (eliminated by Q4).
- DuoAttention pattern training (v2).
- 8B models (same VRAM ceiling; deferred).

## File touch list

| File | Action |
|---|---|
| `src/flashquest/kernel/kv_quant.py` | Modify |
| `src/flashquest/kernel/sparse_int4_fwd.py` | Create |
| `src/flashquest/eager/criticality.py` | Modify |
| `src/flashquest/eager/sparse_int4.py` | Create |
| `src/flashquest/eager/llama_persistent_patch.py` | Modify |
| `src/flashquest/cache/persistent_int4.py` | Create |
| `src/flashquest/cache/persistent_int8.py` | Modify (add `kv_bits = 8` class attribute) |
| `src/flashquest/runtime/chat.py` | Modify (`--kv-bits` flag) |
| `scripts/bench_flashquest.py` | Modify (`--kv-bits` flag) |
| `scripts/phase6_run_ruler_4k_int4.py` | Create |
| `tests/test_kv_quant_int4.py` | Create |
| `tests/test_persistent_int4.py` | Create |
| `tests/test_sparse_int4.py` | Create |
| `tests/test_eager_criticality.py` | Modify |
| `tests/test_chat.py` | Modify |
| `benchmarks/phase6_ruler_4k_int4.json` | Create at gate run |
| `benchmarks/phase6_headtohead_int4.{json,md}` | Create at re-test |
| `docs/PHASES/phase-6-notes.md` | Append task 5 section |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 5; update §11.4 |

## Open questions

None. The four brainstorm decisions resolve scope, quant scheme, storage, and rollback. INT4 modules are parallel siblings of the INT8 path; dispatcher branches on `cache.kv_bits`; quality gate is the existing RULER NIAH 4k harness re-run with `--kv-bits 4`.
