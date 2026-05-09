# Phase 9 — Prompt-Lookup Decoding (Greedy Chain) — Design Spec

**Date:** 2026-05-08
**Status:** spec r1 (pre-codex review)
**Phase:** 9 (after Phase 8a foundation, Phase 8b dropped by profile)
**Target:** ≥15 tok/s @ 32k decode on Llama-3.2-3B-AWQ + Quest sparse INT4, RTX 3050 Ti Laptop sm_86 4GB WSL2

---

## §1 Goal

Add lossless greedy speculative decoding to the existing Quest sparse-attention runtime via **Prompt-Lookup Decoding** (PLD). PLD speculates by mining n-gram matches from the *prompt*, which exactly fits the targeted long-context retrieval / RAG / summarize workload.

Throughput target: **≥15 tok/s @ 32k decode** (2.4× over today's 6.29 tok/s clean baseline). Floor target: **≥18 tok/s @ 8k decode** (1.8× over 10.06 tok/s).

Quality target: **bit-identical output token sequence** to non-spec greedy decoding wherever the verify step's argmax decision is robust to BF16 LSE-merge noise. RULER NIAH 4k must hold 100/100 single + ≥85% multivalue.

Out of scope (Phase 9):
- Sampling. Greedy-only. Sampling needs the Leviathan/Chen rejection sampler trick — deferred.
- Tree-PLD (multiple drafts at branching). Chain-only.
- Lookahead/Jacobi n-gram mining from generated tail. Deferred to potential Phase 9b.
- Self-speculative early-exit (layer-skip draft). Ruled out for this phase.
- INT8/Turbo cache PLD support. INT4 only — matches Phase 8a's `use_compact_kernel` constraint.

## §2 Context

**Phase 8a (tag `phase-8a`) shipped:**
- Compact-list INT4 sparse Triton kernel with sentinel-padding ABI: `selected_page_ids: int32[B, H_q, BUCKET_MAX]`, `-1` sentinel, load-time masking, `qk = NEG_INF` for invalid slots. **Decode-only (S_q=1 hard assert at line 181 of `kernel/sparse_int4_fwd_compact.py`).**
- GPU-resident sort-based `build_compact_selection`.
- `.item()`-free `select_pages_vectorized(..., k_max_static=...)`.
- AWQ layout assertion (verified `qweight (in, out//8)` packing along OUT axis).
- Flag-gated `--use_compact_kernel` integration in `patch_llama_for_quest_persistent`.
- Honest finding: standalone speedup is ~1× (bool-mask kernel was already efficient). Foundation work, not a wallclock win on its own.

**Phase 8b (`be20e3a`) was dropped by profile** — CUDA Graph upper bound was 1.02× because GPU is already 98% saturated at 8k decode. Cache view redesign Phase 8b would have done is **not** carried forward to Phase 9 — PLD's accept-count semantics don't need GPU-resident `_seen_tokens` or fixed-max views.

**Today's clean baselines (no host contention):**
- 8k decode: **10.06 tok/s** (`scripts/phase8b_profile_decode.py`)
- 32k decode: **6.29 tok/s** (extrapolated from same script + Phase 7 32k bench under clean conditions)

**Cumulative roadmap math (with Phase 9 hitting 2.4×):**

| Phase | Speedup | Cumulative tok/s @ 32k |
|---|---|---|
| baseline today | — | 6.29 |
| Phase 9 PLD (this spec) | 2.4× | 15.1 |
| Phase 10 DuoAttention/CATS | 1.5× | 22.6 |
| Phase 11 lookahead (gen-side) | 1.3× | 29.4 |

The user's "OOM gain to ~30 tok/s" target on the 32k decode workload is achievable through Phase 9+10+11.

## §3 Approach overview

PLD greedy chain:
1. After every step, save the model's argmax at the just-committed position as `next_argmax_buffer` (the "free token" prediction).
2. On a step, take the last `K_match` (default 3) committed tokens, search `prompt_ids` for that pattern (rightmost match for recency).
3. If found, propose the next `N_draft` (default 5) tokens from the prompt as `[D_0, D_1, ..., D_{N_draft-1}]`. PLD requires `D_0 == next_argmax_buffer` for the draft to be admissible (this is the verification of D_0); if not, fall back to single-token decode.
4. If admissible: forward `[next_argmax_buffer, D_1, D_2, ..., D_{N_draft-1}]` of length `S_q = N_draft` through the model.
   - K/V for these positions written to a per-layer **sandbox** (BF16, not yet committed to the persistent cache).
   - Sparse Quest attention reads cache at positions `0..N-1` (committed) with the **compact INT4 kernel extended to S_q > 1** (this is the key new kernel work).
   - Dense BF16 attention reads sandbox at positions `N..N+N_draft-1` (the in-flight verify K/Vs) with `is_causal=True`. Tiny `N_draft × N_draft` matmul.
   - Per-query LSE-merge of sparse + dense outputs.
5. After full forward + lm_head: `argmax(logits[i])` for `i ∈ [0, N_draft)` predicts position `N+i+1`. Walk-and-accept against `D_{i+1}`. Accept count `M` = longest matching prefix of `D_1..D_{N_draft-1}` against `argmax(logits[0..N_draft-2])`.
6. Total accepted this step: `M+1` drafts (D_0 verified by previous step, plus `M` more) + 1 free token (`argmax(logits[M])`) = **M+2 emitted tokens**.
7. Cache commit: `cache.commit_draft(M+1)` writes the first `M+1` sandbox K/Vs (one per accepted draft including D_0) to the persistent cache. The free token's K/V is held as `next_input_token` for the next step (it does not get a fresh forward this step). `next_argmax_buffer` is updated to be... → see §4.5 for the precise free-token bookkeeping.

**Why this fits our stack uniquely well:**
- The sparse-over-cache + dense-over-tail + LSE-merge is exactly the same pattern as `_bf16_dense_attn_with_lse` + `_merge_two_attentions` already in `llama_persistent_patch.py` for the partial-page tail. Sandbox replaces the partial-page tail in verify mode.
- Sandbox commit goes through existing `update_quantized` (no rollback math).
- 32k context is exactly where prompt-lookup hit-rate is highest (longer prompt = more n-gram coverage).

## §4 Architecture

### §4.1 PLD draft proposer — `flashquest/specdec/pld.py`

```
def propose_draft(
    prompt_ids: torch.Tensor,       # (S_prompt,) int64 on CPU
    history_tail: torch.Tensor,     # (K_match,) int64 last committed tokens, on CPU
    *,
    K_match: int = 3,
    N_draft: int = 5,
) -> torch.Tensor | None:           # (N_draft,) int64 on CPU, or None
    """Find the rightmost K_match-gram match in prompt_ids; return next N_draft tokens.

    Returns None if (a) history_tail length < K_match,
                   (b) no match found in prompt_ids,
                   (c) match is at the very end (no continuation).
    """
```

Implementation: vectorized scan over `prompt_ids` for the K_match-gram in `history_tail[-K_match:]`. Use rightmost match (highest index). If match starts at index `i`, return `prompt_ids[i+K_match : i+K_match+N_draft]`.

Edge case: if there's a partial match at the end (only `M < N_draft` tokens after match), pad to None (return None) — we want full-length drafts. Or alternatively return what's available; v1 chooses pad-to-None to keep verify shape static.

Performance: O(S_prompt) per call, runs on CPU. At S_prompt=32k that's 32k × K_match comparisons = ~100k ops, ~50µs. Negligible compared to verify forward.

### §4.2 Cache sandbox + commit API — `flashquest/cache/persistent_int4.py`

Add to `PersistentInt4KVCache.__init__`:

```
self.MAX_DRAFT = 8     # bound; configurable per-cache or constant
shape_sandbox = (num_layers, batch_size, num_kv_heads, self.MAX_DRAFT, head_dim)
self.K_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
self.V_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
self._sandbox_count = [0] * num_layers   # per-layer sandbox population
```

Memory footprint at MAX_DRAFT=8: `28 × 1 × 8 × 8 × 128 × 2 (BF16) × 2 (K+V) ≈ 920 KB` — negligible vs the 925 MB main cache.

New methods:

```
def add_draft(self, K_new: torch.Tensor, V_new: torch.Tensor, layer_idx: int) -> None:
    """Write fresh K/V into per-layer sandbox slots [0..S_new-1]. Does NOT advance _seen_tokens."""
    S_new = K_new.shape[2]
    if S_new > self.MAX_DRAFT:
        raise ValueError(f"S_new={S_new} > MAX_DRAFT={self.MAX_DRAFT}")
    self.K_sandbox[layer_idx, :, :, :S_new, :] = K_new
    self.V_sandbox[layer_idx, :, :, :S_new, :] = V_new
    self._sandbox_count[layer_idx] = S_new

def get_views_with_sandbox(self, layer_idx: int) -> dict:
    """Existing get_views() result + sandbox K/V views."""
    views = self.get_views(layer_idx)
    s = self._sandbox_count[layer_idx]
    views["K_sandbox"] = self.K_sandbox[layer_idx, :, :, :s, :]
    views["V_sandbox"] = self.V_sandbox[layer_idx, :, :, :s, :]
    views["sandbox_count"] = s
    return views

def commit_draft(self, accept_count: int, layer_idx: int) -> None:
    """Commit first `accept_count` sandbox positions to persistent cache."""
    if accept_count == 0:
        self._sandbox_count[layer_idx] = 0
        return
    s = self._sandbox_count[layer_idx]
    if accept_count > s:
        raise ValueError(f"accept_count={accept_count} > sandbox_count={s}")
    K_commit = self.K_sandbox[layer_idx, :, :, :accept_count, :]
    V_commit = self.V_sandbox[layer_idx, :, :, :accept_count, :]
    self.update_quantized(K_commit, V_commit, layer_idx)
    self._sandbox_count[layer_idx] = 0

def discard_draft(self, layer_idx: int) -> None:
    """No-op (sandbox is overwritten on next add_draft); kept for API clarity."""
    self._sandbox_count[layer_idx] = 0
```

`_seen_tokens` remains a Python `list[int]`. PLD's accept-count semantics are entirely managed via sandbox; no GPU-resident scalar required.

### §4.3 Verify-mode forward branch — `eager/llama_persistent_patch.py`

Extend `make_quest_persistent_forward` with a third dispatch branch (alongside existing `S_q > 1` prefill and `S_q = 1` decode):

```
S_q = q.shape[2]
verify_mode = getattr(self, "_pld_verify_active", False)

if S_q > 1 and not verify_mode:
    # Prefill (existing path) — dense SDPA, writes to cache via update_quantized
    ...
elif S_q == 1:
    # Single-token decode (existing path) — sparse Quest + partial-page merge
    ...
else:
    # S_q > 1 AND verify_mode — PLD verify path (NEW)
    cache.add_draft(k, v, layer_idx=self.layer_idx)
    views = cache.get_views_with_sandbox(self.layer_idx)
    completed_len = views["completed_len"]
    if completed_len == 0:
        # First decode steps before any complete page — sandbox-only attention
        K_full = views["K_sandbox"]
        V_full = views["V_sandbox"]
        attn_output = bf16_dense_attn_causal(q, K_full, V_full)
    else:
        # Per-Q-head selection on the LAST query (highest recency) — shared across S_q
        scores = _criticality_scores(q[:, :, -1:, :], views)   # use Q[N_draft-1] for selection
        sel = select_pages_vectorized(
            scores, retention=retention_per_q,
            num_sinks=num_sinks, window_pages=window_pages,
            k_max_static=k_max_static,
        )
        sel_compact = build_compact_selection(sel, BUCKET_MAX=bucket_max_static)
        # Compact kernel S_q > 1 extension (see §4.4)
        O_sparse, lse_sparse = flash_attn_sparse_int4_fwd_compact(
            q, views["K_packed"], views["K_scale"], views["K_mn"],
            views["V_packed"], views["V_scale"], views["V_mn"],
            selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
        )    # O_sparse: (B, H_q, S_q, D); lse_sparse: (B, H_q, S_q)
        # Dense over sandbox — small causal attn
        O_draft, lse_draft = bf16_dense_attn_causal_with_lse(
            q, views["K_sandbox"], views["V_sandbox"],
        )    # O_draft: (B, H_q, S_q, D); lse_draft: (B, H_q, S_q)
        # LSE-merge per Q (extended to S_q > 1)
        attn_output = _merge_two_attentions_sq(
            O_sparse, lse_sparse, O_draft, lse_draft,
        )   # (B, H_q, S_q, D)
```

`bf16_dense_attn_causal_with_lse(Q, K, V)` is a helper — extension of existing `_bf16_dense_attn_with_lse` from S_q=1 to S_q>1 with `is_causal=True`. SDPA call returns the attention output; LSE is computed from `qk.max(-1)` and `log(p.sum(-1))`. ~5 lines.

`_merge_two_attentions_sq` extends `_merge_two_attentions` to S_q > 1 — it's the same online-softmax merge formula applied per query position. Currently the helper does `lse_a.unsqueeze(-1)` for S_q=1 broadcast; we change it to operate over the S_q axis directly.

### §4.4 Compact INT4 kernel S_q>1 extension — `kernel/sparse_int4_fwd_compact.py`

Add `S_q: tl.constexpr` to kernel signature and an inner Q-loop:

```
@triton.jit
def _sparse_attn_fwd_kernel_int4_compact(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr,
    sm_scale,
    ...,
    H_q, H_kv, S_kv, S_q,                    # S_q is now a runtime arg as well as constexpr
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PACKED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUCKET_MAX: tl.constexpr,
    SQ_MAX: tl.constexpr,                    # NEW — kernel compile-time bound on S_q
    WRITE_LSE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H_q
    h_q = pid_bh % H_q
    n_rep = H_q // H_kv
    h_kv = h_q // n_rep

    offs_n = tl.arange(0, PAGE_SIZE)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dp = tl.arange(0, HEAD_DIM_PACKED)
    offs_sq = tl.arange(0, SQ_MAX)            # query-axis indices
    sq_mask = offs_sq < S_q                   # mask SQ_MAX > S_q

    # Load Q[B, H_q, S_q, D] for all S_q queries at once
    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh
        + offs_sq[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=sq_mask[:, None], other=0.0)   # (SQ_MAX, HEAD_DIM)

    # Per-Q accumulators
    m_i = tl.full((SQ_MAX,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((SQ_MAX,), dtype=tl.float32)
    acc = tl.zeros((SQ_MAX, HEAD_DIM), dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504

    for i in range(0, BUCKET_MAX):
        # Load page (same as S_q=1 kernel) — selection is shared across S_q
        sel_off = b * stride_selb + h_q * stride_selh + i * stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0
        p_safe = tl.where(page_valid, p, 0)
        page_start = p_safe * PAGE_SIZE
        n_idx = page_start + offs_n
        valid_kv = (n_idx < S_kv) & page_valid

        # Load K-tile (page_size × head_dim) ONCE — reused across all S_q queries
        k_byte_ptrs = (
            K_packed_ptr + b * stride_kb + h_kv * stride_kh
            + n_idx[:, None] * stride_ks + offs_dp[None, :] * stride_kdp
        )
        k_byte = tl.load(k_byte_ptrs, mask=valid_kv[:, None], other=0)
        k_lo = (k_byte & 0xF).to(tl.uint8)
        k_hi = ((k_byte >> 4) & 0xF).to(tl.uint8)
        k_int_2 = tl.join(k_lo, k_hi)
        k_int = tl.reshape(k_int_2, (PAGE_SIZE, HEAD_DIM))

        ks_ptrs = K_scale_ptr + b * stride_ksb + h_kv * stride_ksh + p_safe * stride_ksp + offs_d * stride_ksd
        km_ptrs = K_mn_ptr + b * stride_kmb + h_kv * stride_kmh + p_safe * stride_kmp + offs_d * stride_kmd
        k_scale = tl.load(ks_ptrs).to(tl.float32)
        k_mn = tl.load(km_ptrs).to(tl.float32)
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]    # (PAGE_SIZE, HEAD_DIM)

        # qk = Q · K^T per Q — produces (SQ_MAX, PAGE_SIZE)
        qk = tl.dot(q.to(tl.float32), tl.trans(k))
        # mask: invalid_kv → NEG_INF; rows beyond S_q stay finite but get masked at LSE write
        qk = tl.where(valid_kv[None, :], qk, float("-inf"))

        qk_scaled = qk * qk_scale
        qk_max = tl.max(qk_scaled, axis=1)            # (SQ_MAX,)
        m_ij = tl.maximum(m_i, qk_max)
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p_softmax = tl.math.exp2(qk_scaled - m_ij_safe[:, None])
        p_softmax = tl.where(qk == float("-inf"), 0.0, p_softmax)

        alpha = tl.math.exp2(m_i - m_ij_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p_softmax, axis=1)
        acc = acc * alpha[:, None]

        # Load V-tile ONCE — reused across all S_q
        v_byte_ptrs = ...
        # (same as S_q=1 kernel)
        v_int = ...
        v_scale = ...; v_mn = ...
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]    # (PAGE_SIZE, HEAD_DIM)

        acc += tl.dot(p_softmax, v)                  # (SQ_MAX, HEAD_DIM)

        m_i = m_ij

    # Normalize + write
    safe_l = tl.where(l_i == 0.0, 1.0, l_i)
    acc = acc / safe_l[:, None]

    o_ptrs = (
        O_ptr + b * stride_ob + h_q * stride_oh
        + offs_sq[:, None] * stride_os + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=sq_mask[:, None])

    if WRITE_LSE:
        lse_val = (m_i + tl.math.log2(safe_l)) * 0.69314718
        lse_val = tl.where(l_i == 0.0, float("-inf"), lse_val)
        l_ptr_bh = (
            L_ptr + b * stride_lb + h_q * stride_lh + offs_sq * stride_ls
        )
        tl.store(l_ptr_bh, lse_val, mask=sq_mask)
```

**Constexpr `SQ_MAX`** is the static upper bound on S_q. **Must be ≥16** because Triton 3.1.0's `tl.dot` requires each dimension ≥16. We set `SQ_MAX=16` and tolerate masked rows when actual S_q < 16. With `MAX_DRAFT=8` the upper half of the SQ axis is always masked — wasted compute is bounded at 2× per-CTA (Q load + accumulator math) but K/V tile loads are unaffected (still loaded once per page, the dominant cost).

**SQ axis masking at the qk product** is required so masked rows don't pollute downstream softmax via leaked zero values:
```
qk = tl.dot(q.to(tl.float32), tl.trans(k))      # (SQ_MAX, PAGE_SIZE)
qk = tl.where(sq_mask[:, None] & valid_kv[None, :], qk, float("-inf"))
```
The SQ-mask alone forces masked rows' qk = -inf → softmax = 0 → no contribution to acc/lse. The output store uses `mask=sq_mask[:, None]` so we don't write past S_q in O.

**Performance argument:** the K-tile (`(PAGE_SIZE × HEAD_DIM)`) and V-tile loads dominate kernel time at decode (memory-bandwidth bound, per Phase 8a's microbench finding). With S_q>1, the K/V loads happen ONCE per page but are reused across all S_q queries — this is precisely the win specdec needs.

**Codegen size:** kernel body is ~150 lines; one new register pressure source is `acc[SQ_MAX, HEAD_DIM]` = 6×128 floats = 3 KB — fits in registers on sm_86.

**S_q=1 backward compatibility:** when `S_q=1` the kernel produces the same numerical output as Phase 8a's per-Q version. Implementation: keep the existing kernel as `_sparse_attn_fwd_kernel_int4_compact_sq1` for the S_q=1 hot path (avoiding any unmasked-axis overhead), dispatch on `sel_3d.dim() == 3 vs 4` or a flag in `flash_attn_sparse_int4_fwd_compact`. This keeps Phase 8a's existing decode bench unaffected.

### §4.5 PLD dispatcher + walk-and-accept — `flashquest/specdec/dispatcher.py`

```
def make_quest_pld_dispatcher(
    model: torch.nn.Module,
    cache: PersistentInt4KVCache,
    prompt_ids: torch.Tensor,            # (S_prompt,) int64
    *,
    K_match: int = 3,
    N_draft: int = 5,
):
    """Wrap `model` (already patched with persistent_int4 attention) with PLD generation."""
    state = {
        "next_input_token": None,        # token to feed at next step (= last accepted)
        "next_argmax_buffer": None,      # model's argmax at next_input_token's position (= "free token")
        "history": [],                   # generated token list for n-gram lookup tail
    }

    def init(prompt_input_ids: torch.Tensor):
        """Run prefill, set next_input_token = prompt's argmax, save next_argmax_buffer."""
        out = model(input_ids=prompt_input_ids, use_cache=True, logits_to_keep=1)
        state["next_argmax_buffer"] = out.logits[:, -1].argmax(dim=-1)   # (B,) — predict at position S_prompt
        state["next_input_token"] = state["next_argmax_buffer"]
        # Note: "predict at position S_prompt" means cache._seen = S_prompt, and the next forward
        # will input next_input_token to fill position S_prompt and produce logits at S_prompt
        # which become the NEW next_argmax_buffer for the step AFTER that.
        #
        # PLD warmup: state["history"] is empty after init. PLD requires K_match committed tokens
        # to form a search-key n-gram, so the first K_match (=3) emitted tokens always go
        # through the single-token decode branch. Steady-state PLD activation begins at step
        # K_match+1.

    def step() -> torch.Tensor:
        """Generate one or more tokens; return all newly-emitted token IDs (CPU int64)."""
        next_in = state["next_input_token"]
        prev_argmax = state["next_argmax_buffer"]   # used to verify D_0
        history_tail = torch.tensor(state["history"][-K_match:], device="cpu") if len(state["history"]) >= K_match else None
        draft = (
            propose_draft(prompt_ids.cpu(), history_tail, K_match=K_match, N_draft=N_draft)
            if history_tail is not None else None
        )

        # Admissibility: D_0 must equal prev_argmax (else can't lossless-accept any drafts).
        # prev_argmax is None right after a PLD step (we never computed argmax for the free token);
        # in that case PLD cannot activate this step and we fall back to single-token decode.
        if (
            draft is None
            or prev_argmax is None
            or draft[0].item() != prev_argmax.item()
        ):
            # Single-token decode (existing path)
            with torch.no_grad():
                out = model(input_ids=next_in.unsqueeze(0), use_cache=True, logits_to_keep=1)
            new_argmax = out.logits[:, -1].argmax(dim=-1)
            emitted = next_in.unsqueeze(0)        # (1,) — next_in commits at this step
            state["history"].append(int(next_in.item()))
            state["next_input_token"] = new_argmax
            state["next_argmax_buffer"] = new_argmax
            return emitted

        # PLD verify path
        # Verify input = [next_in (= D_0), D_1, D_2, ..., D_{N_draft-1}] of length N_draft
        verify_input = torch.cat([next_in.unsqueeze(0), draft[1:]], dim=0).unsqueeze(0)   # (1, N_draft)
        _set_pld_verify_active(model, True)
        with torch.no_grad():
            out = model(input_ids=verify_input, use_cache=True)   # logits (1, N_draft, vocab)
        _set_pld_verify_active(model, False)

        # Walk-and-accept: argmax(logits[i]) predicts position N+i+1 (= D_{i+1})
        argmax_seq = out.logits.argmax(dim=-1).squeeze(0)   # (N_draft,)
        accept_count = 0
        for i in range(N_draft - 1):
            if argmax_seq[i].item() == draft[i + 1].item():
                accept_count += 1
            else:
                break
        # M = accept_count drafts AFTER D_0 verified by prev_argmax
        # Total accepted in this step: D_0 (= next_in) + accept_count more drafts + 1 free
        total_accepted = 1 + accept_count          # D_0 + accept_count post-D_0 drafts
        free_token = argmax_seq[accept_count]      # at position N + total_accepted
        # Commit total_accepted K/Vs from sandbox (the first total_accepted positions)
        # NOTE: layer-by-layer is handled by the model's verify pass via cache.add_draft;
        #       the dispatcher commits all layers via a single helper:
        for layer_idx in range(cache.num_layers):
            cache.commit_draft(total_accepted, layer_idx)
        # Discard the unused (rejected) sandbox tail
        for layer_idx in range(cache.num_layers):
            cache.discard_draft(layer_idx)         # no-op cleanup

        # Emitted tokens this step: D_0, D_1, ..., D_{accept_count}, free_token
        emitted = torch.cat([next_in.unsqueeze(0), draft[1:1 + accept_count], free_token.unsqueeze(0)], dim=0)
        for t in emitted.tolist():
            state["history"].append(int(t))
        # Free token becomes next_input_token; we don't yet have its argmax, so next step
        # will fall back to single-token decode (one cycle of cold-start before PLD can resume)
        state["next_input_token"] = free_token
        state["next_argmax_buffer"] = None         # forces next step to single-token decode
        return emitted

    return init, step
```

**`next_argmax_buffer` lifecycle nuance:**
- After single-token decode (S_q=1): `next_argmax_buffer = logits.argmax()` ← we have it for free.
- After PLD verify (S_q>1): the free token is the argmax at position `N + total_accepted`. We don't have its argmax (would need ANOTHER forward over the free token alone). So PLD step always sets `next_argmax_buffer = None`, forcing the *next* step to be a single-token decode that recomputes it.
- Consequence: PLD step always followed by 1 single-token step. Best-case cadence: PLD (M+2 emitted) → single (1 emitted) → PLD → single → … = (M+3) tokens per 2 steps. At M=4 mean accept (target), that's 7 tokens per (1 PLD + 1 decode) = 7 emitted per (1.1× single-step cost + 1× single-step cost) ≈ 7 / 2.1 = 3.3× single-step throughput. Matches our 2.4× target with margin.

Alternative (deferred): include the free token in a follow-up forward to reclaim `next_argmax_buffer`, so PLD steps can cascade. Adds ~1× single-step cost per PLD step (4× emitted / ~2.1× cost = 1.9× — same as alternative path). Not necessarily a win; profile in Task 1.

**`_pld_verify_active` plumbing:**
Set on every patched LlamaAttention module before/after PLD verify forward. A single boolean attribute on each attention layer; the layer's forward branches on `getattr(self, "_pld_verify_active", False)`. Helper `_set_pld_verify_active(model, value)` walks the modules.

## §5 Data flow trace — one PLD step

```
state.next_input_token = T (= prev step's free token, K/V NOT in cache)
state.next_argmax_buffer = A (= model's prediction at T's position; computed prior step)
state.history = [..., T_{N-2}, T_{N-1}, T]   # T is most recent

  ┌─────────────────────────────────────────────────────────────┐
  │ propose_draft(prompt, history_tail=[T_{N-2}, T_{N-1}, T])   │
  │   → finds match in prompt at position p                     │
  │   → returns draft = [D_0=p+3, D_1=p+4, ..., D_4=p+7]        │
  └─────────────────────────────────────────────────────────────┘

  D_0 == A?  → admissibility check
    no  → single-token decode of T, exit
    yes → continue verify

  ┌─────────────────────────────────────────────────────────────┐
  │ verify_input = [T, D_1, D_2, D_3, D_4]   # S_q = 5          │
  │ for layer ∈ 0..27:                                          │
  │   q, k, v ← projections + RoPE                              │
  │   cache.add_draft(k, v, layer)            # sandbox write   │
  │   views ← cache.get_views_with_sandbox(layer)               │
  │   sel ← select_pages_vectorized(views, q[..., -1, :], …)    │
  │   sel_compact ← build_compact_selection(sel, BUCKET_MAX)    │
  │   O_sparse, lse_sparse ← compact_kernel_sq5(q, views, sel)  │
  │   O_draft,  lse_draft  ← bf16_dense_causal(q, K_sb, V_sb)   │
  │   O ← merge_per_q(O_sparse, lse_sparse, O_draft, lse_draft) │
  │   ↓ residual + MLP                                          │
  │ hidden_states[B, 5, hidden]                                 │
  │ logits[B, 5, vocab] ← lm_head(hidden_states)                │
  └─────────────────────────────────────────────────────────────┘

  argmax_seq = argmax(logits[0]) ... argmax(logits[4])

  walk-and-accept:
    argmax_seq[0] vs D_1 → match? continue. count=1
    argmax_seq[1] vs D_2 → match? continue. count=2
    argmax_seq[2] vs D_3 → mismatch. break. accept_count=2
  free_token = argmax_seq[2]
  total_accepted = 1 (D_0) + 2 = 3

  for layer ∈ 0..27: cache.commit_draft(3, layer)
  emitted = [T, D_1, D_2, free_token]   # 4 tokens
  state.history.extend(emitted)
  state.next_input_token = free_token
  state.next_argmax_buffer = None        # next step must be single-token decode
```

## §6 Task 1 — profile-first (entry criterion)

Per the saved profile-first lesson (`feedback_profile_before_speedup_specs.md`), Phase 9's first task is empirical, not architectural. We don't write the kernel S_q>1 extension or the sandbox API until we've measured PLD's accept rate and verify-cost on our specific workload + hardware.

**Implementation:** `benchmarks/phase9_task1_pld_profile.py`
- Use the **existing dense SDPA prefill path** as the verify forward (S_q=N_draft dense over the dequantized cache+sandbox; no sparse, no kernel changes).
- Naive single-layer cache mock: dequantize cache once at start of step, run dense SDPA on (Q, K_full, V_full) where K_full = dequant(K_packed) || K_sandbox.
- Walk-and-accept logic identical to §4.5.

**Measurement:**
- Workloads (greedy decoding, 256-token output each):
  1. PG-essay summarize 8k input — 20 prompts from `data/PaulGrahamEssays.json`
  2. RULER NIAH single 4k — `flashquest.eval.niah_single`, 20 instances
  3. HumanEval 100 prompts (code completion) — load from HF datasets
  4. MT-bench-100 first turns (free-form chat) — load from HF datasets
- Per workload, 100 verify steps minimum, record:
  - mean accept count per step (`M+2` per the §3 math) at K_match=3, N_draft=5
  - 95th-percentile per-step wall time at S_q=5 vs S_q=1 dense forward
  - hit rate (% of steps where PLD draft is admissible — i.e., D_0 == prev_argmax)

**Gates (decided BEFORE writing kernels):**
- Mean accept count ≥ 2.0 averaged over (PG-summarize + RULER) — the targeted retrieval workloads
- S_q=5 dense verify cost ≤ 1.3× S_q=1 dense baseline cost on Llama-3.2-3B-AWQ at 8k context
- Hit rate (admissible step %) ≥ 30% on PG-summarize + RULER

If all three clear: proceed to Tasks 2-N. Otherwise: stop, report findings, decide whether to re-scope to Lookahead or Self-spec early-exit, or drop Phase 9 entirely.

## §7 Phase 9 success gates (full)

| Gate | Threshold | Source |
|---|---|---|
| Task 1 accept-rate (PG-summ + RULER) | mean ≥ 2.0 | `benchmarks/phase9_task1_pld_profile.py` |
| Task 1 verify cost ratio | S_q=5 ≤ 1.3× S_q=1 | same |
| Task 1 admissibility hit rate | ≥ 30% | same |
| Unit + parity tests | all green | `tests/test_*.py` |
| RULER NIAH single 4k | 100/100 vs dense ref | `python -m flashquest.eval.runner --task niah_single --pld-on` |
| RULER NIAH multivalue 4k | ≥ 85% (≥17/20) | same |
| 32k summarize decode | ≥ 15 tok/s | `benchmarks/phase9_decode_bench.py --ctx 32k` |
| 8k summarize decode | ≥ 18 tok/s | same |

## §8 Test plan (TDD)

| Test file | What it asserts |
|---|---|
| `tests/test_pld_proposer.py` | n-gram match: no match → None; rightmost match wins; K_match > history → None; pad partial → None |
| `tests/test_cache_sandbox.py` | `add_draft` + `commit_draft(M)` ≡ direct `update_quantized(K[:M])`; `commit_draft(0)` no-op; cross-page boundary commits handled correctly |
| `tests/test_compact_kernel_sq_gt_1.py` | S_q=N kernel parity vs S_q=1 kernel run N times; numeric tolerance ≤ 1e-2 BF16; S_q=1 codepath byte-equivalent to Phase 8a |
| `tests/test_compact_kernel_sq_gt_1_sentinels.py` | `selected_page_ids` with `-1` sentinels: no OOB, qk = -inf for invalid slots, output unchanged |
| `tests/test_pld_verify_parity.py` | Force all drafts to mismatch (`D_i = prev_argmax`-shifted) → emits 1 token (free token) ≡ non-spec single-token decode bit-for-bit |
| `tests/test_pld_walk_and_accept.py` | Drafts `[A,B,C,D]` vs argmax `[A,B,X,D]` → accept_count=2, free=`X`; all-match → accept_count=N-1, free=`logits[N-1].argmax`; all-mismatch → accept_count=0 |
| `tests/test_pld_dispatcher_e2e.py` | Llama-3.2-3B-AWQ on synthetic prompt with verbatim continuation → PLD path emits identical token sequence to non-spec greedy |
| `tests/test_persistent_patch_pld_verify.py` | Patched LlamaAttention forward in verify mode (`_pld_verify_active=True`) does NOT call `update_quantized` and DOES call `add_draft`; sandbox cleared after `commit_draft` |

## §9 RULER quality gate

PLD with greedy decoding is mathematically equivalent to non-spec greedy decoding when the verify step's argmax decision is deterministic. The only source of non-equivalence is **BF16 LSE-merge numeric drift** (sparse + sandbox attention paths use different precision than the dense prefill reference).

For RULER NIAH (retrieval), the answer token is unambiguous (logit margin ≫ BF16 noise), so PLD and non-spec greedy MUST emit identical tokens.

**RULER 4k @ Llama-3.2-3B-AWQ + INT4 KV + PLD on:**
- `niah_single`: 100/100 vs non-spec dense reference
- `niah_multivalue`: ≥ 85% (≥17/20), matching Phase 7's K3-V3 benchmark

If RULER produces non-identical tokens between PLD and non-spec: **P0 bug** — investigate walk-and-accept, sandbox commit, or LSE-merge numerics. Do not ship.

## §10 Decode benchmark

`benchmarks/phase9_decode_bench.py`:
- Backends: `non-spec` (Phase 8a baseline, `use_compact_kernel=False` and `use_compact_kernel=True` for parity), `PLD-greedy` (this phase) at K_draft ∈ {3, 5, 8}.
- Contexts: 8k, 16k, 32k.
- Workloads: summarize (PG-essay), RAG (NIAH-style query+context).
- Output length: 256 tokens.
- 5 runs per cell, report mean ± stddev tok/s.
- One backend at a time, `nice -n 19`, `run_in_background` + Monitor (per saved feedback `avoid_wsl_lag`).
- Pick K_draft per (ctx, workload) optimum; document tradeoffs.

## §11 File layout

NEW (Phase 9):
- `src/flashquest/specdec/__init__.py`
- `src/flashquest/specdec/pld.py` — `propose_draft`
- `src/flashquest/specdec/dispatcher.py` — `make_quest_pld_dispatcher`, `_set_pld_verify_active`, walk-and-accept
- `benchmarks/phase9_task1_pld_profile.py` — Task 1 profile + accept-rate
- `benchmarks/phase9_decode_bench.py` — full PLD vs baseline at multiple ctx + workloads
- `tests/test_pld_proposer.py`
- `tests/test_cache_sandbox.py`
- `tests/test_compact_kernel_sq_gt_1.py`
- `tests/test_compact_kernel_sq_gt_1_sentinels.py`
- `tests/test_pld_verify_parity.py`
- `tests/test_pld_walk_and_accept.py`
- `tests/test_pld_dispatcher_e2e.py`
- `tests/test_persistent_patch_pld_verify.py`

EXTENDED (Phase 9):
- `src/flashquest/cache/persistent_int4.py` — sandbox tensors + `add_draft` / `get_views_with_sandbox` / `commit_draft` / `discard_draft`
- `src/flashquest/eager/llama_persistent_patch.py` — verify-mode dispatch branch + `_pld_verify_active` plumbing + `_merge_two_attentions_sq` extension
- `src/flashquest/kernel/sparse_int4_fwd_compact.py` — S_q>1 kernel variant (keep S_q=1 as a separate codegen path)
- `docs/PHASES/phase-9-notes.md` — phase journal (created at end)

UNCHANGED:
- `src/flashquest/eager/selection.py` — `select_pages_vectorized` already supports the per-step single call we need (verify uses Q[N-1] for selection)
- `src/flashquest/quant/awq_layout.py`
- `src/flashquest/cache/persistent_int8.py` (PLD is INT4-only)

## §12 Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Profile gate fails (accept rate < 2.0)** | Stop. Do not proceed to kernel work. Report findings; pivot to Lookahead or self-spec. Saved memory `feedback_profile_before_speedup_specs.md` enforces this. |
| **S_q>1 kernel slower than expected** | Microbench at Task 4 (kernel completion) before integration. If S_q=5 sparse cost ≥ 1.5× S_q=1 cost, batch via 5 separate S_q=1 calls instead (preserves correctness, kills the kernel-amortization win, but ships). |
| **LSE-merge numeric drift breaks RULER bit-identity** | Add a tolerance escape: relax bit-identity to "argmax-identical with prob ≥99% across 1000 prompts." If even that fails, investigate the specific layer where drift exceeds expected BF16 bound. |
| **Sandbox + main-cache view inconsistency at page boundaries** | `commit_draft` wraps existing `update_quantized` which already handles partial-page logic. Test cross-page commit explicitly in `test_cache_sandbox.py`. |
| **`next_argmax_buffer = None` cadence kills throughput** | Profile in Task 1: measure (PLD step + single decode) vs (2× single decode) ratio. If the cadence cost dominates, implement free-token follow-up forward in v2. |
| **VRAM headroom at 32k tight (~5.5 GB peak today; 4 GB hardware)** | Sandbox is 920 KB — negligible. Main risk is verify-step's S_q=5 attention having larger activation footprint; budget for ~50 MB extra activations. If OOM at 32k, reduce N_draft to 3. |
| **PLD admissibility hit rate < 30% in chat-style queries** | Targeted workload is RAG/summarize; if user runs chat we just decode normally (the dispatcher's fallback is single-token decode = current path). Document this in `phase-9-notes.md`. |
| **Codex r1 catches a kernel correctness bug** | Lifted from Phase 8a discipline: full r1 review before plan-writing. If r1 surfaces dealbreakers, rewrite spec to r2 before proceeding. Maximum 3 codex review cycles per Phase 8a precedent. |

## §13 Open questions for codex r1 review

Things to specifically ask codex to stress-test:
1. **`next_argmax_buffer` correctness**: is the dispatcher's "PLD verifies D_0 by comparing against prev_argmax" mathematically equivalent to "PLD includes T_anchor in input and verifies D_0 against logits[0].argmax"? Specifically does the sparse Quest selection in this step affect the prev_argmax that was computed in the previous step? (Should not — selection is a function of Q + K_views which haven't changed.)
2. **Selection sharing across S_q>1**: is using `q[:, :, -1:, :]` (last query) for `select_pages_vectorized` a correctness hazard? The selection is per-H_q which is already approximate; sharing across S_q is one more layer. Argument FOR: the queries are sequential within a single decode step, so their attention patterns over the same cache are nearly identical. Argument AGAINST: queries earlier in the sequence are positionally more similar to old context than later queries, so their top-k pages might differ.
3. **Kernel `dot` vs `sum-of-product` + SQ_MAX min size**: Phase 8a's S_q=1 kernel uses `tl.sum(q[None, :] * k, axis=1)` for the QK product. Our S_q>1 kernel switches to `tl.dot(q, tl.trans(k))` because `tl.sum-of-broadcast` doesn't extend to a 2D Q. Does `tl.dot` handle the float32 path correctly on sm_86? (Phase 2's dense kernel uses tl.dot on BF16 inputs with float32 accumulator — should be the same.) Also: Triton 3.1.0's `tl.dot` requires each dim ≥16. We set `SQ_MAX=16` with masking for S_q<16. Concrete codex check: does Triton 3.1.0 actually enforce that 16-min, or does it pad transparently? If it pads, we could use SQ_MAX=8 to halve the wasted Q-load + accumulator work. If it errors, our SQ_MAX=16 is mandatory.
4. **Cache invariant after a partially-accepted PLD step**: cache._seen_tokens advances by `total_accepted` (1 + accept_count, ≤ N_draft). Sandbox count is reset. If the K/V at sandbox slots `total_accepted..N_draft-1` are "stale" — the next step's `add_draft` will overwrite. Is there any read pathway between PLD steps that would see the stale data? (None I can find — `K_sandbox` is only read inside `get_views_with_sandbox`, which is only called by the verify-mode forward branch, which always calls `add_draft` first.)
5. **`commit_draft` per-layer vs at-once**: dispatcher iterates 28 layers calling `commit_draft(M, layer_idx)`. Each commit calls `update_quantized` which does its own torch CUDA work. 28 sequential commits = 28× CPU dispatch latency. Should we batch into a single layer-axis-aware commit? (Marginal at decode where we're memory-bound, but might matter at small ctx.)
6. **Page-flush race between layers**: `update_quantized` mutates `K_partial`/`V_partial` via `torch.cat` which allocates new tensors. If two layers' commits race (they shouldn't in a single-threaded eager forward, but worth confirming), would the cache state corrupt? (No — Python GIL serializes the commits.)

## §14 References

- **PLD original blog**: Saxena, "Prompt Lookup Decoding," Nov 2023. https://github.com/apoorvumang/prompt-lookup-decoding
- **PLD paper**: "Prompt Lookup Decoding for Faster LLM Inference" (Saxena et al., 2023)
- **HF `assisted_generation` PLD impl** (reference for input-shape conventions): `transformers/src/transformers/generation/utils.py`, `prompt_lookup` candidate-generator path
- **Leviathan-Chen rejection sampler** (for Phase 9b sampling support, not v1): "Fast Inference from Transformers via Speculative Decoding," Leviathan et al., 2023
- **Phase 8a foundation work**: `docs/PHASES/phase-8a-notes.md`, tag `phase-8a`
- **Phase 8b kill record**: `docs/PHASES/phase-8b-killed.md`, profile script `scripts/phase8b_profile_decode.py`
- **Profile-first memory**: `~/.claude/projects/-home-hoang-code-personal-active-flashquest/memory/feedback_profile_before_speedup_specs.md`
