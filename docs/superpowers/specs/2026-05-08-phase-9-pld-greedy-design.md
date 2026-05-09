# Phase 9 — Prompt-Lookup Decoding (Greedy Chain) — Design Spec

**Date:** 2026-05-08 (r1) → 2026-05-09 (r2, r2.1)
**Status:** spec r2.1 (post-codex r2; addresses HIGH-severity UNION-overflow concern + 3 minor residuals from codex r2 review)
**Phase:** 9 (after Phase 8a foundation, Phase 8b dropped by profile)
**Target:** ≥15 tok/s @ 32k decode on Llama-3.2-3B-AWQ + Quest sparse INT4, RTX 3050 Ti Laptop sm_86 4GB WSL2

---

## §1 Goal

Add lossless-by-construction greedy speculative decoding to the existing Quest sparse-attention runtime via **Prompt-Lookup Decoding** (PLD). PLD speculates by mining n-gram matches from the *prompt*, which exactly fits the targeted long-context retrieval / RAG / summarize workload.

Throughput target: **≥15 tok/s @ 32k decode** (2.4× over today's 6.29 tok/s clean baseline). Floor target: **≥18 tok/s @ 8k decode** (1.8× over 10.06 tok/s).

**Quality target (revised from r1):** Phase 9 does **not** claim bit-identical equivalence to non-spec greedy. PLD verify uses a *shared* selection scheme over multiple queries (UNION of per-Q top-k, see §4.4), which is a different attention-approximation than non-spec single-step Quest. The actual quality contract:
- **RULER NIAH 4k @ Llama-3.2-3B-AWQ + INT4 + PLD on**: 100/100 single, ≥85% multivalue (matching Phase 7's K3-V3 numbers).
- **Argmax matches non-spec greedy** with empirical probability ≥ 99% on representative prompts (target measured in Task 1; reported per-workload in `phase-9-notes.md`).
- **No catastrophic divergence**: PLD path must not produce token sequences that drift > 1 token away from non-spec greedy on retrieval workloads.

PLD's lossless guarantee in standard literature relies on the verify path running the *same* model the original would. Our verify path runs the *same Quest-sparse model with shared UNION selection* — slightly different from per-Q selection but provably ≥ per-Q in attention coverage (UNION ⊇ per-Q top-k). The empirical-quality contract is what we ship against.

Out of scope (Phase 9):
- Sampling. Greedy-only. Sampling needs the Leviathan/Chen rejection sampler trick — deferred.
- Tree-PLD (multiple drafts at branching). Chain-only.
- Lookahead/Jacobi n-gram mining from generated tail. Deferred to potential Phase 9b.
- Self-speculative early-exit (layer-skip draft). Ruled out for this phase.
- INT8/Turbo cache PLD support. INT4 only — matches Phase 8a's `use_compact_kernel` constraint.

## §2 Context

**Phase 8a (tag `phase-8a`) shipped:**
- Compact-list INT4 sparse Triton kernel with sentinel-padding ABI: `selected_page_ids: int32[B, H_q, BUCKET_MAX]`, `-1` sentinel, load-time masking, `qk = NEG_INF` for invalid slots. **Decode-only (S_q=1 hard assert).**
- GPU-resident sort-based `build_compact_selection`.
- `.item()`-free `select_pages_vectorized(..., k_max_static=...)`.
- AWQ layout assertion (`qweight (in, out//8)` packed along OUT axis).
- Flag-gated `--use_compact_kernel` integration in `patch_llama_for_quest_persistent`.
- Honest finding: standalone speedup is ~1× (bool-mask kernel was already efficient). Foundation work, not a wallclock win on its own.

**Phase 8b (`be20e3a`) dropped by profile** — CUDA Graph upper bound was 1.02× because GPU is already 98% saturated at 8k decode. Cache view redesign Phase 8b would have done is **not** carried forward to Phase 9 — PLD's accept-count semantics don't need GPU-resident `_seen_tokens`.

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

**Phase 9 ROI math (revised — corrects r1's M+2-per-step error):**

PLD chain emits **M+1 tokens per PLD step**, where M is the count of accepted post-D_0 drafts (M ∈ [0, N_draft−1]). The free token (`argmax(logits[M])`) is *held* as `next_input_token` for the next step, **not emitted in the PLD step itself** (codex r1 dealbreaker #2: emitting both in PLD step caused duplicate-token bug).

The cadence is:
- PLD step (cost ≈ 1.5× single-decode in BW-bound regime) emits M+1 tokens.
- Forced single-decode step next (cost = 1× single-decode) emits 1 token (the held free_token).
- Per (PLD + single) cycle: 2 steps, 2.5× single-decode cost, M+2 tokens.

Gain over single-only baseline (M+2 single-steps emit M+2 tokens at M+2 single-step cost):
- gain = (M+2) × 1.0 / 2.5 = **(M+2) / 2.5**

| Mean M | Gain | tok/s @ 32k baseline 6.29 |
|---|---|---|
| 2 | 1.6× | 10.1 (below target) |
| 3 | 2.0× | 12.6 (below target) |
| **4** | **2.4×** | **15.1 (target)** |
| 5 (= max for K_draft=5) | 2.8× | 17.6 |

**Implication:** hitting 15 tok/s requires mean M ≥ 4, i.e. ≥ 80% of post-D_0 drafts accepted on average. This is aggressive; the published PLD literature reports ~50-70% accept rates on typical workloads. We expect long-context retrieval to be at the *upper end* of literature (output = verbatim copy of prompt regions), but Task 1's empirical gate is what truly informs whether this target survives.

## §3 Approach overview

PLD greedy chain. **Verify input is `[D_0, D_1, ..., D_{N_draft-1}]`** (length S_q = N_draft); D_0 = `next_argmax_buffer` (= last-step argmax, not yet committed).

Per step:
1. State at start: `cache._seen = N`, `next_input_token = T` (= last single-decode's argmax, K/V not yet in cache), `next_argmax_buffer = T` if last step was single-decode, else `None`. `committed_history` = list of all already-committed token IDs.
2. PLD admissibility check:
   - If `len(committed_history) < K_match`: fall back to single-decode (history too short for n-gram lookup).
   - If `next_argmax_buffer is None`: fall back (last step was PLD; we don't know argmax for free token; can't verify D_0 by equality).
   - Search `prompt_ids` for the rightmost match of `committed_history[-K_match:]`. If no match: fall back.
   - If match found at prompt index p, candidate draft = `prompt_ids[p+K_match : p+K_match+N_draft]`. If insufficient prompt remains (< N_draft tokens after match): fall back.
   - If `draft[0] != next_argmax_buffer`: fall back (D_0 cannot be verified by previous step's argmax; PLD's lossless invariant fails).
   - **Page-boundary guard**: if `(N + N_draft) // page_size > N // page_size`: fall back. PLD must not cross a page boundary because the verify path keeps sandbox K/V in BF16 and a non-spec single-decode at the same boundary would have quantized to INT4. (Costs ~6-8% of steps near boundary; alternative is simulating quantize-on-boundary in verify, which we defer.)
3. If admissible: build verify input `[D_0, D_1, ..., D_{N_draft-1}]` (S_q = N_draft). Note: `D_0 == next_input_token == next_argmax_buffer == T`.
4. Forward verify input through model:
   - Per layer: compute Q, K_new, V_new for all N_draft positions.
   - `cache.add_draft(K_new, V_new, layer_idx)` — write to per-layer BF16 sandbox (positions 0..N_draft−1 in sandbox = positions N..N+N_draft−1 in full cache).
   - **Sparse over completed pages** (positions 0..completed_len−1): compact-list INT4 kernel **extended to S_q > 1** with **UNION selection** (§4.4). One kernel invocation, per-(B, H_q) program, S_q query rows, BUCKET_MAX_UNION pages.
   - **Dense over (committed partial-page tail || sandbox)**: BF16 attention. K_dense = `cat(views['K_partial'], views['K_sandbox'], dim=2)` of length `partial_len + N_draft`. Same for V_dense. Causal mask: query at position `partial_len + i` (in dense_tail coords) attends to dense_tail[0..(partial_len + i)].
   - **LSE-merge per-query**: combine sparse + dense outputs via online-softmax merge, indexed per S_q axis.
5. After full forward + lm_head: `argmax(logits[i])` for `i ∈ [0, N_draft)` predicts the token at position `N+i+1`. Walk-and-accept:
   - For `i = 0, 1, ..., N_draft-2`: if `argmax(logits[i]) == draft[i+1]`: accept (continue). Else: break.
   - M = index where the walk broke (0 ≤ M ≤ N_draft − 1).
   - free_token = `argmax(logits[M])` (always accepted; if M = N_draft − 1, free_token = `argmax(logits[N_draft-1])`).
6. Emitted this step: `[D_0, D_1, ..., D_M]` (M+1 tokens; **does NOT include free_token**).
7. Cache commit: `cache.commit_draft(M+1, layer_idx)` for each layer. Atomic two-phase: preflight all 28 layers' sandbox states first, then commit (raises before any mutation if a layer is mis-staged).
8. State update: `next_input_token = free_token`, `next_argmax_buffer = None` (forces next step to single-decode), `committed_history.extend([D_0, D_1, ..., D_M])`.

**Why this fits our stack:**
- The sparse-over-cache + dense-over-tail + LSE-merge pattern is exactly the existing single-decode shape, generalized to S_q > 1 over `(partial_page_tail || sandbox)`.
- Sandbox commit goes through existing `update_quantized` (no rollback math); page-boundary guard ensures no cross-page commits during verify.
- 32k context is exactly where prompt-lookup hit-rate is highest.

## §4 Architecture

### §4.1 PLD draft proposer — `flashquest/specdec/pld.py`

```
def propose_draft(
    prompt_ids: torch.Tensor,           # (S_prompt,) int64 on CPU
    history_tail: torch.Tensor,         # (K_match,) int64 on CPU
    *,
    K_match: int = 3,
    N_draft: int = 5,
) -> torch.Tensor | None:               # (N_draft,) int64 on CPU, or None
    """Find rightmost K_match-gram match in prompt_ids; return next N_draft tokens or None.

    Returns None if (a) history_tail length < K_match, (b) no match, (c) match at end (insufficient continuation).
    """
```

Implementation: rolling-hash or vectorized scan over `prompt_ids` for the K_match-gram in `history_tail[-K_match:]`. Rightmost match (highest index). At S_prompt=32k, K_match=3: ~32k integer compares = ~50 µs. Negligible.

### §4.2 Cache sandbox + commit API — `flashquest/cache/persistent_int4.py`

Add to `__init__`:

```
self.MAX_DRAFT = 8                                               # static upper bound on N_draft
shape_sandbox = (num_layers, batch_size, num_kv_heads, MAX_DRAFT, head_dim)
self.K_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
self.V_sandbox = torch.zeros(shape_sandbox, dtype=torch.bfloat16, device=dev)
self._sandbox_count = [0] * num_layers
```

Footprint at MAX_DRAFT=8: `28 × 1 × 8 × 8 × 128 × 2 bytes (BF16) × 2 (K+V) ≈ 920 KB` — negligible vs 925 MB main cache.

Methods:

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
    """Existing get_views() result + sandbox K/V views. Sandbox K/V keys are BF16 tensors of shape
    (B, H_kv, sandbox_count, D)."""
    views = self.get_views(layer_idx)
    s = self._sandbox_count[layer_idx]
    views["K_sandbox"] = self.K_sandbox[layer_idx, :, :, :s, :]
    views["V_sandbox"] = self.V_sandbox[layer_idx, :, :, :s, :]
    views["sandbox_count"] = s
    return views

def preflight_commit(self, accept_count: int, layer_idx: int) -> None:
    """Validate sandbox is ready for accept_count commit. Does NOT mutate state. Raises on bad input."""
    if accept_count < 0:
        raise ValueError(f"accept_count={accept_count} negative")
    s = self._sandbox_count[layer_idx]
    if accept_count > s:
        raise ValueError(f"accept_count={accept_count} > sandbox_count={s}")
    seen = self._seen_tokens[layer_idx]
    page_size = self.page_size
    # Page-boundary guard: PLD spec forbids cross-page commits; assert it
    if (seen + accept_count) // page_size != seen // page_size and accept_count > 0:
        raise RuntimeError(
            f"commit_draft({accept_count}) on layer {layer_idx} would cross page boundary "
            f"(seen={seen}, page_size={page_size}); page-boundary guard violated"
        )

def commit_draft(self, accept_count: int, layer_idx: int) -> None:
    """Commit first `accept_count` sandbox positions to persistent cache.
    Caller is responsible for invoking preflight_commit on ALL layers first."""
    if accept_count == 0:
        self._sandbox_count[layer_idx] = 0
        return
    K_commit = self.K_sandbox[layer_idx, :, :, :accept_count, :]
    V_commit = self.V_sandbox[layer_idx, :, :, :accept_count, :]
    self.update_quantized(K_commit, V_commit, layer_idx)
    self._sandbox_count[layer_idx] = 0

def commit_draft_all_layers(self, accept_count: int) -> None:
    """Two-phase atomic commit across all layers. Preflight first → then commit.
    Raises (without mutating) if any layer's preflight fails."""
    for layer_idx in range(self.num_layers):
        self.preflight_commit(accept_count, layer_idx)
    for layer_idx in range(self.num_layers):
        self.commit_draft(accept_count, layer_idx)
```

`discard_draft` is dropped (commit_draft already clears sandbox count; codex r1 nit #3).

`_seen_tokens` remains a Python `list[int]`; PLD does not require GPU-resident scalar state.

### §4.3 Verify-mode forward branch — `eager/llama_persistent_patch.py`

Extend `make_quest_persistent_forward` with a third dispatch branch alongside existing `S_q > 1` (prefill dense) and `S_q = 1` (decode sparse):

```
S_q = q.shape[2]
verify_mode = getattr(self, "_pld_verify_active", False)

if S_q > 1 and not verify_mode:
    # Prefill (existing path)
    ...
elif S_q == 1:
    # Single-token decode (existing path)
    ...
else:
    # S_q > 1 AND verify_mode — PLD verify (NEW)
    cache.add_draft(k, v, layer_idx=self.layer_idx)
    views = cache.get_views_with_sandbox(self.layer_idx)
    completed_len = views["completed_len"]
    partial_len = views["partial_len"]
    sandbox_count = views["sandbox_count"]
    assert sandbox_count == S_q

    # Sparse over completed pages (S_q > 1, UNION selection — see §4.4)
    if completed_len > 0:
        # Per-Q scores; UNION mask over S_q axis
        scores = _criticality_scores(q, views)            # (B, H_q, S_q, P_completed)
        sel_per_q = select_pages_vectorized(
            scores, retention=retention_per_q,
            num_sinks=num_sinks, window_pages=window_pages,
            k_max_static=k_max_static,
        )                                                 # (B, H_q, S_q, P_completed) bool
        # Score-prioritized UNION-with-truncation. See §4.4 for the rationale: on
        # overflow we keep the BUCKET_MAX_UNION highest-priority union pages and
        # FORCE sinks + window pages to always be retained. No CUDA sync.
        sel_compact = build_compact_union_selection(
            sel_per_q, scores,
            num_sinks=num_sinks, window_pages=window_pages,
            completed_len=completed_len, page_size=page_size,
            BUCKET_MAX_UNION=bucket_max_union_static,
        )                                                 # (B, H_q, BUCKET_MAX_UNION) int32
        O_sparse, lse_sparse = flash_attn_sparse_int4_fwd_compact(
            q, views["K_packed"], views["K_scale"], views["K_mn"],
            views["V_packed"], views["V_scale"], views["V_mn"],
            selected_page_ids=sel_compact,
            page_size=page_size, return_lse=True,
        )                                                 # O: (B, H_q, S_q, D); LSE: (B, H_q, S_q)
    else:
        # No completed pages yet — sparse half is empty.
        B_, H_q_, _, D_ = q.shape
        O_sparse = torch.zeros_like(q)
        lse_sparse = torch.full((B_, H_q_, S_q), float("-inf"), device=q.device, dtype=torch.float32)

    # Dense over (committed partial-page tail || sandbox)
    K_dense = torch.cat([views["K_partial"], views["K_sandbox"]], dim=2)
    V_dense = torch.cat([views["V_partial"], views["V_sandbox"]], dim=2)
    # Length partial_len + sandbox_count. Each query at sandbox index i is at
    # cache_position = completed_len + partial_len + i. In dense_tail coords, query
    # is at offset partial_len + i; it attends to dense_tail[0..(partial_len + i)].
    O_draft, lse_draft = bf16_dense_attn_offset_causal_with_lse(
        q, K_dense, V_dense, q_offset=partial_len,
    )                                                     # O: (B, H_q, S_q, D); LSE: (B, H_q, S_q)

    # Merge per-query
    if completed_len > 0:
        attn_output = _merge_two_attentions_sq(
            O_sparse, lse_sparse, O_draft, lse_draft,
        )
    else:
        attn_output = O_draft
```

`bf16_dense_attn_offset_causal_with_lse(Q, K, V, q_offset)`:
- Q shape: (B, H_q, S_q, D). K, V shape: (B, H_kv, S_kv, D). S_kv = q_offset + S_q.
- For query i (i ∈ [0, S_q)): valid kv positions are [0, q_offset + i]. Construct attention_mask `(S_q, S_kv)` with True for valid positions.
- SDPA call with the explicit mask + return both output and LSE.
- LSE computed from the same `qk` as `m + log(sum(exp(qk - m)))` per query. ~10 lines.

`_merge_two_attentions_sq(O_a, lse_a, O_b, lse_b)`:
- Inputs: O_a, O_b shape (B, H_q, S_q, D); lse_a, lse_b shape (B, H_q, S_q).
- `m = max(lse_a, lse_b)` per query.
- `wa = exp(lse_a - m).unsqueeze(-1)`, `wb = exp(lse_b - m).unsqueeze(-1)`.
- Output: `(wa * O_a.float() + wb * O_b.float()) / (wa + wb)` cast back to BF16.
- Reduces to existing `_merge_two_attentions` when S_q = 1.

### §4.4 Compact INT4 kernel S_q > 1 + UNION selection

#### Selection: score-prioritized UNION

Per-Q top-k pages are computed normally (`select_pages_vectorized` is already vectorized over S_q). The UNION across the S_q axis is `sel_per_q.any(dim=2)`. Quality property: each query's effective page set in the kernel is `UNION ⊇ per-Q top-k`. Attention coverage is at least as high as per-Q. NOT bit-identical to non-spec greedy (different attention values), but ≥ per-Q quality.

`BUCKET_MAX_UNION` is a constexpr static upper bound. The UNION of N_draft per-Q top-ks has expected size between BUCKET_MAX (high overlap) and N_draft × BUCKET_MAX (no overlap). At adjacent decode positions (separated by ≤ N_draft-1 = 4 token positions), the overlap is empirically ≥ 80% (Quest selection is dominated by global retrieval matches, not local positional shifts). Set `BUCKET_MAX_UNION = ⌈1.5 × BUCKET_MAX⌉` initially; **Task 2 of the plan microbenches the actual UNION size distribution** and adjusts the constant if needed.

**Score-prioritized truncation** (addresses codex r2 HIGH residual #1):
On the rare event that runtime UNION size exceeds `BUCKET_MAX_UNION`, the helper `build_compact_union_selection` keeps the **highest-priority pages by max-score across S_q**, and **force-includes sinks + window** so they cannot be truncated. The Phase 8a helper `build_compact_selection` (which sorts by page index) is **not** suitable here — index-sorted truncation could drop a sink or a high-priority retrieval page in favor of a low-priority page that happens to have a smaller index. The score-prioritized helper preserves Quest's selection priority semantics under overflow.

```python
# In flashquest/eager/selection.py (NEW helper)
def build_compact_union_selection(
    sel_per_q: torch.Tensor,         # bool[B, H_q, S_q, P]
    scores: torch.Tensor,            # float[B, H_q, S_q, P] — Quest criticality
    *,
    num_sinks: int,
    window_pages: int,
    completed_len: int,
    page_size: int,
    BUCKET_MAX_UNION: int,
) -> torch.Tensor:                   # int32[B, H_q, BUCKET_MAX_UNION]
    """Score-prioritized UNION selection with sinks + window force-included.

    On overflow (UNION size > BUCKET_MAX_UNION): keep the BUCKET_MAX_UNION
    highest-priority pages by max-score across S_q, with sinks + window forced.
    On underflow (UNION size < BUCKET_MAX_UNION): pad with -1 sentinels (the
    kernel handles sentinels via load-time masking + qk = -inf).
    """
    union_mask = sel_per_q.any(dim=2)                    # (B, H_q, P)
    max_scores = scores.amax(dim=2)                      # (B, H_q, P) — priority within UNION

    # Force-include sinks (positions [0, num_sinks)) and window (last window_pages completed pages)
    n_complete_pages = completed_len // page_size
    P = union_mask.shape[-1]
    forced_mask = torch.zeros_like(union_mask)
    forced_mask[..., :num_sinks] = True
    if n_complete_pages > window_pages:
        forced_mask[..., n_complete_pages - window_pages : n_complete_pages] = True
    elif n_complete_pages > 0:
        forced_mask[..., :n_complete_pages] = True

    # Combined mask: union_mask | forced_mask. Forced pages get +inf priority so they're never dropped.
    in_selection = union_mask | forced_mask
    priority = torch.where(in_selection, max_scores, torch.full_like(max_scores, float("-inf")))
    priority = torch.where(forced_mask, torch.full_like(priority, float("inf")), priority)

    # topk by priority descending; ties broken by lower page index (deterministic)
    top_pages = priority.topk(BUCKET_MAX_UNION, dim=-1).indices    # (B, H_q, BUCKET_MAX_UNION)

    # Mark selected slot as -1 sentinel if its priority is -inf (i.e., not in selection)
    top_priority = priority.gather(-1, top_pages)
    out = torch.where(
        top_priority == float("-inf"),
        torch.full_like(top_pages, -1),
        top_pages,
    )
    return out.to(torch.int32)
```

This helper is GPU-resident, no CUDA sync. The `topk` call returns indices sorted by descending priority (ties broken deterministically); the kernel doesn't care about ordering since it iterates all BUCKET_MAX_UNION slots anyway.

**On overflow vs underflow:**
- Underflow (UNION + forced size < BUCKET_MAX_UNION): some slots are sentinel `-1`. Kernel skips them via load-time masking. Cheap (skips K-tile + V-tile loads via mask).
- Overflow (UNION + forced size > BUCKET_MAX_UNION): truncates the *lowest-priority UNION pages*. Sinks + window are always preserved. Quality cost is bounded by "the dropped page would have contributed to softmax; we lose its mass." Empirically the lowest-score UNION pages contribute negligibly to softmax (their scores are low *because* they don't match Q well). Profile in Task 2 to confirm.

#### Kernel ABI

```
@triton.jit
def _sparse_attn_fwd_kernel_int4_compact_sq_gt_1(
    Q_ptr, K_packed_ptr, V_packed_ptr, O_ptr, L_ptr,
    K_scale_ptr, K_mn_ptr, V_scale_ptr, V_mn_ptr,
    selected_page_ids_ptr,                        # (B, H_q, BUCKET_MAX_UNION) int32 — UNION
    sm_scale,
    stride_qb, stride_qh, stride_qs, stride_qd,   # Q has explicit S_q stride now
    stride_kb, stride_kh, stride_ks, stride_kdp,
    stride_vb, stride_vh, stride_vs, stride_vdp,
    stride_ob, stride_oh, stride_os, stride_od,
    stride_lb, stride_lh, stride_ls,
    stride_ksb, stride_ksh, stride_ksp, stride_ksd,
    stride_kmb, stride_kmh, stride_kmp, stride_kmd,
    stride_vsb, stride_vsh, stride_vss,
    stride_vmb, stride_vmh, stride_vms,
    stride_selb, stride_selh, stride_seli,
    H_q, H_kv, S_kv, S_q,                         # S_q is a runtime arg
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PACKED: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BUCKET_MAX_UNION: tl.constexpr,
    SQ_MAX: tl.constexpr,                         # static upper bound on S_q (=16, see Performance)
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
    offs_sq = tl.arange(0, SQ_MAX)
    sq_mask = offs_sq < S_q                       # (SQ_MAX,) bool

    # Load Q for all SQ_MAX rows; masked rows set to 0
    q_ptrs = (
        Q_ptr + b * stride_qb + h_q * stride_qh
        + offs_sq[:, None] * stride_qs + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=sq_mask[:, None], other=0.0)         # (SQ_MAX, HEAD_DIM)

    m_i = tl.full((SQ_MAX,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((SQ_MAX,), dtype=tl.float32)
    acc = tl.zeros((SQ_MAX, HEAD_DIM), dtype=tl.float32)

    qk_scale = sm_scale * 1.44269504              # log2(e)

    for i in range(0, BUCKET_MAX_UNION):
        sel_off = b * stride_selb + h_q * stride_selh + i * stride_seli
        p = tl.load(selected_page_ids_ptr + sel_off)
        page_valid = p >= 0
        p_safe = tl.where(page_valid, p, 0)

        page_start = p_safe * PAGE_SIZE
        n_idx = page_start + offs_n
        valid_kv = (n_idx < S_kv) & page_valid

        # === K-tile load ONCE per page; reused across all SQ_MAX queries ===
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
        k = k_int.to(tl.float32) * k_scale[None, :] + k_mn[None, :]   # (PAGE_SIZE, HEAD_DIM)

        # === QK product: tl.dot with input_precision="ieee" (NOT TF32 default) ===
        # Output shape (SQ_MAX, PAGE_SIZE). tl.dot requires each dim ≥16; SQ_MAX=16, PAGE_SIZE=64, HEAD_DIM=128 — all ≥16.
        qk = tl.dot(q.to(tl.float32), tl.trans(k), input_precision="ieee")
        # Mask both axes: invalid SQ rows AND invalid KV positions → -inf
        qk = tl.where(sq_mask[:, None] & valid_kv[None, :], qk, float("-inf"))
        qk_scaled = qk * qk_scale

        qk_max = tl.max(qk_scaled, axis=1)
        m_ij = tl.maximum(m_i, qk_max)
        m_ij_safe = tl.where(m_ij == float("-inf"), 0.0, m_ij)
        p_softmax = tl.math.exp2(qk_scaled - m_ij_safe[:, None])
        p_softmax = tl.where(qk == float("-inf"), 0.0, p_softmax)

        alpha = tl.math.exp2(m_i - m_ij_safe)
        alpha = tl.where(m_i == float("-inf"), 0.0, alpha)

        l_i = l_i * alpha + tl.sum(p_softmax, axis=1)
        acc = acc * alpha[:, None]

        # === V-tile load ONCE per page; reused ===
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
        v = v_int.to(tl.float32) * v_scale[:, None] + v_mn[:, None]   # (PAGE_SIZE, HEAD_DIM)

        # === PV: tl.dot with ieee precision ===
        acc += tl.dot(p_softmax, v, input_precision="ieee")           # (SQ_MAX, HEAD_DIM)

        m_i = m_ij

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
        l_ptrs = L_ptr + b * stride_lb + h_q * stride_lh + offs_sq * stride_ls
        tl.store(l_ptrs, lse_val, mask=sq_mask)
```

#### Performance argument & constraints

- **`tl.dot` minimum size**: Triton 3.1.0 requires each operand dim ≥16. `SQ_MAX=16`, `PAGE_SIZE=64`, `HEAD_DIM=128` — all satisfy. Using `SQ_MAX=8` would fail compilation; we explicitly chose 16 with masked rows for `S_q < 16`. **Task 2 includes a Triton compile + run test for SQ_MAX={8, 16}** to confirm this empirically (codex r1 integration risk #1).
- **`input_precision="ieee"`**: pins fp32 precision over default TF32. Costs maybe 2× the dot throughput but preserves numerics expected by the LSE-merge math. (Codex r1 correctness bug #2.)
- **K/V tile reuse across SQ_MAX**: each page's K-tile (`PAGE_SIZE × HEAD_DIM = 64 × 128 = 16 KB`) and V-tile (same) is loaded ONCE and reused across all `SQ_MAX` queries via `tl.dot`. This is the BW amortization win — without it, S_q>1 sparse cost would scale linearly with S_q.
- **Register pressure**: `acc[SQ_MAX, HEAD_DIM] = 16 × 128 fp32 = 8 KiB` per CTA. Plus `q[SQ_MAX, HEAD_DIM] = 8 KiB`, `k[PAGE_SIZE, HEAD_DIM] = 32 KiB`, `v[PAGE_SIZE, HEAD_DIM] = 32 KiB`, `qk[SQ_MAX, PAGE_SIZE] = 4 KiB`, `m_i, l_i`, `alpha`, `m_ij`, `qk_max` (all SQ_MAX scalars or vectors). Total live ~80-100 KiB per CTA. sm_86 has 64 KB shared mem + ~256 KB register file per SM with 4 warps; this is at the upper edge of register pressure and may force spills. **Task 2 microbenches actual occupancy** — if spilling kills throughput, fall back to `SQ_MAX=8` with software workaround for `tl.dot` (e.g., emit pad rows but mark them invalid).
- **S_q=1 backward compatibility**: Phase 8a's S_q=1 kernel (`_sparse_attn_fwd_kernel_int4_compact`) is preserved unchanged. The new kernel is `_sparse_attn_fwd_kernel_int4_compact_sq_gt_1`; the Python wrapper dispatches based on `S_q == 1` vs `S_q > 1`. Phase 8a's existing decode bench uses the S_q=1 path and is byte-equivalent.

### §4.5 PLD dispatcher + walk-and-accept — `flashquest/specdec/dispatcher.py`

```
def make_quest_pld_dispatcher(
    model: torch.nn.Module,
    cache: PersistentInt4KVCache,
    prompt_ids: torch.Tensor,            # (S_prompt,) int64 on CPU
    *,
    K_match: int = 3,
    N_draft: int = 5,
    page_size: int = 64,
):
    """Wrap `model` (already patched with persistent_int4 attention) with PLD generation."""
    state = {
        "next_input_token": None,        # (1,) int64 — token to feed at next step. K/V NOT in cache.
        "next_argmax_buffer": None,      # (1,) int64 — same value as next_input_token in single-decode steady state.
                                         # Tracked separately because it is set to None after a PLD step
                                         # (we don't have a model argmax for the held free_token until next step).
                                         # PLD admissibility checks `draft[0] == next_argmax_buffer` to lossless-verify D_0.
        "committed_history": [],         # all already-committed token IDs (Python list of int).
    }

    def init(prompt_input_ids: torch.Tensor) -> None:
        """Run prefill, set next_input_token = prompt's argmax."""
        with torch.no_grad():
            out = model(input_ids=prompt_input_ids, use_cache=True, logits_to_keep=1)
        next_argmax = out.logits[:, -1].argmax(dim=-1)            # (B=1,) — argmax at position S_prompt-1, predicts position S_prompt
        state["next_input_token"] = next_argmax
        state["next_argmax_buffer"] = next_argmax                  # in single-decode steady-state, these are equal
        state["committed_history"] = list(prompt_input_ids[0].tolist())   # prompt is already committed in prefill
        # PLD warmup note: PLD admissibility requires len(committed_history) ≥ K_match (always true post-prefill,
        # since prompt has ≥3 tokens). The other PLD admissibility checks (n-gram match in prompt, draft[0] equal
        # to next_argmax_buffer, page-boundary guard) gate per-step.

    def step() -> torch.Tensor:
        """Generate ≥1 tokens; return all newly-committed token IDs (CPU int64)."""
        next_in = state["next_input_token"]
        prev_argmax = state["next_argmax_buffer"]
        # Layer-lockstep invariant: every patched LlamaAttention.forward calls
        # update_quantized(layer_idx) with the same S_new in the same order, so all
        # 28 _seen_tokens entries advance identically. Assert it explicitly to catch
        # any future drift from a refactor that breaks lockstep.
        assert all(s == cache._seen_tokens[0] for s in cache._seen_tokens), (
            f"layer _seen_tokens out of lockstep: {cache._seen_tokens}"
        )
        seen = cache._seen_tokens[0]
        history_tail = (
            torch.tensor(state["committed_history"][-K_match:], dtype=torch.int64)
            if len(state["committed_history"]) >= K_match else None
        )
        draft = (
            propose_draft(prompt_ids, history_tail, K_match=K_match, N_draft=N_draft)
            if history_tail is not None else None
        )

        # PLD admissibility: D_0 must equal prev_argmax; cannot cross a page boundary.
        page_boundary_violated = (
            (seen + N_draft) // page_size != seen // page_size
        )
        admissible = (
            draft is not None
            and prev_argmax is not None
            and draft[0].item() == int(prev_argmax.item())
            and not page_boundary_violated
        )

        if not admissible:
            # Single-token decode (existing path)
            with torch.no_grad():
                out = model(input_ids=next_in.unsqueeze(0), use_cache=True, logits_to_keep=1)
            new_argmax = out.logits[:, -1].argmax(dim=-1)
            emitted = next_in.unsqueeze(0)
            state["committed_history"].append(int(next_in.item()))
            state["next_input_token"] = new_argmax
            state["next_argmax_buffer"] = new_argmax
            return emitted

        # PLD verify path — input is [D_0, D_1, ..., D_{N_draft-1}], S_q = N_draft.
        # NOTE: D_0 == next_in == prev_argmax (admissibility ensures this).
        verify_input = draft.to(next_in.device).unsqueeze(0)             # (1, N_draft)

        _set_pld_verify_active(model, True)
        try:
            with torch.no_grad():
                out = model(input_ids=verify_input, use_cache=True)       # logits (1, N_draft, vocab)
        finally:
            _set_pld_verify_active(model, False)

        # Walk-and-accept: argmax(logits[i]) predicts position N+i+1 == draft[i+1]
        argmax_seq = out.logits.argmax(dim=-1).squeeze(0)                 # (N_draft,) int64
        accept_count_post_d0 = 0
        for i in range(N_draft - 1):
            if int(argmax_seq[i].item()) == int(draft[i + 1].item()):
                accept_count_post_d0 += 1
            else:
                break
        # M = accept_count_post_d0
        # Free token at position N + 1 + M = argmax_seq[M] (held over for next step)
        free_token = argmax_seq[accept_count_post_d0:accept_count_post_d0 + 1]   # (1,)
        # Total accepted draft K/Vs to commit (D_0 + M post-D_0): M+1
        total_accepted_kvs = 1 + accept_count_post_d0

        # Two-phase atomic commit (preflight all layers → mutate)
        cache.commit_draft_all_layers(total_accepted_kvs)

        # Emitted this step: D_0 + first M post-D_0 drafts. NOT free_token (held over).
        accepted_ids = draft[: total_accepted_kvs]                        # (M+1,) on CPU
        state["committed_history"].extend(int(t) for t in accepted_ids.tolist())
        state["next_input_token"] = free_token
        state["next_argmax_buffer"] = None                                # forces next step to single-decode
        return accepted_ids

    return init, step
```

**`_set_pld_verify_active`** sets `module._pld_verify_active = bool` on every patched LlamaAttention. The forward branch reads this attribute. Wrapped in `try/finally` so an exception during forward never leaves the model in verify mode (codex r1 integration risk #5).

**State invariants (kept consistent across §3, §4.5, §5):**
- `cache._seen` always = number of K/V slots committed.
- `next_input_token` holds the most-recently-decided token whose K/V is **not yet** in cache; it will be input to the next forward pass.
- `next_argmax_buffer` is the same value as `next_input_token` *in single-decode steady-state* (single-decode produces the next-token argmax which is then used for both purposes). After a PLD step, `next_argmax_buffer = None` — we don't have a fresh model argmax for the held free_token. PLD admissibility checks `draft[0] == next_argmax_buffer`, which lossless-verifies D_0 against what the previous step's logits already decided.
- `committed_history` lists all already-committed tokens. Length = `cache._seen_tokens[0]`.

## §5 Data flow trace — one PLD step

```
state at entry (after a single-decode step that prepared the buffers):
  cache._seen = N                                  (positions 0..N-1 committed in cache)
  next_input_token = T                             (= last single-decode's argmax; K/V NOT in cache yet)
  next_argmax_buffer = T                           (same as next_input_token in single-decode steady-state)
  committed_history = [..., emit_{N-3}, emit_{N-2}, emit_{N-1}]   # length N, ends with the most recent committed token

  ┌────────────────────────────────────────────────────────────────┐
  │ history_tail = committed_history[-K_match:] = [t1, t2, t3]      │
  │ propose_draft(prompt, history_tail) →                            │
  │   finds match at prompt index p; returns                         │
  │   draft = [D_0, D_1, D_2, D_3, D_4]                              │
  │     D_0 = prompt[p+3]                                            │
  │     D_4 = prompt[p+7]                                            │
  └────────────────────────────────────────────────────────────────┘

  Admissibility:
    D_0 == prev_argmax (= T)?              → match (admissible)
    (N + N_draft) crosses page boundary?   → no (else fall back)

  ┌────────────────────────────────────────────────────────────────┐
  │ verify_input = [D_0, D_1, D_2, D_3, D_4]   # S_q = 5            │
  │ (Note: D_0 == next_input_token == T; K/V will be written this   │
  │ step at position N. K/V for D_1..D_4 written at N+1..N+4.)      │
  │                                                                  │
  │ for layer ∈ 0..27:                                              │
  │   q, k, v ← projections + RoPE                                  │
  │   cache.add_draft(k, v, layer)         # writes sandbox[0..4]   │
  │   views ← cache.get_views_with_sandbox(layer)                   │
  │   if completed_len > 0:                                          │
  │     scores ← _criticality_scores(q, views)  # per-Q             │
  │     sel_per_q ← select_pages_vectorized(scores, …)              │
  │     sel_union ← sel_per_q.any(dim=2)                            │
  │     sel_compact ← build_compact_selection(sel_union, BMK_UNION) │
  │     O_sparse, lse_sparse ← compact_kernel_sq5(q, views, sel)    │
  │   K_dense ← cat(K_partial, K_sandbox)   # length partial_len+5  │
  │   V_dense ← cat(V_partial, V_sandbox)                           │
  │   O_draft, lse_draft ← bf16_dense_offset_causal(q, K_dense,    │
  │                       V_dense, q_offset=partial_len)            │
  │   O ← merge(O_sparse, lse_sparse, O_draft, lse_draft)           │
  │   ↓ residual + MLP                                               │
  │ hidden_states[1, 5, hidden]                                     │
  │ logits[1, 5, vocab] ← lm_head(hidden_states)                    │
  └────────────────────────────────────────────────────────────────┘

  argmax_seq = [a_0, a_1, a_2, a_3, a_4]
    a_0 ?= D_1 → match; a_1 ?= D_2 → match; a_2 ?= D_3 → MISMATCH; break
  M (post-D_0 accept count) = 2
  free_token = a_2  (held; emitted next step)
  total_accepted_kvs = 1 + 2 = 3   (commit D_0, D_1, D_2)

  cache.commit_draft_all_layers(3)   # writes positions N..N+2 to main cache

  emitted = [D_0, D_1, D_2]   # 3 tokens (= M+1)
  committed_history.extend([D_0, D_1, D_2])
  next_input_token = a_2 (= free_token)    # K/V NOT in cache
  next_argmax_buffer = None                # next step forced to single-decode

state at exit:
  cache._seen = N + 3
  emitted_this_step = 3 tokens
```

Next step (forced single-decode, M=2 means we forced this rather than another PLD attempt):
```
  next_in = a_2 (= free_token); prev_argmax = None
  Admissibility: prev_argmax is None → fall back

  forward([a_2], S_q=1) writes K/V at position N+3, computes logits → argmax = b
  emitted = [a_2]   # 1 token
  next_input_token = b; next_argmax_buffer = b
```

Per-cycle: 2 steps, 4 tokens. Cycle gain = 4 / 2.5 = **1.6×** at M=2 (matches the §2 ROI table).

## §6 Task 1 — profile-first (entry criterion)

Per the saved profile-first lesson (`feedback_profile_before_speedup_specs.md`), Phase 9's first task is empirical, not architectural. We don't write the kernel S_q>1 extension or the sandbox API until we've measured PLD's accept rate and verify-cost on our specific workload + hardware.

**Implementation:** `benchmarks/phase9_task1_pld_profile.py`
- Use the **existing dense SDPA prefill path** as the verify forward (S_q=N_draft dense over `dequant(cache) || sandbox`; no sparse, no kernel changes).
- Naive single-layer cache mock: dequantize cache once at start of step, run dense SDPA on `(Q, K_full, V_full)`.
- Walk-and-accept logic identical to §4.5; emit accounting per the M+1 corrected math.

**Measurement (per workload, greedy decoding, 256-token output, 100 verify steps):**
- Workloads:
  1. PG-essay summarize 8k input — 20 prompts from `data/PaulGrahamEssays.json`
  2. RULER NIAH single 4k — `flashquest.eval.niah_single`, 20 instances
  3. HumanEval 100 prompts (code completion)
  4. MT-bench-100 first turns (free-form chat)
- Per workload, record:
  - Mean post-D_0 accept count `M_avg` per PLD step (range 0..N_draft−1)
  - Mean tokens-per-step (`(M+1) on PLD steps, 1 on single-decode steps`)
  - Per-cycle gain: `(M+2) / 2.5` for the (PLD + single) cadence
  - Hit rate: % of steps where PLD draft is admissible (n-gram match found AND D_0 == prev_argmax AND not page-boundary)
  - 95th-percentile per-step wall time at S_q=N_draft vs S_q=1 dense forward

**Entry gate (decided BEFORE writing kernels, must clear to proceed):**
- `M_avg ≥ 2` averaged over (PG-summarize + RULER) — minimum 1.6× gain
- S_q=5 dense verify cost ≤ 1.3× S_q=1 dense baseline cost on Llama-3.2-3B-AWQ at 8k context
- Hit rate ≥ 30% on PG-summarize + RULER

**Aspirational target gate** (clear to claim 15 tok/s @ 32k):
- `M_avg ≥ 4` (= 80% of post-D_0 drafts accepted)

If entry gate clears but aspirational gate doesn't: ship Phase 9 with reduced expectations (1.6-2.0× gain) per "Continue with reduced expectations" precedent from Phase 8a Task 5.

If entry gate fails: stop. Report findings. Decide whether to re-scope to Lookahead, self-spec early-exit, or drop Phase 9 entirely.

## §7 Phase 9 success gates (full)

| Gate | Threshold | Source |
|---|---|---|
| Task 1 entry: accept-rate (PG-summ + RULER) | M_avg ≥ 2 | `benchmarks/phase9_task1_pld_profile.py` |
| Task 1 entry: verify cost ratio | S_q=5 ≤ 1.3× S_q=1 dense | same |
| Task 1 entry: admissibility hit rate | ≥ 30% | same |
| Unit + parity + e2e tests | all green | `tests/test_*.py` |
| RULER NIAH single 4k | 100/100 | `python -m flashquest.eval.runner --task niah_single --pld-on` |
| RULER NIAH multivalue 4k | ≥ 85% (≥17/20) | same |
| Argmax-equivalence to non-spec | ≥ 99% on 1000-prompt sample | `benchmarks/phase9_argmax_drift.py` |
| 32k summarize decode | **≥ 15 tok/s** (target) OR ≥ 10 tok/s (reduced-expectation floor) | `benchmarks/phase9_decode_bench.py --ctx 32k` |
| 8k summarize decode | **≥ 18 tok/s** (target) OR ≥ 14 tok/s (reduced-expectation floor) | same |

## §8 Test plan (TDD)

| Test file | What it asserts |
|---|---|
| `tests/test_pld_proposer.py` | n-gram match: no match → None; rightmost match wins; K_match > history → None; insufficient continuation → None |
| `tests/test_cache_sandbox.py` | `add_draft` writes correct slots; `commit_draft_all_layers(M)` produces same cache state as direct `update_quantized(K[:M])` per layer; `commit_draft_all_layers(0)` no-op; preflight raises before any mutation when one layer is mis-staged; **page-boundary preflight raises** if accept_count would cross |
| `tests/test_compact_kernel_sq_gt_1.py` | S_q=N kernel parity vs S_q=1 kernel run N times: numeric tolerance ≤ 1e-2 BF16 (with `input_precision="ieee"`); S_q=1 codepath byte-equivalent to Phase 8a |
| `tests/test_compact_kernel_sq_gt_1_sentinels.py` | `selected_page_ids` UNION with `-1` sentinels: no OOB, qk = -inf for invalid slots, output unchanged |
| `tests/test_pld_verify_parity.py` | Force all post-D_0 drafts mismatched → emits **2 tokens** (D_0 in PLD step, free_token in next step), token sequence ≡ non-spec single-token decode of D_0 followed by single-decode of free_token |
| `tests/test_pld_walk_and_accept.py` | Drafts `[A,B,C,D,E]` vs argmax `[B,C,X,D,E]` → M=2, free=`X`, emit=`[A,B,C]`; all-match → M=N_draft-1, free=argmax_seq[N_draft-1]; all-mismatch (post-D_0) → M=0, free=argmax_seq[0], emit=`[A]` |
| `tests/test_pld_admissibility.py` | next_argmax_buffer=None → fall back; D_0 != next_argmax → fall back; page-boundary crossed → fall back; happy path → admit |
| `tests/test_pld_dispatcher_e2e.py` | Llama-3.2-3B-AWQ on synthetic prompt-with-verbatim-continuation → PLD path emits identical token sequence to non-spec greedy (assuming our quality contract holds — see §9) |
| `tests/test_persistent_patch_pld_verify.py` | Patched LlamaAttention forward in verify mode (`_pld_verify_active=True`) does NOT call `update_quantized` and DOES call `add_draft`; sandbox cleared after `commit_draft_all_layers`; `try/finally` ensures flag reset on forward exception |
| `tests/test_dense_offset_causal.py` | `bf16_dense_attn_offset_causal_with_lse(Q, K, V, q_offset)` with `S_q=1, q_offset=k` matches existing `_bf16_dense_attn_with_lse` (regression); `S_q>1, q_offset=0` matches reference SDPA causal; offset-causal mask shape correctness |
| `tests/test_kernel_tldot_ieee.py` | Triton compile + run smoke for `SQ_MAX=8` and `SQ_MAX=16` to confirm tl.dot dim-min behavior; `input_precision="ieee"` produces bit-identical output to manual fp32 sum-of-product reference within tolerance |
| `tests/test_compact_union_selection.py` | Score-prioritized UNION: union ⊇ per-Q top-k; sinks + window always preserved (force_mask = +inf priority); on overflow drops lowest-score UNION pages, NOT lowest page index; on underflow pads with -1 sentinels; deterministic tie-break |

## §9 Quality gate

PLD verify uses UNION selection over per-Q top-k, which strictly extends each query's effective page coverage compared to per-Q selection. PLD path output is therefore not bit-identical to non-spec greedy (which uses per-Q selection at each step). The quality contract is:

1. **RULER NIAH 4k @ Llama-3.2-3B-AWQ + INT4 + PLD on**:
   - `niah_single`: 100/100. Retrieval answer is unambiguous; UNION-induced drift cannot change argmax of a high-margin token.
   - `niah_multivalue`: ≥ 85% (≥17/20), matching Phase 7 K3-V3 baseline.

2. **Argmax-equivalence**: on a 1000-prompt sample (PG-summarize + RULER + HumanEval), measure fraction of generated tokens where PLD output matches non-spec greedy output token-for-token. Target ≥ 99%.

3. **No catastrophic divergence**: spot-check 50 generated continuations between PLD and non-spec; manually verify no semantic drift > 1 token boundary.

If RULER fails or argmax-equivalence drops below 95%: **P0 quality regression**. Investigate UNION-select drift, sandbox commit, or LSE-merge numerics. Do not ship.

## §10 Decode benchmark

`benchmarks/phase9_decode_bench.py`:
- Backends: `non-spec` (Phase 8a baseline, both `use_compact_kernel=False` and `=True`), `PLD-greedy` at K_draft ∈ {3, 5, 8} (sweep).
- Contexts: 8k, 16k, 32k.
- Workloads: summarize (PG-essay), RAG (NIAH-style query+context).
- Output length: 256 tokens.
- 5 runs per cell, report mean ± stddev tok/s.
- One backend at a time, `nice -n 19`, `run_in_background` + Monitor (per saved feedback `avoid_wsl_lag`).

## §11 File layout

NEW (Phase 9):
- `src/flashquest/specdec/__init__.py`
- `src/flashquest/specdec/pld.py` — `propose_draft`
- `src/flashquest/specdec/dispatcher.py` — `make_quest_pld_dispatcher`, `_set_pld_verify_active`, walk-and-accept
- `benchmarks/phase9_task1_pld_profile.py` — Task 1 entry-gate
- `benchmarks/phase9_argmax_drift.py` — quality measurement (Task ~10)
- `benchmarks/phase9_decode_bench.py` — full PLD vs baseline bench
- All Phase 9 test files listed in §8

EXTENDED (Phase 9):
- `src/flashquest/cache/persistent_int4.py` — sandbox tensors + `add_draft` / `get_views_with_sandbox` / `preflight_commit` / `commit_draft` / `commit_draft_all_layers`
- `src/flashquest/eager/llama_persistent_patch.py` — verify-mode dispatch branch + `_pld_verify_active` plumbing + `_merge_two_attentions_sq` extension + `bf16_dense_attn_offset_causal_with_lse` helper + `bucket_max_union_static` precompute
- `src/flashquest/kernel/sparse_int4_fwd_compact.py` — new `_sparse_attn_fwd_kernel_int4_compact_sq_gt_1` kernel + dispatcher (S_q=1 path unchanged)
- `src/flashquest/eager/selection.py` — new `build_compact_union_selection` helper (score-prioritized UNION, sinks + window force-included; replaces `build_compact_selection` for verify-mode dispatch only — single-decode path still uses `build_compact_selection`)
- `docs/PHASES/phase-9-notes.md` — phase journal (created at end)

UNCHANGED:
- `src/flashquest/quant/awq_layout.py`
- `src/flashquest/cache/persistent_int8.py` (PLD is INT4-only)

## §12 Risks & mitigations

| Risk | Mitigation |
|---|---|
| **Profile gate fails (M_avg < 2)** | Stop. Pivot to Lookahead or self-spec, or drop Phase 9. Saved memory `feedback_profile_before_speedup_specs.md` enforces this. |
| **M_avg ∈ [2, 4) — "reduced-expectations" zone** | Ship Phase 9 with adjusted target (1.6-2.0×). Document in `phase-9-notes.md`. Roadmap math gets re-derived. (Phase 8a precedent.) |
| **S_q>1 kernel slower than expected (register spills, occupancy collapse)** | Microbench at Task 2 (kernel completion) before integration. If S_q=5 sparse cost ≥ 1.5× S_q=1 cost, reduce SQ_MAX from 16 → 8 with `tl.dot` workaround (pad rows but mark invalid via mask), accepting the masked-row work overhead. |
| **UNION selection size > BUCKET_MAX_UNION static bound** | Truncate via `build_compact_selection`. Profile in Task 2 to set the bound at the 99th-percentile of measured UNION size. |
| **LSE-merge numeric drift breaks RULER** | `input_precision="ieee"` in tl.dot is the first-line defense. If still failing, increase `bf16_dense_attn_offset_causal_with_lse` to fp32-accumulator path (we already do for `_bf16_dense_attn_with_lse`). |
| **Page-boundary guard reduces PLD admissibility** | At page_size=64, N_draft=5: ~6-8% of steps blocked by boundary alone, on top of n-gram + admissibility. Acceptable. Alternative (simulate quantize-on-boundary) deferred to Phase 9b. |
| **VRAM headroom at 32k tight** | Sandbox is 920 KB — negligible. Verify-step S_q=5 attention activations ~50 MB extra. If OOM at 32k, reduce N_draft to 3 (and adjust gates accordingly). |
| **Codex r2 catches another correctness bug** | Phase 8 precedent allowed up to 3 codex review cycles. We expect r2 to be cleaner than r1 since dealbreakers are addressed. Maximum 1 more rewrite if r2 surfaces dealbreakers. |
| **`tl.dot` ieee precision halves throughput** | Acceptable: BW-bound kernel means dot-throughput cost is sub-dominant. If ieee-induced slowdown > 30%, profile fp32 accumulation manually (sum-of-product) for the SQ_MAX×PAGE_SIZE dot. |

## §13 Open questions for codex r2 review

After r2 corrections, items for codex to stress-test:

1. **§4.3 verify dense tail**: is `K_partial || K_sandbox` of length `partial_len + N_draft` the right tail? Specifically, when `completed_len = 0` (initial decode steps): `K_partial` may be EMPTY, `K_sandbox` is the only tail, and the offset-causal helper handles `S_kv = N_draft, q_offset = 0`. Same code path; correct?

2. **§4.4 UNION quality**: codex r1 dealbreaker #4 was about per-Q vs shared selection. UNION addresses it by making each query's effective page set ⊇ per-Q. But the resulting attention output for query `i` is `attention(Q_i, UNION)` not `attention(Q_i, top-k(Q_i))`. Quality is provably ≥ per-Q in expected mass coverage but can be != non-spec at the argmax level. Is the §1 quality contract phrased correctly to handle this nuance?

3. **§4.5 dispatcher `cache._seen_tokens[0]` access**: r2.1 adds an explicit `assert all(s == cache._seen_tokens[0] for s in cache._seen_tokens)` before the page-boundary check (codex r2 LOW residual #4). Catches any future regression that breaks layer-lockstep on `update_quantized`.

4. **§3 page-boundary guard arithmetic**: `(N + N_draft) // page_size > N // page_size` — correctly identifies any verify span that would *cause* a page-completion-and-quantization during commit. Mechanics: at N=63, N_draft=1, the verify writes K at position 63 in BF16 sandbox; on commit, `update_quantized` advances seen 63→64, page 0 is now full and gets quantized into INT4 — but the verify forward already attended to that K in BF16. A non-spec single-decode at the same step would write at position 63 in K_partial (BF16), then `update_quantized` quantizes page 0 BEFORE attention runs (per `eager/llama_persistent_patch.py` line 193: update_quantized → get_views → attend). So non-spec attention reads INT4 K[63]; PLD-verify attention reads BF16 K[63]. Drift. The guard `(63+1)//64=1 > 63//64=0` → BLOCKED. ✓ Edge case at N=64 (first position of page 1), N_draft=5: writes positions 64..68, all in page 1, no boundary crossed → `(64+5)//64=1 = 64//64=1` → admissible. ✓

5. **§4.5 walk-and-accept loop**: `for i in range(N_draft - 1)` — verify range bound. We compare `argmax_seq[i] == draft[i+1]` for `i ∈ [0, N_draft-2]`, i.e. checks `draft[1]` through `draft[N_draft-1]`. Total post-D_0 drafts = N_draft-1. Maximum M = N_draft-1 (all post-D_0 accepted). ✓ Free token = `argmax_seq[M]` for M ∈ [0, N_draft-1]; index always valid since `argmax_seq` has length `N_draft`. ✓

6. **§4.4 register pressure**: SQ_MAX=16 × HEAD_DIM=128 fp32 acc = 8 KiB. With Q (8 KiB), K (32 KiB), V (32 KiB), QK_softmax (4 KiB), V matmul scratch, m_i, l_i — total ≈ 90 KiB. sm_86 register file is 64 KB per thread block (256 threads, num_warps=4 default). This *will* force shared-memory or local-memory spills. Codex check: is the kernel's expected occupancy realistic, or should we plan for SQ_MAX=8 with the tl.dot workaround as the actual implementation?

## §14 References

- **PLD original blog**: Saxena, "Prompt Lookup Decoding," Nov 2023. https://github.com/apoorvumang/prompt-lookup-decoding
- **PLD paper**: "Prompt Lookup Decoding for Faster LLM Inference" (Saxena et al., 2023)
- **HF `assisted_generation` PLD impl**: `transformers/src/transformers/generation/utils.py`, `prompt_lookup` candidate-generator path
- **Leviathan-Chen rejection sampler** (for Phase 9b sampling support, not v1): "Fast Inference from Transformers via Speculative Decoding," Leviathan et al., 2023
- **Triton `tl.dot` precision**: https://triton-lang.org/main/python-api/generated/triton.language.dot.html — `input_precision="ieee"` for fp32 (vs default TF32 on Ampere+)
- **Phase 8a foundation work**: `docs/PHASES/phase-8a-notes.md`, tag `phase-8a`
- **Phase 8b kill record**: `docs/PHASES/phase-8b-killed.md`
- **Profile-first memory**: `~/.claude/projects/-home-hoang-code-personal-active-flashquest/memory/feedback_profile_before_speedup_specs.md`
- **Codex r1 review (this spec)**: r1 raised 4 dealbreakers, 3 correctness bugs, 5 integration risks, 4 nits. r2 addressed all 16.
- **Codex r2 review**: confirmed r1 closure on all 12 substantive findings + nits, raised 1 HIGH (UNION-overflow truncation) + 3 minor residuals. r2.1 (this revision) addresses all 4 via surgical edits: score-prioritized UNION helper (`build_compact_union_selection`); cleaner `next_argmax_buffer` wording; tightened §13 page-boundary explanation; explicit layer-lockstep assertion in dispatcher.
