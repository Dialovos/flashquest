"""Phase 12 task 8 — EAGLE-3 chain speculative-decoding dispatcher.

Greedy, lossless-by-contract chain spec-decode over the real Quest sparse INT4
cache. Reuses the Phase 9 PLD state machine verbatim (verify input
``[bonus, d_1, ..., d_{n-1}]``; walk-and-accept; the held free/bonus token;
``commit_draft_all_layers(m+1)``; the page-boundary fallback to a single decode)
— only the *drafter* changes (EAGLE-3 ``propose_chain`` instead of the n-gram
proposer) and the *verify* runs the sparse-union + dense-tail arm
(``set_verify_active``) already wired into ``llama_persistent_patch``.

Calling convention (verified against transformers 4.57 + ``PersistentInt4KVCache``):

* The patched target forward is driven through the ordinary
  ``model(input_ids=..., past_key_values=cache, use_cache=True,
  output_hidden_states=True)`` entry point. ``LlamaModel.forward`` derives
  ``cache_position`` from ``cache.get_seq_length()`` (= ``_seen_tokens[0]`` = N),
  so the ``S_q = n_draft`` verify pass gets the correct absolute RoPE positions
  ``[N, N+1, ..., N+n-1]`` for the bonus + drafts — matching the verify arm's
  ``q_offset = partial_len`` dense-tail causal mask.
* ``set_verify_active(model, True)`` flips every patched ``LlamaAttention`` onto
  the verify branch, which stages the ``n_draft`` K/V in the per-layer sandbox
  (``add_draft``) and attends sparse-union(completed) ⊕ dense(partial-tail ‖
  sandbox). ``commit_draft_all_layers(m+1)`` then commits the bonus + ``m``
  accepted drafts and resets ``_sandbox_count`` (the remaining ``n-1-m`` staged
  drafts are dropped).

Incremental draft-KV reuse is a later optimization (task 7); here ``propose_chain``
re-prefills the head from scratch each step — correct, slower.
"""
from __future__ import annotations

import torch

from .eagle_draft import EAGLE3_FUSION_LAYERS, fuse_target_hidden


def _walk_accept(verify_input_ids, target_argmax, n_draft):
    """Pure greedy walk-and-accept (Phase 9 §4.5 semantics).

    Args:
        verify_input_ids: length-``n_draft`` sequence ``[bonus, d_1, ..., d_{n-1}]``
            (ints or a 1-D int tensor). ``verify_input_ids[0]`` is the already-decided
            bonus; ``verify_input_ids[i]`` for ``i>=1`` is draft ``d_i``.
        target_argmax: length-``n_draft`` sequence; ``target_argmax[i]`` is the
            target's greedy token AFTER consuming ``verify_input_ids[i]`` (i.e. the
            token that should equal ``d_{i+1}`` for ``i < n_draft-1``).
        n_draft: chain depth (== len(verify_input_ids)).

    Returns:
        ``(m, accepted_ids, new_bonus)`` where
          * ``m`` = count of accepted post-bonus drafts, ``0 <= m <= n_draft-1``;
          * ``accepted_ids`` = ``[bonus, d_1, ..., d_m]`` (``m+1`` ints) — committed
            this step;
          * ``new_bonus`` = ``target_argmax[m]`` (int) — the target's correction /
            continuation, held for the next step (NOT emitted this step).
    """
    vin = [int(x) for x in (verify_input_ids.tolist()
                            if torch.is_tensor(verify_input_ids) else verify_input_ids)]
    tgt = [int(x) for x in (target_argmax.tolist()
                            if torch.is_tensor(target_argmax) else target_argmax)]
    if len(vin) != n_draft or len(tgt) != n_draft:
        raise ValueError(
            f"_walk_accept expects length n_draft={n_draft}; got "
            f"verify_input={len(vin)}, target_argmax={len(tgt)}"
        )
    m = 0
    for i in range(n_draft - 1):
        if tgt[i] == vin[i + 1]:
            m += 1
        else:
            break
    accepted_ids = vin[: m + 1]            # [bonus, d_1, ..., d_m]
    new_bonus = tgt[m]                     # always valid: 0 <= m <= n_draft-1
    return m, accepted_ids, new_bonus


def make_quest_specdec(model, cache, draft, *, n_draft=4, page_size=64,
                       fusion_layers=EAGLE3_FUSION_LAYERS, prefill_chunk=256):
    """Wrap a Quest-patched target + INT4 cache + EAGLE-3 head for greedy chain
    speculative decoding.

    Args:
        model: the target ``LlamaForCausalLM`` already patched with
            ``patch_llama_for_quest_persistent`` (verify arm present).
        cache: the ``PersistentInt4KVCache`` shared by the patch.
        draft: an ``EagleDraft`` (its ``propose_chain`` / ``fuse_target_hidden``).
        n_draft: chain depth (verify ``S_q``). Must be ``<= cache.MAX_DRAFT``.
        page_size: cache page size (for the admissibility / page-boundary guard).
        fusion_layers: HF hidden-state indices EAGLE-3 fuses (low/mid/high).
        prefill_chunk: chunk size for the chunked base-model prefill in ``init``
            (bounds peak activation memory on the 4 GB GPU).

    Returns:
        ``(init, step)``. ``init(prompt_ids)`` runs the dense prefill (populates
        the sparse cache + seeds ``fused_seq`` / ``context_ids`` / ``bonus``).
        ``step()`` emits ``>=1`` newly-committed full-vocab token ids (CPU int64).
    """
    from ..eager.llama_persistent_patch import set_verify_active

    if n_draft > cache.MAX_DRAFT:
        raise ValueError(f"n_draft={n_draft} > cache.MAX_DRAFT={cache.MAX_DRAFT}")

    state = {
        "bonus": None,         # (1,) int64 on device — target's last greedy token; KV NOT in cache.
        "fused_seq": None,     # (1, N, 3H) fused target hidden over every verified position.
        "context_ids": None,   # (1, N) int64 on device — all verified token ids.
    }

    @torch.no_grad()
    def init(prompt_ids: torch.Tensor) -> None:
        """Dense prefill: populate the sparse cache, seed fused_seq/context_ids/bonus.

        Mirrors ``task1b_acceptance._greedy_reference``'s prefill: the base model
        is run in append-only chunks (no full ``[1, P, vocab]`` logits, no all-
        positions 29-layer hidden states held at once). The patched non-verify
        ``S_q>1`` arm quantises each chunk into the INT4 cache exactly as a single
        full prefill would. The bonus = argmax of the last-token logits.
        """
        set_verify_active(model, False)
        dev = model.device
        ids = prompt_ids.to(dev).long()
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        base = model.model
        P = ids.shape[1]
        past = None
        fused_chunks = []
        last_h = None
        for s in range(0, P, prefill_chunk):
            e = min(s + prefill_chunk, P)
            bout = base(ids[:, s:e], past_key_values=past,
                        output_hidden_states=True, use_cache=True)
            past = bout.past_key_values
            fused_chunks.append(fuse_target_hidden(bout.hidden_states, fusion_layers))
            last_h = bout.last_hidden_state[:, -1:, :]
            del bout
        bonus = model.lm_head(last_h)[:, -1].argmax(dim=-1)        # (1,)
        state["bonus"] = bonus
        state["fused_seq"] = torch.cat(fused_chunks, dim=1)        # (1, P, 3H)
        state["context_ids"] = ids                                # (1, P)

    @torch.no_grad()
    def _single_decode() -> torch.Tensor:
        """Non-spec S_q=1 decode of the held bonus. Commits the bonus' KV, sets
        the new bonus, returns ``[bonus]`` (1 token, CPU int64)."""
        set_verify_active(model, False)
        bonus = state["bonus"]                                    # (1,)
        out = model(input_ids=bonus.unsqueeze(0), past_key_values=cache,
                    use_cache=True, output_hidden_states=True)
        fused_new = fuse_target_hidden(out.hidden_states, fusion_layers)  # (1,1,3H)
        new_bonus = out.logits[:, -1].argmax(dim=-1)              # (1,)
        emitted = bonus.detach().to("cpu", torch.int64).reshape(1)
        state["context_ids"] = torch.cat([state["context_ids"], bonus.view(1, 1)], dim=1)
        state["fused_seq"] = torch.cat([state["fused_seq"], fused_new], dim=1)
        state["bonus"] = new_bonus
        return emitted

    @torch.no_grad()
    def step() -> torch.Tensor:
        """Generate >=1 tokens; return all newly-committed token ids (CPU int64)."""
        if state["bonus"] is None:
            raise RuntimeError("make_quest_specdec.step() called before init()")
        # Lockstep invariant: every patched layer advanced _seen identically.
        seen = cache._seen_tokens[0]
        assert all(s == seen for s in cache._seen_tokens), (
            f"layer _seen_tokens out of lockstep: {cache._seen_tokens}"
        )
        N = seen

        # Page-boundary admissibility: committing up to n_draft tokens (bonus + up
        # to n_draft-1 drafts) must not cross a page. The verify arm keeps sandbox
        # K/V in BF16; a cross-page commit would quantise differently from the
        # non-spec path. On violation, fall back to a single non-spec decode.
        if (N + n_draft) // page_size != N // page_size:
            return _single_decode()

        bonus = state["bonus"]                                    # (1,) device int64
        # Draft: EAGLE-3 greedy chain over the verified prefix (from scratch).
        chain = draft.propose_chain(
            state["fused_seq"][:, :N], bonus, n_draft=n_draft,
            context_ids=state["context_ids"][:, :N], chunk=prefill_chunk,
        )                                                         # (n_draft,) full vocab, device
        chain = chain.to(bonus.device).long()

        # Verify input = [bonus, d_1, ..., d_{n-1}] (length n_draft).
        verify_input = torch.cat([bonus.view(1), chain[:-1]]).view(1, n_draft)

        set_verify_active(model, True)
        try:
            out = model(input_ids=verify_input, past_key_values=cache,
                        use_cache=True, output_hidden_states=True)
        finally:
            set_verify_active(model, False)

        logits = out.logits                                       # (1, n_draft, vocab)
        fused_new = fuse_target_hidden(out.hidden_states, fusion_layers)  # (1, n_draft, 3H)
        tgt = logits.argmax(dim=-1)[0]                            # (n_draft,) device int64

        # verify_input_ids for the walk = [bonus, d_1, ..., d_{n-1}] (== verify_input row).
        m, accepted_ids, new_bonus = _walk_accept(
            verify_input[0], tgt, n_draft,
        )

        # Commit bonus + m accepted drafts (m+1 sandbox slots) across all layers.
        cache.commit_draft_all_layers(m + 1)

        # Grow verified state by exactly m+1 (lockstep with the cache _seen advance).
        accepted_t = torch.tensor(accepted_ids, dtype=torch.int64,
                                  device=state["context_ids"].device).view(1, m + 1)
        state["context_ids"] = torch.cat([state["context_ids"], accepted_t], dim=1)
        state["fused_seq"] = torch.cat([state["fused_seq"], fused_new[:, : m + 1]], dim=1)
        state["bonus"] = tgt[m: m + 1]                            # held; not emitted

        return torch.tensor(accepted_ids, dtype=torch.int64)      # (m+1,) CPU

    return init, step
