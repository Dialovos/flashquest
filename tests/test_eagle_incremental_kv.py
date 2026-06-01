"""Phase 12 task 7 — incremental draft-KV maintenance: the decisive correctness
check that the incremental head KV reproduces the from-scratch prefill EXACTLY.

The optimization (``EagleDraft.seed`` / ``propose_from`` / ``advance``) grows the
EAGLE-3 head's KV by only the newly-committed tokens instead of re-prefilling the
whole verified prefix every dispatcher step. It is a SPEED + long-context-OOM
enabler that must be byte-for-byte lossless w.r.t. the task-8 from-scratch
``propose_chain``.

These tests are fast, synthetic, and CPU-only: they build a TINY vendored EAGLE-3
``Model`` with random weights (no checkpoint download) and feed random fused
hidden + token ids. The assertion is pure determinism — incremental == from-scratch
— which holds for ANY weights, so random init is sufficient and the equivalence is
the real EAGLE head forward, not a stub. The ``lm_head`` is scaled up so the greedy
chain has DISTINCT tokens (a degenerate all-same chain would pass even if both
paths were identically broken), and we hard-assert the chain is non-constant.

The EAGLE shift convention under test (verified against ``prefill`` + Codex): draft
position ``p`` consumes ``(target_fused[p], verified_token[p+1])``; the held bonus
is ``token[N]`` and pairs with ``fused[N-1]`` — NOT ``fused[N]`` (the classic
off-by-one). ``advance`` commits the deferred ``(fused[N-1], bonus)`` + the accepted
span and re-defers the tail, reproducing a from-scratch shifted prefill.
"""
import sys
from pathlib import Path

import pytest
import torch

# Vendored EAGLE-3 model dir (same resolution as eagle_draft._import_vendored_model).
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "eagle" / "eagle" / "model"


def _build_tiny_draft(seed: int = 0, H: int = 48, V: int = 128):
    """A tiny CPU EagleDraft (random weights, no download). fc path is exercised
    because the fused width 3*H differs from the embed width H (as in production:
    9216 -> 3072)."""
    if str(_VENDOR) not in sys.path:
        sys.path.insert(0, str(_VENDOR))
    import cnets  # type: ignore
    from configs import EConfig  # type: ignore

    from flashquest.specdec.eagle_draft import EagleDraft

    torch.manual_seed(seed)
    cfg = EConfig(
        vocab_size=V, hidden_size=H, intermediate_size=96,
        num_hidden_layers=1, num_attention_heads=6, num_key_value_heads=2,
        max_position_embeddings=512, rms_norm_eps=1e-6, pad_token_id=0,
    )
    cfg.draft_vocab_size = V        # vocab == draft_vocab => d2t is +0 (trivial map)
    cfg.rope_theta = 10000.0
    cfg.pretraining_tp = 1
    model = cnets.Model(cfg, load_emb=False, path=None, bias=False,
                        total_tokens=8, depth=3, top_k=4, threshold=1.0)
    with torch.no_grad():
        # Scale the head so argmax varies position-to-position (discriminating chain).
        model.lm_head.weight.mul_(6.0)
        model.embed_tokens.weight.normal_(0.0, 1.0)
    model = model.to(torch.float32).eval()
    model.init_tree()
    return EagleDraft(model=model, device="cpu", dtype=torch.float32), H, V


# ---------------------------------------------------------------------------
# Decisive check: incremental (seed + advance + propose_from) == from-scratch.
# ---------------------------------------------------------------------------

def test_incremental_equals_from_scratch_single_token():
    """The task-template scenario: a context of length T, the LAST token accepted
    via advance, then propose_from must equal propose_chain over the full prefix.

    Mirrors the prefill shift exactly:
      chain_full = propose_chain(fused[:T], bonus, n, ctx[:T])
      state = seed(fused[:T-1], ctx[T-1], ctx[:T-1])   # bonus deferred
      state = advance(state, [ctx[T-1]], fused[T-1:T]) # commit position T-1
      state["bonus"] = bonus                            # the held proposal input
      chain_inc = propose_from(state, n)
      assert chain_inc == chain_full
    """
    draft, H, V = _build_tiny_draft(seed=1)
    torch.manual_seed(3)
    T, n_draft = 9, 5
    fused = torch.randn(1, T, 3 * H)
    ctx = torch.randint(0, V, (1, T))
    bonus = int(torch.randint(0, V, (1,)).item())

    chain_full = draft.propose_chain(
        fused[:, :T], bonus, n_draft=n_draft, context_ids=ctx[:, :T], chunk=512,
    )
    # Discriminating: a constant chain would pass even if both paths were broken.
    assert len(set(chain_full.tolist())) > 1, (
        f"degenerate chain {chain_full.tolist()} — not discriminating; "
        f"adjust seed/lm_head scale"
    )

    state = draft.seed(fused[:, : T - 1], int(ctx[0, T - 1].item()),
                       ctx[:, : T - 1], chunk=512)
    assert state["seen"] == T - 1                       # bonus deferred, not committed
    assert state["past"][0][0].shape[2] == T - 2        # KV covers 0..T-3

    state = draft.advance(state, ctx[:, T - 1].view(1), fused[:, T - 1 : T],
                          chunk=512)
    assert state["seen"] == T                            # committed position T-1
    assert state["past"][0][0].shape[2] == T - 1         # KV covers 0..T-2
    assert state["bonus"] is None                        # advance clears it
    state["bonus"] = torch.tensor([bonus])               # caller sets held bonus

    chain_inc = draft.propose_from(state, n_draft=n_draft)
    assert torch.equal(chain_full, chain_inc), (
        f"incremental chain {chain_inc.tolist()} != from-scratch "
        f"{chain_full.tolist()}"
    )


def test_propose_from_does_not_mutate_state():
    """propose_from must be repeatable on the SAME state (functional KV: no clone
    needed, no corruption). Two calls => identical chains, KV length unchanged."""
    draft, H, V = _build_tiny_draft(seed=2)
    torch.manual_seed(5)
    T, n_draft = 8, 4
    fused = torch.randn(1, T, 3 * H)
    ctx = torch.randint(0, V, (1, T))
    bonus = int(torch.randint(0, V, (1,)).item())

    state = draft.seed(fused[:, : T - 1], int(ctx[0, T - 1].item()), ctx[:, : T - 1])
    state = draft.advance(state, ctx[:, T - 1].view(1), fused[:, T - 1 : T])
    state["bonus"] = torch.tensor([bonus])

    kv_len_before = state["past"][0][0].shape[2]
    chain1 = draft.propose_from(state, n_draft=n_draft)
    chain2 = draft.propose_from(state, n_draft=n_draft)
    assert torch.equal(chain1, chain2), "propose_from is not idempotent (mutated state)"
    assert state["past"][0][0].shape[2] == kv_len_before, "propose_from grew state['past']"
    assert state["bonus"] is not None and int(state["bonus"]) == bonus, "bonus changed"


def test_chunked_seed_matches_unchunked():
    """seed/advance with a small chunk (crossing chunk boundaries) reproduces the
    same chain as a single full pass — guards the chunked _run_span / RoPE-from-
    cached-length invariant."""
    H, V = 48, 128
    torch.manual_seed(7)
    T, n_draft = 11, 5
    fused = torch.randn(1, T, 3 * H)
    ctx = torch.randint(0, V, (1, T))
    bonus = int(torch.randint(0, V, (1,)).item())

    # Reference: unchunked from-scratch.
    draft_ref, _, _ = _build_tiny_draft(seed=9, H=H, V=V)
    chain_ref = draft_ref.propose_chain(
        fused[:, :T], bonus, n_draft=n_draft, context_ids=ctx[:, :T], chunk=512,
    )

    # Incremental with chunk=3 (multiple boundaries inside the prefix).
    draft_c, _, _ = _build_tiny_draft(seed=9, H=H, V=V)
    state = draft_c.seed(fused[:, : T - 1], int(ctx[0, T - 1].item()),
                         ctx[:, : T - 1], chunk=3)
    state = draft_c.advance(state, ctx[:, T - 1].view(1), fused[:, T - 1 : T], chunk=3)
    state["bonus"] = torch.tensor([bonus])
    chain_c = draft_c.propose_from(state, n_draft=n_draft)
    assert torch.equal(chain_ref, chain_c), (
        f"chunked seed/advance chain {chain_c.tolist()} != unchunked "
        f"{chain_ref.tolist()}"
    )


@pytest.mark.parametrize("A", [1, 2, 3])
def test_multi_token_advance_equals_from_scratch(A):
    """The real dispatcher case: accept a span of A tokens ([bonus, d_1..d_{A-1}])
    in ONE advance, then propose_from == from-scratch prefill over the grown prefix.

    Setup: verified prefix length S; the accepted span occupies positions S..S+A-1
    holding ctx2[S..S+A-1]; the new held bonus is set after advance. The reference
    is propose_chain(fused2[:S+A], new_bonus, ctx2[:S+A]).
    """
    draft, H, V = _build_tiny_draft(seed=4)
    torch.manual_seed(11 + A)
    S, n_draft = 6, 5
    T2 = S + A
    fused2 = torch.randn(1, T2, 3 * H)
    ctx2 = torch.randint(0, V, (1, T2))
    new_bonus = int(torch.randint(0, V, (1,)).item())

    chain_ref = draft.propose_chain(
        fused2[:, :T2], new_bonus, n_draft=n_draft, context_ids=ctx2[:, :T2], chunk=512,
    )

    # Incremental: seed at S with bonus = ctx2[S] (the first accepted token), advance
    # the accepted span ctx2[S..T2-1] / fused2[S..T2-1], then set the new held bonus.
    state = draft.seed(fused2[:, :S], int(ctx2[0, S].item()), ctx2[:, :S])
    assert state["seen"] == S
    accepted = ctx2[:, S:T2].reshape(-1)                # [bonus, d_1..d_{A-1}]
    state = draft.advance(state, accepted, fused2[:, S:T2])
    assert state["seen"] == T2                          # advanced by exactly A
    assert state["past"][0][0].shape[2] == T2 - 1       # KV covers 0..T2-2
    state["bonus"] = torch.tensor([new_bonus])
    chain_inc = draft.propose_from(state, n_draft=n_draft)
    assert torch.equal(chain_ref, chain_inc), (
        f"A={A}: incremental {chain_inc.tolist()} != from-scratch {chain_ref.tolist()}"
    )


# ---------------------------------------------------------------------------
# Guards: misuse raises clearly.
# ---------------------------------------------------------------------------

def test_propose_from_without_bonus_raises():
    """advance clears the bonus; proposing before the caller re-sets it must fail
    loudly (catches the dispatcher forgetting state['draft']['bonus'] = new_bonus)."""
    draft, H, V = _build_tiny_draft(seed=6)
    torch.manual_seed(13)
    T = 6
    fused = torch.randn(1, T, 3 * H)
    ctx = torch.randint(0, V, (1, T))
    state = draft.seed(fused[:, : T - 1], int(ctx[0, T - 1].item()), ctx[:, : T - 1])
    state = draft.advance(state, ctx[:, T - 1].view(1), fused[:, T - 1 : T])
    assert state["bonus"] is None
    with pytest.raises(RuntimeError):
        draft.propose_from(state, n_draft=3)


def test_advance_length_mismatch_raises():
    """accepted_tokens and accepted_fused_hidden must span the same positions."""
    draft, H, V = _build_tiny_draft(seed=8)
    torch.manual_seed(17)
    T = 6
    fused = torch.randn(1, T, 3 * H)
    ctx = torch.randint(0, V, (1, T))
    state = draft.seed(fused[:, : T - 1], int(ctx[0, T - 1].item()), ctx[:, : T - 1])
    with pytest.raises(ValueError):
        # 1 token but 2 fused rows.
        draft.advance(state, ctx[:, T - 1].view(1), fused[:, T - 2 : T])
