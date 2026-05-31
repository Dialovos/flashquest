"""Phase 12 task 8 — synthetic unit tests for the spec-decode dispatcher.

Tests the walk-and-accept + commit accounting in isolation (no GPU model).
End-to-end losslessness (spec output == non-spec greedy on a real model) is
task 9's equivalence test.
"""
import pytest
import torch

from flashquest.specdec.dispatcher import _walk_accept


# ---------------------------------------------------------------------------
# walk-and-accept (pure helper) — hand-worked cases
# ---------------------------------------------------------------------------

def test_walk_accept_full_accept():
    """All tgt[i] == d_{i+1}: m = n_draft-1, accepted = bonus + all drafts."""
    n = 4
    bonus = 100
    # chain = [d_1, d_2, d_3, d_4]
    chain = [11, 12, 13, 14]
    verify_input = [bonus, chain[0], chain[1], chain[2]]      # [100, 11, 12, 13]
    # tgt[i] is the token AFTER verify_input[i]; full accept => tgt[i]==chain[i+1] for i<3,
    # and tgt[3] is the continuation past the last accepted draft.
    tgt = [11, 12, 13, 999]                                   # tgt[0..2] match chain[1..3]
    m, accepted, new_bonus = _walk_accept(verify_input, tgt, n)
    assert m == n - 1 == 3
    assert accepted == [100, 11, 12, 13]                      # bonus + d_1..d_3
    assert new_bonus == 999                                   # tgt[3], held continuation


def test_walk_accept_mismatch_at_i1():
    """tgt[0] matches d_1, tgt[1] != d_2: walk breaks at i=1 => m=1."""
    n = 4
    bonus = 100
    chain = [11, 12, 13, 14]
    verify_input = [bonus, 11, 12, 13]
    tgt = [11, 777, 13, 14]                                   # i=0 ok (11==11); i=1 fail (777!=12)
    m, accepted, new_bonus = _walk_accept(verify_input, tgt, n)
    assert m == 1
    assert accepted == [100, 11]                              # bonus + d_1
    assert new_bonus == 777                                   # tgt[1], the correction


def test_walk_accept_immediate_mismatch():
    """tgt[0] != d_1: m=0, accepted=[bonus], new_bonus=tgt[0]."""
    n = 4
    bonus = 100
    chain = [11, 12, 13, 14]
    verify_input = [bonus, 11, 12, 13]
    tgt = [555, 12, 13, 14]                                   # i=0 fail (555 != 11)
    m, accepted, new_bonus = _walk_accept(verify_input, tgt, n)
    assert m == 0
    assert accepted == [100]                                  # bonus only
    assert new_bonus == 555                                   # tgt[0]


def test_walk_accept_accepts_tensor_inputs():
    """Tensors are accepted (mirrors the dispatcher's call site)."""
    n = 4
    verify_input = torch.tensor([100, 11, 12, 13], dtype=torch.int64)
    tgt = torch.tensor([11, 12, 777, 14], dtype=torch.int64)  # break at i=2 => m=2
    m, accepted, new_bonus = _walk_accept(verify_input, tgt, n)
    assert m == 2
    assert accepted == [100, 11, 12]
    assert new_bonus == 777


def test_walk_accept_length_mismatch_raises():
    with pytest.raises(ValueError):
        _walk_accept([1, 2, 3], [1, 2, 3, 4], 4)
    with pytest.raises(ValueError):
        _walk_accept([1, 2, 3, 4], [1, 2, 3], 4)


def test_walk_accept_n_draft_1_degenerate():
    """n_draft=1: no drafts to accept, m=0, accepted=[bonus], new_bonus=tgt[0]."""
    m, accepted, new_bonus = _walk_accept([42], [99], 1)
    assert m == 0
    assert accepted == [42]
    assert new_bonus == 99


# ---------------------------------------------------------------------------
# commit accounting — CPU minimal stub (no GPU)
# ---------------------------------------------------------------------------

class _FakeCache:
    """Minimal stand-in exercising the dispatcher's commit contract on CPU:
    add_draft stages n_draft, commit_draft_all_layers(m+1) advances _seen by m+1
    and clears the sandbox. Mirrors PersistentInt4KVCache's accounting."""

    MAX_DRAFT = 8

    def __init__(self, num_layers=2, page_size=64):
        self.num_layers = num_layers
        self.page_size = page_size
        self._seen_tokens = [0] * num_layers
        self._sandbox_count = [0] * num_layers
        self.commit_calls = []

    def add_draft(self, K_new, V_new, layer_idx):
        s = K_new.shape[2] if torch.is_tensor(K_new) else int(K_new)
        if s > self.MAX_DRAFT:
            raise ValueError("over MAX_DRAFT")
        self._sandbox_count[layer_idx] = s

    def commit_draft_all_layers(self, accept_count):
        for li in range(self.num_layers):
            if accept_count > self._sandbox_count[li]:
                raise ValueError("accept_count > sandbox_count")
            seen = self._seen_tokens[li]
            if accept_count > 0 and (seen + accept_count) // self.page_size != seen // self.page_size:
                raise RuntimeError("page boundary")
        self.commit_calls.append(accept_count)
        for li in range(self.num_layers):
            self._seen_tokens[li] += accept_count
            self._sandbox_count[li] = 0


def test_fake_cache_commit_advances_by_m_plus_1():
    """add_draft(n_draft) then commit(m+1): _seen advances by exactly m+1,
    sandbox cleared. This is the accounting the dispatcher's commit path relies on."""
    c = _FakeCache(num_layers=3, page_size=64)
    n_draft, m = 4, 2
    for li in range(c.num_layers):
        c.add_draft(n_draft, n_draft, li)
    assert all(s == n_draft for s in c._sandbox_count)
    c.commit_draft_all_layers(m + 1)
    assert c.commit_calls == [m + 1]
    assert all(s == m + 1 for s in c._seen_tokens)            # advanced by m+1
    assert all(s == 0 for s in c._sandbox_count)              # sandbox dropped


def test_fake_cache_commit_zero_accept():
    """m=0 (immediate mismatch) commits just the bonus (m+1=1)."""
    c = _FakeCache(num_layers=2, page_size=64)
    for li in range(c.num_layers):
        c.add_draft(4, 4, li)
    c.commit_draft_all_layers(1)                              # bonus only
    assert all(s == 1 for s in c._seen_tokens)
    assert all(s == 0 for s in c._sandbox_count)


# ---------------------------------------------------------------------------
# commit accounting — real PersistentInt4KVCache (GPU)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_cache_commit_accounting():
    """The real cache: add_draft(n_draft) + commit(m+1) advances _seen by m+1
    and resets the sandbox — the exact contract the dispatcher commits against."""
    from flashquest.cache.persistent_int4 import PersistentInt4KVCache

    c = PersistentInt4KVCache(batch_size=1, num_layers=2, num_kv_heads=8,
                              head_dim=128, max_seq_len=4096, page_size=64,
                              device="cuda")
    n_draft, m = 4, 2

    def _kv(S):
        return (torch.randn(1, 8, S, 128, dtype=torch.bfloat16, device="cuda"),
                torch.randn(1, 8, S, 128, dtype=torch.bfloat16, device="cuda"))

    # Seed one full page so the (N + n_draft) commit stays within a page.
    for li in range(c.num_layers):
        c.update_quantized(*_kv(64), li)
    assert all(s == 64 for s in c._seen_tokens)

    for li in range(c.num_layers):
        c.add_draft(*_kv(n_draft), li)
    assert all(s == n_draft for s in c._sandbox_count)

    c.commit_draft_all_layers(m + 1)
    assert all(s == 64 + (m + 1) for s in c._seen_tokens)     # advanced by m+1
    assert all(s == 0 for s in c._sandbox_count)              # remaining drafts dropped
