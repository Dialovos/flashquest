"""Phase 11c — effective-retention resolution for the chat CLI.

Calibrated K3-V3 needs retention 0.25 to clear the RULER NIAH multivalue gate
(100/100/95 at 0.25 vs 85% at 0.20); INT4/INT8 stay at 0.20. An explicit
--retention always overrides. These lock that policy in.
"""
from __future__ import annotations

from types import SimpleNamespace

from flashquest.runtime.chat import _parse_args, _resolve_retention


def _args(**kw):
    base = dict(retention=None, kv_bits=4, codebook="calibrated")
    base.update(kw)
    return SimpleNamespace(**base)


def test_retention_default_arg_is_none_sentinel():
    """--retention parses to None when unset, so the mode-aware default applies."""
    assert _parse_args(["--model", "x", "--context", "1024", "-i"]).retention is None


def test_int4_unset_resolves_to_020():
    assert _resolve_retention(_args(kv_bits=4)) == 0.20


def test_int8_unset_resolves_to_020():
    assert _resolve_retention(_args(kv_bits=8)) == 0.20


def test_calibrated_k3v3_unset_resolves_to_025():
    """The Phase 11c fix: calibrated 3-bit defaults to 0.25 (clears the gate)."""
    assert _resolve_retention(_args(kv_bits=3, codebook="calibrated")) == 0.25


def test_paper_k3v3_unset_stays_020():
    """Paper codebook is not gate-cleared at 0.25; it keeps the 0.20 default."""
    assert _resolve_retention(_args(kv_bits=3, codebook="paper")) == 0.20


def test_explicit_retention_always_wins():
    """An explicit --retention overrides the mode-aware default, even for calib 3-bit."""
    assert _resolve_retention(_args(kv_bits=3, codebook="calibrated", retention=0.10)) == 0.10
    assert _resolve_retention(_args(kv_bits=4, retention=0.5)) == 0.5
