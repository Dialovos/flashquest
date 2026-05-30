"""Phase 11 — calibrated codebook artifact invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _artifact_dir() -> Path:
    return Path(__file__).parents[1] / "src/flashquest/turbo"


def test_llama_3_2_3b_codebook_artifact_exists():
    """Calibration artifact ships in the package."""
    p = _artifact_dir() / "codebook_llama_3_2_3b.pt"
    assert p.exists(), f"Run scripts/phase11_calibrate_codebook.py to generate {p}"


def test_llama_3_2_3b_codebook_monotonic():
    """Every (layer, K|V) row sorted ascending. Critical for bit-split decode."""
    from flashquest.turbo.codebook import load_codebook

    cb = load_codebook("casperhansen/llama-3.2-3b-instruct-awq")
    diffs = cb[:, :, 1:] - cb[:, :, :-1]
    bad = (diffs < 0).nonzero()
    assert len(bad) == 0, f"non-monotonic codepoints at {bad.tolist()}"


def test_llama_3_2_3b_codebook_sidecar_metadata():
    """Sidecar JSON records calibration provenance."""
    p = _artifact_dir() / "codebook_llama_3_2_3b.json"
    assert p.exists()
    meta = json.loads(p.read_text())
    assert meta["model_id"] == "casperhansen/llama-3.2-3b-instruct-awq"
    assert meta["corpus"] == "PaulGrahamEssays"
    assert meta["n_tokens_per_layer"] >= 50_000
    assert "calibration_commit" in meta
