"""Tests for DuoAttention pattern loader."""
from pathlib import Path

import pytest
import torch

VENDOR_PATTERN = Path(__file__).resolve().parents[1] / "vendor" / "duo-attention" / "attn_patterns" / "Meta-Llama-3.1-8B-Instruct" / "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10" / "full_attention_heads.tsv"


def test_load_llama_3_1_8b_shape():
    """EP4/EP5: Llama-3.1-8B has 32 layers × 8 KV heads."""
    from flashquest.duo.pattern import load_duo_pattern

    pattern = load_duo_pattern(VENDOR_PATTERN)
    assert pattern.shape == (32, 8)
    assert pattern.dtype == torch.bool


def test_threshold_at_half():
    """EP6: values >= 0.5 round up to retrieval (True), < 0.5 to streaming."""
    from flashquest.duo.pattern import load_duo_pattern

    pattern = load_duo_pattern(VENDOR_PATTERN)
    assert pattern.any(), "expected at least one retrieval head"
    assert (~pattern).any(), "expected at least one streaming head"


def test_threshold_kwarg(tmp_path: Path):
    """User can override threshold; default is 0.5."""
    from flashquest.duo.pattern import load_duo_pattern

    p = tmp_path / "tiny.tsv"
    p.write_text("0.4\t0.6\n0.5\t0.51\n")

    default = load_duo_pattern(p)
    assert default.tolist() == [[False, True], [True, True]]

    strict = load_duo_pattern(p, threshold=0.51)
    assert strict.tolist() == [[False, True], [False, True]]


def test_missing_file_raises(tmp_path: Path):
    """EP11: missing file raises FileNotFoundError."""
    from flashquest.duo.pattern import load_duo_pattern

    with pytest.raises(FileNotFoundError):
        load_duo_pattern(tmp_path / "nope.tsv")


def test_empty_file_raises(tmp_path: Path):
    """EP11: empty TSV raises ValueError."""
    from flashquest.duo.pattern import load_duo_pattern

    p = tmp_path / "empty.tsv"
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_duo_pattern(p)
