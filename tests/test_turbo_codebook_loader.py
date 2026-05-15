"""Phase 11 — codebook loader tests."""
from __future__ import annotations

import pytest
import torch


def test_load_codebook_returns_correct_shape():
    """Llama-3.2-3B-AWQ codebook ships at (28, 2, 8) fp32."""
    from flashquest.turbo.codebook import load_codebook

    cb = load_codebook("casperhansen/llama-3.2-3b-instruct-awq")
    assert cb.shape == (28, 2, 8), f"got {tuple(cb.shape)}"
    assert cb.dtype == torch.float32
    # Each (layer, kv) row must be sorted ascending (bit-split invariant).
    for li in range(28):
        for kv in range(2):
            assert torch.all(cb[li, kv, :-1] <= cb[li, kv, 1:]), \
                f"layer {li} kv {kv} not sorted: {cb[li, kv].tolist()}"


def test_load_codebook_unknown_model_raises_keyerror():
    """Models without a shipped artifact raise KeyError cleanly."""
    from flashquest.turbo.codebook import load_codebook

    with pytest.raises(KeyError):
        load_codebook("nonexistent/model-id")


def test_paper_codebook_constant():
    """The paper codebook is exposed for fallback callers; matches kv_quant.py."""
    from flashquest.turbo.codebook import PAPER_CODEBOOK
    from flashquest.kernel.kv_quant import K_TURBO_CODEBOOK

    assert torch.allclose(PAPER_CODEBOOK.cpu(), K_TURBO_CODEBOOK.cpu())
    assert PAPER_CODEBOOK.shape == (8,)
