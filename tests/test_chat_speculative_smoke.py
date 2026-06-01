"""Phase 12 task 10 — `flashquest chat --speculative` single-shot smoke.

Exercises the spec-decode path end-to-end through the chat CLI: load the AWQ
target, patch it with the INT4 sparse cache + verify arm, load the EAGLE-3 draft
head, then run the dispatcher loop and stream committed tokens to stdout.

This is a fast/small smoke (24 new tokens, ctx 512, n_draft 4). It asserts the
streamed output is non-empty and coherent (real text, not gibberish/all-EOS),
without raising. End-to-end losslessness vs non-spec greedy is task 9's
equivalence gate; here we only confirm the CLI wiring works.
"""
import pytest
import torch

TARGET = "casperhansen/llama-3.2-3b-instruct-awq"

pytestmark = pytest.mark.slow


def _have_target() -> bool:
    try:
        from transformers import AutoConfig

        AutoConfig.from_pretrained(TARGET)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not _have_target(), reason="AWQ target not available offline")
def test_chat_speculative_single_shot(capsys):
    """`flashquest chat --speculative` single-shot streams coherent, non-empty
    text on a short prompt at a small context (no exception)."""
    from flashquest.runtime import chat as chat_mod

    chat_mod.main([
        "--model", TARGET,
        "--context", "512",
        "--prompt", "List three primary colors, separated by commas.",
        "--max-new-tokens", "24",
        "--speculative",
        "--n-draft", "4",
        # --kv-bits defaults to 4 (required by --speculative); patched path on.
    ])

    out = capsys.readouterr().out
    text = out.strip()

    # Non-empty.
    assert len(text) > 0, "spec-decode produced no streamed output"

    # Coherent: real decoded text, not gibberish/all-EOS. Special tokens are
    # stripped during streaming, so an all-EOS run would surface as empty/blank
    # text — already caught above. Require some alphabetic content and at least a
    # couple of distinct word-like tokens (rules out a single repeated glyph).
    alpha = sum(c.isalpha() for c in text)
    assert alpha >= 3, f"output looks like gibberish (alpha={alpha}): {text!r}"
    words = [w for w in text.split() if any(c.isalpha() for c in w)]
    assert len(words) >= 2, f"output not word-like: {text!r}"
