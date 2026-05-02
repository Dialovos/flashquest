"""ER1-ER6: RULER NIAH 4k subset eval — generators + scoring."""
import re

import pytest

from flashquest.eval.niah import (
    NEEDLE_TEMPLATE,
    PROMPT_TEMPLATE,
    make_prompt,
    random_number,
    random_uuid,
    score,
)


def test_random_number_format():
    """7-digit numeric string; deterministic given seed."""
    import random
    rng = random.Random(0)
    n = random_number(rng, num_digits=7)
    assert n.isdigit()
    assert len(n) == 7


def test_random_uuid_format():
    """uuid4 string; deterministic given seed."""
    import random
    rng = random.Random(0)
    u = random_uuid(rng)
    assert re.match(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", u)


def test_needle_template_format():
    """RULER's verbatim needle string."""
    assert NEEDLE_TEMPLATE == (
        "One of the special magic {type_needle_v} for {key} is: {value}."
    )


def test_prompt_template_substring():
    """RULER's verbatim prompt opener."""
    assert "Some special magic" in PROMPT_TEMPLATE
    assert "{context}" in PROMPT_TEMPLATE
    assert "What are all the special magic" in PROMPT_TEMPLATE


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")


def test_er1_single_fits_ctx(tok):
    """ER1: prompt fits in ctx_len after tokenization, with sane density."""
    prompt, keys = make_prompt("single", ctx_len=4096, tokenizer=tok, seed=0)
    n_tokens = len(tok(prompt).input_ids)
    assert n_tokens <= 4096, f"prompt too long: {n_tokens} tokens"
    assert n_tokens >= 3000, f"prompt too short: {n_tokens} (target ~4000)"
    assert len(keys) == 1


def test_er2_seed_determinism(tok):
    """ER2: same seed → same prompt + keys; different seed → different keys."""
    p0a, k0a = make_prompt("single", ctx_len=2048, tokenizer=tok, seed=42)
    p0b, k0b = make_prompt("single", ctx_len=2048, tokenizer=tok, seed=42)
    p1, k1 = make_prompt("single", ctx_len=2048, tokenizer=tok, seed=43)
    assert p0a == p0b
    assert k0a == k0b
    assert k0a != k1   # different seed → different needle value
