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


def test_er3_multikey_distractors_distinct(tok):
    """ER3: multikey has 4 keys, returns expected_keys = [target_value]
    (target key matches the question). Distractor keys must not equal target key."""
    prompt, expected = make_prompt("multikey", ctx_len=4096, tokenizer=tok, seed=0)
    assert len(expected) == 1
    target_value = expected[0]
    # The needle for the target key/value must appear in prompt.
    assert target_value in prompt
    # Count distinct UUIDs in the prompt — should be exactly 4.
    uuid_re = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    uuids = set(re.findall(uuid_re, prompt))
    assert len(uuids) == 4, f"expected 4 distinct keys, got {len(uuids)}"


def test_multivalue_returns_4_values(tok):
    """multivalue: 1 key, 4 values; expected = all 4 values; all must appear in prompt."""
    prompt, expected = make_prompt("multivalue", ctx_len=4096, tokenizer=tok, seed=0)
    assert len(expected) == 4
    for v in expected:
        assert v in prompt
    uuid_re = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    uuids = set(re.findall(uuid_re, prompt))
    assert len(uuids) == 1


def test_er4_score_substring_match():
    """ER4: score() is exact substring match per RULER; handles trailing model output."""
    assert score("the answer is 1234567 because reasons", ["1234567"]) is True
    assert score(" 1234567.", ["1234567"]) is True
    assert score("12345", ["1234567"]) is False
    assert score(
        "v1=1111111 and v2=2222222 v3=3333333 v4=4444444",
        ["1111111", "2222222", "3333333", "4444444"],
    ) is True
    assert score(
        "only 1111111 and 2222222 are present",
        ["1111111", "2222222", "3333333", "4444444"],
    ) is False
    assert score("anything", []) is True


def test_er6_run_niah_zero_samples(tok):
    """ER6: n_samples=0 returns empty result, doesn't crash; model never called."""
    from flashquest.eval.runner import run_niah
    out = run_niah(model=None, tokenizer=tok, task="single",
                   n_samples=0, ctx_len=512, seed=0)
    assert out["hits"] == 0
    assert out["total"] == 0
    assert out["task"] == "single"
    assert out["samples"] == []


@pytest.mark.slow
def test_er5_smoke_llama_3_2_1b_ctx512_n2():
    """ER5: end-to-end smoke on Llama-3.2-1B at ctx=512 with n=2; <90s."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flashquest.eval.runner import run_niah

    name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    t0 = time.perf_counter()
    out = run_niah(model, tokenizer, task="single", n_samples=2,
                   ctx_len=512, seed=0, max_new_tokens=64)
    elapsed = time.perf_counter() - t0
    assert elapsed < 90, f"smoke too slow: {elapsed:.1f}s"
    assert out["total"] == 2
    assert 0 <= out["hits"] <= 2
