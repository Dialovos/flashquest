"""ER1-ER6: RULER NIAH 4k subset eval — generators + scoring."""
import re

import pytest

from flashquest.eval.niah import (
    NEEDLE_TEMPLATE,
    PROMPT_TEMPLATE,
    random_number,
    random_uuid,
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
