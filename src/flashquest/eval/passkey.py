"""Synthetic long-context retrieval: hide a 5-digit passkey in filler text,
ask the model to retrieve it. Standard sparse-attention quality probe."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass


FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. "
    "There and back again. "
)


@dataclass
class PasskeyExample:
    text: str
    passkey: str
    depth_pct: float  # where in [0, 1] of the filler the passkey was inserted


def make_example(
    *,
    rng: random.Random,
    tokenizer,
    target_total_tokens: int,
    depth_pct: float,
) -> PasskeyExample:
    """Build a passkey prompt with the secret at ~`depth_pct` of the filler.

    Sizes the filler in *actual tokens* using the provided tokenizer so the
    final prompt lands near `target_total_tokens` (within ~5 %).
    """
    passkey = f"{rng.randint(10000, 99999)}"
    needle = (
        f" The pass key is {passkey}. Remember it. {passkey} is the pass key. "
    )
    intro = (
        "There is an important info hidden inside a lot of irrelevant text. "
        "Find it and memorize it. I will quiz you about the important info there.\n\n"
    )
    outro = "\n\nWhat is the pass key? The pass key is "

    fixed_overhead = len(tokenizer.encode(intro + needle + outro, add_special_tokens=False))
    filler_budget = max(0, target_total_tokens - fixed_overhead)
    tokens_per_unit = len(tokenizer.encode(FILLER, add_special_tokens=False)) or 1
    n_units = max(1, filler_budget // tokens_per_unit)

    pre_units = int(n_units * depth_pct)
    post_units = n_units - pre_units
    pre = FILLER * pre_units
    post = FILLER * post_units
    body = pre + needle + post
    prompt = intro + body + outro
    return PasskeyExample(text=prompt, passkey=passkey, depth_pct=depth_pct)


def score(generation: str, passkey: str) -> bool:
    """A generation passes if the passkey appears as the first 5-digit run."""
    m = re.search(r"\d{5}", generation)
    return bool(m) and m.group(0) == passkey
