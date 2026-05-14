"""RULER NIAH 4k subset — 3 tasks: single, multikey, multivalue.

Prompt template + needle string vendored verbatim from
NVIDIA/RULER/scripts/data/synthetic/niah.py (Apache 2.0).
Our generator harness avoids the wonderwords/nltk/tqdm dependency tree;
we use uuids for keys (RULER's `type_needle_k=uuids`) and 7-digit numbers
for values (RULER's default `type_needle_v=numbers`).
"""
from __future__ import annotations

import json
import random
import re
import uuid
from pathlib import Path
from typing import Iterable

# === Verbatim from NVIDIA/RULER/scripts/data/synthetic/niah.py ===
NEEDLE_TEMPLATE = "One of the special magic {type_needle_v} for {key} is: {value}."

# RULER's default template (singular grammar applied if num_needle_q*num_needle_v==1).
PROMPT_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the "
    "provided text? The special magic {type_needle_v} for {query} mentioned in the "
    "provided text are"
)
# === End verbatim ===

TYPE_NEEDLE_V = "numbers"  # RULER's default; we keep "numbers" plural in template per RULER protocol.


def random_number(rng: random.Random, num_digits: int = 7) -> str:
    lower = 10 ** (num_digits - 1)
    upper = 10 ** num_digits - 1
    return str(rng.randint(lower, upper))


def random_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128), version=4))


def _singularize(template: str) -> str:
    """RULER's grammar fixup when num_needle_q * num_needle_v == 1."""
    template = template.replace("Some", "A")
    template = template.replace("are all", "is")
    template = template.replace("are", "is")
    template = template.replace("answers", "answer")
    return template


_HAYSTACK_CACHE: list[str] | None = None


def _load_haystack_words() -> list[str]:
    """Load Paul Graham essays from the committed corpus.

    Path resolution: <repo>/data/PaulGrahamEssays.json. Bundled by
    scripts/fetch_ruler_corpus.sh from gkamradt/LLMTest_NeedleInAHaystack
    (subset of RULER's URL list, no html2text dep).
    """
    global _HAYSTACK_CACHE
    if _HAYSTACK_CACHE is not None:
        return _HAYSTACK_CACHE
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    else:
        raise RuntimeError("could not locate repo root from " + str(here))
    corpus_path = repo_root / "data" / "PaulGrahamEssays.json"
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"NIAH corpus not at {corpus_path}; run "
            f"bash scripts/fetch_ruler_corpus.sh"
        )
    text = json.loads(corpus_path.read_text())["text"]
    words = re.sub(r"\s+", " ", text).split(" ")
    _HAYSTACK_CACHE = words
    return words


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split — '. ' / '! ' / '? ' boundaries only."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _build_context(rng: random.Random, num_words: int, needles: list[str]) -> str:
    """Insert needles at random sentence positions in a prefix of the haystack."""
    haystack = _load_haystack_words()
    if num_words <= len(haystack):
        text = " ".join(haystack[:num_words])
    else:
        repeats = (num_words + len(haystack) - 1) // len(haystack)
        text = " ".join((haystack * repeats)[:num_words])
    sentences = _split_sentences(text)
    n_sents = len(sentences)
    depths = sorted(rng.sample(range(0, n_sents + 1), len(needles)))
    out: list[str] = []
    last = 0
    for i, d in enumerate(depths):
        out.append(" ".join(sentences[last:d]))
        out.append(needles[i])
        last = d
    out.append(" ".join(sentences[last:]))
    return " ".join(p for p in out if p)


def _budget_haystack_words(
    template_singular: str,
    type_needle_v: str,
    key: str,
    needle: str,
    tokenizer,
    ctx_len: int,
    margin_tokens: int = 256,
) -> int:
    """Binary-search the haystack word count so the rendered prompt fits in
    ctx_len - margin_tokens. margin_tokens covers the assistant's answer."""
    target = ctx_len - margin_tokens
    haystack = _load_haystack_words()
    lo, hi = 100, min(len(haystack) * 2, ctx_len * 4)
    best = lo
    for _ in range(20):
        mid = (lo + hi) // 2
        rendered = template_singular.format(
            type_needle_v=type_needle_v.rstrip("s"),
            context=_build_context(random.Random(0), mid, [needle]),
            query=key,
        )
        n = len(tokenizer(rendered).input_ids)
        if n <= target:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def make_prompt(
    task: str,
    ctx_len: int,
    tokenizer,
    seed: int,
) -> tuple[str, list[str]]:
    """Generate a NIAH prompt + expected keys. Tasks: single, multikey, multivalue."""
    rng = random.Random(seed)

    if task == "single":
        key = random_uuid(rng)
        value = random_number(rng)
        needle = NEEDLE_TEMPLATE.format(
            type_needle_v=TYPE_NEEDLE_V, key=key, value=value
        )
        template = _singularize(PROMPT_TEMPLATE)
        n_words = _budget_haystack_words(
            template, TYPE_NEEDLE_V, key, needle, tokenizer, ctx_len
        )
        context = _build_context(rng, n_words, [needle])
        prompt = template.format(
            type_needle_v=TYPE_NEEDLE_V.rstrip("s"),
            context=context,
            query=key,
        )
        return prompt, [value]

    if task == "multikey":
        # 4 keys total (1 target + 3 distractors), 1 value each, query the target.
        num_needle_k = 4
        keys = [random_uuid(rng) for _ in range(num_needle_k)]
        values = [random_number(rng) for _ in range(num_needle_k)]
        needles = [
            NEEDLE_TEMPLATE.format(type_needle_v=TYPE_NEEDLE_V, key=keys[i], value=values[i])
            for i in range(num_needle_k)
        ]
        target_idx = rng.randrange(num_needle_k)
        target_key = keys[target_idx]
        target_value = values[target_idx]
        rng.shuffle(needles)
        template = _singularize(PROMPT_TEMPLATE)
        n_words = _budget_haystack_words(
            template, TYPE_NEEDLE_V, target_key, needles[0], tokenizer, ctx_len
        )
        context = _build_context(rng, n_words, needles)
        prompt = template.format(
            type_needle_v=TYPE_NEEDLE_V.rstrip("s"),
            context=context,
            query=target_key,
        )
        return prompt, [target_value]

    if task == "multivalue":
        # 1 key, 4 values; query that key; must retrieve all 4 values.
        num_needle_v = 4
        key = random_uuid(rng)
        values = [random_number(rng) for _ in range(num_needle_v)]
        needles = [
            NEEDLE_TEMPLATE.format(type_needle_v=TYPE_NEEDLE_V, key=key, value=v)
            for v in values
        ]
        rng.shuffle(needles)
        # num_q * num_v = 1 * 4 = 4 ≠ 1: keep the PLURAL template (no singularize).
        template = PROMPT_TEMPLATE
        n_words = _budget_haystack_words(
            template, TYPE_NEEDLE_V, key, needles[0], tokenizer, ctx_len
        )
        context = _build_context(rng, n_words, needles)
        prompt = template.format(
            type_needle_v=TYPE_NEEDLE_V,  # plural
            context=context,
            query=key,
        )
        return prompt, list(values)

    raise NotImplementedError(f"task={task!r} not implemented yet")


def score(generated: str, expected_keys: Iterable[str]) -> bool:
    """RULER NIAH scoring: case-sensitive substring; ALL expected_keys must appear."""
    return all(k in generated for k in expected_keys)
