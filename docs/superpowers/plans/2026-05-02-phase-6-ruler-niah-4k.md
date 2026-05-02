# Phase 6 Task 2 — RULER NIAH 4k Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken un-seeded passkey eval with the SPEC's actual quality gate — RULER NIAH (single, multikey, multivalue) at ctx=4k on Llama-3.2-3B-AWQ, retention=0.25 vs dense SDPA, ≥85% pass.

**Architecture:** Hybrid — own generator harness (no `nltk`/`wonderwords`/`tqdm` dependencies) but RULER's prompt template + scoring rules vendored verbatim from `https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/niah.py`. Three task generators (single, multikey, multivalue) call shared primitives. All-retrieval head_pattern eliminates head-pattern variance. Runner is a thin loop around `model.generate(...)`.

**Tech Stack:** PyTorch, transformers, autoawq (already in pyproject). No new runtime deps — pure-Python generator + score functions.

**Reference:** `docs/superpowers/specs/2026-05-02-phase-6-ruler-niah-4k-design.md`. RULER NIAH source: `https://raw.githubusercontent.com/NVIDIA/RULER/main/scripts/data/synthetic/niah.py` (Apache 2.0).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `vendor/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json` | Filler corpus (existing in RULER repo) | Fetch via vendor_clone.sh |
| `scripts/vendor_clone.sh` | Add RULER repo | Modify |
| `src/flashquest/eval/__init__.py` | Re-export NIAH generators + scoring + runner | Create |
| `src/flashquest/eval/niah.py` | `make_prompt(task, ctx_len, tok, seed)` + `score(...)` | Create |
| `src/flashquest/eval/runner.py` | `run_niah(model, tok, task, n_samples, ...)` | Create |
| `tests/test_eval_niah.py` | ER1-ER6 unit tests | Create |
| `scripts/phase6_run_ruler_4k.py` | CLI: dense + patched runs across 3 tasks | Create |
| `benchmarks/phase6_ruler_4k.json` | Run output | Create at end |
| `docs/PHASES/phase-6-notes.md` | Append task 2 section | Modify at end |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 2 | Modify at end |

---

## Task 1: Vendor RULER + Paul Graham essays corpus

**Files:**
- Modify: `scripts/vendor_clone.sh`

- [ ] **Step 1: Add RULER to vendor_clone.sh**

Append at the end of `scripts/vendor_clone.sh` (before the closing line if any):

```bash
clone_or_pull https://github.com/NVIDIA/RULER.git                       RULER
```

- [ ] **Step 2: Run vendor sync**

Run:
```
bash scripts/vendor_clone.sh
```
Expected: clones `vendor/RULER`. Verify: `ls vendor/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json` exists.

- [ ] **Step 3: Commit**

```
git add scripts/vendor_clone.sh
git commit -m "phase 6 task 2: vendor RULER for NIAH corpus"
```

---

## Task 2: NIAH primitives (random key/value, needle template)

**Files:**
- Create: `src/flashquest/eval/__init__.py`
- Create: `src/flashquest/eval/niah.py`
- Create: `tests/test_eval_niah.py`

- [ ] **Step 1: Write the failing test for primitive generators**

Create `tests/test_eval_niah.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail (ImportError)**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py -v
```
Expected: ImportError on `flashquest.eval.niah`.

- [ ] **Step 3: Create `src/flashquest/eval/__init__.py`**

```python
"""Eval harnesses for flashquest. Phase 6 task 2: RULER NIAH 4k subset."""
from .niah import (
    NEEDLE_TEMPLATE,
    PROMPT_TEMPLATE,
    make_prompt,
    score,
)
from .runner import run_niah

__all__ = [
    "NEEDLE_TEMPLATE",
    "PROMPT_TEMPLATE",
    "make_prompt",
    "score",
    "run_niah",
]
```

- [ ] **Step 4: Create `src/flashquest/eval/niah.py` with primitives only**

```python
"""RULER NIAH 4k subset — 3 tasks: single, multikey, multivalue.

Prompt template + needle string vendored verbatim from
NVIDIA/RULER/scripts/data/synthetic/niah.py (Apache 2.0).
Our generator harness avoids the wonderwords/nltk/tqdm dependency tree;
we use uuids for keys (RULER's `type_needle_k=uuids`) and 7-digit numbers
for values (RULER's default `type_needle_v=numbers`).
"""
from __future__ import annotations

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

TYPE_NEEDLE_V = "numbers"   # RULER's default; we keep "numbers" plural in template per RULER protocol.


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
```

- [ ] **Step 5: Run tests to verify primitives pass; full-prompt tests still fail with NameError on make_prompt**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py -v
```
Expected: 4 of the simple tests PASS; `make_prompt` and `score` import succeeds (in `__init__.py`) but no tests use them yet so collection succeeds. (If `runner.run_niah` import fails, mark as expected — Task 6 fixes.) For now, comment the `from .runner import run_niah` line in `__init__.py` if it blocks.

Actually fix immediately: in `__init__.py`, leave `runner` import commented until Task 6:
```python
# from .runner import run_niah   # uncommented in Task 6
```

Re-run; tests pass.

- [ ] **Step 6: Commit**

```
git add src/flashquest/eval/__init__.py src/flashquest/eval/niah.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: NIAH primitives (uuids, numbers, RULER templates verbatim)"
```

---

## Task 3: `make_prompt("single", ...)` + ER1, ER2

**Files:**
- Modify: `src/flashquest/eval/niah.py`
- Modify: `tests/test_eval_niah.py`

- [ ] **Step 1: Write the failing test for niah_single**

Append to `tests/test_eval_niah.py`:

```python
@pytest.fixture
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")


def test_er1_single_fits_ctx(tok):
    """ER1: prompt + answer fits in ctx_len after tokenization."""
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
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er1_single_fits_ctx -v
```
Expected: FAIL with `ImportError: cannot import name 'make_prompt'` (we removed it from `__init__.py`) OR `NameError`.

Fix: ensure `make_prompt` is exported. We'll define it in this task.

- [ ] **Step 3: Implement `_load_haystack_words` and `make_prompt("single", ...)`**

Append to `src/flashquest/eval/niah.py`:

```python
_HAYSTACK_CACHE: list[str] | None = None


def _load_haystack_words() -> list[str]:
    """Load Paul Graham essays from the vendored RULER corpus.

    Path resolution:
        <repo>/vendor/RULER/scripts/data/synthetic/json/PaulGrahamEssays.json
    """
    global _HAYSTACK_CACHE
    if _HAYSTACK_CACHE is not None:
        return _HAYSTACK_CACHE
    import json
    here = Path(__file__).resolve()
    # Walk up to repo root (where pyproject.toml lives).
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    else:
        raise RuntimeError("could not locate repo root from " + str(here))
    corpus_path = (
        repo_root / "vendor" / "RULER" / "scripts" / "data" / "synthetic"
        / "json" / "PaulGrahamEssays.json"
    )
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"RULER essay corpus not at {corpus_path}; run "
            f"bash scripts/vendor_clone.sh"
        )
    text = json.loads(corpus_path.read_text())["text"]
    words = re.sub(r"\s+", " ", text).split(" ")
    _HAYSTACK_CACHE = words
    return words


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence split — '. ' boundaries only. Matches the granularity
    needed for needle insertion; we don't need nltk's full tokenizer."""
    # Re-glue '. ' splits with the period kept.
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _build_context(rng: random.Random, num_words: int, needles: list[str]) -> str:
    """Insert needles at random sentence positions in a prefix of the haystack."""
    haystack = _load_haystack_words()
    if num_words <= len(haystack):
        text = " ".join(haystack[:num_words])
    else:
        # RULER's repeat-and-slice fallback for short corpora.
        repeats = (num_words + len(haystack) - 1) // len(haystack)
        text = " ".join((haystack * repeats)[:num_words])
    sentences = _split_sentences(text)
    n_sents = len(sentences)
    # 40-position depth grid like RULER.
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
        template = _singularize(PROMPT_TEMPLATE)  # num_q*num_v=1 → singular
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

    raise NotImplementedError(f"task={task!r} not implemented yet")
```

- [ ] **Step 4: Run ER1 + ER2**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er1_single_fits_ctx tests/test_eval_niah.py::test_er2_seed_determinism -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eval/niah.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: niah_single generator (ER1, ER2)"
```

---

## Task 4: `make_prompt("multikey", ...)` + ER3

**Files:**
- Modify: `src/flashquest/eval/niah.py`
- Modify: `tests/test_eval_niah.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_eval_niah.py`:

```python
def test_er3_multikey_distractors_distinct(tok):
    """ER3: multikey has 4 keys, returns expected_keys = [target_value]
    (target key matches the question). Distractor keys must not equal target key."""
    prompt, expected = make_prompt("multikey", ctx_len=4096, tokenizer=tok, seed=0)
    assert len(expected) == 1
    target_value = expected[0]
    # The needle string for the target key/value must appear in prompt.
    assert target_value in prompt
    # Count distinct UUIDs in the prompt (rough check for 4 keys).
    uuids = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", prompt))
    assert len(uuids) == 4, f"expected 4 distinct keys, got {len(uuids)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er3_multikey_distractors_distinct -v
```
Expected: FAIL with `NotImplementedError: task='multikey'`.

- [ ] **Step 3: Implement multikey branch**

In `src/flashquest/eval/niah.py`, replace the `raise NotImplementedError` at the end of `make_prompt` with:

```python
    if task == "multikey":
        # 4 keys total, 1 value each, query 1 (the target).
        num_needle_k, num_needle_v, num_needle_q = 4, 1, 1
        keys = [random_uuid(rng) for _ in range(num_needle_k)]
        values = [[random_number(rng)] for _ in range(num_needle_k)]
        needles = [
            NEEDLE_TEMPLATE.format(
                type_needle_v=TYPE_NEEDLE_V, key=keys[i], value=values[i][0]
            )
            for i in range(num_needle_k)
        ]
        rng2 = random.Random(seed)
        rng2.shuffle(needles)
        target_idx = rng.randrange(num_needle_k)
        target_key = keys[target_idx]
        target_value = values[target_idx][0]
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

    raise NotImplementedError(f"task={task!r} not implemented yet")
```

- [ ] **Step 4: Run ER3**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er3_multikey_distractors_distinct -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eval/niah.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: niah_multikey generator (ER3)"
```

---

## Task 5: `make_prompt("multivalue", ...)`

**Files:**
- Modify: `src/flashquest/eval/niah.py`
- Modify: `tests/test_eval_niah.py`

- [ ] **Step 1: Write the test**

Append to `tests/test_eval_niah.py`:

```python
def test_multivalue_returns_4_values(tok):
    """multivalue: 1 key, 4 values; expected = all 4 values; all must appear in prompt."""
    prompt, expected = make_prompt("multivalue", ctx_len=4096, tokenizer=tok, seed=0)
    assert len(expected) == 4
    for v in expected:
        assert v in prompt
    # 1 unique key (UUID).
    uuids = set(re.findall(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", prompt))
    assert len(uuids) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_multivalue_returns_4_values -v
```
Expected: FAIL with NotImplementedError.

- [ ] **Step 3: Implement multivalue branch**

In `src/flashquest/eval/niah.py`, replace the `raise NotImplementedError` at the end of `make_prompt` with:

```python
    if task == "multivalue":
        # 1 key, 4 values, query that 1 key — must retrieve all 4 values.
        num_needle_v = 4
        key = random_uuid(rng)
        values = [random_number(rng) for _ in range(num_needle_v)]
        needles = [
            NEEDLE_TEMPLATE.format(type_needle_v=TYPE_NEEDLE_V, key=key, value=v)
            for v in values
        ]
        rng2 = random.Random(seed)
        rng2.shuffle(needles)
        # Plural grammar: num_q * num_v = 1 * 4 = 4 ≠ 1, so keep PLURAL template.
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
```

- [ ] **Step 4: Run all generator tests**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py -v -k "single or multikey or multivalue or random or template"
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eval/niah.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: niah_multivalue generator"
```

---

## Task 6: `score()` + ER4

**Files:**
- Modify: `src/flashquest/eval/niah.py`
- Modify: `tests/test_eval_niah.py`

- [ ] **Step 1: Write ER4 test**

Append to `tests/test_eval_niah.py`:

```python
def test_er4_score_substring_match():
    """ER4: score() is exact substring match per RULER; handles trailing model output."""
    # Single key — 1 expected value.
    assert score("the answer is 1234567 because reasons", ["1234567"]) is True
    assert score(" 1234567.", ["1234567"]) is True
    assert score("12345", ["1234567"]) is False
    # Multivalue — all expected values must appear.
    assert score("v1=1111111 and v2=2222222 v3=3333333 v4=4444444", ["1111111", "2222222", "3333333", "4444444"]) is True
    assert score("only 1111111 and 2222222 are present", ["1111111", "2222222", "3333333", "4444444"]) is False
    # Empty expected → trivially true.
    assert score("anything", []) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er4_score_substring_match -v
```
Expected: FAIL — `score` import already in `__init__.py` from Task 2 but the function isn't defined yet.

- [ ] **Step 3: Implement `score`**

Append to `src/flashquest/eval/niah.py`:

```python
def score(generated: str, expected_keys: Iterable[str]) -> bool:
    """RULER NIAH scoring: case-sensitive substring; ALL expected_keys must appear."""
    return all(k in generated for k in expected_keys)
```

- [ ] **Step 4: Run ER4**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er4_score_substring_match -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```
git add src/flashquest/eval/niah.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: score() — substring match per RULER (ER4)"
```

---

## Task 7: `runner.run_niah` + ER5, ER6

**Files:**
- Create: `src/flashquest/eval/runner.py`
- Modify: `src/flashquest/eval/__init__.py`
- Modify: `tests/test_eval_niah.py`

- [ ] **Step 1: Write ER5 + ER6 (smoke + edge)**

Append to `tests/test_eval_niah.py`:

```python
def test_er6_run_niah_zero_samples(tok):
    """ER6: n_samples=0 returns empty result, doesn't crash."""
    from flashquest.eval.runner import run_niah
    # Use a stub model that won't be called.
    out = run_niah(model=None, tokenizer=tok, task="single",
                   n_samples=0, ctx_len=512, seed=0)
    assert out["hits"] == 0
    assert out["total"] == 0
    assert out["task"] == "single"


@pytest.mark.slow
def test_er5_smoke_llama_3_2_1b_ctx512_n2():
    """ER5: end-to-end smoke on Llama-3.2-1B at ctx=512 with n=2; <60s."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flashquest.eval.runner import run_niah

    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda().eval()

    t0 = time.perf_counter()
    out = run_niah(model, tok, task="single", n_samples=2,
                   ctx_len=512, seed=0, max_new_tokens=64)
    elapsed = time.perf_counter() - t0
    assert elapsed < 90, f"smoke too slow: {elapsed:.1f}s"
    assert out["total"] == 2
    assert 0 <= out["hits"] <= 2
```

- [ ] **Step 2: Create `src/flashquest/eval/runner.py`**

```python
"""Run a NIAH task across n_samples; return hits/total."""
from __future__ import annotations

import torch

from .niah import make_prompt, score


@torch.no_grad()
def run_niah(
    model,
    tokenizer,
    task: str,
    n_samples: int,
    ctx_len: int,
    seed: int = 0,
    max_new_tokens: int = 128,
) -> dict:
    """Run `task` on `model` for `n_samples` prompts. Returns:
        {"task": str, "hits": int, "total": int, "samples": [{prompt, keys, generated, hit}, ...]}
    """
    samples = []
    hits = 0
    for i in range(n_samples):
        prompt, expected = make_prompt(task, ctx_len, tokenizer, seed=seed * 10000 + i)
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        hit = score(text, expected)
        if hit:
            hits += 1
        samples.append({
            "prompt_tokens": int(ids.shape[1]),
            "expected": list(expected),
            "generated": text,
            "hit": bool(hit),
        })
    return {"task": task, "hits": hits, "total": n_samples, "samples": samples}
```

- [ ] **Step 3: Uncomment runner re-export**

Edit `src/flashquest/eval/__init__.py` to uncomment `from .runner import run_niah`.

- [ ] **Step 4: Run ER6 (fast)**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er6_run_niah_zero_samples -v
```
Expected: PASS.

- [ ] **Step 5: Run ER5 (slow smoke)**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_eval_niah.py::test_er5_smoke_llama_3_2_1b_ctx512_n2 -v
```
Expected: PASS in <90s.

- [ ] **Step 6: Run ALL eval tests + the broader fast suite for regressions**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -v -m "not slow"
```
Expected: all PASS (no regressions in Phases 1–6 unit coverage).

- [ ] **Step 7: Commit**

```
git add src/flashquest/eval/runner.py src/flashquest/eval/__init__.py tests/test_eval_niah.py
git commit -m "phase 6 task 2: run_niah loop + smoke (ER5, ER6)"
```

---

## Task 8: CLI runner — `phase6_run_ruler_4k.py`

**Files:**
- Create: `scripts/phase6_run_ruler_4k.py`

- [ ] **Step 1: Write the CLI script**

```python
"""Phase 6 task 2: RULER NIAH 4k subset on Llama-3.2-3B-AWQ.

Runs three tasks (niah_single, niah_multikey, niah_multivalue) under two
backends (dense SDPA, patched flashquest with retention=0.25). Computes
patched_hits / dense_hits per task. Gate: ratio >= 0.85 for every task.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from flashquest.eval.runner import run_niah


TASKS = ["single", "multikey", "multivalue"]


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_dense(model_name: str, ctx_len: int, n_samples: int, seed: int, max_new: int) -> dict:
    from flashquest.runtime.awq_load import load_awq_model
    print(f"\n=== dense (vanilla SDPA) ===")
    model, tok = load_awq_model(model_name)
    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        r = run_niah(model, tok, task=task, n_samples=n_samples,
                     ctx_len=ctx_len, seed=seed, max_new_tokens=max_new)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok
    _free()
    return out


def run_patched(
    model_name: str, ctx_len: int, n_samples: int, seed: int, max_new: int,
    retention: float, num_sinks: int, window_pages: int, page_size: int,
) -> dict:
    from flashquest.cache import PersistentInt8KVCache
    from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
    from flashquest.runtime.awq_load import load_awq_model

    print(f"\n=== patched (retention={retention}, all-retrieval head_pattern) ===")
    model, tok = load_awq_model(model_name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=ctx_len + max_new + 128, page_size=page_size, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=retention, num_sinks=num_sinks,
        window_pages=window_pages, page_size=page_size,
    )

    out = {}
    for task in TASKS:
        t0 = time.perf_counter()
        # Reset cache between samples is handled inside run_niah's first model.generate
        # by HF re-initialising past_key_values; we additionally zero our seen counter
        # via a pre-call hook. For simplicity, we just zero after each task.
        cache._seen_tokens = [0] * cache.num_layers
        r = run_niah(model, tok, task=task, n_samples=n_samples,
                     ctx_len=ctx_len, seed=seed, max_new_tokens=max_new)
        wall = time.perf_counter() - t0
        print(f"  {task}: hits={r['hits']}/{r['total']} wall={wall:.0f}s")
        out[f"niah_{task}"] = {"hits": r["hits"], "total": r["total"], "wall_s": wall}
    del model, tok, cache
    _free()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=128)
    args = p.parse_args()

    dense = run_dense(args.model, args.ctx_len, args.n_samples, args.seed, args.max_new_tokens)
    patched = run_patched(
        args.model, args.ctx_len, args.n_samples, args.seed, args.max_new_tokens,
        args.retention, args.num_sinks, args.window_pages, args.page_size,
    )

    tasks_out = {}
    all_pass = True
    for t in TASKS:
        k = f"niah_{t}"
        d, q = dense[k]["hits"], patched[k]["hits"]
        ratio = q / d if d > 0 else 0.0
        passed = ratio >= 0.85
        if not passed:
            all_pass = False
        tasks_out[k] = {
            "dense_hits": d, "patched_hits": q,
            "total": dense[k]["total"], "ratio": ratio, "pass": passed,
            "dense_wall_s": dense[k]["wall_s"],
            "patched_wall_s": patched[k]["wall_s"],
        }

    result = {
        "model": args.model, "ctx_len": args.ctx_len, "n_samples": args.n_samples,
        "retention": args.retention, "num_sinks": args.num_sinks,
        "window_pages": args.window_pages, "page_size": args.page_size,
        "seed": args.seed, "head_pattern": "all-retrieval",
        "tasks": tasks_out,
        "gate": "≥85% vs dense at retention=0.25 + all-retrieval head_pattern",
        "all_pass": all_pass,
    }
    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_ruler_4k.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    print(f"\nAll-pass: {all_pass}")
    for t, info in tasks_out.items():
        print(f"  {t}: {info['patched_hits']}/{info['dense_hits']} = {info['ratio']:.2%} {'PASS' if info['pass'] else 'FAIL'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test the CLI on a small ctx_len + n_samples (no commit yet)**

Run a quick check that the script loads without error:
```
source .venv/bin/activate && python -c "import importlib.util, sys; spec = importlib.util.spec_from_file_location('m', 'scripts/phase6_run_ruler_4k.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```
git add scripts/phase6_run_ruler_4k.py
git commit -m "phase 6 task 2: phase6_run_ruler_4k.py CLI — dense + patched, gate >=85%"
```

---

## Task 9: Full eval run on Llama-3.2-3B-AWQ at ctx=4k, n=20

**Files:** none (run-only)

- [ ] **Step 1: Run the eval (long: ~2.5 hr per config = ~5 hr total)**

Run:
```
source .venv/bin/activate && python -u scripts/phase6_run_ruler_4k.py 2>&1 | tee /tmp/phase6_ruler_4k.log
```
Expected: each task prints `hits=X/20`. Final line: `All-pass: True` or `False`.

If `All-pass: False`: do NOT proceed; the gate failed. Diagnose by:
1. Inspecting `benchmarks/phase6_ruler_4k.json` for which task failed.
2. Trying retention=0.50 (looser top-k) to see if quality is rentention-bound.
3. Examining failing samples' `generated` text to see what the model said vs the expected key.

- [ ] **Step 2: Verify JSON**

Run:
```
cat benchmarks/phase6_ruler_4k.json | python -m json.tool | head -30
```
Expected: `"all_pass": true` and `"ratio"` fields ≥ 0.85.

- [ ] **Step 3: Commit**

```
git add benchmarks/phase6_ruler_4k.json
git commit -m "phase 6 task 2: RULER 4k results — all 3 tasks >=85% vs dense"
```

(If all_pass=False, do NOT commit; raise the issue back to the plan author.)

---

## Task 10: Update notes + DOC + README + SPEC; tag

**Files:**
- Modify: `docs/PHASES/phase-6-notes.md`
- Modify: `DOC.md`
- Modify: `README.md`
- Modify: `docs/SPEC.md`

- [ ] **Step 1: Append task 2 section to `docs/PHASES/phase-6-notes.md`**

Append at end of file:

```markdown
## Phase 6 task 2 — RULER NIAH 4k subset eval

**Started:** 2026-05-02
**Status:** complete (tag `phase-6-task-2`)
**Spec:** [../superpowers/specs/2026-05-02-phase-6-ruler-niah-4k-design.md](../superpowers/specs/2026-05-02-phase-6-ruler-niah-4k-design.md)

Replaces the broken passkey methodology with the SPEC's actual quality
gate. Three NIAH tasks (single, multikey, multivalue) at ctx=4 k on
Llama-3.2-3B-AWQ, retention=0.25, all-retrieval head_pattern, n=20 samples
per task per backend. Acceptance: patched_hits / dense_hits ≥ 0.85.

Decisions: hybrid harness (own generator, RULER prompt template +
scoring vendored verbatim from `vendor/RULER/scripts/data/synthetic/niah.py`);
all-retrieval head_pattern (eliminates the un-seeded torch.rand variance
that broke the passkey eval); dense baseline = vanilla SDPA, no patches.

Result (from `benchmarks/phase6_ruler_4k.json` — fill in measured numbers):

| task | dense | patched | ratio | pass |
|---|---|---|---|---|
| niah_single | (TODO) | (TODO) | (TODO) | (TODO) |
| niah_multikey | (TODO) | (TODO) | (TODO) | (TODO) |
| niah_multivalue | (TODO) | (TODO) | (TODO) | (TODO) |
```

Replace the `(TODO)` cells with actual values from the run.

- [ ] **Step 2: Append task 2 row to `DOC.md` Phases list**

Find the "Phase 6 task 1 — Criticality + top-k fix" line and append after it:

```markdown
- **Phase 6 task 2 — RULER NIAH 4k subset eval** ✅ **complete (tag `phase-6-task-2`)**. `flashquest.eval.{niah, runner}` with three task generators (single, multikey, multivalue) and a substring-match scorer, all following the RULER NIAH protocol verbatim (`vendor/RULER/scripts/data/synthetic/niah.py`). Replaces the broken passkey eval. Run on Llama-3.2-3B-AWQ at ctx=4k, retention=0.25, all-retrieval head_pattern, n=20: all 3 tasks ≥85% vs dense SDPA. See `docs/PHASES/phase-6-notes.md`.
```

- [ ] **Step 3: Append RULER NIAH section to `README.md` (after Phase 6 task 1 section)**

Append:

```markdown
## Phase 6 task 2 — RULER NIAH 4k subset

The Phase 5 / 6 task 1a passkey eval was un-seeded `torch.rand` for the
DuoAttention head_pattern; results were brittle to the random pattern.
RULER NIAH replaces it as the SPEC's quality gate.

Three tasks, each n=20 samples at ctx=4 k on Llama-3.2-3B-AWQ:
`niah_single` (1 needle, retrieve 1 value), `niah_multikey` (4 keys,
retrieve target value), `niah_multivalue` (1 key, 4 values, retrieve all).
Generated via our own harness using the RULER prompt template + scoring
rules verbatim (`src/flashquest/eval/niah.py`); all-retrieval head_pattern
isolates sparse-attention quality from DuoAttention variance.

Acceptance: patched_hits / dense_hits ≥ 0.85 for every task.

| task | dense (SDPA) | patched (Quest INT8) | ratio | pass |
|---|---|---|---|---|
| niah_single | (TODO) | (TODO) | (TODO) | (TODO) |
| niah_multikey | (TODO) | (TODO) | (TODO) | (TODO) |
| niah_multivalue | (TODO) | (TODO) | (TODO) | (TODO) |

Re-run via `python scripts/phase6_run_ruler_4k.py`. Release-grade with
`--n-samples 64`.
```

Fill in the `(TODO)` values from `benchmarks/phase6_ruler_4k.json`.

- [ ] **Step 4: Tick task 2 in `docs/SPEC.md`**

Find the SPEC §6 "Phase 6 — Polish & release" priority list and replace task 2's entry:

```markdown
2. ~~**RULER 4 k subset eval.**~~ **DONE (2026-05-02, tag `phase-6-task-2`)** — `flashquest.eval.niah` (3 tasks: single, multikey, multivalue), `flashquest.eval.run_niah`, `scripts/phase6_run_ruler_4k.py`. RULER protocol verbatim; all-retrieval head_pattern; dense SDPA baseline. n=20 default, n=64 release-grade. Result on Llama-3.2-3B-AWQ at ctx=4 k, retention=0.25: all 3 tasks ≥ 85 % vs dense. See `docs/PHASES/phase-6-notes.md`.
```

- [ ] **Step 5: Commit and tag**

```
git add docs/PHASES/phase-6-notes.md DOC.md README.md docs/SPEC.md
git commit -m "phase 6 task 2: complete — DOC + README + SPEC updated with measured ratios"
git tag phase-6-task-2
git log --oneline phase-6-task-1..HEAD
```

---

## Self-review (post-write)

1. **Spec coverage** — design §Architecture, §Components, §Data flow, §Edge cases, §Validation gates: each maps to a task. niah primitives → Task 2; single/multikey/multivalue → Tasks 3/4/5; score → Task 6; runner + ER5/ER6 → Task 7; CLI + dense + patched + gate → Task 8; full run → Task 9; docs + tag → Task 10.
2. **Placeholder scan** — `(TODO)` markers appear only in Task 10 doc updates where the engineer fills measured numbers from the JSON. Documented.
3. **Type consistency** — `make_prompt` returns `tuple[str, list[str]]` everywhere; `score(text, expected_keys: Iterable[str]) -> bool` everywhere; `run_niah` returns `{"task", "hits", "total", "samples"}`; CLI assembles `tasks` dict mirroring the spec's example JSON. Consistent.

No issues found.
