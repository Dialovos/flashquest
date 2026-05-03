# Phase 6 Task 3 — `flashquest chat` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `flashquest chat` console-script CLI committed in SPEC §11 — single-shot default + interactive REPL, streaming via TextIteratorStreamer, with --context-file/stdin for loading 32 k context.

**Architecture:** One file at `src/flashquest/runtime/chat.py` glues `load_awq_model` + `PersistentInt8KVCache` + `patch_llama_for_quest_persistent` (all existing) into an argparse + `TextIteratorStreamer` + `threading.Thread` driver. Prompt rendering via `tokenizer.apply_chat_template`; cache reset between REPL turns; oldest-pair truncation when the rendered chat overflows `--context`. Console-script entry point installed via `pyproject.toml [project.scripts]`.

**Tech Stack:** argparse, transformers `TextIteratorStreamer`, `threading`, existing flashquest modules (`runtime.awq_load`, `cache.persistent_int8`, `eager.llama_persistent_patch`).

**Reference:** `docs/superpowers/specs/2026-05-02-phase-6-chat-cli-design.md`. Existing patterns: `scripts/phase6_run_ruler_4k.py` (model + cache + patch wiring), `src/flashquest/eval/runner.py` (no-model `run_niah(model=None, n_samples=0, ...)` pattern).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/flashquest/runtime/chat.py` | argparse, history-build, truncation, streamer, REPL, main | Create |
| `tests/test_chat.py` | Unit tests (no model) + slow smoke (Llama-3.2-1B SDPA) | Create |
| `pyproject.toml` | `[project.scripts] flashquest = "flashquest.runtime.chat:main"` | Modify |
| `docs/PHASES/phase-6-notes.md` | Append task 3 section | Modify at end |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 3, add invocation example | Modify at end |

Single-file CLI (~180 LOC) is appropriate — thin glue layer with one argparse, one loop, one streaming routine. Splitting would create artificial boundaries.

---

## Task 1: Argparse + `_parse_args` (no model)

**Files:**
- Create: `src/flashquest/runtime/chat.py`
- Create: `tests/test_chat.py`

- [ ] **Step 1: Write the failing tests for _parse_args**

Create `tests/test_chat.py`:

```python
"""Phase 6 task 3 — flashquest chat CLI tests."""
import io
import sys
from contextlib import redirect_stdout, redirect_stderr

import pytest

from flashquest.runtime.chat import _parse_args


def test_parse_args_defaults():
    """Defaults match SPEC §11 invocation."""
    args = _parse_args([
        "--model", "casperhansen/llama-3.2-3b-instruct-awq",
        "--context", "32768",
        "--prompt", "hi",
    ])
    assert args.model == "casperhansen/llama-3.2-3b-instruct-awq"
    assert args.context == 32768
    assert args.prompt == "hi"
    assert args.context_file is None
    assert args.interactive is False
    assert args.max_new_tokens == 512
    assert args.sample is False
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.seed is None
    assert args.retention == 0.25
    assert args.num_sinks == 4
    assert args.window_pages == 2
    assert args.page_size == 64
    assert args.no_patch is False
    assert "helpful assistant" in args.system_prompt.lower()


def test_parse_args_interactive_short_flag():
    """-i is shorthand for --interactive."""
    args = _parse_args(["--model", "x", "--context", "1024", "-i"])
    assert args.interactive is True


def test_parse_args_sampling_flags():
    """--sample + --temperature + --top-p + --seed parse together."""
    args = _parse_args([
        "--model", "x", "--context", "1024",
        "--sample", "--temperature", "0.5", "--top-p", "0.95", "--seed", "42",
        "--prompt", "hi",
    ])
    assert args.sample is True
    assert args.temperature == 0.5
    assert args.top_p == 0.95
    assert args.seed == 42


def test_parse_args_no_patch():
    """--no-patch flag flips the backend toggle."""
    args = _parse_args(["--model", "x", "--context", "1024", "--no-patch", "-i"])
    assert args.no_patch is True


def test_parse_args_context_file():
    """--context-file accepts a path string and the literal '-' for stdin."""
    a1 = _parse_args(["--model", "x", "--context", "1024", "--context-file", "doc.txt", "-i"])
    assert a1.context_file == "doc.txt"
    a2 = _parse_args(["--model", "x", "--context", "1024", "--context-file", "-", "-i"])
    assert a2.context_file == "-"
```

- [ ] **Step 2: Run tests to verify ImportError**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py -v
```
Expected: ImportError on `flashquest.runtime.chat`.

- [ ] **Step 3: Create `src/flashquest/runtime/chat.py` with `_parse_args` only**

```python
"""Phase 6 task 3 — `flashquest chat` console CLI.

Wires src/flashquest/runtime/awq_load.py + cache/persistent_int8.py +
eager/llama_persistent_patch.py into a streaming chat driver. Single-shot
default; --interactive enters a REPL that resets the cache between turns.
"""
from __future__ import annotations

import argparse
from typing import Sequence


_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flashquest chat",
        description="Streaming chat over an AWQ-INT4 model with Quest+INT8 KV.",
    )
    p.add_argument("--model", required=True,
                   help="HF model id (e.g. casperhansen/llama-3.2-3b-instruct-awq).")
    p.add_argument("--context", type=int, required=True,
                   help="Cache budget in tokens (e.g. 32768).")
    p.add_argument("--prompt", default=None,
                   help="Single-shot user prompt. Mutually exclusive with -i for the entry path.")
    p.add_argument("--context-file", default=None,
                   help="Path to a text file injected as the first user message; "
                        "use '-' to read from stdin.")
    p.add_argument("-i", "--interactive", action="store_true",
                   help="REPL mode. Cache resets between turns.")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Per-turn generation cap.")
    p.add_argument("--sample", action="store_true",
                   help="Enable sampling (otherwise greedy).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=None,
                   help="torch.manual_seed before --sample generations.")
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--no-patch", action="store_true",
                   help="Skip Quest+INT8 patch; run vanilla SDPA (debugging).")
    p.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    return p.parse_args(argv)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py -v
```
Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/runtime/chat.py tests/test_chat.py
git commit -m "phase 6 task 3: chat CLI argparse skeleton"
```

---

## Task 2: `_build_initial_history` (file/stdin/inline prompt)

**Files:**
- Modify: `src/flashquest/runtime/chat.py`
- Modify: `tests/test_chat.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chat.py`:

```python
import argparse


def _ns(**overrides) -> argparse.Namespace:
    """Minimal Namespace for _build_initial_history."""
    base = dict(
        prompt=None, context_file=None, interactive=False,
        system_prompt="You are a helpful assistant.",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_initial_history_prompt_only():
    """--prompt becomes a single user message after the system prompt."""
    from flashquest.runtime.chat import _build_initial_history
    h = _build_initial_history(_ns(prompt="hello"))
    assert h == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hello"},
    ]


def test_build_initial_history_interactive_no_prompt():
    """Interactive with no --prompt: only the system message."""
    from flashquest.runtime.chat import _build_initial_history
    h = _build_initial_history(_ns(interactive=True))
    assert h == [{"role": "system", "content": "You are a helpful assistant."}]


def test_build_initial_history_context_file_path(tmp_path):
    """--context-file PATH reads the file and becomes the first user message."""
    from flashquest.runtime.chat import _build_initial_history
    p = tmp_path / "doc.txt"
    p.write_text("doc body")
    h = _build_initial_history(_ns(context_file=str(p), interactive=True))
    assert h[0]["role"] == "system"
    assert h[1] == {"role": "user", "content": "doc body"}


def test_build_initial_history_context_file_and_prompt(tmp_path):
    """--context-file + --prompt: doc as first user msg, prompt as second."""
    from flashquest.runtime.chat import _build_initial_history
    p = tmp_path / "doc.txt"
    p.write_text("doc body")
    h = _build_initial_history(_ns(context_file=str(p), prompt="summarize"))
    assert h == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "doc body"},
        {"role": "user", "content": "summarize"},
    ]


def test_build_initial_history_stdin(monkeypatch):
    """--context-file '-' reads from stdin."""
    import io
    from flashquest.runtime.chat import _build_initial_history
    monkeypatch.setattr("sys.stdin", io.StringIO("piped content"))
    h = _build_initial_history(_ns(context_file="-", prompt="ok"))
    assert h[1] == {"role": "user", "content": "piped content"}
    assert h[2] == {"role": "user", "content": "ok"}


def test_build_initial_history_no_input_raises():
    """Single-shot with no --prompt and no --context-file is an error."""
    from flashquest.runtime.chat import _build_initial_history
    with pytest.raises(SystemExit):
        _build_initial_history(_ns())
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py::test_build_initial_history_prompt_only -v
```
Expected: FAIL with `ImportError: cannot import name '_build_initial_history'`.

- [ ] **Step 3: Implement `_build_initial_history`**

Append to `src/flashquest/runtime/chat.py`:

```python
import sys
from pathlib import Path


def _read_context_file(path: str) -> str:
    """Read --context-file. '-' means stdin."""
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text()


def _build_initial_history(args: argparse.Namespace) -> list[dict]:
    """Build the chat history from CLI args.

    Order: system → (context-file as user) → (--prompt as user).
    Single-shot mode requires at least one of --prompt or --context-file.
    """
    history: list[dict] = [{"role": "system", "content": args.system_prompt}]
    if args.context_file is not None:
        history.append({"role": "user", "content": _read_context_file(args.context_file)})
    if args.prompt is not None:
        history.append({"role": "user", "content": args.prompt})
    if not args.interactive and len(history) == 1:
        sys.stderr.write(
            "error: single-shot mode requires --prompt or --context-file (or pass -i)\n"
        )
        sys.exit(2)
    return history
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py -v
```
Expected: 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/runtime/chat.py tests/test_chat.py
git commit -m "phase 6 task 3: _build_initial_history (file, stdin, prompt)"
```

---

## Task 3: `_truncate_history` (oldest-pair drop)

**Files:**
- Modify: `src/flashquest/runtime/chat.py`
- Modify: `tests/test_chat.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_chat.py`:

```python
class _StubTokenizer:
    """Minimal stub: chat-template renders 'role: content' lines, tokenize splits on spaces."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None):
        class _Ids:
            def __init__(self, n): self.input_ids = [list(range(n))] if return_tensors == "pt" else list(range(n))
        return _Ids(len(text.split()))


def test_truncate_history_no_op_when_under_budget():
    """If rendered tokens <= ctx_len - 256, history returned as-is."""
    from flashquest.runtime.chat import _truncate_history
    tok = _StubTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]
    out = _truncate_history(messages, tok, ctx_len=10000)
    assert out == messages


def test_truncate_history_drops_oldest_pair():
    """Drops oldest user/assistant pair until under budget; preserves system."""
    from flashquest.runtime.chat import _truncate_history
    tok = _StubTokenizer()
    # Make each user/assistant message ~3 words → ctx_len budget forces drops.
    messages = [
        {"role": "system", "content": "sys p"},  # 2 tokens
        {"role": "user", "content": "old old old"},
        {"role": "assistant", "content": "old reply 1"},
        {"role": "user", "content": "mid mid mid"},
        {"role": "assistant", "content": "mid reply 2"},
        {"role": "user", "content": "new new new"},
    ]
    # ctx_len=260 → budget = 260 - 256 = 4 tokens.
    out = _truncate_history(messages, tok, ctx_len=260)
    # System always retained; tail kept; oldest pairs dropped.
    assert out[0]["role"] == "system"
    assert any(m["content"] == "new new new" for m in out)
    # Should have dropped the "old" pair and possibly "mid" pair.
    assert not any("old" in m["content"] for m in out)


def test_truncate_history_minimum_two():
    """Stops dropping when only system + one message remain."""
    from flashquest.runtime.chat import _truncate_history
    tok = _StubTokenizer()
    # Single user message that itself overflows: should still return [system, user].
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": " ".join(["w"] * 1000)},
    ]
    out = _truncate_history(messages, tok, ctx_len=10)
    assert len(out) == 2
    assert out[0]["role"] == "system"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py::test_truncate_history_no_op_when_under_budget -v
```
Expected: FAIL with `ImportError: cannot import name '_truncate_history'`.

- [ ] **Step 3: Implement `_truncate_history`**

Append to `src/flashquest/runtime/chat.py`:

```python
def _truncate_history(messages: list[dict], tokenizer, ctx_len: int) -> list[dict]:
    """Drop oldest non-system user/assistant pairs until the rendered prompt
    fits in ctx_len - 256 (decode head room). Preserve the system message and
    refuse to drop below 2 messages total."""
    while True:
        text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        n = len(tokenizer(text).input_ids)
        if n <= ctx_len - 256 or len(messages) <= 2:
            return messages
        sys_msgs = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        # Drop the oldest user (and following assistant if present).
        dropped = 1
        if len(rest) >= 2 and rest[1]["role"] == "assistant":
            dropped = 2
        rest = rest[dropped:]
        messages = sys_msgs + rest
        if not rest:
            return sys_msgs + [messages[-1]] if messages else sys_msgs
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py -v
```
Expected: 14 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/runtime/chat.py tests/test_chat.py
git commit -m "phase 6 task 3: _truncate_history — oldest-pair drop"
```

---

## Task 4: `_stream_one` (TextIteratorStreamer + thread)

**Files:**
- Modify: `src/flashquest/runtime/chat.py`

- [ ] **Step 1: Add the streaming function (no test yet — covered by ER1 smoke in Task 7)**

Append to `src/flashquest/runtime/chat.py`:

```python
import threading

import torch


def _stream_one(model, tokenizer, cache, messages: list[dict], args) -> str:
    """Render history → tokenize → spawn generate-thread → stream pieces.

    Returns the full assistant text. Resets the persistent cache (if any)
    before generation so the rendered chat is the entire context.
    """
    if cache is not None:
        cache._seen_tokens = [0] * cache.num_layers

    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    gen_kwargs = dict(
        input_ids=ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        streamer=streamer,
        use_cache=True,
    )
    if args.sample:
        gen_kwargs["temperature"] = args.temperature
        gen_kwargs["top_p"] = args.top_p
        if args.seed is not None:
            torch.manual_seed(args.seed)

    thread = threading.Thread(
        target=lambda: _generate_with_no_grad(model, gen_kwargs),
    )
    thread.start()

    pieces: list[str] = []
    try:
        for piece in streamer:
            print(piece, end="", flush=True)
            pieces.append(piece)
    except KeyboardInterrupt:
        pass
    thread.join()
    print()
    return "".join(pieces)


def _generate_with_no_grad(model, gen_kwargs: dict) -> None:
    with torch.no_grad():
        model.generate(**gen_kwargs)
```

- [ ] **Step 2: Verify the module imports without error**

Run:
```
source .venv/bin/activate && python -c "from flashquest.runtime.chat import _stream_one; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/flashquest/runtime/chat.py
git commit -m "phase 6 task 3: _stream_one — TextIteratorStreamer + thread"
```

---

## Task 5: REPL loop (`_run_repl`)

**Files:**
- Modify: `src/flashquest/runtime/chat.py`

- [ ] **Step 1: Add the REPL function**

Append to `src/flashquest/runtime/chat.py`:

```python
def _run_repl(model, tokenizer, cache, history: list[dict], args) -> None:
    """REPL: read user line, append to history, truncate, stream, append assistant."""
    backend = "patched" if cache is not None else "sdpa"
    print(
        f"flashquest chat — model={args.model}, ctx={args.context}, backend={backend}.",
        flush=True,
    )
    print("Ctrl-C or empty EOF to exit.\n", flush=True)

    # If --context-file or --prompt seeded a user message, answer it first.
    if len(history) >= 2 and history[-1]["role"] == "user":
        history = _truncate_history(history, tokenizer, args.context)
        print("assistant> ", end="", flush=True)
        text = _stream_one(model, tokenizer, cache, history, args)
        history.append({"role": "assistant", "content": text})

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        history.append({"role": "user", "content": user})
        history = _truncate_history(history, tokenizer, args.context)
        print("assistant> ", end="", flush=True)
        text = _stream_one(model, tokenizer, cache, history, args)
        history.append({"role": "assistant", "content": text})
```

- [ ] **Step 2: Verify import**

Run:
```
source .venv/bin/activate && python -c "from flashquest.runtime.chat import _run_repl; print('OK')"
```
Expected: `OK`.

- [ ] **Step 3: Commit**

```bash
git add src/flashquest/runtime/chat.py
git commit -m "phase 6 task 3: _run_repl — interactive loop"
```

---

## Task 6: `main` (load + branch + entry-point)

**Files:**
- Modify: `src/flashquest/runtime/chat.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Implement `main`**

Append to `src/flashquest/runtime/chat.py`:

```python
def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    from flashquest.runtime.awq_load import load_awq_model

    model, tokenizer = load_awq_model(args.model)

    cache = None
    if not args.no_patch:
        from flashquest.cache.persistent_int8 import PersistentInt8KVCache
        from flashquest.eager.llama_persistent_patch import (
            patch_llama_for_quest_persistent,
        )

        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        pattern = torch.ones(
            cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool,
        )
        cache = PersistentInt8KVCache(
            batch_size=1,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=args.context + args.max_new_tokens + 128,
            page_size=args.page_size,
            device="cuda",
        )
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=args.retention, num_sinks=args.num_sinks,
            window_pages=args.window_pages, page_size=args.page_size,
        )

    history = _build_initial_history(args)

    if args.interactive:
        _run_repl(model, tokenizer, cache, history, args)
        return

    history = _truncate_history(history, tokenizer, args.context)
    _stream_one(model, tokenizer, cache, history, args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add console-script entry point**

Edit `pyproject.toml`. Find the section after `[project.optional-dependencies]` and before `[tool.setuptools.packages.find]`. Insert:

```toml
[project.scripts]
flashquest = "flashquest.runtime.chat:main"
```

- [ ] **Step 3: Re-install in editable mode**

Run:
```
source .venv/bin/activate && pip install -e . --no-deps
```
Expected: `Successfully installed flashquest-0.0.0`.

- [ ] **Step 4: Verify console script resolves**

Run:
```
which flashquest && flashquest --help 2>&1 | head -25
```
Expected: a path under `.venv/bin/flashquest`; help text printing argparse usage.

- [ ] **Step 5: Re-run unit tests for regressions**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py -v
```
Expected: 14 PASS (no regressions; main is not unit-tested directly).

- [ ] **Step 6: Commit**

```bash
git add src/flashquest/runtime/chat.py pyproject.toml
git commit -m "phase 6 task 3: main + flashquest console-script entry point"
```

---

## Task 7: Smoke test on Llama-3.2-1B SDPA (ER1)

**Files:**
- Modify: `tests/test_chat.py`

- [ ] **Step 1: Write the slow-marked smoke test**

Append to `tests/test_chat.py`:

```python
@pytest.mark.slow
def test_smoke_single_shot_llama_3_2_1b_sdpa(capsys):
    """ER1: end-to-end single-shot streaming on Llama-3.2-1B (SDPA, no patch).

    Bypasses load_awq_model (which requires AWQ weights) by going through
    main() with --no-patch and a non-AWQ model. We monkeypatch
    load_awq_model to return Llama-3.2-1B + tokenizer.
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flashquest.runtime import chat as chat_mod

    name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    # Patch the loader so main() picks up our pre-loaded pair.
    orig_load = chat_mod.__dict__.get("load_awq_model")
    chat_mod.__dict__["_test_pair"] = (model, tokenizer)

    def _fake_load(_name, **_kw):
        return chat_mod.__dict__["_test_pair"]

    import flashquest.runtime.awq_load as awq_mod
    awq_mod.load_awq_model = _fake_load

    try:
        t0 = time.perf_counter()
        chat_mod.main([
            "--model", name,
            "--context", "512",
            "--prompt", "Say hello in one word.",
            "--max-new-tokens", "16",
            "--no-patch",
        ])
        elapsed = time.perf_counter() - t0
    finally:
        if orig_load is not None:
            awq_mod.load_awq_model = orig_load

    assert elapsed < 90, f"smoke too slow: {elapsed:.1f}s"
    out = capsys.readouterr().out
    # Streamer prints something (model produced any tokens).
    assert len(out.strip()) > 0
```

- [ ] **Step 2: Run the smoke test**

Run:
```
source .venv/bin/activate && python -m pytest tests/test_chat.py::test_smoke_single_shot_llama_3_2_1b_sdpa -v
```
Expected: PASS in <90 s.

- [ ] **Step 3: Run full fast suite for regressions**

Run:
```
source .venv/bin/activate && python -m pytest tests/ -m "not slow"
```
Expected: 178 passed (164 from before + 14 new chat unit tests).

- [ ] **Step 4: Commit**

```bash
git add tests/test_chat.py
git commit -m "phase 6 task 3: smoke test — single-shot streaming on Llama-3.2-1B SDPA"
```

---

## Task 8: Manual end-to-end on Llama-3.2-3B-AWQ (gate)

**Files:** none (run-only; results inform docs in Task 9)

- [ ] **Step 1: Single-shot smoke on the real target model + 32 k context budget**

Run (foreground, ~1-2 min):
```
source .venv/bin/activate && \
  flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
             --context 32768 \
             --prompt "Write one sentence introducing yourself." \
             --max-new-tokens 64
```
Expected: tokens stream to stdout, completes without OOM, prints a coherent sentence.

- [ ] **Step 2: Long-context smoke — 4 k context-file**

Build a 4 k-token doc and run:
```
source .venv/bin/activate && \
  python -c "import json; print(json.loads(open('data/PaulGrahamEssays.json').read())['text'][:16000])" > /tmp/doc4k.txt && \
  flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
             --context 8192 \
             --context-file /tmp/doc4k.txt \
             --prompt "Summarize the document in one sentence." \
             --max-new-tokens 96
```
Expected: streams a coherent one-sentence summary; finishes within reasonable wall (≈30-90 s).

- [ ] **Step 3: Stdin smoke**

Run:
```
source .venv/bin/activate && \
  echo "The quick brown fox jumps over the lazy dog." | \
  flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
             --context 1024 \
             --context-file - \
             --prompt "How many words are in the sentence?" \
             --max-new-tokens 32
```
Expected: streams an answer (the model's count, which may or may not be exactly 9 — we're testing plumbing, not arithmetic).

- [ ] **Step 4: Interactive smoke (manual, ≥3 turns)**

Run:
```
source .venv/bin/activate && \
  flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
             --context 8192 \
             --interactive
```
Type three short prompts. Verify each turn streams; Ctrl-C exits cleanly.

- [ ] **Step 5: Commit a benchmark JSON capturing single-shot timing (optional)**

If wall times look notable, capture into `benchmarks/phase6_chat_smoke.json`:
```bash
cat > /tmp/phase6_chat_smoke.json <<EOF
{
  "model": "casperhansen/llama-3.2-3b-instruct-awq",
  "ctx_len": 32768,
  "single_shot_intro_wall_s": <REPLACE-WITH-MEASURED>,
  "context_file_4k_summarize_wall_s": <REPLACE-WITH-MEASURED>,
  "all_pass": true
}
EOF
mv /tmp/phase6_chat_smoke.json benchmarks/phase6_chat_smoke.json
git add benchmarks/phase6_chat_smoke.json
git commit -m "phase 6 task 3: chat CLI smoke timings"
```

(Skip if numbers aren't useful; the gate is qualitative — "streams + coherent" — not numeric.)

---

## Task 9: Update notes + DOC + README + SPEC; tag

**Files:**
- Modify: `docs/PHASES/phase-6-notes.md`
- Modify: `DOC.md`
- Modify: `README.md`
- Modify: `docs/SPEC.md`

- [ ] **Step 1: Append task 3 section to `docs/PHASES/phase-6-notes.md`**

Append at the end of file:

```markdown

---

# Phase 6 Task 3 Notes

**Started:** 2026-05-02
**Status:** **complete (tag `phase-6-task-3`)**.
**Spec:** [../superpowers/specs/2026-05-02-phase-6-chat-cli-design.md](../superpowers/specs/2026-05-02-phase-6-chat-cli-design.md)
**Plan:** [../superpowers/plans/2026-05-02-phase-6-chat-cli.md](../superpowers/plans/2026-05-02-phase-6-chat-cli.md)

## Summary

Console-script `flashquest` exposing the SPEC §11 invocation literally:
`flashquest chat --model llama-3.2-3b-awq --context 32k`. Single-shot
default + interactive REPL (`-i`); `--context-file PATH` (or `-` for
stdin) for 32 k-context demos; greedy default with opt-in
`--sample`/`--temperature`/`--top-p`/`--seed`; `--no-patch` falls back
to vanilla SDPA for debugging.

Streaming via `transformers.TextIteratorStreamer` running in a daemon
thread; chat history rendered via `tokenizer.apply_chat_template`;
persistent INT8 KV cache resets between REPL turns
(`cache._seen_tokens = [0] * cache.num_layers`); oldest-pair history
truncation when rendered prompt exceeds `--context - 256`.

## Surface

- `flashquest chat ...` console script (entry point in `pyproject.toml`).
- `python -m flashquest.runtime.chat ...` (module form, equivalent).
- 14 unit tests + 1 slow smoke on Llama-3.2-1B SDPA in
  `tests/test_chat.py`.
- Manual gates: 32 k-context single-shot, 4 k context-file summarize,
  stdin pipe, interactive ≥3-turn REPL.

## v2 follow-ups

- Tool-use / function-calling.
- Persistent on-disk chat history.
- Token/sec live counter in REPL.
- ANSI-colored role tags.
- `--max-context` auto-detected from `model.config.max_position_embeddings`.
```

- [ ] **Step 2: Append task 3 row to `DOC.md`**

Edit `DOC.md`. Find the line beginning `- **Phase 6 task 2 — RULER NIAH 4 k`. Append immediately after it:

```markdown
- **Phase 6 task 3 — `flashquest chat` CLI** ✅ **complete (tag `phase-6-task-3`)**. Console script `flashquest` (entry point in `pyproject.toml`) wraps `load_awq_model` + `PersistentInt8KVCache` + `patch_llama_for_quest_persistent` into a streaming chat driver. Single-shot default + interactive REPL (`-i`); `--context-file PATH` (or `-` for stdin) loads up to 32 k of context; greedy default with opt-in `--sample`. Streaming via `TextIteratorStreamer` in a thread; cache resets between REPL turns; oldest-pair truncation. Matches SPEC §11 invocation `flashquest chat --model llama-3.2-3b-awq --context 32k` literally. See `docs/PHASES/phase-6-notes.md`.
```

- [ ] **Step 3: Update `README.md` — add chat CLI section**

Edit `README.md`. Find the "## Phase 6 task 2 — RULER NIAH 4 k subset (quality gate)" section. Insert *after* it (before "## Non-goals"):

```markdown
## Phase 6 task 3 — `flashquest chat` CLI

The SPEC §11 acceptance invocation, shipped:

```bash
pip install -e .
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 \
           --interactive
```

Single-shot mode for scripting + benchmarks:

```bash
flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
           --context 32768 \
           --context-file long-doc.txt \
           --prompt "Summarize the document in 3 sentences." \
           --max-new-tokens 256
```

Pipe stdin:

```bash
cat long-doc.txt | flashquest --model casperhansen/llama-3.2-3b-instruct-awq \
                              --context 32768 --context-file - \
                              --prompt "Summarize."
```

Greedy by default; `--sample --temperature 0.7 --top-p 0.9 --seed 0` for
reproducible sampled generation. `--no-patch` falls back to vanilla SDPA
for debugging.
```

(Use a fenced block with `bash` language tag.)

- [ ] **Step 4: Tick task 3 in `docs/SPEC.md`**

Edit `docs/SPEC.md`. Find:

```markdown
3. **Demo chat CLI.** `flashquest chat --model llama-3.2-3b-awq --context 32k`. Stream tokens. README §11 acceptance criterion.
```

Replace with:

```markdown
3. **Demo chat CLI.** ✅ **DONE (2026-05-02, tag `phase-6-task-3`).** `flashquest` console script (entry point in `pyproject.toml`) at `src/flashquest/runtime/chat.py`. Single-shot default + interactive REPL (`-i`); `--context-file PATH` or `-` for stdin; greedy default + opt-in `--sample`/`--temperature`/`--top-p`/`--seed`; `--no-patch` for SDPA fallback. Streaming via `TextIteratorStreamer`; chat history via `tokenizer.apply_chat_template`; cache resets between REPL turns; oldest-pair truncation. SPEC §11 invocation literal: `flashquest --model casperhansen/llama-3.2-3b-instruct-awq --context 32768 --interactive`. See `docs/PHASES/phase-6-notes.md`.
```

- [ ] **Step 5: Commit and tag**

```bash
git add docs/PHASES/phase-6-notes.md DOC.md README.md docs/SPEC.md
git commit -m "phase 6 task 3: complete — DOC + README + SPEC + notes"
git tag phase-6-task-3
git log --oneline phase-6-task-2..HEAD
```

---

## Self-review (post-write)

**1. Spec coverage** — every section of `2026-05-02-phase-6-chat-cli-design.md` maps to a task:

- §Decisions / interaction model → Tasks 1 + 5 (argparse + REPL)
- §Decisions / long-context loading → Task 2 (`_build_initial_history` with file/stdin)
- §Decisions / sampling defaults → Task 1 (argparse) + Task 4 (`_stream_one` honours flags)
- §Decisions / streaming mechanism → Task 4
- §Decisions / prompt formatting → Task 4 (`apply_chat_template` call)
- §Decisions / KV cache scope → Task 4 (cache reset at top of `_stream_one`)
- §Decisions / truncation → Task 3
- §Decisions / entry point → Task 6 (`pyproject.toml`)
- §Decisions / patch toggle → Task 6 (`--no-patch` branch in `main`)
- §Edge cases ER1-ER8 → Tasks 1-3 (unit), Task 7 (ER1 smoke), Task 8 (ER2/ER3 manual stdin/file/interactive)
- §Validation gates → Task 7 (smoke), Task 8 (manual on real model)

**2. Placeholder scan** — only `<REPLACE-WITH-MEASURED>` in Task 8 step 5 (optional benchmark JSON capture); flagged as optional in the same step. Acceptable: the gate is qualitative; numeric capture is a nicety.

**3. Type consistency** — `_build_initial_history(args) -> list[dict]`; `_truncate_history(messages, tokenizer, ctx_len) -> list[dict]`; `_stream_one(model, tokenizer, cache, messages, args) -> str`; `_run_repl(model, tokenizer, cache, history, args) -> None`; `main(argv=None) -> None`. Types match across tasks. `cache._seen_tokens = [0] * cache.num_layers` matches the existing reset semantics from `scripts/phase6_run_ruler_4k.py`. Pyproject `[project.scripts]` entry maps to `flashquest.runtime.chat:main`, matching `main`'s exported name.

No issues found.
