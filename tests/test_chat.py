"""Phase 6 task 3 — flashquest chat CLI tests."""
import argparse
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
