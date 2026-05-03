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
