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
    assert args.retention == 0.20
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


def test_parse_args_kv_bits_default_4():
    """INT4 promoted to default after RULER 4k @ INT4 gate cleared 100/100/100."""
    args = _parse_args(["--model", "x", "--context", "1024", "-i"])
    assert args.kv_bits == 4


def test_parse_args_kv_bits_8_explicit():
    """INT8 fallback still selectable via --kv-bits 8."""
    args = _parse_args(["--model", "x", "--context", "1024", "-i", "--kv-bits", "8"])
    assert args.kv_bits == 8


def test_parse_args_kv_bits_rejects_other_values():
    with pytest.raises(SystemExit):
        _parse_args(["--model", "x", "--context", "1024", "-i", "--kv-bits", "16"])


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


class _StubTokenizer:
    """Minimal stub: chat-template renders 'role: content' lines, tokenize splits on spaces."""

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def __call__(self, text, return_tensors=None):
        class _Ids:
            def __init__(self, n):
                self.input_ids = [list(range(n))] if return_tensors == "pt" else list(range(n))
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
    messages = [
        {"role": "system", "content": "sys p"},
        {"role": "user", "content": "old old old"},
        {"role": "assistant", "content": "old reply 1"},
        {"role": "user", "content": "mid mid mid"},
        {"role": "assistant", "content": "mid reply 2"},
        {"role": "user", "content": "new new new"},
    ]
    # ctx_len=260 → budget = 260 - 256 = 4 tokens.
    out = _truncate_history(messages, tok, ctx_len=260)
    assert out[0]["role"] == "system"
    assert any(m["content"] == "new new new" for m in out)
    assert not any("old" in m["content"] for m in out)


def test_truncate_history_minimum_two():
    """Stops dropping when only system + one message remain."""
    from flashquest.runtime.chat import _truncate_history
    tok = _StubTokenizer()
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": " ".join(["w"] * 1000)},
    ]
    out = _truncate_history(messages, tok, ctx_len=10)
    assert len(out) == 2
    assert out[0]["role"] == "system"


@pytest.mark.slow
def test_smoke_single_shot_llama_3_2_1b_sdpa(capsys, monkeypatch):
    """ER1: end-to-end single-shot streaming on Llama-3.2-1B (SDPA, no patch).

    Bypasses load_awq_model (which requires AWQ weights) by monkeypatching
    flashquest.runtime.awq_load.load_awq_model to return the pre-loaded pair.
    """
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flashquest.runtime import chat as chat_mod
    import flashquest.runtime.awq_load as awq_mod

    name = "unsloth/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    monkeypatch.setattr(awq_mod, "load_awq_model", lambda _name, **_kw: (model, tokenizer))

    t0 = time.perf_counter()
    chat_mod.main([
        "--model", name,
        "--context", "512",
        "--prompt", "Say hello in one word.",
        "--max-new-tokens", "16",
        "--no-patch",
    ])
    elapsed = time.perf_counter() - t0

    assert elapsed < 90, f"smoke too slow: {elapsed:.1f}s"
    out = capsys.readouterr().out
    assert len(out.strip()) > 0
