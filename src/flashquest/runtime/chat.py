"""Phase 6 task 3 — `flashquest chat` console CLI.

Wires src/flashquest/runtime/awq_load.py + cache/persistent_int8.py +
eager/llama_persistent_patch.py into a streaming chat driver. Single-shot
default; --interactive enters a REPL that resets the cache between turns.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Sequence

import torch


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
                   help="Single-shot user prompt.")
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
    p.add_argument(
        "--retention", type=float, default=0.20,
        help="Quest top-k page retention. Default 0.20 (Phase 10). "
        "Use 0.10 for ~24%% faster decode on single-needle retrieval workloads "
        "(multi-needle quality degrades — RULER multivalue 65%% at 0.10 vs 95%% at 0.20).",
    )
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--kv-bits", type=int, choices=[4, 8, 3], default=4,
                   help="KV cache bit width. 4 = KIVI-INT4 (default, RULER 100/100/100). "
                        "3 = TurboQuant K3-V3 (Phase 7). 8 = KIVI-INT8.")
    p.add_argument("--no-patch", action="store_true",
                   help="Skip Quest+INT8 patch; run vanilla SDPA (debugging).")
    p.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    return p.parse_args(argv)


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
        dropped = 1
        if len(rest) >= 2 and rest[1]["role"] == "assistant":
            dropped = 2
        rest = rest[dropped:]
        messages = sys_msgs + rest
        if not rest:
            return messages


def _generate_with_no_grad(model, gen_kwargs: dict) -> None:
    with torch.no_grad():
        model.generate(**gen_kwargs)


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
        target=_generate_with_no_grad, args=(model, gen_kwargs),
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


def _run_repl(model, tokenizer, cache, history: list[dict], args) -> None:
    """REPL: read user line, append to history, truncate, stream, append assistant."""
    backend = "patched" if cache is not None else "sdpa"
    print(
        f"flashquest chat — model={args.model}, ctx={args.context}, backend={backend}.",
        flush=True,
    )
    print("Ctrl-C or empty EOF to exit.\n", flush=True)

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


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)

    from flashquest.runtime.awq_load import load_awq_model

    model, tokenizer = load_awq_model(args.model)

    cache = None
    if not args.no_patch:
        if args.kv_bits == 3:
            from flashquest.cache.persistent_turbo import PersistentTurboKVCache as CacheCls
        elif args.kv_bits == 4:
            from flashquest.cache.persistent_int4 import PersistentInt4KVCache as CacheCls
        else:
            from flashquest.cache.persistent_int8 import PersistentInt8KVCache as CacheCls
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
        cache = CacheCls(
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
