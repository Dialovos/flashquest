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
        "--retention", type=float, default=None,
        help="Quest top-k page retention. Default 0.20 for INT4/INT8; 0.25 for "
        "--kv-bits 3 --codebook calibrated (Phase 11c: 0.25 clears the RULER "
        "multivalue gate at 100/100/95, vs 85%% at 0.20). Explicit value overrides. "
        "Use 0.10 for ~24%% faster decode on single-needle workloads "
        "(multi-needle quality degrades — RULER multivalue 65%% at 0.10).",
    )
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument(
        "--codebook", choices=("calibrated", "paper"), default="calibrated",
        help="TurboQuant codebook for --kv-bits 3. 'calibrated' = per-layer fit "
             "to the model's activations (Phase 11); 'paper' = data-oblivious "
             "Lloyd-Max (Phase 7). Falls back to paper with a warning if no "
             "calibration artifact ships for the model. Default 'calibrated'.",
    )
    p.add_argument("--kv-bits", type=int, choices=[4, 8, 3], default=4,
                   help="KV cache bit width. 4 = KIVI-INT4 (default, RULER 100/100/100). "
                        "3 = TurboQuant K3-V3 (Phase 7). 8 = KIVI-INT8.")
    p.add_argument("--no-patch", action="store_true",
                   help="Skip Quest+INT8 patch; run vanilla SDPA (debugging).")
    p.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    p.add_argument(
        "--speculative", action="store_true",
        help="Enable EAGLE-3 chain speculative decoding (Phase 12). Requires the "
             "patched path (not --no-patch) and --kv-bits 4 (the verify arm + "
             "dispatcher are INT4-only). Greedy + lossless-by-contract.",
    )
    p.add_argument("--n-draft", type=int, default=4,
                   help="Speculative chain depth (verify S_q). Default 4.")
    p.add_argument("--draft-model", default="thoughtworks/Llama-3.2-3B-Instruct-Eagle3",
                   help="EAGLE-3 draft head checkpoint (used with --speculative).")
    return p.parse_args(argv)


def _resolve_retention(args) -> float:
    """Resolve effective retention from --retention + mode.

    Phase 11c: calibrated K3-V3 (`--kv-bits 3 --codebook calibrated`) needs
    retention 0.25 to clear the RULER NIAH gate (single/multikey/multivalue =
    100/100/95 at 0.25, vs multivalue 85% at 0.20). INT4/INT8 stay at 0.20
    (already 100/100/95 there). An explicit --retention always wins.
    """
    if args.retention is not None:
        return args.retention
    if args.kv_bits == 3 and args.codebook == "calibrated":
        return 0.25
    return 0.20


def _validate_speculative(args) -> None:
    """Fail fast on incompatible --speculative combinations.

    The spec verify arm + dispatcher are INT4-only and live on the patched path,
    so --speculative needs the patched build (not --no-patch) and --kv-bits 4.
    """
    if not args.speculative:
        return
    if args.no_patch:
        sys.stderr.write(
            "error: --speculative requires the patched path; "
            "it is incompatible with --no-patch\n"
        )
        sys.exit(2)
    if args.kv_bits != 4:
        sys.stderr.write(
            f"error: --speculative requires --kv-bits 4 (got {args.kv_bits}); "
            "the verify arm + spec-decode dispatcher are INT4-only\n"
        )
        sys.exit(2)


def _stop_token_ids(model, tokenizer) -> set[int]:
    """Stop-token set for the spec loop: every EOS / end-of-turn id the normal
    model.generate path would halt on.

    Collects ``tokenizer.eos_token_id`` and ``model.generation_config
    .eos_token_id`` (either may be an int or a list — Llama-3 ships
    ``[<|end_of_text|>, <|eom_id|>, <|eot_id|>]``) plus a defensive lookup of
    ``<|eot_id|>``. The non-spec path delegates this to generate(); here we
    replicate it so an emitted EOS truncates the stream identically.
    """
    ids: set[int] = set()

    def _add(v):
        if v is None:
            return
        if isinstance(v, (list, tuple, set)):
            for x in v:
                _add(x)
        else:
            ids.add(int(v))

    _add(getattr(tokenizer, "eos_token_id", None))
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        _add(getattr(gen_cfg, "eos_token_id", None))
    try:
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != getattr(tokenizer, "unk_token_id", None):
            _add(eot)
    except Exception:
        pass
    return ids


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


def _stream_one_spec(model, tokenizer, cache, draft, messages: list[dict], args) -> str:
    """Spec-decode counterpart of ``_stream_one``: render history → tokenize →
    run the EAGLE-3 chain dispatcher, streaming committed tokens to stdout.

    Mirrors ``_stream_one``'s UX (decoded text per emission, flushed) and
    contract (resets the persistent cache first; returns the full assistant
    text). Stops when total emitted >= --max-new-tokens OR an emitted id is a
    stop token (EOS / end-of-turn), truncating the final step at the stop id.
    """
    from flashquest.specdec.dispatcher import make_quest_specdec

    if cache is not None:
        cache._seen_tokens = [0] * cache.num_layers

    text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    prompt_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)

    stop_ids = _stop_token_ids(model, tokenizer)
    max_new = args.max_new_tokens

    init, step = make_quest_specdec(
        model, cache, draft, n_draft=args.n_draft, page_size=args.page_size,
    )
    init(prompt_ids)

    # Incremental detokenization: keep every generated id, decode the running
    # list, and print only the newly-revealed suffix. Decoding the cumulative
    # sequence (rather than per-token) lets the tokenizer stitch multi-token
    # UTF-8 / BPE pieces — the same robustness TextIteratorStreamer provides.
    gen_ids: list[int] = []
    printed = ""
    try:
        while len(gen_ids) < max_new:
            emitted = [int(x) for x in step().tolist()]

            stopped = False
            for tid in emitted:
                if tid in stop_ids:
                    stopped = True
                    break
                gen_ids.append(tid)
                if len(gen_ids) >= max_new:
                    break

            new_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Guard against printing a half-formed multi-byte char (decode ends
            # in the U+FFFD replacement marker until the next id completes it).
            if new_text.endswith("�"):
                continue
            if len(new_text) > len(printed):
                piece = new_text[len(printed):]
                print(piece, end="", flush=True)
                printed = new_text

            if stopped:
                break
    except KeyboardInterrupt:
        pass

    # Flush any tail held back by the U+FFFD guard.
    final_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    if len(final_text) > len(printed):
        print(final_text[len(printed):], end="", flush=True)
        printed = final_text
    print()
    return printed


def _emit(model, tokenizer, cache, draft, messages: list[dict], args) -> str:
    """Stream one assistant turn via the spec-decode loop (--speculative) or the
    standard generate-thread streamer. Both reset the cache and return the text."""
    if args.speculative:
        return _stream_one_spec(model, tokenizer, cache, draft, messages, args)
    return _stream_one(model, tokenizer, cache, messages, args)


def _run_repl(model, tokenizer, cache, history: list[dict], args, draft=None) -> None:
    """REPL: read user line, append to history, truncate, stream, append assistant."""
    backend = "patched" if cache is not None else "sdpa"
    if args.speculative:
        backend += "+spec"
    print(
        f"flashquest chat — model={args.model}, ctx={args.context}, backend={backend}.",
        flush=True,
    )
    print("Ctrl-C or empty EOF to exit.\n", flush=True)

    if len(history) >= 2 and history[-1]["role"] == "user":
        history = _truncate_history(history, tokenizer, args.context)
        print("assistant> ", end="", flush=True)
        text = _emit(model, tokenizer, cache, draft, history, args)
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
        text = _emit(model, tokenizer, cache, draft, history, args)
        history.append({"role": "assistant", "content": text})


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    _validate_speculative(args)
    args.retention = _resolve_retention(args)

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
        cache_kwargs = dict(
            batch_size=1,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            max_seq_len=args.context + args.max_new_tokens + 128,
            page_size=args.page_size,
            device="cuda",
        )
        if args.kv_bits == 3 and args.codebook == "calibrated":
            cache_kwargs["model_id"] = args.model
        cache = CacheCls(**cache_kwargs)
        patch_llama_for_quest_persistent(
            model, cache=cache, head_pattern=pattern,
            retention=args.retention, num_sinks=args.num_sinks,
            window_pages=args.window_pages, page_size=args.page_size,
        )

    # Build the EAGLE-3 draft head once (reused across every turn). The cache is
    # guaranteed present here: _validate_speculative rejects --speculative with
    # --no-patch, and --kv-bits 4 builds PersistentInt4KVCache above.
    draft = None
    if args.speculative:
        from flashquest.specdec import load_eagle3_draft

        draft = load_eagle3_draft(
            args.draft_model, device="cuda", dtype=torch.bfloat16,
            embed_weight=model.model.embed_tokens.weight,
        )

    history = _build_initial_history(args)

    if args.interactive:
        _run_repl(model, tokenizer, cache, history, args, draft=draft)
        return

    history = _truncate_history(history, tokenizer, args.context)
    _emit(model, tokenizer, cache, draft, history, args)


if __name__ == "__main__":
    main()
