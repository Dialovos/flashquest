# Phase 6 task 3 — `flashquest chat` CLI

**Date:** 2026-05-02
**Status:** Design approved; ready for implementation plan
**SPEC reference:** `docs/SPEC.md` §6 task 3 ("Demo chat CLI. `flashquest chat --model llama-3.2-3b-awq --context 32k`. Stream tokens. README §11 acceptance criterion.") and §11 acceptance bullet 2.

## Problem

The v1.0 acceptance criterion (`docs/SPEC.md` §11) is:

> 1. `pip install -e .`
> 2. `flashquest chat --model llama-3.2-3b-awq --context 32k`
> 3. Tokens stream at ~10 tok/s with 32 k of context loaded, fitting in 4 GB VRAM.

We have everything below the CLI: AWQ loader (`flashquest.runtime.load_awq_model`), persistent INT8 KV cache (`flashquest.cache.PersistentInt8KVCache`), patched eager attention (`flashquest.eager.llama_persistent_patch.patch_llama_for_quest_persistent`), all-retrieval head_pattern, validated quality (RULER NIAH ≥85 % vs dense, Phase 6 task 2), and 5.14 tok/s decode at 32 k (Phase 6 task 1; v1.0 target 10 tok/s gated on Phase 6 task 5+ INT4 KV / kernel-fused criticality).

Missing piece: a single-binary CLI that wires those into a streaming chat loop. Phase 6 task 4 (head-to-head benchmark vs llama.cpp / vLLM) and the eventual v1.0 release tag depend on this CLI existing and working reproducibly.

## Decisions

| Question | Decision |
|---|---|
| Interaction model | **Both — single-shot default; `-i`/`--interactive` enters REPL** (option C). Single-shot is what the docs/benchmarks demonstrate; REPL is one extra `while True` loop on top of the same per-turn code path. |
| Long-context loading | **`--context-file PATH` (or `-` for stdin)** (option C). `--context-file` is the demo-friendly path (reproducible, scriptable); stdin is one branch of the same file-reading code. |
| Sampling defaults | **Greedy default + `--sample`/`--temperature`/`--top-p`/`--seed` opt-in flags** (option C). Greedy keeps benchmark/demo recordings reproducible and matches Phase 6 eval wiring; sampling flags are ~15 LOC of argparse. |
| Streaming mechanism | `transformers.TextIteratorStreamer` + a `threading.Thread` wrapping `model.generate(...)`. Mainline iterates the streamer and `print(piece, end="", flush=True)`. Stock pattern, ~15 LOC. |
| Prompt formatting | `tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)`. Llama-3.2-Instruct tokenizers ship the chat template; no manual `<|begin_of_text|>` / role-tag handling. |
| KV cache scope | Patched mode binds a single `PersistentInt8KVCache` to the model. In REPL we **reset the cache between turns** (re-render the full chat history on each turn). Cleaner than incremental state because (a) the patched prefill is dense and reasonably fast at ~32 k, (b) it sidesteps cache-state bugs across history edits, (c) it matches the run_niah `pre_sample` semantics already validated in Phase 6 task 2. |
| Truncation | If the rendered chat exceeds `--context`, drop the *oldest assistant/user pairs* until it fits (keep system + most recent turn). Tokenizer's `truncation=True, max_length=ctx` would silently chop mid-message; explicit pair-dropping is more honest. |
| Entry point | `pyproject.toml [project.scripts] flashquest = "flashquest.runtime.chat:main"`. Installed as a console script via `pip install -e .`. Matches SPEC §11 invocation exactly. |
| Patch / dense toggle | `--no-patch` flag falls back to vanilla SDPA (same wiring as `phase6_run_ruler_4k.py`'s `run_dense`). Useful for "is the patch wrong, or is the model dumb?" debugging. |

## Architecture

```
src/flashquest/runtime/
├── chat.py          # NEW — `flashquest chat` entry point + main loop
└── awq_load.py      # existing — load_awq_model (reused unchanged)

tests/
└── test_chat.py     # NEW — argparse, prompt format, streaming smoke

pyproject.toml       # MODIFY — add [project.scripts] flashquest = "flashquest.runtime.chat:main"
```

No other files change. The CLI uses public APIs already exported.

## Components

### `chat.py`

Single module, four functions + `main`:

```python
def _parse_args() -> argparse.Namespace: ...
def _build_initial_history(args) -> list[dict]: ...
def _truncate_history(messages, tokenizer, ctx_len: int) -> list[dict]: ...
def _stream_one(model, tokenizer, cache, messages, args) -> str:
    """Render → tokenize → spawn generate-thread → iterate streamer →
    print + collect → return assistant text."""
def main() -> None:
    """Parse args, load model, optionally patch, build initial history,
    branch on args.interactive."""
```

Total ~180 LOC including comments. Single file is appropriate; this is a thin glue layer over libraries we already own.

### Argparse signature

```bash
flashquest chat \
  --model casperhansen/llama-3.2-3b-instruct-awq \
  --context 32768 \
  [--prompt TEXT | --context-file PATH | -]    # one of these (or interactive)
  [-i | --interactive]
  [--max-new-tokens 512]
  [--sample] [--temperature 0.7] [--top-p 0.9] [--seed 0]
  [--retention 0.25] [--num-sinks 4] [--window-pages 2] [--page-size 64]
  [--no-patch]
  [--system-prompt "You are a helpful assistant."]
```

`--context` is the cache budget; `--max-new-tokens` is per-turn generation.

### Argparse-routing logic

```
if args.context_file:
    ctx_text = read_file_or_stdin(args.context_file)
    history = [{role: system, content: args.system_prompt},
               {role: user, content: ctx_text}]
elif args.prompt:
    history = [{role: system, ...}]   # ctx empty; prompt becomes first user msg
else:
    history = [{role: system, ...}]   # interactive must be true; REPL drives input
```

### Streaming loop (per turn)

```python
def _stream_one(model, tok, cache, messages, args) -> str:
    if cache is not None:
        cache._seen_tokens = [0] * cache.num_layers
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=ids, max_new_tokens=args.max_new_tokens,
        do_sample=args.sample, streamer=streamer, use_cache=True,
    )
    if args.sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p)
        if args.seed is not None:
            torch.manual_seed(args.seed)
    thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    pieces: list[str] = []
    try:
        for piece in streamer:
            print(piece, end="", flush=True)
            pieces.append(piece)
    except KeyboardInterrupt:
        # Stop the generate thread cooperatively. Streamer raises StopIteration
        # when the thread sets _is_finished; we can also signal via thread.join.
        # Simplest: let the thread finish naturally (max_new_tokens caps it).
        # User sees partial output; we still join.
        pass
    thread.join()
    print()
    return "".join(pieces)
```

### REPL

```python
def _run_repl(model, tok, cache, history, args):
    print(f"flashquest chat — model={args.model}, ctx={args.context}, "
          f"backend={'patched' if cache else 'sdpa'}. Ctrl-C exits.")
    if len(history) >= 2 and history[-1]["role"] == "user":
        # context-file or initial prompt provided — answer it first
        text = _stream_one(model, tok, cache, history, args)
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
        history = _truncate_history(history, tok, args.context)
        print("assistant> ", end="", flush=True)
        text = _stream_one(model, tok, cache, history, args)
        history.append({"role": "assistant", "content": text})
```

### Truncation

```python
def _truncate_history(messages, tokenizer, ctx_len: int) -> list[dict]:
    # Keep system message + most-recent user. Drop oldest user/assistant pairs
    # until rendered prompt fits in ctx_len - 256 (decode head room).
    while True:
        text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        n = len(tokenizer(text).input_ids)
        if n <= ctx_len - 256 or len(messages) <= 2:
            return messages
        # Drop the oldest non-system message pair.
        sys_msgs = [m for m in messages if m["role"] == "system"]
        rest = [m for m in messages if m["role"] != "system"]
        rest = rest[2:]  # drop oldest user + assistant
        messages = sys_msgs + rest
```

## Data flow

```
argv → _parse_args
       │
       ├─ args.no_patch=True ──→ load_awq_model(sdpa)             ──┐
       │                                                              │
       └─ args.no_patch=False ─→ load_awq_model + PersistentInt8KVCache + patch_llama_for_quest_persistent
                                                                      │
                                                                      ▼
       _build_initial_history(args)                                   │
                                                                      ▼
       args.interactive ─yes─→ _run_repl(model, tok, cache, history, args)
                       └─no──→ _stream_one(model, tok, cache, history, args) → print → exit
```

## Edge cases / tests

| ID | Case | Test |
|---|---|---|
| ER1 | `--prompt "..."` only — single-shot, prints to stdout, exits 0 | `test_chat.py` smoke (slow-marked, Llama-3.2-1B SDPA, ctx=512) |
| ER2 | `echo "..." \| flashquest chat --context-file -` — stdin path works | `test_chat.py` |
| ER3 | `--context-file path.txt --prompt "summarize"` — file injects as user msg | unit test for `_build_initial_history` |
| ER4 | History exceeds `--context`: oldest pairs dropped, system retained | unit test for `_truncate_history` (no model load) |
| ER5 | `--no-patch` skips cache + patch wiring | unit test inspects `_main_no_patch_branch` (or smoke that `cache is None`) |
| ER6 | `--sample --seed 0` deterministic vs `--sample --seed 1` differ | smoke test on tiny model, n=2 |
| ER7 | `Ctrl-C` during streaming exits cleanly (thread joined) | manual test (documented; not pytest) |
| ER8 | `flashquest --help` prints usage and exits 0 | argparse smoke |

## Validation gates

| Gate | Target |
|---|---|
| Unit tests for `_truncate_history`, `_build_initial_history` | green (no model load) |
| Smoke test for single-shot streaming on Llama-3.2-1B at ctx=512 (slow-marked) | <60 s, prints something, exits 0 |
| Manual: `flashquest chat --model llama-3.2-3b-awq --context 32768 --context-file <32k-doc.txt> --prompt "summarize"` | streams tokens, completes within reasonable wall, output coherent |
| Manual: `flashquest chat -i --model llama-3.2-3b-awq --context 32768` REPL works (≥3 turns) | streams; cache resets between turns; coherent |
| `pip install -e .` exposes `flashquest` console script | `which flashquest` resolves |

If all gates pass → tag `phase-6-task-3`, update DOC/README/SPEC.

## Non-goals

- Tool use / function calling.
- Multi-modal (image/audio).
- Multi-user / serving.
- Web UI.
- Persistent chat history on disk.
- DuoAttention pattern training (still all-retrieval).
- The 10 tok/s v1.0 target — that's gated on Phase 6 task 5+ (INT4 KV, kernel-fused criticality). This task's wall-time SLO is "subjectively usable for streaming", i.e. ≥4 tok/s at 32 k (already cleared).
- Head-to-head vs llama.cpp / vLLM — Phase 6 task 4.

## File touch list

| File | Action |
|---|---|
| `src/flashquest/runtime/chat.py` | Create |
| `tests/test_chat.py` | Create |
| `pyproject.toml` | Modify — add `[project.scripts] flashquest = "flashquest.runtime.chat:main"` |
| `docs/PHASES/phase-6-notes.md` | Append task 3 section |
| `DOC.md` | Append task 3 row |
| `README.md` | Add chat CLI section + invocation example |
| `docs/SPEC.md` | Tick task 3 |

## Open questions

None. The decisions above resolve every ambiguity in the SPEC §6 task 3 + §11 invocation. Sampling, truncation, REPL behaviour, and streaming mechanism are all explicit.
