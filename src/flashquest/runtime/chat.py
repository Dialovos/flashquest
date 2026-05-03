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
    p.add_argument("--retention", type=float, default=0.25)
    p.add_argument("--num-sinks", type=int, default=4)
    p.add_argument("--window-pages", type=int, default=2)
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--no-patch", action="store_true",
                   help="Skip Quest+INT8 patch; run vanilla SDPA (debugging).")
    p.add_argument("--system-prompt", default=_DEFAULT_SYSTEM_PROMPT)
    return p.parse_args(argv)
