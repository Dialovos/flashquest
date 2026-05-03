"""vLLM single-request decode bench, parametric over --max-model-len.

Catches OOM during LLM(...) construction and llm.generate(...) and writes a
per-cell JSON record so the orchestrator can keep going.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path


def _record_skeleton(model_id: str, max_model_len: int) -> dict:
    return {
        "backend": "vLLM 0.7.3",
        "quant": "AWQ-INT4, FP16 KV",
        "ctx_len": max_model_len,
        "decode_tok_s": None,
        "prefill_tok_s": None,
        "peak_vram_mib": None,
        "wall_s": None,
        "oom": False,
        "error": None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    record = _record_skeleton(args.model, args.max_model_len)
    t_start = time.perf_counter()

    import torch  # imported here so OOM during import is caught below

    try:
        from vllm import LLM, SamplingParams

        torch.cuda.reset_peak_memory_stats()

        llm = LLM(
            model=args.model,
            quantization="awq",
            dtype="float16",
            gpu_memory_utilization=0.95,
            max_model_len=args.max_model_len,
            enforce_eager=False,
            swap_space=0,
        )

        target_in = max(64, int(args.max_model_len * 0.8))
        prompt = ("The quick brown fox jumps over the lazy dog. " * (target_in // 8 + 1))
        tok = llm.get_tokenizer()
        ids = tok.encode(prompt)[:target_in]
        prompt = tok.decode(ids, skip_special_tokens=True)

        llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0.0))
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.0))
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        out = outputs[0]
        n_in = len(out.prompt_token_ids)
        n_out = len(out.outputs[0].token_ids)
        elapsed = t1 - t0

        record["decode_tok_s"] = n_out / elapsed if elapsed > 0 else None
        record["prefill_tok_s"] = n_in / elapsed if elapsed > 0 else None
        record["peak_vram_mib"] = int(torch.cuda.max_memory_allocated() / 1024 / 1024)

    except Exception as exc:
        msg = str(exc).lower()
        if (
            "out of memory" in msg
            or "kv cache" in msg
            or "no available" in msg
            or "memory for the cache" in msg
        ):
            record["oom"] = True
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        record["wall_s"] = time.perf_counter() - t_start
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
