"""vLLM decode + prefill timing at 8k context, single request.

vLLM does not load Q4_K_M GGUF directly, so this baseline uses
Llama-3.2-3B AWQ-INT4 — same model, comparable bit-width, different
quant method. Document this caveat when comparing against the
llama.cpp Q4_K_M number.
"""
import json
import time
from pathlib import Path

import torch
from vllm import LLM, SamplingParams


def main() -> None:
    model_id = "hugging-quants/Llama-3.2-3B-Instruct-AWQ-INT4"
    llm = LLM(
        model=model_id,
        quantization="awq",
        dtype="float16",
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        enforce_eager=False,
        swap_space=0,
    )

    # ~7000 input tokens leaves headroom for 128 generated tokens within 8k.
    prompt = ("The quick brown fox jumps over the lazy dog. " * 1000)[:30000]

    # Warm-up
    llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    torch.cuda.synchronize()

    # Measure
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], SamplingParams(max_tokens=128, temperature=0.0))
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    out = outputs[0]
    n_in = len(out.prompt_token_ids)
    n_out = len(out.outputs[0].token_ids)
    elapsed = t1 - t0

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    result = {
        "model": model_id,
        "input_tokens": n_in,
        "output_tokens": n_out,
        "elapsed_s": elapsed,
        "tok_s_total": (n_in + n_out) / elapsed,
        "decode_tok_s_approx": n_out / elapsed,
        "peak_vram_mb": peak_mb,
    }
    print(json.dumps(result, indent=2))

    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "vllm_8k.json"
    out_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
