"""Single-request vLLM bench using V0 RequestMetrics phase timestamps."""
from __future__ import annotations

import argparse
import os
import time
from importlib.metadata import version

from bench_common import (
    add_run_arguments,
    is_oom,
    new_record,
    provenance,
    run_config,
    summarize_samples,
    synthetic_token_ids,
    validate_run_arguments,
    vllm_sample,
    write_benchmark_record,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_run_arguments(p)
    p.add_argument("--kv-cache-dtype", choices=["auto", "fp8"], default="fp8")
    args = p.parse_args()
    validate_run_arguments(p, args)
    config = run_config(args.ctx_len, args.n_decode, args.reps, args.seed,
                        model=args.model, kv_cache_dtype=args.kv_cache_dtype)
    record = new_record("vLLM", f"AWQ-INT4, {args.kv_cache_dtype} KV", config)
    record["provenance"] = provenance()
    t_start = time.perf_counter()
    try:
        # V1 releases can omit RequestMetrics. A missing timing API is an error,
        # not permission to label end-to-end throughput as decode throughput.
        os.environ.setdefault("VLLM_USE_V1", "0")
        from vllm import LLM, SamplingParams

        record["versions"] = {"vllm": version("vllm")}
        llm = LLM(
            model=args.model, quantization="awq", dtype="float16",
            kv_cache_dtype=args.kv_cache_dtype, gpu_memory_utilization=0.95,
            max_model_len=args.ctx_len + args.n_decode, swap_space=0,
            enable_prefix_caching=False, disable_log_stats=False,
        )
        ids = synthetic_token_ids(len(llm.get_tokenizer()), args.ctx_len, args.seed)
        prompt = {"prompt_token_ids": ids}
        params = SamplingParams(max_tokens=args.n_decode, temperature=0.0, ignore_eos=True)
        llm.generate([prompt], params, use_tqdm=False)
        for _ in range(args.reps):
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], params, use_tqdm=False)
            elapsed = time.perf_counter() - t0
            record["samples"].append(vllm_sample(outputs[0], args.ctx_len,
                                                args.n_decode, elapsed))
        summarize_samples(record)
        # GPU workers may be separate processes. Parent allocator counters cannot
        # establish their memory usage, so leave those fields unmeasured.
    except Exception as exc:  # noqa: BLE001 — record backend/import failures for the matrix
        record["oom"] = is_oom(str(exc))
        record["error"] = f"{type(exc).__name__}: {exc}"
    record["wall_s"] = time.perf_counter() - t_start
    exported = write_benchmark_record(args.out, record)
    print(exported)
    return int(record["error"] is not None)


if __name__ == "__main__":
    raise SystemExit(main())
