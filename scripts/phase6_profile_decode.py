"""Phase 6 task 1 prerequisite: profile Phase 5's decode loop at 32k to
identify which Python wrapper op dominates wall time.

Two passes:
  1. torch.profiler over a few decode steps — gets aggregate CUDA + CPU
     time per op (dequantize_k, compute_page_summary, page_scores,
     select_pages, flash_attn_sparse_fwd, _bf16_dense_attn_with_lse).
  2. Manual per-op timing with explicit cuda.synchronize() — separates
     real op cost from sync-stall artifacts that the profiler conflates.

Writes benchmarks/phase6_profile.json with both views.
"""
from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.criticality import page_scores
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages
from flashquest.kernel import flash_attn_sparse_fwd
from flashquest.kernel.kv_quant import dequantize_k, dequantize_v
from flashquest.runtime.awq_load import load_awq_model


N_PREFILL = 32768
N_DECODE_PROFILED = 4
N_DECODE_TIMED = 8
PAGE_SIZE = 64
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


@contextmanager
def cuda_timer(label: str, accum: dict):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    accum.setdefault(label, []).append(dt)


@torch.no_grad()
def manual_per_op_timing(
    model, tok, cache, head_pattern, next_tok, n_decode: int,
) -> dict:
    """Hand-instrumented timing of each step inside one decode forward pass.

    Caller has already run prefill + a couple warmup decode steps.
    Picks layer 0 mid-decode as a representative slice.
    """
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    layer_idx = 0
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    n_rep = H_q // H_kv

    pattern_layer = head_pattern[layer_idx].to("cuda")
    pattern_per_q = pattern_layer.repeat_interleave(n_rep)
    retention_per_q = torch.where(
        pattern_per_q,
        torch.full((H_q,), RETENTION, device="cuda"),
        torch.zeros(H_q, device="cuda"),
    )

    accum: dict = {}
    end_to_end_steps_s: list[float] = []

    for step in range(n_decode):
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        _ = model(next_tok, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        end_to_end_steps_s.append(time.perf_counter() - t_step)

        # Re-run the layer-0 sub-steps with our own timing wrappers using
        # the cache state captured during the real forward pass.
        views = cache.get_views(layer_idx)
        Q = torch.randn(1, H_q, 1, head_dim, device="cuda", dtype=torch.bfloat16)

        with cuda_timer("dequantize_k", accum):
            K_dq = dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=PAGE_SIZE)

        with cuda_timer("repeat_interleave_K", accum):
            K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)

        with cuda_timer("compute_page_summary", accum):
            page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=PAGE_SIZE)

        with cuda_timer("page_scores", accum):
            scores = page_scores(Q.float(), page_min, page_max)

        with cuda_timer("select_pages", accum):
            sel = select_pages(
                scores, retention=retention_per_q,
                num_sinks=NUM_SINKS, window_pages=WINDOW_PAGES,
            )

        with cuda_timer("flash_attn_sparse_fwd", accum):
            O, lse = flash_attn_sparse_fwd(
                Q, views["K_uint8"], views["K_scale"], views["K_mn"],
                views["V_uint8"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=PAGE_SIZE, return_lse=True,
            )

        next_tok = torch.tensor([[1]], device="cuda")  # dummy advance

    means_ms = {k: 1000 * (sum(v) / len(v)) for k, v in accum.items()}
    end_to_end_ms = 1000 * (sum(end_to_end_steps_s) / len(end_to_end_steps_s))
    return {
        "per_step_ms": means_ms,
        "n_steps": n_decode,
        "end_to_end_step_ms": end_to_end_ms,
        "decode_tok_per_s": 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0,
    }


@torch.no_grad()
def profiler_pass(model, next_tok, n_decode: int) -> tuple[dict, torch.Tensor]:
    """CPU-only torch.profiler pass (CUPTI unavailable in WSL2 → no CUDA timing)."""
    activities = [ProfilerActivity.CPU]
    with profile(activities=activities, record_shapes=False) as prof:
        for _ in range(n_decode):
            with record_function("decode_step"):
                out = model(next_tok, use_cache=True, logits_to_keep=1)
                next_tok = out.logits[:, -1:].argmax(dim=-1)
        torch.cuda.synchronize()

    by_op_cpu: dict = {}
    for ev in prof.key_averages():
        by_op_cpu[ev.key] = ev.self_cpu_time_total / 1000.0  # us → ms
    top_cpu = sorted(by_op_cpu.items(), key=lambda kv: -kv[1])[:30]
    return {"n_decode": n_decode, "top_cpu_ms": top_cpu}, next_tok


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    print(f"Loading {name} ...")
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL + N_DECODE_PROFILED + N_DECODE_TIMED + 128,
        page_size=PAGE_SIZE, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=RETENTION, num_sinks=NUM_SINKS,
        window_pages=WINDOW_PAGES, page_size=PAGE_SIZE,
    )

    print("\n=== prefill (one-time, reused across both passes) ===")
    cache._seen_tokens = [0] * cache.num_layers
    ids = synthetic_prompt_ids(tok, N_PREFILL)
    t0 = time.perf_counter()
    out = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    print(f"prefill: {time.perf_counter() - t0:.1f} s")
    next_tok = out.logits[:, -1:].argmax(dim=-1)
    # 2 warmup decode steps to populate the partial-page tail.
    for _ in range(2):
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(dim=-1)

    print("\n=== torch.profiler pass (CPU-only — CUPTI unavailable in WSL2) ===")
    prof_result, next_tok = profiler_pass(model, next_tok, N_DECODE_PROFILED)
    print("Top CPU ops (self time, ms):")
    for k, v in prof_result["top_cpu_ms"][:20]:
        print(f"  {v:7.2f} ms  {k}")

    print("\n=== manual per-op timing (layer 0, single decode step) ===")
    manual = manual_per_op_timing(model, tok, cache, pattern, next_tok, N_DECODE_TIMED)
    total_ms = sum(manual["per_step_ms"].values())
    for k, v in sorted(manual["per_step_ms"].items(), key=lambda kv: -kv[1]):
        pct = 100 * v / total_ms if total_ms > 0 else 0
        print(f"  {v:7.3f} ms  ({pct:5.1f} %)  {k}")
    print(f"  ----")
    print(f"  {total_ms:7.3f} ms  total per layer 0 sub-step measured")
    print(f"  Across {cfg.num_hidden_layers} layers / decode token: ~{total_ms * cfg.num_hidden_layers:.1f} ms")
    print(f"  End-to-end one decode step (model forward): {manual['end_to_end_step_ms']:.1f} ms "
          f"=> {manual['decode_tok_per_s']:.3f} tok/s")

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_profile.json"
    out.write_text(json.dumps({
        "model": name, "n_prefill": N_PREFILL,
        "torch_profiler": prof_result, "manual": manual,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
