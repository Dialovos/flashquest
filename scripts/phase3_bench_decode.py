"""Synthetic decode-loop perf: phase 3 sparse INT8 vs phase 2 dense.

Setup: Llama-3.2-3B geometry (H_q=24, H_kv=8, D=64). Pre-populate an 8 k
KV cache. Time N=64 decode steps. Compare tok/s.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages
from flashquest.kernel import flash_attn_fwd, flash_attn_sparse_fwd
from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v


N_DECODE_STEPS = 64
S_KV = 8192
PAGE_SIZE = 64
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2


def _setup():
    torch.manual_seed(0)
    B, H_q, H_kv, D = 1, 24, 8, 64
    K = torch.randn(B, H_kv, S_KV, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S_KV, D, dtype=torch.bfloat16, device="cuda")
    K_u8, K_s, K_m = quantize_k(K, page_size=PAGE_SIZE)
    V_u8, V_s, V_m = quantize_v(V)
    return B, H_q, H_kv, D, K, V, K_u8, K_s, K_m, V_u8, V_s, V_m


def _bench_dense(B, H_q, H_kv, D, K, V):
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    n_rep = H_q // H_kv
    Kr = K.repeat_interleave(n_rep, dim=1)
    Vr = V.repeat_interleave(n_rep, dim=1)
    for _ in range(5):
        flash_attn_fwd(Q, Kr, Vr, causal=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_STEPS):
        flash_attn_fwd(Q, Kr, Vr, causal=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_DECODE_STEPS * 1000


def _bench_sparse(B, H_q, H_kv, D, K_u8, K_s, K_m, V_u8, V_s, V_m):
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    K_dq = dequantize_k(K_u8, K_s, K_m, page_size=PAGE_SIZE)
    K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
    pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=PAGE_SIZE)
    scores = page_scores(Q.float(), pmin, pmax)
    sel = select_pages(scores, RETENTION, NUM_SINKS, WINDOW_PAGES)

    for _ in range(5):
        flash_attn_sparse_fwd(Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=PAGE_SIZE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_DECODE_STEPS):
        flash_attn_sparse_fwd(Q, K_u8, K_s, K_m, V_u8, V_s, V_m, selection_mask=sel, page_size=PAGE_SIZE)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_DECODE_STEPS * 1000


def main() -> None:
    B, H_q, H_kv, D, K, V, K_u8, K_s, K_m, V_u8, V_s, V_m = _setup()

    dense_ms = _bench_dense(B, H_q, H_kv, D, K, V)
    sparse_ms = _bench_sparse(B, H_q, H_kv, D, K_u8, K_s, K_m, V_u8, V_s, V_m)

    speedup = dense_ms / sparse_ms
    result = {
        "shape": {"B": B, "H_q": H_q, "H_kv": H_kv, "S_kv": S_KV, "D": D},
        "config": {
            "page_size": PAGE_SIZE, "retention": RETENTION,
            "num_sinks": NUM_SINKS, "window_pages": WINDOW_PAGES,
            "n_decode_steps": N_DECODE_STEPS,
        },
        "phase2_dense_ms_per_step": dense_ms,
        "phase3_sparse_ms_per_step": sparse_ms,
        "sparse_speedup": speedup,
        "phase3_target_speedup": 1.5,
        "passes_phase3_target": speedup >= 1.5,
    }
    print(json.dumps(result, indent=2))

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase3_perf.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
