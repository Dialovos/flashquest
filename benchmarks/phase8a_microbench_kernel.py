"""Microbench: compact INT4 kernel vs. bool-mask kernel at 32k decode shape.

Gate: compact >= 1.2x bool-mask kernel-wall, otherwise reconsider Phase 8a.

Usage: python benchmarks/phase8a_microbench_kernel.py
"""
from __future__ import annotations

import time

import torch

from flashquest.eager.selection import build_compact_selection
from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd
from flashquest.kernel.sparse_int4_fwd_compact import flash_attn_sparse_int4_fwd_compact


def time_call(fn, warmup=20, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("requires CUDA")
    torch.manual_seed(0)
    device = "cuda"

    # 32k decode shape — Llama-3.2-3B
    D, page_size = 128, 64
    B, H_q, H_kv = 1, 24, 8
    P = 32768 // page_size  # 512
    S_kv = P * page_size

    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device=device)
    K_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    V_packed = torch.randint(0, 256, (B, H_kv, S_kv, D // 2), dtype=torch.uint8, device=device)
    K_scale = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device).abs()
    K_mn = torch.randn(B, H_kv, P, D, dtype=torch.bfloat16, device=device)
    V_scale = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device).abs()
    V_mn = torch.randn(B, H_kv, S_kv, dtype=torch.bfloat16, device=device)

    # 25% retention -> top-k = 128 pages out of 512
    sel_mask = torch.zeros(B, H_q, 1, P, dtype=torch.bool, device=device)
    for h in range(H_q):
        idx = torch.randperm(P, device=device)[:128]
        sel_mask[0, h, 0, idx] = True

    sel_compact = build_compact_selection(sel_mask, BUCKET_MAX=134)  # 128 + 4 sinks + 2 window

    def _full():
        flash_attn_sparse_int4_fwd(
            Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
            selection_mask=sel_mask, page_size=page_size, return_lse=True,
        )

    def _compact():
        flash_attn_sparse_int4_fwd_compact(
            Q, K_packed, K_scale, K_mn, V_packed, V_scale, V_mn,
            selected_page_ids=sel_compact, page_size=page_size, return_lse=True,
        )

    t_full = time_call(_full)
    t_compact = time_call(_compact)

    speedup = t_full / t_compact
    print(f"Bool-mask kernel:    {t_full * 1e6:.1f} us / call")
    print(f"Compact-list kernel: {t_compact * 1e6:.1f} us / call")
    print(f"Speedup:             {speedup:.2f}x")
    print()
    if speedup >= 1.2:
        print("GATE PASS — proceed to integration (Task 7).")
    else:
        print("GATE FAIL — investigate before continuing.")
        print("  Possible causes: Triton autotune mismatch, num_warps/num_stages")
        print("  not yet tuned for the smaller loop, or memory bandwidth saturated.")


if __name__ == "__main__":
    main()
