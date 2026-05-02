"""Diagnostic: compare Phase 5 (dequant chain) vs Phase 6 (algebraic) selection
masks on real cache data after a long prefill.

The persistent_e2e test at S=512 forces almost all pages via sinks+window,
which masks selection-mask differences. This runs at S>=1024 with retention
that actually filters, and diffs the resulting masks element-wise.
"""
from __future__ import annotations

import torch

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.criticality import page_scores, page_scores_int8
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages, select_pages_vectorized
from flashquest.kernel.kv_quant import dequantize_k


def main():
    torch.manual_seed(0)
    B, H_kv, H_q, D, page_size = 1, 8, 24, 128, 64
    n_rep = H_q // H_kv
    S = 2048  # 32 pages — selection actually filters at retention=0.25
    retention = 0.25
    num_sinks, window_pages = 4, 2

    cache = PersistentInt8KVCache(
        batch_size=B, num_layers=1,
        num_kv_heads=H_kv, head_dim=D,
        max_seq_len=S + 256, page_size=page_size, device="cuda",
    )

    # Synthetic prefill K/V
    K = torch.randn(B, H_kv, S, D, device="cuda", dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, device="cuda", dtype=torch.bfloat16)
    cache.update_quantized(K, V, layer_idx=0)
    views = cache.get_views(0)

    # Synthetic decode-step Q
    Q = torch.randn(B, H_q, 1, D, device="cuda", dtype=torch.bfloat16)

    # Per-head retention (50/50 retrieval/streaming for variety)
    retention_per_q = torch.zeros(H_q, device="cuda")
    retention_per_q[: H_q // 2] = retention

    # === Phase 5 path ===
    K_dq = dequantize_k(views["K_uint8"], views["K_scale"], views["K_mn"], page_size=page_size)
    K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
    page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
    scores_p5 = page_scores(Q.float(), page_min, page_max)
    sel_p5 = select_pages(
        scores_p5, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )

    # === Phase 6 path ===
    scores_p6 = page_scores_int8(Q, views["K_scale"], views["K_mn"])
    sel_p6 = select_pages_vectorized(
        scores_p6, retention=retention_per_q,
        num_sinks=num_sinks, window_pages=window_pages,
    )

    # === Diff scores ===
    scores_diff = (scores_p5 - scores_p6).abs()
    print(f"scores shapes: P5={scores_p5.shape}, P6={scores_p6.shape}")
    print(f"scores max abs diff: {scores_diff.max().item():.4f}")
    print(f"scores rel diff (vs |p5|): {(scores_diff / scores_p5.abs().clamp_min(1e-6)).mean().item():.4e}")

    # === Diff selection masks ===
    print(f"sel shapes: P5={sel_p5.shape}, P6={sel_p6.shape}")
    n_diff = (sel_p5 != sel_p6).sum().item()
    print(f"selection mask diff: {n_diff} differing entries out of {sel_p5.numel()}")
    if n_diff > 0:
        # Per-head breakdown
        diff_per_h = (sel_p5 != sel_p6).sum(dim=(0, 2, 3))
        print(f"per-head diff count: {diff_per_h.tolist()}")
        # Show count per head: how many selected by P5, P6, both, neither
        for h in range(H_q):
            p5_count = sel_p5[0, h, 0].sum().item()
            p6_count = sel_p6[0, h, 0].sum().item()
            both = (sel_p5[0, h, 0] & sel_p6[0, h, 0]).sum().item()
            print(f"  head {h:2d}: P5 sel={p5_count}, P6 sel={p6_count}, both={both}, retention={retention_per_q[h].item():.2f}")

    # === Sanity: per-head k_per_h ===
    k_per_h_loop = []
    for h in range(H_q):
        r = retention_per_q[h].item()
        if r >= 1.0:
            k_per_h_loop.append(scores_p5.shape[-1])
        elif r <= 0.0:
            k_per_h_loop.append(0)
        else:
            import math
            k_per_h_loop.append(math.ceil(r * scores_p5.shape[-1]))
    print(f"k_per_h (loop): {k_per_h_loop}")

    P = scores_p5.shape[-1]
    k_per_h_vec = (retention_per_q * P).ceil().long().clamp(min=0, max=P)
    print(f"k_per_h (vec):  {k_per_h_vec.tolist()}")


if __name__ == "__main__":
    main()
