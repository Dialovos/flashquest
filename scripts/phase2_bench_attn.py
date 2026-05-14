"""Bench the Phase 2 dense Triton kernel vs flash_attn 2.7.4 + torch SDPA.

Shape pinned to the Phase 0 reference (S=8192, H_q=24, H_kv=8, D=64, BF16),
which is Llama-3.2-3B's geometry. Reports ms / forward, ratio vs FA-2,
ratio vs SDPA. SPEC §6 Phase 2 win condition: within 30% of FA-2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from flash_attn import flash_attn_func

from flashquest.kernel import flash_attn_fwd


def time_fn(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iter * 1000  # ms


def main() -> None:
    torch.manual_seed(0)
    B, H_q, H_kv, S, D = 1, 24, 8, 8192, 64
    dtype = torch.bfloat16

    Q_bhd = torch.randn(B, H_q, S, D, dtype=dtype, device="cuda")
    K_bhd = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")
    V_bhd = torch.randn(B, H_kv, S, D, dtype=dtype, device="cuda")

    Q_nhd = Q_bhd.transpose(1, 2).contiguous()
    K_nhd = K_bhd.transpose(1, 2).contiguous()
    V_nhd = V_bhd.transpose(1, 2).contiguous()

    flash_ms = time_fn(lambda: flash_attn_func(Q_nhd, K_nhd, V_nhd, causal=True))
    triton_ms = time_fn(lambda: flash_attn_fwd(Q_bhd, K_bhd, V_bhd, causal=True))

    n_rep = H_q // H_kv
    Kr = K_bhd.repeat_interleave(n_rep, dim=1)
    Vr = V_bhd.repeat_interleave(n_rep, dim=1)
    sdpa_ms = time_fn(
        lambda: torch.nn.functional.scaled_dot_product_attention(Q_bhd, Kr, Vr, is_causal=True)
    )

    result = {
        "shape": {
            "B": B, "H_q": H_q, "H_kv": H_kv, "S": S, "D": D,
            "dtype": "bfloat16", "causal": True,
        },
        "flash_attn_2_ms": flash_ms,
        "triton_kernel_ms": triton_ms,
        "sdpa_ms": sdpa_ms,
        "triton_over_flash": triton_ms / flash_ms,
        "triton_over_sdpa": triton_ms / sdpa_ms,
        "spec_target_ratio": 1.30,
        "passes_spec_target": (triton_ms / flash_ms) <= 1.30,
    }
    print(json.dumps(result, indent=2))

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase2_perf.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
