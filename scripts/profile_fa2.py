"""Run a single FA-2 forward on a representative shape; print timing."""
import time

import torch
from flash_attn import flash_attn_func


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Llama-3.2-3B shape: 24 heads, head_dim=64; pick seq=8192, batch=1.
    B, H_q, H_kv, S_q, S_kv, D = 1, 24, 8, 8192, 8192, 64
    q = torch.randn(B, S_q, H_q, D, dtype=dtype, device=device)
    k = torch.randn(B, S_kv, H_kv, D, dtype=dtype, device=device)
    v = torch.randn(B, S_kv, H_kv, D, dtype=dtype, device=device)

    # Warm-up
    for _ in range(3):
        flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()

    n_iter = 20
    t0 = time.perf_counter()
    for _ in range(n_iter):
        out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    avg_ms = (t1 - t0) / n_iter * 1000
    print(f"FA-2 fwd (B={B}, S={S_q}, H_q={H_q}, H_kv={H_kv}, D={D}) avg: {avg_ms:.3f} ms")
    print(f"output sum (sanity): {out.float().sum().item():.4f}")


if __name__ == "__main__":
    main()
