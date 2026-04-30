"""Smoke test: verify Triton tl.dot accepts INT8 operands on sm_86.

Answers SPEC §8 Open Question 1.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _int8_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k, other=0).to(tl.int8)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k, other=0).to(tl.int8)
        acc += tl.dot(a, b, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required"
    cc = torch.cuda.get_device_capability(0)
    assert cc == (8, 6), f"Expected sm_86, got sm_{cc[0]}{cc[1]}"

    M, N, K = 128, 128, 128
    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device="cuda")
    c = torch.empty((M, N), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
    _int8_matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
    )
    torch.cuda.synchronize()

    # PyTorch CUDA doesn't implement int @ int matmul; use float64 reference
    # (range fits exactly: 128 * 7 * 7 = 6272 << 2^53 mantissa).
    expected = (a.to(torch.float64) @ b.to(torch.float64)).to(torch.int32)
    max_err = (c - expected).abs().max().item()
    print(f"shape: M={M} N={N} K={K}, BLOCK=64x64x64, sm_{cc[0]}{cc[1]}")
    print(f"torch={torch.__version__}, triton={triton.__version__}")
    print(f"max abs error: {max_err}")
    assert max_err == 0, "INT8 mma should be exact on integers"
    print("OK: Triton tl.dot with INT8 operands works on sm_86")


if __name__ == "__main__":
    main()
