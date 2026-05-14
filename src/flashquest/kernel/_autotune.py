"""Single-source-of-truth autotune configs for sm_86. Phase 2 dense forward."""
from __future__ import annotations

import triton

# sm_86 has 48 KB SMEM / SM. BLOCK_M = BLOCK_N = 64, double-buffered K/V loads
# at BF16, head_dim=64 -> 2 * (64 * 64 * 2) = 16 KB per buffer * 2 buffers + Q tile
# = ~24 KB. Leaves ~24 KB margin. head_dim=128 doubles SMEM use -> still fits.
FORWARD_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 64},
        num_warps=4,
        num_stages=2,
    ),
]
