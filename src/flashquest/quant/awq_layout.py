"""AWQ tensor-layout constants + assertion helper.

Locks Phase 8a Triton kernels against the AutoAWQ tensor layout. Run
`python scripts/phase8a_audit_awq_layout.py` to print actual shapes from a
loaded checkpoint and verify these constants match.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

# AutoAWQ defaults (Llama-3.2-3B-AWQ uses these)
AWQ_GROUP_SIZE = 128       # in-axis group for scales/zeros
AWQ_PACK_FACTOR = 8        # 8 INT4 values packed into one int32 along the in-axis


@dataclass(frozen=True)
class AWQLayout:
    """Shapes of an AutoAWQ Linear's quantized state.

    AutoAWQ packs along the OUT axis (verified by audit on
    casperhansen/llama-3.2-3b-instruct-awq, 2026-05-08).

    For an `nn.Linear(in_features=K, out_features=N)`:
        qweight: (K, N // AWQ_PACK_FACTOR) int32 — 8 INT4 values per int32 along N
        scales:  (K // AWQ_GROUP_SIZE, N) bf16/fp16 — per-group scale (K-axis groups)
        qzeros:  (K // AWQ_GROUP_SIZE, N // AWQ_PACK_FACTOR) int32 — packed along N
    """
    in_features: int
    out_features: int

    @property
    def qweight_shape(self) -> tuple[int, int]:
        return (self.in_features, self.out_features // AWQ_PACK_FACTOR)

    @property
    def scales_shape(self) -> tuple[int, int]:
        return (self.in_features // AWQ_GROUP_SIZE, self.out_features)

    @property
    def qzeros_shape(self) -> tuple[int, int]:
        return (self.in_features // AWQ_GROUP_SIZE,
                self.out_features // AWQ_PACK_FACTOR)


def assert_awq_layout(
    layer: torch.nn.Module,
    in_features: int,
    out_features: int,
    *,
    name: str = "<unnamed>",
) -> AWQLayout:
    """Assert that an AutoAWQ Linear's qweight/scales/qzeros match expectations.

    Returns the AWQLayout for downstream kernel use.
    """
    expected = AWQLayout(in_features, out_features)

    qw = getattr(layer, "qweight", None)
    sc = getattr(layer, "scales", None)
    qz = getattr(layer, "qzeros", None)
    if qw is None or sc is None or qz is None:
        raise ValueError(
            f"{name}: layer is not an AutoAWQ-quantized Linear "
            f"(missing qweight/scales/qzeros)"
        )

    if tuple(qw.shape) != expected.qweight_shape:
        raise ValueError(
            f"{name}: qweight shape {tuple(qw.shape)} != expected {expected.qweight_shape}"
        )
    if tuple(sc.shape) != expected.scales_shape:
        raise ValueError(
            f"{name}: scales shape {tuple(sc.shape)} != expected {expected.scales_shape}"
        )
    if tuple(qz.shape) != expected.qzeros_shape:
        raise ValueError(
            f"{name}: qzeros shape {tuple(qz.shape)} != expected {expected.qzeros_shape}"
        )
    if qw.dtype != torch.int32:
        raise ValueError(f"{name}: qweight dtype {qw.dtype} != int32")
    if qz.dtype != torch.int32:
        raise ValueError(f"{name}: qzeros dtype {qz.dtype} != int32")

    return expected
