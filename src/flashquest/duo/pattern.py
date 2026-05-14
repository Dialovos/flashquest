"""Loader for DuoAttention pre-trained head classifications.

The upstream format (vendor/duo-attention/attn_patterns/<model>/<run>/
full_attention_heads.tsv) is a tab-separated file with one row per layer
and one column per KV head. Values are floats in [0, 1] indicating the
probability the head should use *full retrieval* attention. We threshold
to bool: True = retrieval head, False = streaming head.
"""
from __future__ import annotations

from pathlib import Path

import torch


def load_duo_pattern(path: str | Path, threshold: float = 0.5) -> torch.Tensor:
    """Read a DuoAttention TSV and return a (num_layers, num_kv_heads) bool tensor.

    Args:
        path: TSV file path.
        threshold: values >= threshold are retrieval (True). Default 0.5.

    Returns:
        Bool tensor on CPU. True = retrieval head, False = streaming head.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"DuoAttention pattern not found: {p}")

    rows: list[list[float]] = []
    expected_cols: int | None = None
    for line_no, line in enumerate(p.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        cols = [float(c) for c in line.split("\t") if c]
        if expected_cols is None:
            expected_cols = len(cols)
        elif len(cols) != expected_cols:
            raise ValueError(
                f"DuoAttention pattern at {p}:{line_no} has {len(cols)} cols, "
                f"expected {expected_cols}"
            )
        rows.append(cols)

    if not rows:
        raise ValueError(f"DuoAttention pattern at {p} is empty")

    return torch.tensor(rows) >= threshold
