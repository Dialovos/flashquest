"""Per-layer TurboQuant codebook loader (Phase 11).

Codebooks are 8-codepoint Lloyd-Max levels fit to actual Llama K/V activations
after per-token RMS scale + Walsh-Hadamard rotation. Per-layer granularity
(28 codebooks for Llama-3.2-3B). Each codebook stored sorted ascending so the
kernel's bit-split (msb << 2) | lsb decode keeps ordinal monotonicity.

Artifact path: src/flashquest/turbo/codebook_<model>.pt
Artifact shape: (num_layers, 2, 8) fp32, axis 1 = (K=0, V=1).
"""
from __future__ import annotations

from pathlib import Path

import torch

PAPER_CODEBOOK = torch.tensor(
    [-2.1519, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1519],
    dtype=torch.float32,
)

_ARTIFACT_MAP = {
    "casperhansen/llama-3.2-3b-instruct-awq": "codebook_llama_3_2_3b.pt",
}


def load_codebook(model_id: str) -> torch.Tensor:
    """Return (num_layers, 2, 8) fp32 codebook for `model_id`.

    Raises:
        KeyError if no calibration artifact ships for `model_id`.
    """
    if model_id not in _ARTIFACT_MAP:
        raise KeyError(
            f"No calibration codebook for {model_id!r}. "
            f"Available: {list(_ARTIFACT_MAP)}. "
            f"Fall back to PAPER_CODEBOOK or run scripts/phase11_calibrate_codebook.py."
        )
    artifact_path = Path(__file__).parent / _ARTIFACT_MAP[model_id]
    if not artifact_path.exists():
        raise KeyError(
            f"Artifact {artifact_path} missing. Run "
            f"`python scripts/phase11_calibrate_codebook.py --model {model_id}` to generate."
        )
    cb = torch.load(artifact_path, map_location="cpu", weights_only=True)
    assert cb.dim() == 3 and cb.shape[1] == 2 and cb.shape[2] == 8, \
        f"Bad codebook shape {tuple(cb.shape)}"
    assert cb.dtype == torch.float32
    return cb
