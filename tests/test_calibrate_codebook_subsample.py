"""Phase 11 — regression guard for the calibration fit hot path.

The earlier calibration stalled because ``fit_per_layer_codebook`` ran 1-D
Lloyd-Max over all ~33.5M rotated scalars per fit (~2.2 s/iter x ~96 iters =
~5 min/fit, ~4.7 h for the full 28-layer K+V run). The fix subsamples up to
``max_samples`` points before clustering. These tests lock that behavior in:

  - ``lloyd_max_1d`` recovers separated clusters and stays monotonic,
  - a subsample fit matches the full-data fit within tolerance, and
  - ``fit_per_layer_codebook`` handles a full-layer-sized input in seconds
    (full-resolution would take minutes), proving the subsample is active.
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts/phase11_calibrate_codebook.py"
_PAPER = np.array([-2.1519, -1.3439, -0.7560, -0.2451,
                   0.2451, 0.7560, 1.3439, 2.1519], dtype=np.float32)


def _load_calib():
    spec = importlib.util.spec_from_file_location("phase11_calib", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lloyd_max_1d_monotonic_and_recovers_clusters():
    calib = _load_calib()
    rng = np.random.default_rng(0)
    data = np.concatenate(
        [rng.normal(m, 0.1, 40_000) for m in (-2.0, -1.0, 0.0, 1.0, 2.0)]
    ).astype(np.float32)
    centers, inertia = calib.lloyd_max_1d(data, init=_PAPER)
    assert centers.shape == (8,)
    assert np.all(np.diff(centers) >= 0), f"non-monotonic: {centers.tolist()}"
    assert centers.min() <= -1.5 and centers.max() >= 1.5
    assert inertia >= 0.0


def test_lloyd_max_1d_subsample_matches_full_within_tol():
    """A subsample fit tracks the full-data fit closely.

    This is a deliberately stiff proxy: 300k of 5M (ratio ~1/17) drifts ~1.7e-2
    from the full fit, while the shipped config (1M of ~33.5M, ratio ~1/34) sits
    near ~4e-3. The 2.5e-2 ceiling proves the subsample is materially equivalent
    (<0.6 % of the ~4.3-wide codepoint span -- negligible for a 3-bit quantizer)
    while still catching a grossly-too-small subsample.
    """
    calib = _load_calib()
    rng = np.random.default_rng(1)
    data = rng.standard_normal(5_000_000).astype(np.float32)
    full, _ = calib.lloyd_max_1d(data, init=_PAPER)
    sub, _ = calib.lloyd_max_1d(
        data[rng.choice(data.size, size=300_000, replace=False)], init=_PAPER
    )
    assert np.max(np.abs(full - sub)) < 2.5e-2


def test_fit_per_layer_codebook_subsamples_large_input_fast():
    """Full-layer-sized input (~33.5M pts) fits in seconds via subsample.

    Pre-fix this called Lloyd-Max over all 33.5M points (~5 min); the generous
    30 s ceiling distinguishes the subsampled path without flaking on load.
    """
    calib = _load_calib()
    cap = [torch.randn(1, 8192, 8, 64, dtype=torch.bfloat16) for _ in range(8)]
    t0 = time.perf_counter()
    cb, residual, n_tokens = calib.fit_per_layer_codebook(
        cap, _PAPER, max_samples=300_000
    )
    dt = time.perf_counter() - t0
    assert cb.shape == (8,)
    assert np.all(np.diff(cb) >= 0), f"non-monotonic: {cb.tolist()}"
    assert n_tokens == 8 * 8192 * 8, "n_tokens must report true count, not subsample size"
    assert dt < 30.0, f"fit took {dt:.1f}s — subsampling not active?"
