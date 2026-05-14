# Phase 11 — TurboQuant Calibrated Codebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace TurboQuant K3-V3's paper-default Lloyd-Max codebook with 28 per-layer calibrated codebooks (K + V each) fit from Llama-3.2-3B-AWQ activations on PaulGrahamEssays. Target: RULER NIAH 4 k multivalue ≥95 % (matches INT4), enabling K3-V3 to replace `--kv-bits 4` as the v1.2 default at unchanged 25 % cache shrink.

**Architecture:** The TurboQuant kernel's `_codebook_lookup_3bit` is parameterized to accept 8 codepoints as `tl.constexpr` (per-launch). The Python wrapper picks `cache.codebook_k[layer_idx]` / `cache.codebook_v[layer_idx]` and unpacks into 16 constexpr floats. Per-layer codebooks compile a separate kernel variant lazily (cached after first run). The quant/dequant helpers in `kv_quant.py` accept the codebook as an argument; cache holds the per-layer codebooks; dispatcher in `llama_persistent_patch.py` threads `layer_idx` into `make_quest_persistent_forward`. Offline calibration ships a 2 KB `.pt` artifact; loader falls back to paper codebook for unknown models.

**Tech Stack:** Python 3.12, PyTorch 2.5.1+cu121, Triton 3.1.0, transformers 4.57.6, AutoAWQ 0.2.9, scikit-learn (k-means for Lloyd-Max). RTX 3050 Ti Laptop sm_86, 4 GB VRAM, WSL2.

**Spec:** `docs/superpowers/specs/2026-05-14-phase-11-turboquant-calibrated-codebook-design.md` (r2, post-codex)

**Critical entry gate (Task 1, profile-first per `feedback_profile_before_speedup_specs.md`):** stop if codepoint divergence < 5 % OR quality-simulator delta is bit-identical across all 3 probe samples. **Do not implement Tasks 2+ if the gate fails — document the kill at `docs/PHASES/phase-11-killed-by-probe.md` and stop.** The granularity pre-signal R is advisory; record it but do not gate on it.

---

## Task 1: Profile-first probe (ENTRY GATE)

This task uses the **existing** Phase 7 quantization helpers + reference path — no new kernel work, no plumbing. The goal is to measure whether calibration produces a meaningfully different codebook AND whether using it changes generated tokens before any integration code is touched.

**Files:**
- Create: `scripts/phase11_calibrate_probe.py`
- Create (output): `benchmarks/phase11/probe.md`
- Create (output): `benchmarks/phase11/probe_codepoints.json` (raw per-layer codepoints, calibrated + paper, for inspection)

- [ ] **Step 1: Write the probe script**

Create `scripts/phase11_calibrate_probe.py`:

```python
"""Phase 11 Task 1 — calibration entry-gate probe.

Loads Llama-3.2-3B-AWQ, captures K/V tensors per layer at 8k prefill on 3 PG
essays (~24 k tokens), fits per-layer + per-head Lloyd-Max codebooks (k=8,
warm-start from paper), then runs RULER NIAH multivalue (3 samples, ctx=4 k)
twice via `_flash_attn_sparse_turbo_fwd_reference`: once with paper codebook,
once with calibrated. Outputs:

  - benchmarks/phase11/probe.md     (one-page verdict + tables)
  - benchmarks/phase11/probe_codepoints.json  (raw per-layer codepoints)

Gate (must clear ALL to proceed to Task 2+):
  - codepoint_divergence_max  ≥ 5%   (per-layer-K and per-layer-V each)
  - quality_simulator_delta   ≥ 1 sample differs in generated tokens (of 3)
Granularity pre-signal R is recorded as an advisory number; not a gate.

Usage: python scripts/phase11_calibrate_probe.py
       --model casperhansen/llama-3.2-3b-instruct-awq
       --essays 3 --out benchmarks/phase11/probe.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from flashquest.kernel.kv_quant import K_TURBO_CODEBOOK
from flashquest.kernel.wht import wht_along_head_dim
from flashquest.runtime.awq_load import load_awq_model
from flashquest.eval.niah import generate_multivalue_samples

# ... (full implementation below)


def capture_activations(model, prompts, num_layers):
    """Run prefill on each prompt; collect post-projection K and V per layer.

    Returns dict: layer_idx -> {"K": [(S, H_kv, D), ...], "V": [...]} (cpu).
    Hooks discard tensors after detaching to cpu to keep RAM bounded.
    """
    acts = {li: {"K": [], "V": []} for li in range(num_layers)}
    handles = []
    for module in model.modules():
        if hasattr(module, "layer_idx") and hasattr(module, "k_proj"):
            li = module.layer_idx
            def make_hook(layer_idx, which):
                def hook(mod, inp, out):
                    # out shape: (B, S, H_kv * D); reshape to (S, H_kv, D)
                    B, S, _ = out.shape
                    H_kv = mod.out_features // model.config.head_dim
                    D = model.config.head_dim
                    acts[layer_idx][which].append(
                        out.detach().reshape(B, S, H_kv, D).cpu().float()
                    )
                return hook
            handles.append(module.k_proj.register_forward_hook(make_hook(li, "K")))
            handles.append(module.v_proj.register_forward_hook(make_hook(li, "V")))

    with torch.no_grad():
        for prompt_ids in prompts:
            model(prompt_ids, use_cache=False)

    for h in handles:
        h.remove()
    return acts


def fit_per_layer_codebook(acts_layer_kv, n_clusters=8, warm_start=None):
    """Apply per-token RMS scale + WHT, flatten, fit k-means.

    Returns: codebook (8,) fp32 sorted ascending.
    """
    # Stack all prompts' tensors: (S_total, H_kv, D)
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0)  # (S_total, H_kv, D)
    # Per-token RMS scale (across head_dim, per (token, head))
    rms = stacked.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked.float() / rms
    # WHT along head_dim
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)  # (S, H_kv, D)
    flat = rotated.reshape(-1).numpy().astype(np.float32)  # (S*H_kv*D,)

    init = warm_start.reshape(-1, 1) if warm_start is not None else "k-means++"
    km = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=300, random_state=0)
    km.fit(flat.reshape(-1, 1))
    codebook = np.sort(km.cluster_centers_.flatten()).astype(np.float32)
    return codebook


def fit_per_head_codebook(acts_layer_kv, n_clusters=8, warm_start=None):
    """Like fit_per_layer_codebook but per-head; returns (H_kv, 8) fp32."""
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0)
    H_kv = stacked.shape[1]
    rms = stacked.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked.float() / rms
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)
    codebooks = np.zeros((H_kv, n_clusters), dtype=np.float32)
    for h in range(H_kv):
        flat = rotated[:, h, :].reshape(-1).numpy().astype(np.float32)
        init = warm_start.reshape(-1, 1) if warm_start is not None else "k-means++"
        km = KMeans(n_clusters=n_clusters, init=init, n_init=1, max_iter=300, random_state=0)
        km.fit(flat.reshape(-1, 1))
        codebooks[h] = np.sort(km.cluster_centers_.flatten())
    return codebooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--essays", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--out", default="benchmarks/phase11/probe.md")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_awq_model(args.model)
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    paper_cb = K_TURBO_CODEBOOK.cpu().numpy().astype(np.float32)

    # Build N prompt tensors from data/PaulGrahamEssays.json at ctx_len truncation.
    import json as _json
    essays = _json.load(open("data/PaulGrahamEssays.json"))["essays"][:args.essays]
    prompts = []
    for essay in essays:
        ids = tok(essay["text"], return_tensors="pt", truncation=True,
                  max_length=args.ctx).input_ids.cuda()
        prompts.append(ids)

    print(f"Capturing activations from {len(prompts)} prompts ...")
    acts = capture_activations(model, prompts, num_layers)

    print(f"Fitting per-layer codebooks (K + V) for {num_layers} layers ...")
    per_layer_k = np.stack([fit_per_layer_codebook(acts[li]["K"], warm_start=paper_cb)
                            for li in range(num_layers)])  # (L, 8)
    per_layer_v = np.stack([fit_per_layer_codebook(acts[li]["V"], warm_start=paper_cb)
                            for li in range(num_layers)])

    print(f"Fitting per-head codebooks (granularity side-check) ...")
    per_head_k = np.stack([fit_per_head_codebook(acts[li]["K"], warm_start=paper_cb)
                           for li in range(num_layers)])  # (L, H_kv, 8)
    per_head_v = np.stack([fit_per_head_codebook(acts[li]["V"], warm_start=paper_cb)
                           for li in range(num_layers)])

    # Codepoint divergence (per-layer vs paper)
    div_k = np.max(np.abs(per_layer_k - paper_cb[None, :]) / np.abs(paper_cb[None, :]))
    div_v = np.max(np.abs(per_layer_v - paper_cb[None, :]) / np.abs(paper_cb[None, :]))

    # Granularity ratio R = max |per-head - per-layer| / max |per-layer - paper|
    EPS = 1e-6
    per_layer_diff = np.max(np.abs(per_layer_k - paper_cb[None, :]))
    per_head_diff = np.max(np.abs(per_head_k - per_layer_k[:, None, :]))
    R = per_head_diff / max(per_layer_diff, EPS)

    # Quality-simulator delta: 3 RULER multivalue samples, paper vs calibrated.
    # This requires running _flash_attn_sparse_turbo_fwd_reference with both
    # codebooks and comparing generated tokens. To keep the probe self-contained,
    # we monkey-patch K_TURBO_CODEBOOK and V_TURBO_CODEBOOK in flashquest.kernel.kv_quant
    # for each run, then read back the generated sequences.
    print("Running quality-simulator delta (3 samples × 2 codebooks) ...")
    sim_delta_count = quality_simulator_delta(
        model, tok, paper_cb, per_layer_k, per_layer_v, n_samples=3,
    )

    # Write probe.md
    verdict_codepoint = "PASS" if (div_k >= 0.05 and div_v >= 0.05) else "KILL"
    verdict_simulator = "PASS" if sim_delta_count >= 1 else "KILL"
    overall = "PROCEED" if verdict_codepoint == "PASS" and verdict_simulator == "PASS" else "KILL"

    md = []
    md.append("# Phase 11 — Calibration Entry-Gate Probe\n")
    md.append(f"**Verdict: {overall}**\n")
    md.append(f"- Codepoint divergence (per-layer vs paper): K = {div_k:.3%}, "
              f"V = {div_v:.3%} → **{verdict_codepoint}** (gate ≥5%).")
    md.append(f"- Quality-simulator delta: {sim_delta_count}/3 samples differ → "
              f"**{verdict_simulator}** (gate ≥1).")
    md.append(f"- Granularity pre-signal R (advisory): {R:.2f} "
              f"(R > 2 across layers → consider Phase 11b per-head).")
    Path(args.out).write_text("\n".join(md))

    json_out = Path(args.out).with_suffix(".codepoints.json")
    json_out.write_text(json.dumps({
        "paper": paper_cb.tolist(),
        "per_layer_k": per_layer_k.tolist(),
        "per_layer_v": per_layer_v.tolist(),
        "per_head_k_summary": {
            "shape": list(per_head_k.shape),
            "max_per_head_dev_from_per_layer": float(per_head_diff),
        },
        "div_k": float(div_k), "div_v": float(div_v),
        "R": float(R), "sim_delta_count": int(sim_delta_count),
    }, indent=2))

    print(f"Wrote {args.out}\nVerdict: {overall}")
    return 0 if overall == "PROCEED" else 1


def quality_simulator_delta(model, tok, paper_cb, per_layer_k, per_layer_v, n_samples):
    """Run n_samples RULER multivalue prompts twice; count samples where generated
    token sequences differ between paper and per-layer-calibrated codebooks.

    Implementation: use flashquest.eval.niah to generate the same 3 multivalue
    prompts; for each, run a short greedy decode (max_new_tokens=32) twice:
    once with paper K_TURBO_CODEBOOK + V_TURBO_CODEBOOK monkey-patched in
    flashquest.kernel.kv_quant, once with per-layer-mean as a single codebook
    swapped in (proxy for the full per-layer; the reference path doesn't accept
    per-layer codebooks yet — that's Task 4). Compare generated token id lists.
    """
    import flashquest.kernel.kv_quant as kvq
    import importlib

    diff_count = 0
    samples = generate_multivalue_samples(n=n_samples, ctx_len=4096, seed=7)
    proxy_cb_k = torch.tensor(per_layer_k.mean(axis=0), dtype=torch.float32, device="cuda")
    proxy_cb_v = torch.tensor(per_layer_v.mean(axis=0), dtype=torch.float32, device="cuda")
    original_K = kvq.K_TURBO_CODEBOOK.clone()
    original_V = kvq.V_TURBO_CODEBOOK.clone()

    def gen(input_ids):
        out = model.generate(input_ids, max_new_tokens=32, do_sample=False,
                             use_cache=True, logits_to_keep=1)
        return out[0, input_ids.shape[-1]:].tolist()

    try:
        for s in samples:
            input_ids = tok(s["prompt"], return_tensors="pt",
                            truncation=True, max_length=4096).input_ids.cuda()
            kvq.K_TURBO_CODEBOOK.copy_(original_K)
            kvq.V_TURBO_CODEBOOK.copy_(original_V)
            paper_gen = gen(input_ids)

            kvq.K_TURBO_CODEBOOK.copy_(proxy_cb_k)
            kvq.V_TURBO_CODEBOOK.copy_(proxy_cb_v)
            calib_gen = gen(input_ids)

            if paper_gen != calib_gen:
                diff_count += 1
    finally:
        kvq.K_TURBO_CODEBOOK.copy_(original_K)
        kvq.V_TURBO_CODEBOOK.copy_(original_V)

    return diff_count


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the probe**

Run: `nice -n 19 .venv/bin/python scripts/phase11_calibrate_probe.py --essays 3 --ctx 8192`

Expected wall: ~25-30 min on the dev box (3 × 8 k prefills + 28 × 2 = 56 k-means fits + 3 × 2 generations).

- [ ] **Step 3: Read the verdict**

Open `benchmarks/phase11/probe.md`. If verdict is `KILL`, **stop the plan**: copy the probe content into `docs/PHASES/phase-11-killed-by-probe.md`, commit, and update `DOC.md` with the phase-11-killed entry. Do not proceed to Task 2.

If verdict is `PROCEED`, continue.

- [ ] **Step 4: Commit probe + output**

```bash
git add scripts/phase11_calibrate_probe.py benchmarks/phase11/
git commit -m "phase 11 task 1: profile-first probe (entry gate) — verdict {PROCEED|KILL}"
```

---

## Task 2: Codebook loader API

**Files:**
- Create: `src/flashquest/turbo/__init__.py`
- Create: `src/flashquest/turbo/codebook.py`
- Test: `tests/test_turbo_codebook_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_turbo_codebook_loader.py`:

```python
"""Phase 11 — codebook loader tests."""
from __future__ import annotations

import pytest
import torch


def test_load_codebook_returns_correct_shape():
    """Llama-3.2-3B-AWQ codebook ships at (28, 2, 8) fp32."""
    from flashquest.turbo.codebook import load_codebook

    cb = load_codebook("casperhansen/llama-3.2-3b-instruct-awq")
    assert cb.shape == (28, 2, 8), f"got {tuple(cb.shape)}"
    assert cb.dtype == torch.float32
    # Each (layer, kv) row must be sorted ascending
    for li in range(28):
        for kv in range(2):
            assert torch.all(cb[li, kv, :-1] <= cb[li, kv, 1:]), \
                f"layer {li} kv {kv} not sorted: {cb[li, kv].tolist()}"


def test_load_codebook_unknown_model_raises_keyerror():
    """Models without a shipped artifact raise KeyError cleanly."""
    from flashquest.turbo.codebook import load_codebook

    with pytest.raises(KeyError):
        load_codebook("nonexistent/model-id")


def test_paper_codebook_constant():
    """The paper codebook is exposed for fallback callers; matches kv_quant.py."""
    from flashquest.turbo.codebook import PAPER_CODEBOOK
    from flashquest.kernel.kv_quant import K_TURBO_CODEBOOK

    assert torch.allclose(PAPER_CODEBOOK.cpu(), K_TURBO_CODEBOOK.cpu())
    assert PAPER_CODEBOOK.shape == (8,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_turbo_codebook_loader.py -v`

Expected: FAIL — `flashquest.turbo` doesn't exist yet.

- [ ] **Step 3: Implement the loader**

Create `src/flashquest/turbo/__init__.py`:

```python
"""TurboQuant calibration artifacts + loader (Phase 11)."""
from .codebook import load_codebook, PAPER_CODEBOOK

__all__ = ["load_codebook", "PAPER_CODEBOOK"]
```

Create `src/flashquest/turbo/codebook.py`:

```python
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
    """Return (num_layers, 2, 8) fp32 codebook for model_id.

    Raises:
        KeyError if no calibration artifact ships for the given model_id.
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
```

- [ ] **Step 4: Run test to verify the unknown-model + paper-codebook tests pass; first test still fails (no artifact yet)**

Run: `.venv/bin/python -m pytest tests/test_turbo_codebook_loader.py -v`

Expected: 2 PASS (`unknown_model`, `paper_codebook`), 1 FAIL (`load_codebook_returns_correct_shape` — artifact not generated yet; that's Task 3). Leave the failing test as-is; it will pass after Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/turbo/ tests/test_turbo_codebook_loader.py
git commit -m "phase 11 task 2: codebook loader API + unknown-model/paper tests"
```

---

## Task 3: Full calibration script + artifact + monotonicity test

**Files:**
- Create: `scripts/phase11_calibrate_codebook.py`
- Create (output): `src/flashquest/turbo/codebook_llama_3_2_3b.pt`
- Create (output): `src/flashquest/turbo/codebook_llama_3_2_3b.json` (sidecar metadata)
- Test: `tests/test_calibrate_codebook_monotonic.py`

- [ ] **Step 1: Write the monotonicity test**

Create `tests/test_calibrate_codebook_monotonic.py`:

```python
"""Phase 11 — calibrated codebook artifact invariants."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def test_llama_3_2_3b_codebook_artifact_exists():
    """Calibration artifact ships in the package."""
    p = Path(__file__).parents[1] / "src/flashquest/turbo/codebook_llama_3_2_3b.pt"
    assert p.exists(), f"Run scripts/phase11_calibrate_codebook.py to generate {p}"


def test_llama_3_2_3b_codebook_monotonic():
    """Every (layer, K|V) row is sorted ascending. Critical for bit-split decode."""
    from flashquest.turbo.codebook import load_codebook

    cb = load_codebook("casperhansen/llama-3.2-3b-instruct-awq")
    diffs = cb[:, :, 1:] - cb[:, :, :-1]
    bad = (diffs < 0).nonzero()
    assert len(bad) == 0, f"non-monotonic codepoints at {bad.tolist()}"


def test_llama_3_2_3b_codebook_sidecar_metadata():
    """Sidecar JSON records calibration provenance."""
    p = Path(__file__).parents[1] / "src/flashquest/turbo/codebook_llama_3_2_3b.json"
    assert p.exists()
    meta = json.loads(p.read_text())
    assert meta["model_id"] == "casperhansen/llama-3.2-3b-instruct-awq"
    assert meta["corpus"] == "PaulGrahamEssays"
    assert meta["n_tokens_per_layer"] >= 50_000
    assert "calibration_commit" in meta
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_calibrate_codebook_monotonic.py -v`

Expected: FAIL — artifact does not exist yet.

- [ ] **Step 3: Write the calibration script**

Create `scripts/phase11_calibrate_codebook.py`:

```python
"""Phase 11 — full per-layer codebook calibration for TurboQuant K3-V3.

Mirrors the probe but at full scale (≥10 essays, ≥50 k cached tokens per layer).
Writes:
  - src/flashquest/turbo/codebook_<model>.pt   (num_layers, 2, 8) fp32
  - src/flashquest/turbo/codebook_<model>.json sidecar metadata

Usage:
  python scripts/phase11_calibrate_codebook.py
    --model casperhansen/llama-3.2-3b-instruct-awq
    --essays 12 --ctx 8192
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from flashquest.kernel.kv_quant import K_TURBO_CODEBOOK
from flashquest.kernel.wht import wht_along_head_dim
from flashquest.runtime.awq_load import load_awq_model


_MODEL_ID_TO_FILENAME = {
    "casperhansen/llama-3.2-3b-instruct-awq": "codebook_llama_3_2_3b",
}


def capture_activations(model, prompts, num_layers, head_dim):
    """Same hook strategy as the probe; targets >= 50k tokens per layer."""
    acts = {li: {"K": [], "V": []} for li in range(num_layers)}
    handles = []
    for module in model.modules():
        if hasattr(module, "layer_idx") and hasattr(module, "k_proj"):
            li = module.layer_idx
            def make_hook(layer_idx, which):
                def hook(mod, inp, out):
                    B, S, total = out.shape
                    H_kv = total // head_dim
                    acts[layer_idx][which].append(
                        out.detach().reshape(B, S, H_kv, head_dim).cpu().float()
                    )
                return hook
            handles.append(module.k_proj.register_forward_hook(make_hook(li, "K")))
            handles.append(module.v_proj.register_forward_hook(make_hook(li, "V")))

    with torch.no_grad():
        for prompt_ids in prompts:
            model(prompt_ids, use_cache=False)

    for h in handles:
        h.remove()
    return acts


def fit_per_layer_codebook_kv(acts_layer_kv, paper_cb, n_clusters=8):
    """Apply Phase 7 transform (per-token RMS + WHT), then k-means warm-started."""
    stacked = torch.cat(acts_layer_kv, dim=1).squeeze(0)  # (S_total, H_kv, D)
    rms = stacked.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    normalized = stacked.float() / rms
    rotated = wht_along_head_dim(normalized.unsqueeze(0)).squeeze(0)
    flat = rotated.reshape(-1).numpy().astype(np.float32)
    km = KMeans(n_clusters=n_clusters, init=paper_cb.reshape(-1, 1),
                n_init=1, max_iter=500, random_state=0)
    km.fit(flat.reshape(-1, 1))
    sorted_cb = np.sort(km.cluster_centers_.flatten()).astype(np.float32)
    residual_rms = float(np.sqrt(km.inertia_ / flat.size))
    n_tokens = stacked.shape[0] * stacked.shape[1]
    return sorted_cb, residual_rms, n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--essays", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=8192)
    args = ap.parse_args()

    if args.model not in _MODEL_ID_TO_FILENAME:
        raise SystemExit(f"Unknown model {args.model}; add to _MODEL_ID_TO_FILENAME.")
    stem = _MODEL_ID_TO_FILENAME[args.model]

    model, tok = load_awq_model(args.model)
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    paper_cb = K_TURBO_CODEBOOK.cpu().numpy().astype(np.float32)

    essays = json.loads(Path("data/PaulGrahamEssays.json").read_text())["essays"][:args.essays]
    prompts = []
    for essay in essays:
        ids = tok(essay["text"], return_tensors="pt", truncation=True,
                  max_length=args.ctx).input_ids.cuda()
        prompts.append(ids)

    print(f"[calibrate] capturing activations from {len(prompts)} essays at ctx={args.ctx}")
    acts = capture_activations(model, prompts, num_layers, head_dim)

    print(f"[calibrate] fitting per-layer K + V codebooks for {num_layers} layers ...")
    cb = np.zeros((num_layers, 2, 8), dtype=np.float32)
    residuals = np.zeros((num_layers, 2), dtype=np.float32)
    n_tokens_each = np.zeros((num_layers, 2), dtype=np.int64)
    for li in range(num_layers):
        cb_k, r_k, n_k = fit_per_layer_codebook_kv(acts[li]["K"], paper_cb)
        cb_v, r_v, n_v = fit_per_layer_codebook_kv(acts[li]["V"], paper_cb)
        cb[li, 0] = cb_k
        cb[li, 1] = cb_v
        residuals[li, 0] = r_k
        residuals[li, 1] = r_v
        n_tokens_each[li] = (n_k, n_v)
        print(f"  layer {li:2d}: K_residual_rms={r_k:.4f}  V_residual_rms={r_v:.4f}")

    cb_t = torch.from_numpy(cb).float()
    out_pt = Path(f"src/flashquest/turbo/{stem}.pt")
    torch.save(cb_t, out_pt)

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    meta = {
        "model_id": args.model,
        "corpus": "PaulGrahamEssays",
        "n_essays": args.essays,
        "ctx": args.ctx,
        "n_tokens_per_layer": int(n_tokens_each.min()),
        "calibration_commit": commit,
        "residual_rms_per_layer": residuals.tolist(),
    }
    out_json = out_pt.with_suffix(".json")
    out_json.write_text(json.dumps(meta, indent=2))

    print(f"\n[calibrate] wrote {out_pt} ({out_pt.stat().st_size} bytes)")
    print(f"[calibrate] wrote {out_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the calibration**

Run: `nice -n 19 .venv/bin/python scripts/phase11_calibrate_codebook.py --essays 12 --ctx 8192`

Expected wall: ~25 min. Watch for any layer's `residual_rms` > 0.5 — that signals a layer where the codebook is a poor fit (kept as informational; not a blocker).

- [ ] **Step 5: Run the monotonicity tests**

Run: `.venv/bin/python -m pytest tests/test_calibrate_codebook_monotonic.py tests/test_turbo_codebook_loader.py -v`

Expected: ALL PASS now (artifact exists, sorted, sidecar in place).

- [ ] **Step 6: Commit**

```bash
git add scripts/phase11_calibrate_codebook.py src/flashquest/turbo/codebook_llama_3_2_3b.pt src/flashquest/turbo/codebook_llama_3_2_3b.json tests/test_calibrate_codebook_monotonic.py
git commit -m "phase 11 task 3: full calibration script + Llama-3.2-3B artifact (per-layer K+V) + monotonicity test"
```

---

## Task 4: Parameterize quant/dequant helpers (`kv_quant.py`)

The quant + dequant helpers currently use the module-level `K_TURBO_CODEBOOK` / `V_TURBO_CODEBOOK` constants. Add an optional `codebook=` parameter to each; defaults preserve Phase 7 behavior.

**Files:**
- Modify: `src/flashquest/kernel/kv_quant.py:327-420`
- Test: `tests/test_kv_quant.py` (extend with calibrated-codebook tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_kv_quant.py`:

```python
def test_quantize_k_turbo_accepts_custom_codebook():
    """quantize_k_turbo(K, codebook=cb) round-trips through dequant with cb."""
    import torch
    from flashquest.kernel.kv_quant import quantize_k_turbo, dequantize_k_turbo

    torch.manual_seed(7)
    K = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device="cuda")
    cb = torch.tensor(
        [-2.0, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 2.0],
        dtype=torch.float32, device="cuda",
    )

    K_msb, K_lsb, K_scale_t, K_scale_r, K_mn_r = quantize_k_turbo(
        K, page_size=64, codebook=cb,
    )
    K_rt = dequantize_k_turbo(K_msb, K_lsb, K_scale_t, head_dim=64, codebook=cb)

    # Round-trip residual vs the paper-codebook round-trip should be similar order.
    paper_K_rt = dequantize_k_turbo(
        *quantize_k_turbo(K, page_size=64)[:3], head_dim=64,
    )
    custom_err = (K.float() - K_rt.float()).abs().mean()
    paper_err = (K.float() - paper_K_rt.float()).abs().mean()
    # Custom codebook isn't optimal for random Gaussian, so error can be higher,
    # but should not blow up; require within 2× of paper baseline.
    assert custom_err < 2 * paper_err, f"custom={custom_err:.4f} paper={paper_err:.4f}"


def test_quantize_v_turbo_accepts_custom_codebook():
    """Symmetric to K case."""
    import torch
    from flashquest.kernel.kv_quant import quantize_v_turbo, dequantize_v_turbo

    torch.manual_seed(7)
    V = torch.randn(1, 8, 64, 64, dtype=torch.bfloat16, device="cuda")
    cb = torch.tensor(
        [-2.0, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 2.0],
        dtype=torch.float32, device="cuda",
    )

    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V, codebook=cb)
    V_rt = dequantize_v_turbo(V_msb, V_lsb, V_scale_t, head_dim=64, codebook=cb)
    paper_V_rt = dequantize_v_turbo(*quantize_v_turbo(V)[:3], head_dim=64)
    custom_err = (V.float() - V_rt.float()).abs().mean()
    paper_err = (V.float() - paper_V_rt.float()).abs().mean()
    assert custom_err < 2 * paper_err
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_kv_quant.py::test_quantize_k_turbo_accepts_custom_codebook tests/test_kv_quant.py::test_quantize_v_turbo_accepts_custom_codebook -v`

Expected: FAIL — `codebook=` not yet accepted.

- [ ] **Step 3: Modify quant + dequant helpers**

Edit `src/flashquest/kernel/kv_quant.py`:

```python
# Update quantize_k_turbo signature + body (around line 327):
def quantize_k_turbo(K: torch.Tensor, page_size: int, codebook: Optional[torch.Tensor] = None):
    """TurboQuant K: WHT → per-token scale → 3-bit Lloyd-Max → bit-split pack.

    Args:
        K: (B, H, S, D) bf16 cuda.
        page_size: pages for criticality raw-stats.
        codebook: (8,) fp32 cuda. Defaults to K_TURBO_CODEBOOK (paper).
    """
    from flashquest.kernel.wht import wht_along_head_dim

    cb = K_TURBO_CODEBOOK if codebook is None else codebook
    if cb.shape != (8,):
        raise ValueError(f"quantize_k_turbo codebook must be (8,); got {tuple(cb.shape)}")

    B, H, S, D = K.shape
    if D % 8 != 0:
        raise ValueError(f"quantize_k_turbo requires head_dim multiple of 8; got {D}")

    K_rot = wht_along_head_dim(K)
    K_rms = K_rot.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    K_scale_turbo = K_rms.clamp_min(_EPS)
    K_normalized = K_rot.float() / K_scale_turbo
    K_idx = _quantize_to_codebook(K_normalized, cb)
    K_msb, K_lsb = _pack_bit_split(K_idx)
    K_scale_raw, K_mn_raw = _scale_mn_per_page_channel_int4(K, page_size)
    return (
        K_msb, K_lsb,
        K_scale_turbo.to(torch.bfloat16),
        K_scale_raw.to(torch.bfloat16),
        K_mn_raw.to(torch.bfloat16),
    )


def dequantize_k_turbo(
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    head_dim: int,
    codebook: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of quantize_k_turbo. Accepts custom codebook (paper default)."""
    from flashquest.kernel.wht import wht_along_head_dim

    cb = K_TURBO_CODEBOOK if codebook is None else codebook
    K_idx = _unpack_bit_split(K_msb, K_lsb, head_dim=head_dim)
    K_rot = cb[K_idx.long()] * K_scale_turbo.float()
    K = wht_along_head_dim(K_rot)
    return K.to(torch.bfloat16)


def quantize_v_turbo(V: torch.Tensor, codebook: Optional[torch.Tensor] = None):
    """K3-V3 V quant with optional codebook override."""
    from flashquest.kernel.wht import wht_along_head_dim

    cb = V_TURBO_CODEBOOK if codebook is None else codebook
    if cb.shape != (8,):
        raise ValueError(f"quantize_v_turbo codebook must be (8,); got {tuple(cb.shape)}")

    B, H, S, D = V.shape
    if D % 8 != 0:
        raise ValueError(f"quantize_v_turbo requires head_dim multiple of 8; got {D}")
    V_rot = wht_along_head_dim(V)
    V_rms = V_rot.float().pow(2).mean(dim=-1, keepdim=True).sqrt()
    V_scale_turbo = V_rms.clamp_min(_EPS)
    V_normalized = V_rot.float() / V_scale_turbo
    V_idx = _quantize_to_codebook(V_normalized, cb)
    V_msb, V_lsb = _pack_bit_split(V_idx)
    return V_msb, V_lsb, V_scale_turbo.to(torch.bfloat16)


def dequantize_v_turbo(
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    head_dim: int,
    codebook: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of quantize_v_turbo (K3-V3) with optional codebook override."""
    from flashquest.kernel.wht import wht_along_head_dim

    cb = V_TURBO_CODEBOOK if codebook is None else codebook
    V_idx = _unpack_bit_split(V_msb, V_lsb, head_dim=head_dim)
    V_rot = cb[V_idx.long()] * V_scale_turbo.float()
    V = wht_along_head_dim(V_rot)
    return V.to(torch.bfloat16)
```

Make sure `from typing import Optional` is imported at top of file (check existing imports first — likely already present).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_kv_quant.py -v`

Expected: ALL PASS (existing tests still pass with `codebook=None` default; new tests pass).

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/kernel/kv_quant.py tests/test_kv_quant.py
git commit -m "phase 11 task 4: parameterize quantize/dequantize_{k,v}_turbo with optional codebook"
```

---

## Task 5: Parameterize Triton kernel + Python wrapper (`sparse_turbo_fwd.py`)

The kernel currently calls `_codebook_lookup_3bit(idx)` with codepoints baked into the function. Add 8 K-codepoint + 8 V-codepoint `tl.constexpr` arguments to the kernel; replace the hardcoded `_codebook_lookup_3bit` with `_codebook_lookup_param`. The wrapper accepts optional `codebook_k`, `codebook_v` tensors.

**Files:**
- Modify: `src/flashquest/kernel/sparse_turbo_fwd.py`
- Test: `tests/test_sparse_turbo_calibrated.py` (new file)

- [ ] **Step 1: Write the failing parity test**

Create `tests/test_sparse_turbo_calibrated.py`:

```python
"""Phase 11 — fused TurboQuant kernel parity under custom codebooks."""
from __future__ import annotations

import math

import pytest
import torch


def _make_quantized_kv(B, H_kv, S, D, codebook_k, codebook_v, page_size, seed):
    """Build (K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t) using custom cbs."""
    from flashquest.kernel.kv_quant import quantize_k_turbo, quantize_v_turbo
    torch.manual_seed(seed)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16, device="cuda")
    K_msb, K_lsb, K_scale_t, _, _ = quantize_k_turbo(K, page_size=page_size, codebook=codebook_k)
    V_msb, V_lsb, V_scale_t = quantize_v_turbo(V, codebook=codebook_v)
    return K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sparse_turbo_fwd_parity_custom_codebook(seed):
    """Fused kernel + reference path bit-equivalent when both use the same custom codebook."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd, _flash_attn_sparse_turbo_fwd_reference,
    )

    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    page_size = 64
    num_pages = S // page_size

    # Random codebook (sorted ascending; required by bit-split invariant).
    torch.manual_seed(seed)
    raw = torch.randn(8, device="cuda").float()
    cb_k = torch.sort(raw)[0]
    raw_v = torch.randn(8, device="cuda").float()
    cb_v = torch.sort(raw_v)[0]

    K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t = _make_quantized_kv(
        B, H_kv, S, D, cb_k, cb_v, page_size, seed
    )

    torch.manual_seed(seed + 100)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_fused, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
        codebook_k=cb_k, codebook_v=cb_v,
    )
    O_ref, _ = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
        codebook_k=cb_k, codebook_v=cb_v,
    )
    err = (O_fused.float() - O_ref.float()).abs().max()
    assert err < 5e-2, f"max abs err {err}"


def test_sparse_turbo_fwd_paper_default_unchanged():
    """Phase 7 behavior preserved when no codebook is passed (default = paper)."""
    from flashquest.kernel.sparse_turbo_fwd import (
        flash_attn_sparse_turbo_fwd, _flash_attn_sparse_turbo_fwd_reference,
    )

    B, H_q, H_kv, S, D = 1, 8, 2, 256, 64
    page_size = 64
    num_pages = S // page_size
    seed = 7

    K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t = _make_quantized_kv(
        B, H_kv, S, D, None, None, page_size, seed,
    )

    torch.manual_seed(seed + 100)
    Q = torch.randn(B, H_q, 1, D, dtype=torch.bfloat16, device="cuda")
    sel = torch.ones(B, H_q, 1, num_pages, dtype=torch.bool, device="cuda")

    O_fused, _ = flash_attn_sparse_turbo_fwd(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
    )
    O_ref, _ = _flash_attn_sparse_turbo_fwd_reference(
        Q, K_msb, K_lsb, K_scale_t, V_msb, V_lsb, V_scale_t,
        selection_mask=sel, page_size=page_size,
    )
    err = (O_fused.float() - O_ref.float()).abs().max()
    assert err < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_sparse_turbo_calibrated.py -v`

Expected: FAIL — `codebook_k=` not accepted by wrapper or reference path.

- [ ] **Step 3: Add codebook parameters to the reference path**

Edit `src/flashquest/kernel/sparse_turbo_fwd.py:26-83` (the `_flash_attn_sparse_turbo_fwd_reference` function):

```python
def _flash_attn_sparse_turbo_fwd_reference(
    Q: torch.Tensor,
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
    codebook_k: Optional[torch.Tensor] = None,
    codebook_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reference path with optional per-layer codebooks (Phase 11)."""
    if Q.dim() != 4:
        raise ValueError(f"Q must be 4D (B, H_q, 1, D); got {Q.shape}")

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"decode-only (S_q={S_q})")

    Bk, H_kv, S_kv, _ = K_msb.shape
    n_rep = H_q // H_kv

    K_full = dequantize_k_turbo(K_msb, K_lsb, K_scale_turbo, head_dim=D, codebook=codebook_k)
    V_full = dequantize_v_turbo(V_msb, V_lsb, V_scale_turbo, head_dim=D, codebook=codebook_v)

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    page_idx = torch.arange(S_kv, device=Q.device) // page_size
    sel = selection_mask[..., page_idx]
    attn_bias = torch.where(sel, 0.0, float("-inf")).float()

    K_rep = K_full.repeat_interleave(n_rep, dim=1)
    V_rep = V_full.repeat_interleave(n_rep, dim=1)

    qk = (Q.float() @ K_rep.float().transpose(-1, -2)) * sm_scale + attn_bias
    m = qk.max(dim=-1, keepdim=True).values
    p = torch.exp(qk - m)
    l = p.sum(dim=-1, keepdim=True)
    O = (p @ V_rep.float()) / l

    lse = None
    if return_lse:
        lse = (m + torch.log(l)).squeeze(-1).to(torch.float32)
    return O.to(torch.bfloat16), lse
```

- [ ] **Step 4: Replace `_codebook_lookup_3bit` with parameterized version**

In `src/flashquest/kernel/sparse_turbo_fwd.py`, replace the hardcoded `_codebook_lookup_3bit` (lines 92-107) with two helpers — one for K, one for V — each taking 8 codepoints as constexpr. Triton's `@triton.jit` propagates constexpr scalars through call boundaries cleanly:

```python
@triton.jit
def _codebook_lookup_param(idx, CP0: tl.constexpr, CP1: tl.constexpr, CP2: tl.constexpr,
                           CP3: tl.constexpr, CP4: tl.constexpr, CP5: tl.constexpr,
                           CP6: tl.constexpr, CP7: tl.constexpr):
    """Map uint8 idx ∈ {0..7} → fp32 codepoint. Codepoints passed as constexpr."""
    return tl.where(idx == 0, CP0,
           tl.where(idx == 1, CP1,
           tl.where(idx == 2, CP2,
           tl.where(idx == 3, CP3,
           tl.where(idx == 4, CP4,
           tl.where(idx == 5, CP5,
           tl.where(idx == 6, CP6,
                              CP7)))))))
```

- [ ] **Step 5: Update the Triton kernel signature + call sites**

Edit `_sparse_attn_fwd_kernel_turbo` (lines 110-251 in `sparse_turbo_fwd.py`): add 16 `tl.constexpr` arguments (K codepoints + V codepoints) right after `WRITE_LSE: tl.constexpr,`. Replace the two existing `_codebook_lookup_3bit(...)` calls (lines 190 and 230) with `_codebook_lookup_param(k_idx, K_CP0, K_CP1, ..., K_CP7)` and `_codebook_lookup_param(v_idx, V_CP0, ..., V_CP7)`. Full edited fragment around the kernel signature:

```python
@triton.jit
def _sparse_attn_fwd_kernel_turbo(
    Q_rot_ptr,
    K_msb_ptr, K_lsb_ptr, V_msb_ptr, V_lsb_ptr,
    O_rot_ptr, L_ptr,
    K_scale_t_ptr, V_scale_t_ptr,
    sel_ptr,
    sm_scale,
    stride_qb, stride_qh, stride_qd,
    stride_kmb, stride_kmh, stride_kms, stride_kmd,
    stride_klb, stride_klh, stride_kls, stride_kld,
    stride_vmb, stride_vmh, stride_vms, stride_vmd,
    stride_vlb, stride_vlh, stride_vls, stride_vld,
    stride_ob, stride_oh, stride_od,
    stride_lb, stride_lh,
    stride_kstb, stride_ksth, stride_ksts,
    stride_vstb, stride_vsth, stride_vsts,
    stride_selb, stride_selh, stride_selp,
    H_q, H_kv, S_kv, NUM_PAGES,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_MSB: tl.constexpr,
    HEAD_DIM_LSB: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WRITE_LSE: tl.constexpr,
    K_CP0: tl.constexpr, K_CP1: tl.constexpr, K_CP2: tl.constexpr, K_CP3: tl.constexpr,
    K_CP4: tl.constexpr, K_CP5: tl.constexpr, K_CP6: tl.constexpr, K_CP7: tl.constexpr,
    V_CP0: tl.constexpr, V_CP1: tl.constexpr, V_CP2: tl.constexpr, V_CP3: tl.constexpr,
    V_CP4: tl.constexpr, V_CP5: tl.constexpr, V_CP6: tl.constexpr, V_CP7: tl.constexpr,
):
    """Decode-only sparse forward, TurboQuant K3-V3 with parameterized codebooks."""
    # ... existing body unchanged through k_idx assembly ...
    k_idx = (k_msb_full.to(tl.int32) << 2) | k_lsb_full.to(tl.int32)
    k_rot = _codebook_lookup_param(k_idx, K_CP0, K_CP1, K_CP2, K_CP3, K_CP4, K_CP5, K_CP6, K_CP7)
    # ... rest unchanged through V section ...
    v_idx = (v_msb_full.to(tl.int32) << 2) | v_lsb_full.to(tl.int32)
    v_rot = _codebook_lookup_param(v_idx, V_CP0, V_CP1, V_CP2, V_CP3, V_CP4, V_CP5, V_CP6, V_CP7)
    # ... rest unchanged.
```

Delete the old `_codebook_lookup_3bit` helper (now unused).

- [ ] **Step 6: Update the Python wrapper**

Edit `flash_attn_sparse_turbo_fwd` (lines 254-346): add `codebook_k`, `codebook_v` kwargs; unpack into 16 floats; pass to kernel launch:

```python
def flash_attn_sparse_turbo_fwd(
    Q: torch.Tensor,
    K_msb: torch.Tensor,
    K_lsb: torch.Tensor,
    K_scale_turbo: torch.Tensor,
    V_msb: torch.Tensor,
    V_lsb: torch.Tensor,
    V_scale_turbo: torch.Tensor,
    *,
    selection_mask: torch.Tensor,
    page_size: int = 64,
    sm_scale: Optional[float] = None,
    return_lse: bool = True,
    codebook_k: Optional[torch.Tensor] = None,
    codebook_v: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Decode-only fused TurboQuant sparse forward (K3-V3) with optional per-layer codebooks."""
    assert Q.is_cuda and Q.dtype == torch.bfloat16
    assert K_msb.dtype == torch.uint8 and K_lsb.dtype == torch.uint8
    assert V_msb.dtype == torch.uint8 and V_lsb.dtype == torch.uint8

    B, H_q, S_q, D = Q.shape
    if S_q != 1:
        raise NotImplementedError(f"flash_attn_sparse_turbo_fwd: decode-only (S_q={S_q})")
    if D not in _SUPPORTED_HEAD_DIMS:
        raise NotImplementedError(f"head_dim={D} not in {_SUPPORTED_HEAD_DIMS}")

    Bk, H_kv, S_kv, _ = K_msb.shape
    assert B == Bk
    assert H_q % H_kv == 0
    num_pages = selection_mask.shape[-1]
    assert selection_mask.shape == (B, H_q, 1, num_pages)
    assert selection_mask.dtype == torch.bool

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    # Resolve codebooks (paper default).
    cb_k = K_TURBO_CODEBOOK if codebook_k is None else codebook_k
    cb_v = V_TURBO_CODEBOOK if codebook_v is None else codebook_v
    if cb_k.shape != (8,) or cb_v.shape != (8,):
        raise ValueError(
            f"codebooks must be (8,); got K={tuple(cb_k.shape)}, V={tuple(cb_v.shape)}"
        )
    cb_k_list = cb_k.detach().cpu().float().tolist()
    cb_v_list = cb_v.detach().cpu().float().tolist()

    Q_rot = wht_along_head_dim(Q)
    Q_2d = Q_rot.squeeze(2).contiguous()
    O_rot_2d = torch.zeros_like(Q_2d)

    L = torch.empty(B, H_q, dtype=torch.float32, device=Q.device) if return_lse else None
    L_ptr = L if L is not None else torch.empty(0, device=Q.device, dtype=torch.float32)
    sl_b, sl_h = (L.stride() if L is not None else (0, 0))

    sel_2d = selection_mask.squeeze(2)

    grid = (B * H_q,)
    _sparse_attn_fwd_kernel_turbo[grid](
        Q_2d,
        K_msb, K_lsb, V_msb, V_lsb,
        O_rot_2d, L_ptr,
        K_scale_turbo, V_scale_turbo,
        sel_2d,
        sm_scale,
        Q_2d.stride(0), Q_2d.stride(1), Q_2d.stride(2),
        K_msb.stride(0), K_msb.stride(1), K_msb.stride(2), K_msb.stride(3),
        K_lsb.stride(0), K_lsb.stride(1), K_lsb.stride(2), K_lsb.stride(3),
        V_msb.stride(0), V_msb.stride(1), V_msb.stride(2), V_msb.stride(3),
        V_lsb.stride(0), V_lsb.stride(1), V_lsb.stride(2), V_lsb.stride(3),
        O_rot_2d.stride(0), O_rot_2d.stride(1), O_rot_2d.stride(2),
        sl_b, sl_h,
        K_scale_turbo.stride(0), K_scale_turbo.stride(1), K_scale_turbo.stride(2),
        V_scale_turbo.stride(0), V_scale_turbo.stride(1), V_scale_turbo.stride(2),
        sel_2d.stride(0), sel_2d.stride(1), sel_2d.stride(2),
        H_q, H_kv, S_kv, num_pages,
        HEAD_DIM=D,
        HEAD_DIM_MSB=D // 8,
        HEAD_DIM_LSB=D // 4,
        PAGE_SIZE=page_size,
        WRITE_LSE=bool(return_lse),
        K_CP0=cb_k_list[0], K_CP1=cb_k_list[1], K_CP2=cb_k_list[2], K_CP3=cb_k_list[3],
        K_CP4=cb_k_list[4], K_CP5=cb_k_list[5], K_CP6=cb_k_list[6], K_CP7=cb_k_list[7],
        V_CP0=cb_v_list[0], V_CP1=cb_v_list[1], V_CP2=cb_v_list[2], V_CP3=cb_v_list[3],
        V_CP4=cb_v_list[4], V_CP5=cb_v_list[5], V_CP6=cb_v_list[6], V_CP7=cb_v_list[7],
        num_warps=4,
        num_stages=2,
    )

    O_rot = O_rot_2d.unsqueeze(2)
    O = wht_along_head_dim(O_rot)
    L_out = L.unsqueeze(2) if L is not None else None
    return O, L_out
```

- [ ] **Step 7: Run parity tests**

Run: `.venv/bin/python -m pytest tests/test_sparse_turbo_calibrated.py tests/test_sparse_turbo.py -v`

Expected: 4 PASS (3 random-codebook seeds + the paper-default-unchanged test); existing `test_sparse_turbo.py` continues to pass (paper default preserved through `codebook_k=None`).

- [ ] **Step 8: Commit**

```bash
git add src/flashquest/kernel/sparse_turbo_fwd.py tests/test_sparse_turbo_calibrated.py
git commit -m "phase 11 task 5: parameterize sparse_turbo_fwd kernel + wrapper with codebook_k / codebook_v"
```

---

## Task 6: Cache codebook fields + dispatcher per-layer wire

`PersistentTurboKVCache` gains `codebook_k`, `codebook_v` `(num_layers, 8)` tensors loaded from the calibration artifact (with paper fallback). `update_quantized` passes the per-layer slice to `quantize_k_turbo` / `quantize_v_turbo`. `make_quest_persistent_forward` accepts `layer_idx` and threads `cache.codebook_k[layer_idx]`, `cache.codebook_v[layer_idx]` into both the dequant helpers (used by prefill SDPA) and `flash_attn_sparse_turbo_fwd`.

**Files:**
- Modify: `src/flashquest/cache/persistent_turbo.py`
- Modify: `src/flashquest/eager/llama_persistent_patch.py`
- Test: `tests/test_persistent_turbo_calibrated_smoke.py` (new file)
- Modify: `tests/test_persistent_int4.py` or similar — no change needed if existing tests pass with Phase 7 default

- [ ] **Step 1: Write the failing smoke test**

Create `tests/test_persistent_turbo_calibrated_smoke.py`:

```python
"""Phase 11 — PersistentTurboKVCache calibration plumbing smoke (1B fallback)."""
from __future__ import annotations

import pytest
import torch


@pytest.mark.slow
def test_persistent_turbo_calibrated_default_loads_codebook_for_3b():
    """When initialized with model_id of a calibrated model, cache loads the artifact."""
    from flashquest.cache import PersistentTurboKVCache

    cache = PersistentTurboKVCache(
        batch_size=1, num_layers=28, num_kv_heads=8, head_dim=64,
        max_seq_len=512, page_size=64, device="cuda",
        model_id="casperhansen/llama-3.2-3b-instruct-awq",
    )
    assert cache.codebook_k.shape == (28, 8)
    assert cache.codebook_v.shape == (28, 8)
    # Calibrated codebooks should NOT equal paper (otherwise the calibration was a no-op).
    from flashquest.turbo.codebook import PAPER_CODEBOOK
    paper = PAPER_CODEBOOK.to(cache.codebook_k.device)
    assert not torch.allclose(cache.codebook_k, paper.expand(28, 8))


@pytest.mark.slow
def test_persistent_turbo_calibrated_falls_back_to_paper_for_unknown_model():
    """Unknown model_id: cache warns and falls back to paper codebook."""
    from flashquest.cache import PersistentTurboKVCache
    from flashquest.turbo.codebook import PAPER_CODEBOOK

    with pytest.warns(UserWarning, match="paper"):
        cache = PersistentTurboKVCache(
            batch_size=1, num_layers=4, num_kv_heads=2, head_dim=64,
            max_seq_len=128, page_size=64, device="cuda",
            model_id="nonexistent/model-id",
        )

    expected = PAPER_CODEBOOK.to(cache.codebook_k.device).expand(4, 8)
    assert torch.allclose(cache.codebook_k, expected)
    assert torch.allclose(cache.codebook_v, expected)


@pytest.mark.slow
def test_persistent_turbo_calibrated_quant_uses_per_layer_codebook():
    """update_quantized() routes the per-layer codebook into quantize_k/v_turbo."""
    import torch
    from flashquest.cache import PersistentTurboKVCache

    cache = PersistentTurboKVCache(
        batch_size=1, num_layers=2, num_kv_heads=2, head_dim=64,
        max_seq_len=128, page_size=64, device="cuda",
        model_id="casperhansen/llama-3.2-3b-instruct-awq",
    )
    torch.manual_seed(0)
    K_new = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")
    V_new = torch.randn(1, 2, 64, 64, dtype=torch.bfloat16, device="cuda")

    # Write to layer 0 with its codebook
    cache.update_quantized(K_new, V_new, layer_idx=0)

    # Verify: dequant of layer 0 cache with layer-0 codebook returns sensible K.
    from flashquest.kernel.kv_quant import dequantize_k_turbo
    K_rt = dequantize_k_turbo(
        cache.K_msb[0], cache.K_lsb[0], cache.K_scale_turbo[0],
        head_dim=64, codebook=cache.codebook_k[0],
    )
    # Roundtrip error should be < 0.5 max (lossy quant but bounded).
    assert (K_new.float() - K_rt.float()).abs().mean() < 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_persistent_turbo_calibrated_smoke.py -v -m slow`

Expected: FAIL — `model_id=` not accepted, no `codebook_k` attribute.

- [ ] **Step 3: Modify cache constructor + update_quantized**

Edit `src/flashquest/cache/persistent_turbo.py:22-50` (constructor):

```python
def __init__(
    self,
    *,
    batch_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    page_size: int = 64,
    device: str | torch.device = "cuda",
    model_id: str | None = None,
    codebook: torch.Tensor | None = None,
):
    """PersistentTurboKVCache with optional per-layer calibrated codebook.

    Codebook resolution (in priority):
      1. explicit `codebook` arg (shape (num_layers, 2, 8) fp32)
      2. `load_codebook(model_id)` if `model_id` is set
      3. paper codebook broadcast to (num_layers, 2, 8), with a one-time warn.
    """
    if head_dim % 8 != 0:
        raise ValueError(
            f"PersistentTurboKVCache requires head_dim multiple of 8 "
            f"(MSB plane packs 8/byte); got {head_dim}"
        )
    if head_dim & (head_dim - 1) != 0:
        raise ValueError(
            f"PersistentTurboKVCache requires head_dim power of 2 (WHT); got {head_dim}"
        )
    # ... [existing field assignments unchanged from line 42 through 75] ...

    # Phase 11: resolve codebook.
    import warnings
    from flashquest.turbo.codebook import PAPER_CODEBOOK, load_codebook

    if codebook is not None:
        cb = codebook.to(dev).float()
        assert cb.shape == (num_layers, 2, 8), \
            f"codebook must be (num_layers={num_layers}, 2, 8); got {tuple(cb.shape)}"
    elif model_id is not None:
        try:
            cb = load_codebook(model_id).to(dev).float()
            if cb.shape[0] != num_layers:
                raise ValueError(
                    f"calibrated codebook has {cb.shape[0]} layers; cache configured for {num_layers}"
                )
        except KeyError as e:
            warnings.warn(
                f"No calibrated codebook for {model_id!r}; falling back to paper. {e}",
                UserWarning,
            )
            paper = PAPER_CODEBOOK.to(dev)
            cb = paper.expand(num_layers, 2, 8).clone()
    else:
        paper = PAPER_CODEBOOK.to(dev)
        cb = paper.expand(num_layers, 2, 8).clone()

    self.codebook_k = cb[:, 0, :].contiguous()  # (num_layers, 8)
    self.codebook_v = cb[:, 1, :].contiguous()
```

Edit `update_quantized` (lines 83-140), changing the two quantize calls to pass per-layer codebooks:

```python
            K_msb, K_lsb, K_scale_t, K_scale_r, K_mn_r = quantize_k_turbo(
                K_complete, page_size=page_size,
                codebook=self.codebook_k[layer_idx],
            )
            V_msb, V_lsb, V_scale_t = quantize_v_turbo(
                V_complete, codebook=self.codebook_v[layer_idx],
            )
```

- [ ] **Step 4: Modify dispatcher to thread layer_idx + codebooks**

Edit `src/flashquest/eager/llama_persistent_patch.py`. In `make_quest_persistent_forward` (line 64), add a `layer_idx: int` argument and use it to slice the cache's codebooks:

```python
def make_quest_persistent_forward(
    *,
    cache,
    head_pattern_layer: torch.Tensor,
    retention: float,
    num_sinks: int,
    window_pages: int,
    page_size: int,
    layer_idx: int = 0,
    use_compact_kernel: bool = False,
):
    """... [keep existing docstring] ..."""
    kv_bits = getattr(cache, "kv_bits", 8)
    head_dim = cache.head_dim
    # ... [keep existing max_seq_len + retention_max + use_compact_kernel preamble] ...

    if kv_bits == 3:
        codebook_k_layer = cache.codebook_k[layer_idx]
        codebook_v_layer = cache.codebook_v[layer_idx]

        def _dequant_k_from_views(views):
            return dequantize_k_turbo(
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                head_dim=head_dim, codebook=codebook_k_layer,
            )

        def _dequant_v_from_views(views):
            return dequantize_v_turbo(
                views["V_msb"], views["V_lsb"], views["V_scale_turbo"],
                head_dim=head_dim, codebook=codebook_v_layer,
            )

        def _criticality_scores(q, views):
            return page_scores_int4_fast(q, views["K_scale_raw"], views["K_mn_raw"])

        def _sparse_fwd_call(q, views, sel):
            return flash_attn_sparse_turbo_fwd(
                q,
                views["K_msb"], views["K_lsb"], views["K_scale_turbo"],
                views["V_msb"], views["V_lsb"], views["V_scale_turbo"],
                selection_mask=sel, page_size=page_size, return_lse=True,
                codebook_k=codebook_k_layer, codebook_v=codebook_v_layer,
            )
    # ... [keep elif kv_bits == 4 and elif kv_bits == 8 branches unchanged] ...
```

In `patch_llama_for_quest_persistent` (line 257), pass `layer_idx=li` when constructing the forward closure:

```python
            fwd = make_quest_persistent_forward(
                cache=cache,
                head_pattern_layer=head_pattern[li].to("cuda"),
                retention=retention,
                num_sinks=num_sinks,
                window_pages=window_pages,
                page_size=page_size,
                layer_idx=li,
                use_compact_kernel=use_compact_kernel,
            )
```

- [ ] **Step 5: Run smoke tests**

Run: `.venv/bin/python -m pytest tests/test_persistent_turbo_calibrated_smoke.py -v -m slow`

Expected: 3 PASS.

- [ ] **Step 6: Run regression suite (non-slow)**

Run: `nice -n 19 .venv/bin/python -m pytest -m "not slow" -q`

Expected: ALL PASS (Phase 7 / 8a / 8b / 10 tests continue to pass — paper-codebook default and per-layer broadcast both round-trip correctly).

- [ ] **Step 7: Commit**

```bash
git add src/flashquest/cache/persistent_turbo.py src/flashquest/eager/llama_persistent_patch.py tests/test_persistent_turbo_calibrated_smoke.py
git commit -m "phase 11 task 6: cache codebook_k/v fields + dispatcher per-layer wiring"
```

---

## Task 7: CLI `--codebook` flag

Add `--codebook {calibrated,paper}` to `flashquest chat`. Default `calibrated` for models with a shipped artifact, falls back to `paper` with a one-line stderr note otherwise.

**Files:**
- Modify: `src/flashquest/runtime/chat.py`
- Modify: `tests/test_chat.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_chat.py`:

```python
def test_parse_args_codebook_default_calibrated():
    """Default `--codebook` is `calibrated` (Phase 11)."""
    args = _parse_args([
        "--model", "casperhansen/llama-3.2-3b-instruct-awq",
        "--context", "1024", "-i",
    ])
    assert args.codebook == "calibrated"


def test_parse_args_codebook_paper_opt_in():
    """`--codebook paper` keeps Phase 7 behavior."""
    args = _parse_args([
        "--model", "x", "--context", "1024", "-i", "--codebook", "paper",
    ])
    assert args.codebook == "paper"


def test_parse_args_codebook_invalid_rejected():
    """Anything other than calibrated/paper exits with non-zero."""
    with pytest.raises(SystemExit):
        _parse_args(["--model", "x", "--context", "1024", "-i", "--codebook", "junk"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_chat.py -v`

Expected: 3 FAIL on new tests; existing tests still pass.

- [ ] **Step 3: Add the flag + plumb to cache construction**

Edit `src/flashquest/runtime/chat.py`. In `_parse_args` (around line 30-90), add:

```python
    parser.add_argument(
        "--codebook", choices=("calibrated", "paper"), default="calibrated",
        help="TurboQuant codebook for --kv-bits 3. 'calibrated' = per-layer fit "
             "to the model's activations (Phase 11); 'paper' = data-oblivious "
             "Lloyd-Max (Phase 7). Default 'calibrated'.",
    )
```

In the body where `PersistentTurboKVCache` is instantiated (around line 220, only when `args.kv_bits == 3`), pass `model_id=` and `codebook=`:

```python
        if args.kv_bits == 3:
            cache_kwargs = dict(
                batch_size=1, num_layers=cfg.num_hidden_layers,
                num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
                max_seq_len=args.context + 128, page_size=args.page_size, device="cuda",
            )
            if args.codebook == "calibrated":
                cache_kwargs["model_id"] = args.model
            # else: fall through to default = paper codebook
            cache = PersistentTurboKVCache(**cache_kwargs)
```

- [ ] **Step 4: Run all chat tests**

Run: `.venv/bin/python -m pytest tests/test_chat.py -v`

Expected: ALL PASS.

- [ ] **Step 5: Commit**

```bash
git add src/flashquest/runtime/chat.py tests/test_chat.py
git commit -m "phase 11 task 7: CLI --codebook {calibrated,paper} flag (default calibrated)"
```

---

## Task 8: First-run compile-time budget check

Confirm the per-layer constexpr explosion (28 kernel variants) compiles under 120 s on a clean Triton cache. This is a sanity check, not a hard test gate.

**Files:**
- (No new files — uses existing tests)

- [ ] **Step 1: Clear Triton cache**

Run: `rm -rf ~/.triton/cache`

- [ ] **Step 2: Time the smoke test cold-start**

Run: `time .venv/bin/python -m pytest tests/test_persistent_turbo_calibrated_smoke.py -v -m slow --durations=20`

Expected: First-run wall ≤120 s. If higher, investigate whether constexpr param count is the issue (28 layers × 16 codepoints = 448 constexpr values per kernel variant). Fallback: pass codebook as a `(8,)` runtime tensor and `tl.load` it inside the kernel (loses ~10 % perf vs constexpr inline, but compiles in 1 variant). Document the fallback decision inline in `sparse_turbo_fwd.py` if it triggers.

- [ ] **Step 3: Record the wall**

Append the timing line to `benchmarks/phase11/compile_budget.md`:

```markdown
# Phase 11 — First-run compile-time budget

Date: YYYY-MM-DD
Triton cache: cleared before run.
Test: `tests/test_persistent_turbo_calibrated_smoke.py -v -m slow`
First-run wall: <fill in> s.
Within budget (≤120 s)? <yes/no>
```

- [ ] **Step 4: Commit**

```bash
git add benchmarks/phase11/compile_budget.md
git commit -m "phase 11 task 8: record first-run compile-time budget (28 layer variants)"
```

---

## Task 9: RULER NIAH 4 k quality gate (manual)

This is THE quality gate. Pass = calibration shipped as default; fail = ship Phase 11 as opt-in or kill.

**Files:**
- Create: `scripts/phase11_run_ruler_4k_calibrated.py`
- Create (output): `benchmarks/phase11/ruler_4k_calibrated.json`

- [ ] **Step 1: Write the eval runner**

Create `scripts/phase11_run_ruler_4k_calibrated.py`:

```python
"""Phase 11 — RULER NIAH 4 k subset eval on TurboQuant K3-V3 calibrated."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from flashquest.cache import PersistentTurboKVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eval.niah import run_niah
from flashquest.runtime.awq_load import load_awq_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--retention", type=float, default=0.20)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--codebook", choices=("calibrated", "paper"), default="calibrated")
    ap.add_argument("--out", default="benchmarks/phase11/ruler_4k_calibrated.json")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    cache_kwargs = dict(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=args.ctx + 128, page_size=64, device="cuda",
    )
    if args.codebook == "calibrated":
        cache_kwargs["model_id"] = args.model
    cache = PersistentTurboKVCache(**cache_kwargs)

    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=args.retention,
    )

    results = {}
    for task in ("niah_single", "niah_multikey", "niah_multivalue"):
        cache._seen_tokens = [0] * cfg.num_hidden_layers  # reset between tasks
        hits, total, samples = run_niah(
            model, tok, task=task, ctx_len=args.ctx, n=args.n, seed=7,
        )
        results[task] = {"hits": hits, "total": total, "rate": hits / total}
        print(f"{task}: {hits}/{total} ({100 * hits / total:.0f} %)")

    out = {
        "model": args.model,
        "ctx": args.ctx,
        "retention": args.retention,
        "kv_bits": 3,
        "codebook": args.codebook,
        "n": args.n,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")

    gate_pass = (
        results["niah_single"]["rate"] >= 0.95
        and results["niah_multikey"]["rate"] >= 0.95
        and results["niah_multivalue"]["rate"] >= 0.95
    )
    print(f"\nGate (all ≥95 %): {'PASS' if gate_pass else 'FAIL'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the calibrated eval**

Run: `nice -n 19 .venv/bin/python scripts/phase11_run_ruler_4k_calibrated.py --codebook calibrated`

Expected wall: ~40 min (Phase 6 task 2 precedent). Watch for the gate-pass line.

- [ ] **Step 3: Run the paper baseline (for comparison)**

Run: `nice -n 19 .venv/bin/python scripts/phase11_run_ruler_4k_calibrated.py --codebook paper --out benchmarks/phase11/ruler_4k_paper.json`

Expected wall: another ~40 min. This gives the apples-to-apples delta vs Phase 7's 100/100/85 baseline.

- [ ] **Step 4: Interpret + decide**

Compare `benchmarks/phase11/ruler_4k_{calibrated,paper}.json`:

- **Gate PASS** (calibrated ≥95 % across all three): Phase 11 ships K3-V3 calibrated as default; INT4 demoted to throughput-priority opt-in (proceed to Task 10).
- **Gate FAIL** (multivalue still <95 %): the codebook is not the lever. Document at `docs/PHASES/phase-11-quality-gate-fail.md` with the per-task hit table and the granularity ratio R from Task 1. Consider Phase 11b (per-head) as the next escalation. Do not proceed to Task 10; keep INT4 as default.

- [ ] **Step 5: Commit results**

```bash
git add scripts/phase11_run_ruler_4k_calibrated.py benchmarks/phase11/ruler_4k_*.json
git commit -m "phase 11 task 9: RULER 4k @ K3-V3 calibrated vs paper (gate: PASS|FAIL)"
```

---

## Task 10: Decode speed gate @ 32 k (manual)

Verify the per-layer constexpr-dispatch overhead is within ±5 % of Phase 7 baseline. Run only if Task 9 passed; otherwise skip.

**Files:**
- Create: `scripts/phase11_bench_decode_32k.py`
- Create (output): `benchmarks/phase11/decode_32k_calibrated.json`

- [ ] **Step 1: Write the bench**

Create `scripts/phase11_bench_decode_32k.py`:

```python
"""Phase 11 — decode tok/s bench at 32 k context, K3-V3 calibrated vs paper."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flashquest.cache import PersistentTurboKVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="casperhansen/llama-3.2-3b-instruct-awq")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--retention", type=float, default=0.20)
    ap.add_argument("--codebook", choices=("calibrated", "paper"), default="calibrated")
    ap.add_argument("--warmup-steps", type=int, default=5)
    ap.add_argument("--measure-steps", type=int, default=25)
    ap.add_argument("--out", default="benchmarks/phase11/decode_32k_calibrated.json")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    model, tok = load_awq_model(args.model)
    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads

    cache_kwargs = dict(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=args.ctx + 128, page_size=64, device="cuda",
    )
    if args.codebook == "calibrated":
        cache_kwargs["model_id"] = args.model
    cache = PersistentTurboKVCache(**cache_kwargs)

    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern, retention=args.retention,
    )

    # Prefill to ctx.
    prompt_ids = torch.randint(0, cfg.vocab_size, (1, args.ctx), dtype=torch.long).cuda()
    with torch.no_grad():
        model(prompt_ids, use_cache=True)
    torch.cuda.synchronize()

    # Warmup.
    for _ in range(args.warmup_steps):
        last = prompt_ids[:, -1:].clone()
        with torch.no_grad():
            out = model(last, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()

    # Measure.
    t0 = time.perf_counter()
    for _ in range(args.measure_steps):
        last = prompt_ids[:, -1:].clone()
        with torch.no_grad():
            out = model(last, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    tok_per_s = args.measure_steps / (t1 - t0)
    peak_mib = torch.cuda.max_memory_allocated() // (1024 * 1024)
    result = {
        "model": args.model,
        "ctx": args.ctx,
        "retention": args.retention,
        "kv_bits": 3,
        "codebook": args.codebook,
        "decode_tok_per_s": tok_per_s,
        "peak_vram_mib": int(peak_mib),
        "measure_steps": args.measure_steps,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"\n{args.codebook}: {tok_per_s:.2f} tok/s @ ctx={args.ctx}, peak {peak_mib} MiB")


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run paper baseline**

Run: `nice -n 19 .venv/bin/python scripts/phase11_bench_decode_32k.py --codebook paper --out benchmarks/phase11/decode_32k_paper.json`

Expected wall: ~10 min. Expected tok/s: ~2.62 (Phase 7 baseline).

- [ ] **Step 3: Run calibrated**

Run: `nice -n 19 .venv/bin/python scripts/phase11_bench_decode_32k.py --codebook calibrated`

Expected wall: ~10 min plus first-run compile (~1-2 min on top). Expected tok/s: within ±5 % of paper baseline (≥2.49).

- [ ] **Step 4: Check the gate**

Compare the two JSONs. Speed delta = `(calibrated / paper - 1) * 100 %`. Required: `≥ -5 %`.

- **Gate PASS** (`calibrated ≥ 2.49 tok/s`): proceed to Task 11.
- **Gate FAIL** (`calibrated < 2.49 tok/s`): the per-layer constexpr dispatch is costlier than expected. Investigate `nvtx`/Triton's `--print-kernel-args` cache to confirm 28 distinct compiled kernels are being launched. If so, consider the runtime-tensor fallback (passing codebook as `(8,)` tensor + `tl.load`); document the regression at `docs/PHASES/phase-11-speed-regression.md`. Do not ship as default.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase11_bench_decode_32k.py benchmarks/phase11/decode_32k_*.json
git commit -m "phase 11 task 10: decode 32k bench, K3-V3 calibrated vs paper (gate: PASS|FAIL)"
```

---

## Task 11: Docs + CHANGELOG + DOC.md + README update (if Task 9 + 10 passed)

Quality gate passed AND speed gate passed → ship calibrated as v1.2 default. Update user-facing docs.

**Files:**
- Modify: `DOC.md`
- Modify: `README.md`
- Modify: `CHANGELOG.md`
- Create: `docs/PHASES/phase-11-notes.md`

- [ ] **Step 1: Write the phase notes**

Create `docs/PHASES/phase-11-notes.md`:

```markdown
# Phase 11 Notes — TurboQuant Calibrated Codebook

**Started:** 2026-05-14
**Completed:** <fill in>
**Status:** complete (tag `phase-11`)
**Plan/Spec:** `docs/superpowers/specs/2026-05-14-phase-11-turboquant-calibrated-codebook-design.md` (r2)

## Summary

Replaced TurboQuant K3-V3's paper-default Lloyd-Max codebook with 28 per-layer
calibrated codebooks (K + V) fit from Llama-3.2-3B-AWQ activations on
PaulGrahamEssays. Closed the multivalue gap from <fill in: paper %> →
<fill in: calibrated %>, hitting the ≥95 % gate. Speed delta vs paper baseline:
<fill in> %.

## Quality gate (RULER NIAH 4 k @ ctx=4 k, n=20, retention=0.20)

| Codebook | single | multikey | multivalue |
|---|---|---|---|
| paper (Phase 7 baseline) | <fill> | <fill> | <fill> |
| calibrated (Phase 11) | <fill> | <fill> | <fill> |

## Speed gate (decode tok/s @ 32 k, retention=0.20)

| Codebook | tok/s | peak VRAM | Δ vs paper |
|---|---|---|---|
| paper | <fill> | <fill> MiB | reference |
| calibrated | <fill> | <fill> MiB | <fill> % |

## Entry-gate probe (Task 1, retained for reference)

- Codepoint divergence: K = <fill>, V = <fill> (gate ≥5 %).
- Quality-simulator delta: <fill>/3 differ (gate ≥1).
- Granularity pre-signal R: <fill> (R > 2 → consider Phase 11b per-head).

## Surface

- `flashquest --kv-bits 3 --codebook calibrated` (default for calibrated models)
- `flashquest --kv-bits 3 --codebook paper` (opt-out to Phase 7 behavior)
- New module `flashquest.turbo.codebook.load_codebook(model_id)`
- Cache `PersistentTurboKVCache(model_id=...)` autoloads calibration
- Calibration artifact `src/flashquest/turbo/codebook_llama_3_2_3b.pt` (~2 KB)
- Calibration script `scripts/phase11_calibrate_codebook.py` for new models

## Tests carried over

- `tests/test_sparse_turbo_calibrated.py` — kernel parity with custom codebook
- `tests/test_turbo_codebook_loader.py` — loader API + paper fallback
- `tests/test_calibrate_codebook_monotonic.py` — sorted-codepoint invariant
- `tests/test_persistent_turbo_calibrated_smoke.py` (slow) — cache wiring smoke
- `tests/test_chat.py::test_parse_args_codebook_*` — CLI flag
- `tests/test_kv_quant.py::test_quantize_{k,v}_turbo_accepts_custom_codebook` — quant param

## Out of scope (still)

- Per-head / per-channel granularity (Phase 11b candidate if multivalue misses 95 %)
- K3-V2 with calibration (capability-extension target, separate phase)
- Calibration for other models (additive: ship per-model `.pt` files)
- Per-token RMS scale calibration (interacts with codebook; do not bundle)
```

- [ ] **Step 2: Append phase-11 entry to DOC.md**

Edit `DOC.md`. Insert after the v1.0 entry:

```markdown
- **Phase 11 — TurboQuant calibrated codebook** ✅ **complete (tag `phase-11`)** — replaced K3-V3's paper Lloyd-Max codebook with 28 per-layer calibrated codebooks (K + V) fit from Llama-3.2-3B-AWQ activations on PaulGrahamEssays. Closed multivalue gap (Phase 7 17/20 → <fill in>/20); RULER NIAH 4 k cleared 100/100/<fill in>. K3-V3 promoted to default (`--kv-bits 3 --codebook calibrated`); `--kv-bits 4` (KIVI-INT4) demoted to throughput-priority opt-in. See `docs/PHASES/phase-11-notes.md`.
```

- [ ] **Step 3: Update README throughput table**

Edit `README.md`. Update the "Throughput — single-cell decode at 32 k" table to add a K3-V3 calibrated row + update default-mode markers:

```markdown
| Cache mode | retention | decode tok/s | prefill tok/s | peak VRAM (MiB) |
|---|---|---|---|---|
| `--kv-bits 3 --codebook calibrated` (default) | 0.20 | <fill> | 77.6 | 6105 |
| `--kv-bits 4` (throughput opt-in) | 0.20 | **8.41** | 65.0 | 5478 |
| `--kv-bits 4` (throughput, single-needle) | 0.10 | 9.91 | 65.0 | 5478 |
| `--kv-bits 3 --codebook paper` (Phase 7 fallback) | 0.20 | 2.62 | 77.6 | 6105 |
```

Update the RULER quality table accordingly with the calibrated row. Update the lead paragraph that mentioned `--kv-bits 4` as default.

- [ ] **Step 4: Append v1.2 to CHANGELOG.md**

Add a new entry at the top of `CHANGELOG.md`:

```markdown
## [v1.2] — <fill in date>

K3-V3 calibrated TurboQuant is now the default. Per-layer codebooks fit to Llama-3.2-3B-AWQ activations on PaulGrahamEssays close the Phase 7 multivalue gap from 85 % to <fill in> %, allowing 25 % smaller KV cache at parity quality with the prior INT4 default.

### Changed

- Default `--kv-bits 3 --codebook calibrated` (was `--kv-bits 4` in v1.0).
- INT4 (`--kv-bits 4`) demoted to throughput-priority opt-in.

### Added

- `flashquest.turbo.codebook.load_codebook(model_id)` loader API + paper fallback.
- `PersistentTurboKVCache(model_id=...)` autoloads the per-layer codebook artifact.
- CLI flag `--codebook {calibrated, paper}` (default `calibrated`).
- Calibration script `scripts/phase11_calibrate_codebook.py` for new models.
- Calibration artifact `src/flashquest/turbo/codebook_llama_3_2_3b.pt` (~2 KB).

### Benchmarks at v1.2 defaults

- 32 k decode: <fill in> tok/s (K3-V3 calibrated, retention=0.20).
- RULER NIAH 4 k: 100/100/<fill in> (single/multikey/multivalue).
```

- [ ] **Step 5: Tag and commit**

```bash
git add DOC.md README.md CHANGELOG.md docs/PHASES/phase-11-notes.md
git commit -m "phase 11: notes + DOC.md/README/CHANGELOG promote K3-V3 calibrated to default"
git tag -a phase-11 -m "Phase 11: per-layer calibrated TurboQuant codebook"
```

Push tag separately when ready: `git push origin phase-11`.

---

## Out-of-scope (intentional)

- **Public release refresh.** Phase 11 is a master-side phase; if Tasks 9/10 pass and we want a v1.2 public release, that's a separate workflow per `feedback_public_release_orphan_branch` (new orphan commit, force-push origin/main, push v1.2 tag at orphan). Do not bundle in this plan.
- **Per-head codebook fallback.** If the quality gate at Task 9 fails, Phase 11b is the natural next phase, but it's not built here.
- **Other models.** Adding Llama-3.1-8B or Mistral calibration is purely additive — run `scripts/phase11_calibrate_codebook.py` for the new model and register it in `_ARTIFACT_MAP` + `_MODEL_ID_TO_FILENAME`. Not in this plan.
