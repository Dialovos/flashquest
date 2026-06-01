"""Phase 12 Task A — the headline 32k decode benchmark for EAGLE-3 spec decode.

Drives the spec-decode dispatcher directly (no chat-CLI text plumbing) for clean
timing on the Llama-3.2-3B-AWQ target + ``PersistentInt4KVCache`` + EAGLE-3 draft
head, at the v1.0 default config (``--kv-bits 4 --retention 0.20``).

Two arms over the SAME patched model + cache config, each from a fresh cache:

  spec      — prefill ~32k via ``make_quest_specdec(...).init``, then time
              ``step()`` until >= N_DECODE tokens are emitted. Reports spec
              tok/s + mean accepted/iter (the real long-ctx, sparse-path,
              incremental-draft-KV acceptance — distinct from the gate-1b dense
              1k probe).
  baseline  — fresh cache, plain non-spec greedy S=1 decode loop over N_DECODE
              tokens (set_verify_active False throughout). Should reproduce the
              v1.0 ~8.41 tok/s @ 32k.

Peak VRAM (``torch.cuda.max_memory_allocated``) is captured with the draft head
resident — this is THE number that resolves the gate-1b VRAM caveat (the gate's
1k probe peaked 3717 MiB; 4k spilled to 4264 MiB via WSL2 host fallback). On a
4 GB card peak may exceed 4095 MiB via WSL2 overcommit; we report the allocator
peak AND a concurrent ``nvidia-smi`` snapshot and are explicit about which is
on-GPU residency.

Run one GPU job at a time, ``nice -n 19``; long run -> run-in-background + poll.
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import (
    patch_llama_for_quest_persistent,
    set_verify_active,
)
from flashquest.runtime.awq_load import load_awq_model
from flashquest.specdec import load_eagle3_draft
from flashquest.specdec.dispatcher import make_quest_specdec

TARGET = "casperhansen/llama-3.2-3b-instruct-awq"
HEAD = "thoughtworks/Llama-3.2-3B-Instruct-Eagle3"
import sys as _sys
N_PREFILL = int(_sys.argv[1]) if len(_sys.argv) > 1 else 32768
N_DECODE = int(_sys.argv[2]) if len(_sys.argv) > 2 else 96
N_DRAFT = 4
RETENTION = 0.20
PAGE_SIZE = 64
PREFILL_CHUNK = 256
# Optional weight-only INT8 draft head (Phase 12 task 12): set
# FLASHQUEST_DRAFT_QUANT=int8 | int8:perchannel | int8:bnb to quantize. Frees
# ~232 MiB so 16k/32k fit on a 4 GB card without a WSL2 host-memory spill.
DRAFT_QUANT = os.environ.get("FLASHQUEST_DRAFT_QUANT") or None


def _nvidia_smi_used_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    """~n_tokens of natural-ish text (so the draft head isn't fed degenerate
    repetition that would inflate or deflate acceptance unrealistically)."""
    text = ("The quick brown fox jumps over the lazy dog. "
            "In a distant land, scholars debated the nature of memory and time. ") * (
        n_tokens // 20 + 4)
    ids = tok(text, return_tensors="pt").input_ids[:, :n_tokens]
    return ids.to("cuda")


def _reset_cache(cache) -> None:
    cache._seen_tokens = [0] * cache.num_layers


@torch.no_grad()
def measure_baseline(model, cache, ids) -> dict:
    """Non-spec greedy S=1 decode over N_DECODE tokens from a fresh cache."""
    set_verify_active(model, False)
    _reset_cache(cache)
    torch.cuda.synchronize()

    t_pre = time.perf_counter()
    out = model(ids, use_cache=True, output_hidden_states=False, logits_to_keep=1)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    next_tok = out.logits[:, -1:].argmax(dim=-1)
    del out
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    for _ in range(N_DECODE):
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_dec

    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tok_per_s": N_DECODE / decode_s,
        "n_decode": N_DECODE,
    }


@torch.no_grad()
def measure_spec(model, cache, draft, ids) -> dict:
    """Spec-decode: prefill via init(), then time step() until >= N_DECODE emitted."""
    _reset_cache(cache)
    init, step = make_quest_specdec(
        model, cache, draft, n_draft=N_DRAFT, page_size=PAGE_SIZE,
        prefill_chunk=PREFILL_CHUNK,
    )
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    init(ids)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t_pre

    emitted_total = 0
    accepts = []  # tokens emitted per iter (1 bonus + accepted drafts)
    torch.cuda.synchronize()
    t_dec = time.perf_counter()
    while emitted_total < N_DECODE:
        committed = step()  # CPU int64, length m+1 (>=1)
        n = int(committed.numel())
        accepts.append(n)
        emitted_total += n
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t_dec

    return {
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "tok_per_s": emitted_total / decode_s,
        "n_emitted": emitted_total,
        "n_iters": len(accepts),
        "mean_accepted_per_iter": statistics.mean(accepts) if accepts else 0.0,
        "accept_histogram": {str(k): accepts.count(k) for k in sorted(set(accepts))},
    }


def main():
    torch.manual_seed(0)
    model, tok = load_awq_model(TARGET)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    pattern = torch.ones(
        cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)

    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL + N_DECODE + 128, page_size=PAGE_SIZE, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=RETENTION, num_sinks=4, window_pages=2, page_size=PAGE_SIZE,
    )

    # Draft head resident for BOTH arms' VRAM accounting (it stays loaded in the
    # shipped --speculative path regardless of which decode runs).
    draft = load_eagle3_draft(
        HEAD, device="cuda", dtype=torch.bfloat16,
        embed_weight=model.model.embed_tokens.weight,
        quantize=DRAFT_QUANT,
    )
    print(f"draft quant: {DRAFT_QUANT}")

    ids = synthetic_prompt_ids(tok, N_PREFILL)
    print(f"prompt tokens: {ids.shape[1]}")

    # --- baseline (fresh cache) ---
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = measure_baseline(model, cache, ids)
    base["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    base["nvidia_smi_used_mib"] = _nvidia_smi_used_mib()
    print(f"[baseline] prefill={base['prefill_s']:.1f}s "
          f"decode={base['tok_per_s']:.2f} tok/s "
          f"peak={base['peak_vram_mib']:.0f} MiB "
          f"smi={base['nvidia_smi_used_mib']} MiB")

    # --- spec (fresh cache, draft head resident) ---
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    spec = measure_spec(model, cache, draft, ids)
    spec["peak_vram_mib"] = torch.cuda.max_memory_allocated() / 2**20
    spec["nvidia_smi_used_mib"] = _nvidia_smi_used_mib()
    print(f"[spec]     prefill={spec['prefill_s']:.1f}s "
          f"decode={spec['tok_per_s']:.2f} tok/s "
          f"mean_accepted/iter={spec['mean_accepted_per_iter']:.3f} "
          f"peak={spec['peak_vram_mib']:.0f} MiB "
          f"smi={spec['nvidia_smi_used_mib']} MiB")

    speedup = spec["tok_per_s"] / base["tok_per_s"] if base["tok_per_s"] else 0.0
    peak_overall = max(base["peak_vram_mib"], spec["peak_vram_mib"])
    print(f"\nspeedup = {speedup:.3f}x  (spec {spec['tok_per_s']:.2f} / "
          f"baseline {base['tok_per_s']:.2f} tok/s)")
    print(f"peak VRAM (max of arms, draft resident) = {peak_overall:.0f} MiB")

    result = {
        "target": TARGET, "head": HEAD, "draft_quant": DRAFT_QUANT,
        "n_prefill": int(ids.shape[1]), "n_decode_target": N_DECODE,
        "n_draft": N_DRAFT, "retention": RETENTION, "kv_bits": 4,
        "page_size": PAGE_SIZE,
        "baseline": base, "spec": spec,
        "speedup": speedup,
        "peak_vram_mib_overall": peak_overall,
        "spec_band": ("default" if speedup >= 1.5 else
                      "opt-in" if speedup >= 1.2 else "off"),
        "notes": (
            "Spec arm uses the incremental-draft-KV dispatcher (use_incremental "
            "default) over the real Quest sparse INT4 cache at 32k — this is the "
            "shipped --speculative path. mean_accepted_per_iter is the long-ctx "
            "sparse-path acceptance (1 bonus + accepted drafts), distinct from the "
            "gate-1b dense 1k probe. Peak VRAM is the allocator max with the EAGLE-3 "
            "head resident; on a 4 GB card peak may exceed 4095 MiB via WSL2 "
            "overcommit (host-memory spill) — see nvidia_smi_used_mib for the "
            "concurrent on-GPU snapshot. spec_band per SPEC §7: >=1.5x default, "
            "1.2-1.5x opt-in, <1.2x off."
        ),
    }
    qsuffix = ("" if not DRAFT_QUANT
               else "_" + DRAFT_QUANT.replace(":", "_"))
    out = Path(f"benchmarks/phase12/decode_{int(ids.shape[1])//1024}k{qsuffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")
    return result


if __name__ == "__main__":
    main()
