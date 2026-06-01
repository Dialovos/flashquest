"""Phase 12 — decisive cost-breakdown probe.

Isolates the four components of a single spec-decode iteration to find
whether the dominant cost is the verify forward (fundamental) or draft +
orchestration (potentially fixable).

THE number: ratio_verify = T_verify_q4_ms / T_decode_q1_ms

Measures:
  T_decode_q1  -- one production S_q=1 decode forward (set_verify_active=False)
  T_verify_q4  -- one verify forward, S_q=4, no draft (set_verify_active=True)
  T_draft4     -- draft.propose_from(state, n_draft=4)
  T_advance    -- draft.advance(state, 4 accepted tokens, fused_hidden)

All measurements: median over N_MEASURE iters after N_WARMUP warmup,
cuda Event timing (device-side, not perf_counter).

Context: ~4096 tokens prefill (fits in VRAM without swap).
"""
from __future__ import annotations

import gc
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch

from flashquest.cache.persistent_int4 import PersistentInt4KVCache
from flashquest.eager.llama_persistent_patch import (
    patch_llama_for_quest_persistent,
    set_verify_active,
)
from flashquest.runtime.awq_load import load_awq_model
from flashquest.specdec import load_eagle3_draft
from flashquest.specdec.eagle_draft import fuse_target_hidden, EAGLE3_FUSION_LAYERS

# ── config ────────────────────────────────────────────────────────────────────
TARGET    = "casperhansen/llama-3.2-3b-instruct-awq"
HEAD      = "thoughtworks/Llama-3.2-3B-Instruct-Eagle3"
N_PREFILL = 4096
N_DRAFT   = 4
RETENTION = 0.20
PAGE_SIZE = 64
N_WARMUP  = 10
N_MEASURE = 30
# Optional weight-only INT8 draft head (Phase 12 task 12): set
# FLASHQUEST_DRAFT_QUANT=int8 | int8:perchannel | int8:bnb to quantize.
DRAFT_QUANT = os.environ.get("FLASHQUEST_DRAFT_QUANT") or None
# ─────────────────────────────────────────────────────────────────────────────


def _nvidia_smi_mib() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def cuda_ms(start_evt: torch.cuda.Event, end_evt: torch.cuda.Event) -> float:
    """Elapsed time in ms between two recorded cuda Events."""
    return start_evt.elapsed_time(end_evt)


def median_cuda_ms(fn, n_warmup=N_WARMUP, n_measure=N_MEASURE) -> float:
    """Run fn() n_warmup+n_measure times; return median ms of measured iters."""
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(n_measure)]
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    for i in range(n_measure):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(cuda_ms(starts[i], ends[i]) for i in range(n_measure))


def synthetic_ids(tok, n: int) -> torch.Tensor:
    text = ("The quick brown fox jumps over the lazy dog. "
            "In a distant land, scholars debated the nature of memory and time. ") * (n // 20 + 4)
    return tok(text, return_tensors="pt").input_ids[:, :n].to("cuda")


@torch.no_grad()
def main():
    torch.manual_seed(42)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading target: {TARGET}")
    model, tok = load_awq_model(TARGET)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads)
    num_layers  = cfg.num_hidden_layers
    num_kv_heads = cfg.num_key_value_heads

    # Cache sized for 4k prefill + 256 headroom (well under 4 GB)
    cache = PersistentInt4KVCache(
        batch_size=1, num_layers=num_layers,
        num_kv_heads=num_kv_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL + 256, page_size=PAGE_SIZE, device="cuda",
    )

    head_pattern = torch.ones(num_layers, num_kv_heads, dtype=torch.bool)
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=head_pattern,
        retention=RETENTION, num_sinks=4, window_pages=2, page_size=PAGE_SIZE,
    )

    print(f"Loading draft: {HEAD}  (quant={DRAFT_QUANT})")
    draft = load_eagle3_draft(
        HEAD, device="cuda", dtype=torch.bfloat16,
        embed_weight=model.model.embed_tokens.weight,
        quantize=DRAFT_QUANT,
    )

    # ── prefill ~4096 tokens ──────────────────────────────────────────────────
    print(f"Prefilling {N_PREFILL} tokens …")
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ids = synthetic_ids(tok, N_PREFILL)
    set_verify_active(model, False)
    # Chunked prefill into the cache (mirrors dispatcher.init)
    CHUNK = 256
    base = model.model
    past = None
    fused_chunks = []
    for s in range(0, N_PREFILL, CHUNK):
        e = min(s + CHUNK, N_PREFILL)
        bout = base(ids[:, s:e], past_key_values=past,
                    output_hidden_states=True, use_cache=True)
        past = bout.past_key_values
        fused_chunks.append(fuse_target_hidden(bout.hidden_states, EAGLE3_FUSION_LAYERS))
        last_h = bout.last_hidden_state[:, -1:, :]
        del bout

    fused_seq = torch.cat(fused_chunks, dim=1)  # (1, P, 3H)
    bonus = model.lm_head(last_h)[:, -1].argmax(dim=-1)  # (1,)
    peak_vram_mib = torch.cuda.max_memory_allocated() / 2**20
    smi_mib = _nvidia_smi_mib()
    print(f"Prefill done. Cache seen={cache._seen_tokens[0]}  "
          f"peak_alloc={peak_vram_mib:.0f} MiB  smi={smi_mib} MiB")

    # Snapshot cache state for reproducible timing loops
    # (we'll re-stage the sandbox; decode arm commits but the cache grows
    # negligibly over 30+10 iters on a 4k base — total drift < 40 toks).
    seen_after_prefill = cache._seen_tokens[0]

    # ── seed draft state once (used for T_draft4 and T_advance) ──────────────
    print("Seeding draft state …")
    draft_state = draft.seed(fused_seq, bonus, ids, chunk=256)
    draft_state["bonus"] = bonus.view(1, 1)  # ensure shape matches propose_from

    # Snapshot: propose_from is non-mutating; advance IS mutating so we'll
    # re-seed for the advance measurement.

    # ── T_decode_q1 ───────────────────────────────────────────────────────────
    # One S_q=1 non-verify decode forward.
    # We do NOT reset _seen_tokens between iters — the cache grows +1/iter
    # (~40 toks over 40 iters on a 4k base = negligible) per the spec.
    print("Measuring T_decode_q1 …")
    next_tok = bonus.view(1, 1)
    set_verify_active(model, False)

    def _decode_q1():
        nonlocal next_tok
        out = model(input_ids=next_tok, past_key_values=cache,
                    use_cache=True, output_hidden_states=False)
        next_tok = out.logits[:, -1:].argmax(dim=-1)

    T_decode_q1_ms = median_cuda_ms(_decode_q1)
    print(f"  T_decode_q1 = {T_decode_q1_ms:.3f} ms  "
          f"(cache seen now {cache._seen_tokens[0]})")

    # ── T_verify_q4 ───────────────────────────────────────────────────────────
    # One verify forward with S_q=4 (no draft emit). Repeated calls to add_draft
    # re-stage the sandbox without committing, so cache state is stable.
    print("Measuring T_verify_q4 …")
    # Build a fixed 4-token verify input: [bonus, d1, d2, d3]
    proposal = draft.propose_from(draft_state, n_draft=N_DRAFT)  # [4]
    verify_input = torch.cat([bonus.view(1), proposal[:-1]]).view(1, N_DRAFT)

    set_verify_active(model, True)

    def _verify_q4():
        model(input_ids=verify_input, past_key_values=cache,
              use_cache=True, output_hidden_states=True)

    T_verify_q4_ms = median_cuda_ms(_verify_q4)
    set_verify_active(model, False)
    print(f"  T_verify_q4 = {T_verify_q4_ms:.3f} ms")

    # ── T_draft4 ──────────────────────────────────────────────────────────────
    # propose_from is non-mutating so the same state is reused.
    print("Measuring T_draft4 …")

    def _draft4():
        draft.propose_from(draft_state, n_draft=N_DRAFT)

    T_draft4_ms = median_cuda_ms(_draft4)
    print(f"  T_draft4    = {T_draft4_ms:.3f} ms")

    # ── T_advance ─────────────────────────────────────────────────────────────
    # advance IS mutating — re-seed a fresh state each iter.
    # We time only the advance call itself; the re-seed happens outside the event.
    print("Measuring T_advance …")
    # Use a fixed accepted span of 4 tokens (bonus + 3 accepted drafts, m=3)
    accepted_tokens = torch.cat([bonus.view(1), proposal[:3]])   # [4]
    # fused hidden for those 4 positions (from fused_seq)
    # positions: last 4 of the verified prefix is a proxy; shape (1,4,3H)
    accepted_fused = fused_seq[:, -N_DRAFT:, :].contiguous()

    # Re-seed once outside timing loop to have a fresh state
    adv_state = draft.seed(fused_seq, bonus, ids, chunk=256)
    adv_state["bonus"] = bonus.view(1, 1)

    adv_starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_WARMUP + N_MEASURE)]
    adv_ends   = [torch.cuda.Event(enable_timing=True) for _ in range(N_WARMUP + N_MEASURE)]

    for i in range(N_WARMUP + N_MEASURE):
        # Re-seed a fresh state for each iter so advance sees the same base
        adv_state = draft.seed(fused_seq, bonus, ids, chunk=256)
        adv_state["bonus"] = bonus.view(1, 1)
        torch.cuda.synchronize()
        adv_starts[i].record()
        adv_state = draft.advance(adv_state, accepted_tokens, accepted_fused, chunk=256)
        adv_ends[i].record()

    torch.cuda.synchronize()
    advance_samples = [cuda_ms(adv_starts[i], adv_ends[i]) for i in range(N_WARMUP, N_WARMUP + N_MEASURE)]
    T_advance_ms = statistics.median(advance_samples)
    print(f"  T_advance   = {T_advance_ms:.3f} ms")

    # ── derived metrics ───────────────────────────────────────────────────────
    ratio_verify           = T_verify_q4_ms / T_decode_q1_ms
    modeled_iteration_ms   = T_verify_q4_ms + T_draft4_ms + T_advance_ms
    modeled_iteration_decodes = modeled_iteration_ms / T_decode_q1_ms
    mean_accepted_per_iter = 2.1  # from Phase 12 task A measurement
    modeled_speedup        = mean_accepted_per_iter / modeled_iteration_decodes

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  T_decode_q1       = {T_decode_q1_ms:.3f} ms  (1 tok, non-verify S_q=1)")
    print(f"  T_verify_q4       = {T_verify_q4_ms:.3f} ms  (verify S_q=4, no draft)")
    print(f"  T_draft4          = {T_draft4_ms:.3f} ms  (propose_from n_draft=4)")
    print(f"  T_advance         = {T_advance_ms:.3f} ms  (advance 4 accepted tokens)")
    print()
    print(f"  ratio_verify      = {ratio_verify:.3f}  ← THE number (verify/decode)")
    print()
    print(f"  modeled_iter_ms   = {modeled_iteration_ms:.3f} ms  "
          f"(verify + draft + advance)")
    print(f"  modeled_iter_dec  = {modeled_iteration_decodes:.3f}x  "
          f"(in units of decode)")
    print(f"  modeled_speedup   = {modeled_speedup:.3f}x  "
          f"(@ 2.1 tok/iter, sanity vs observed ~0.91x)")
    print()
    print(f"  peak_vram_mib     = {peak_vram_mib:.0f} MiB  (prefill, draft resident)")
    print(f"  smi_mib           = {smi_mib} MiB")

    # Cost breakdown (% of modeled iteration)
    pct_verify  = 100 * T_verify_q4_ms  / modeled_iteration_ms
    pct_draft   = 100 * T_draft4_ms     / modeled_iteration_ms
    pct_advance = 100 * T_advance_ms    / modeled_iteration_ms
    print()
    print(f"  iter cost split:  verify={pct_verify:.1f}%  "
          f"draft={pct_draft:.1f}%  advance={pct_advance:.1f}%")

    result = {
        "target": TARGET,
        "head": HEAD,
        "draft_quant": DRAFT_QUANT,
        "n_prefill": N_PREFILL,
        "n_draft": N_DRAFT,
        "retention": RETENTION,
        "kv_bits": 4,
        "page_size": PAGE_SIZE,
        "n_warmup": N_WARMUP,
        "n_measure": N_MEASURE,
        "timing": {
            "T_decode_q1_ms": round(T_decode_q1_ms, 4),
            "T_verify_q4_ms": round(T_verify_q4_ms, 4),
            "T_draft4_ms":    round(T_draft4_ms, 4),
            "T_advance_ms":   round(T_advance_ms, 4),
        },
        "ratio_verify": round(ratio_verify, 4),
        "modeled_iteration_ms": round(modeled_iteration_ms, 4),
        "modeled_iteration_decodes": round(modeled_iteration_decodes, 4),
        "mean_accepted_per_iter_assumed": mean_accepted_per_iter,
        "modeled_speedup": round(modeled_speedup, 4),
        "cost_pct": {
            "verify":  round(pct_verify, 2),
            "draft":   round(pct_draft, 2),
            "advance": round(pct_advance, 2),
        },
        "peak_vram_mib": round(peak_vram_mib, 1),
        "smi_mib": smi_mib,
        "notes": (
            "T_decode_q1: S_q=1 non-verify forward, cache grows +1/iter "
            "(~40 toks over 40 iters on 4k base, negligible). "
            "T_verify_q4: S_q=4 verify arm, add_draft re-stages sandbox "
            "without committing — repeated at same cache state. "
            "T_draft4: propose_from is non-mutating, same state reused. "
            "T_advance: re-seeded state each iter to keep m+1=4 advance "
            "at a fixed base. modeled_speedup uses mean_accepted=2.1 from "
            "Phase 12 task A 4k measurement."
        ),
    }

    suffix = ("" if not DRAFT_QUANT
              else "_" + DRAFT_QUANT.replace(":", "_"))
    out = Path(f"benchmarks/phase12/iteration_breakdown{suffix}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out}")
    return result


if __name__ == "__main__":
    main()
