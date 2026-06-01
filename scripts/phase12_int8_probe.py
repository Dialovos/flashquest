"""Phase 12 task 12 — decisive INT8-head probe (fast, small ctx).

Loads the EAGLE-3 head twice (bf16 and INT8), drives a real depth-4 chain from
the SAME target fused hidden states over a short prompt, and reports:

  * resident VRAM of each head (does int8 actually shrink the head?),
  * per-call propose cost (bf16 vs int8 draft step),
  * draft-token argmax agreement bf16-vs-int8 (the acceptance proxy: do the
    int8 head's greedy predictions match the bf16 head's?),
  * an exact greedy-replay acceptance for BOTH heads vs the target (mean
    accepted/iter — must stay >= ~2.0).

One GPU job; small ctx so it's quick. Run with `nice -n 19`.
"""
from __future__ import annotations

import gc
import json
import statistics
import sys
from pathlib import Path

import torch

from flashquest.runtime.awq_load import load_awq_model
from flashquest.specdec import EAGLE3_FUSION_LAYERS, load_eagle3_draft
from flashquest.specdec.eagle_draft import fuse_target_hidden

TARGET = "casperhansen/llama-3.2-3b-instruct-awq"
HEAD = "thoughtworks/Llama-3.2-3B-Instruct-Eagle3"
N_DRAFT = 4
CTX = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 64
PREFILL_CHUNK = 256


def _niah_prompt(tok, ctx_tokens):
    magic = 4173927
    needle = f"The special access code is {magic}. Remember it. "
    filler = ("The grass is green and the sky is blue. People walk their dogs "
              "in the park on sunny afternoons. ")
    fill_len = len(tok(filler, add_special_tokens=False).input_ids)
    n_fill = max(1, (ctx_tokens - 96) // max(1, fill_len))
    haystack = filler * n_fill
    cut = int(len(haystack) * 0.6)
    doc = haystack[:cut] + needle + haystack[cut:]
    msgs = [{"role": "user",
             "content": doc + "\n\nWhat is the special access code? "
                              "Answer with just the number."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    return ids[:, :ctx_tokens]


@torch.no_grad()
def _greedy_reference(model, ids, max_new, fusion_layers):
    base = model.model
    P = ids.shape[1]
    past = None
    fused_chunks = []
    last_h = None
    for s in range(0, P, PREFILL_CHUNK):
        e = min(s + PREFILL_CHUNK, P)
        bout = base(ids[:, s:e], past_key_values=past, output_hidden_states=True,
                    use_cache=True)
        past = bout.past_key_values
        fused_chunks.append(fuse_target_hidden(bout.hidden_states, fusion_layers))
        last_h = bout.last_hidden_state[:, -1:, :]
        del bout
    nxt = model.lm_head(last_h)[:, -1].argmax(dim=-1, keepdim=True)
    gen = []
    for _ in range(max_new):
        gen.append(nxt)
        out = model(nxt, past_key_values=past, output_hidden_states=True,
                    use_cache=True)
        past = out.past_key_values
        fused_chunks.append(fuse_target_hidden(out.hidden_states, fusion_layers))
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    full_ids = torch.cat([ids] + gen, dim=1)
    fused_seq = torch.cat(fused_chunks, dim=1)
    return full_ids, fused_seq


@torch.no_grad()
def _replay(draft, full_ids, fused_seq, P):
    """Greedy chain-spec replay -> (accepts list, all draft chains list)."""
    total = full_ids.shape[1]
    accepts = []
    chains = []
    tip = P - 1
    while True:
        bonus_idx = tip + 1
        if bonus_idx >= total:
            break
        bonus_tok = full_ids[:, bonus_idx]
        ctx_ids = full_ids[:, :bonus_idx]
        fused = fused_seq[:, :bonus_idx, :]
        chain = draft.propose_chain(fused, bonus_tok, n_draft=N_DRAFT,
                                    context_ids=ctx_ids, chunk=PREFILL_CHUNK)
        chains.append(chain.detach().cpu())
        m = 0
        for j in range(N_DRAFT):
            tgt_idx = bonus_idx + 1 + j
            if tgt_idx >= total:
                break
            if chain[j].item() == full_ids[0, tgt_idx].item():
                m += 1
            else:
                break
        accepts.append(m)
        tip = bonus_idx + m
    return accepts, chains


def _head_linear_mib(draft):
    """Resident bytes of the head's quantizable Linears (handles int8 wrap)."""
    from flashquest.specdec.eagle_quant import _QUANT_TARGETS, _get_submodule
    tot = 0.0
    for dotted in _QUANT_TARGETS:
        parent, attr = _get_submodule(draft.model, dotted)
        mod = getattr(parent, attr)
        w = getattr(mod, "weight", None)
        if w is None and hasattr(mod, "int8"):
            w = mod.int8.weight
        if w is not None:
            tot += w.numel() * w.element_size() / 2**20
    return tot


def _time_propose(draft, fused_seq, full_ids, P, iters=30, warmup=8):
    bonus_idx = P
    bonus_tok = full_ids[:, bonus_idx]
    ctx_ids = full_ids[:, :bonus_idx]
    fused = fused_seq[:, :bonus_idx, :]
    # seed an incremental state and time propose_from (the real step path)
    state = draft.seed(fused, bonus_tok, ctx_ids, chunk=PREFILL_CHUNK)
    state["bonus"] = bonus_tok.view(1, 1)

    def fn():
        draft.propose_from(state, n_draft=N_DRAFT)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@torch.no_grad()
def main():
    torch.manual_seed(0)
    dev = "cuda"
    model, tok = load_awq_model(TARGET)
    L = len(model.model.layers)
    fusion_layers = (2, L // 2, L - 3)
    assert tuple(fusion_layers) == tuple(EAGLE3_FUSION_LAYERS)

    ids = _niah_prompt(tok, CTX).to(dev)
    P = ids.shape[1]
    print(f"ctx={P} max_new={MAX_NEW}")

    # one target greedy reference, reused for both heads
    full_ids, fused_seq = _greedy_reference(model, ids, MAX_NEW, fusion_layers)

    out = {"ctx": P, "max_new": MAX_NEW, "n_draft": N_DRAFT,
           "target": TARGET, "head": HEAD}

    results = {}
    for label, quant in [("bf16", None),
                         ("int8_perchannel", "int8:perchannel"),
                         ("int8_bnb", "int8:bnb")]:
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 2**20
        draft = load_eagle3_draft(HEAD, device=dev, dtype=torch.bfloat16,
                                  embed_weight=model.model.embed_tokens.weight,
                                  quantize=quant)
        torch.cuda.synchronize()
        after = torch.cuda.memory_allocated() / 2**20
        lin_mib = _head_linear_mib(draft)
        accepts, chains = _replay(draft, full_ids, fused_seq, P)
        mean_acc = statistics.mean(1 + m for m in accepts) if accepts else 0.0
        # peak allocated DURING a propose (captures any transient dequant scratch
        # — the metric the decode bench reports for fit/no-fit).
        torch.cuda.reset_peak_memory_stats()
        t_prop = _time_propose(draft, fused_seq, full_ids, P)
        peak_during_propose = torch.cuda.max_memory_allocated() / 2**20
        results[label] = {
            "head_resident_delta_mib": round(after - before, 1),
            "head_linear_mib": round(lin_mib, 1),
            "mean_accepted_per_iter": round(mean_acc, 4),
            "n_iters": len(accepts),
            "accept_hist": {str(k): accepts.count(k) for k in sorted(set(accepts))},
            "propose4_ms": round(t_prop, 4),
            "peak_alloc_during_propose_mib": round(peak_during_propose, 1),
            "chains": [c.tolist() for c in chains],
        }
        print(f"[{label}] head_delta={after-before:.1f} MiB  "
              f"linears={lin_mib:.1f} MiB  mean_acc/iter={mean_acc:.3f}  "
              f"propose4={t_prop:.2f} ms  peak_propose={peak_during_propose:.0f} MiB  "
              f"iters={len(accepts)}")
        del draft
        gc.collect(); torch.cuda.empty_cache()

    # argmax agreement vs bf16 (acceptance proxy) for each int8 backend
    def _agree(a, b):
        n = min(len(a), len(b)); tm = tt = p0m = p0t = 0
        for i in range(n):
            for j in range(min(len(a[i]), len(b[i]))):
                tt += 1
                if a[i][j] == b[i][j]:
                    tm += 1
                if j == 0:
                    p0t += 1
                    if a[i][j] == b[i][j]:
                        p0m += 1
        return round(tm / max(1, tt), 4), round(p0m / max(1, p0t), 4)

    bf = results["bf16"]["chains"]
    agreements = {}
    for label in ("int8_perchannel", "int8_bnb"):
        tok_a, pos0_a = _agree(bf, results[label]["chains"])
        agreements[label] = {"token": tok_a, "pos0": pos0_a}
        print(f"chain agreement bf16-vs-{label}: token={tok_a:.1%} pos0={pos0_a:.1%}")
    out["chain_agreement_vs_bf16"] = agreements

    for label in results:
        results[label].pop("chains", None)
    out["heads"] = results
    out["vram_saved_mib"] = {
        label: round(results["bf16"]["head_resident_delta_mib"]
                     - results[label]["head_resident_delta_mib"], 1)
        for label in ("int8_perchannel", "int8_bnb")
    }
    print(f"\nVRAM saved (head resident): {out['vram_saved_mib']}")
    print(f"acceptance: " + "  ".join(
        f"{k}={results[k]['mean_accepted_per_iter']:.3f}" for k in results))
    print(f"propose4 ms: " + "  ".join(
        f"{k}={results[k]['propose4_ms']:.1f}" for k in results))

    Path("benchmarks/phase12").mkdir(parents=True, exist_ok=True)
    p = Path("benchmarks/phase12/int8_probe.json")
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
