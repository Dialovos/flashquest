"""Phase 12 gate 1b — EAGLE-3 draft acceptance probe.

Measures the token-acceptance rate of the ``thoughtworks/Llama-3.2-3B-Instruct-Eagle3``
draft head against the AWQ Llama-3.2-3B target, under greedy decoding with chain
depth ``n_draft=4``.

Method (greedy reference replay — exact for greedy verification):
  1. Generate the target's greedy continuation once (KV-cached, S=1 decode),
     recording the fused 3-layer hidden state (HF ``output_hidden_states``
     indices [2, L//2, L-3], concatenated) at every position.
  2. Walk that reference sequence as a chain-spec decoder: at the tip of the
     verified prefix, the target's fused hidden over the prefix + the bonus token
     seed a depth-4 draft chain. Because the target decodes greedily, "the target
     accepts draft d_j" iff d_j equals the target's known greedy token at that
     position. Count m = leading matches; emitted-this-iter = 1 (free/bonus
     token) + m. Advance the tip by m+1 and repeat. This yields the same
     acceptance statistics as a real tree/chain verify pass without re-running
     the target per proposal.

Reports raw ``mean_accepted`` (= mean tokens emitted per iter = 1 + accepted
drafts), the accept histogram, ``draft_step_ratio`` (one isolated draft step /
one isolated target decode step), ``top1_agreement`` (sanity), and ``peak_mib``.
The controller computes ``net = mean_accepted / (0.87 + draft_step_ratio)`` and
the verdict.

Target is plain dense AWQ here; sparse-path + long-ctx acceptance is a Part-B
equivalence concern, NOT this gate.
"""
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
# The vendored draft attention materialises a full [1, H, q_len, kv_len] fp32
# matrix (no flash/SDPA), so a single q_len=T prefill is O(T^2) and OOMs the 4 GB
# GPU at 4k. We feed the context in append-only chunks of this size (bit-for-bit
# identical to a full prefill under regular RoPE), bounding peak memory.
PREFILL_CHUNK = 256


def _load_pg_prompt(tok, ctx_tokens):
    """A PaulGrahamEssays summarize prompt truncated to ~ctx_tokens."""
    data = json.loads(Path("data/PaulGrahamEssays.json").read_text())
    essay = data["text"]
    body_budget = ctx_tokens - 64  # headroom for the chat wrapper
    body_ids = tok(essay, add_special_tokens=False).input_ids[:body_budget]
    body = tok.decode(body_ids)
    msgs = [{"role": "user",
             "content": "Summarize the following essay:\n\n" + body}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt")
    return ids[:, :ctx_tokens]


def _make_niah_prompt(tok, ctx_tokens):
    """A RULER-NIAH-style needle-in-a-haystack prompt truncated to ~ctx_tokens."""
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
    """Greedy-decode `max_new` tokens (KV-cached). Returns:

        full_ids:    LongTensor[1, P+G]    prompt + greedy continuation
        fused_seq:   Tensor[1, P+G, 3H]    fused hidden at EVERY position
                     (so fused_seq[:, :i+1] are the hiddens for prefix ids[:i+1],
                     and fused_seq[:, i] is the hidden that PRODUCES full_ids[i+1]).
    """
    # Prompt pass via the *base* model (no lm_head) in append-only chunks so we
    # never hold the full [1, P, vocab] logits (~1 GB at 4k) nor all P positions'
    # 29-layer hidden states at once (~700 MB at 4k). We keep only the fused
    # 3-layer hidden per position (3H wide).
    base = model.model
    P = ids.shape[1]
    chunk = 256
    past = None
    fused_chunks = []
    last_h = None
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        bout = base(ids[:, s:e], past_key_values=past, output_hidden_states=True,
                    use_cache=True)
        past = bout.past_key_values
        fused_chunks.append(fuse_target_hidden(bout.hidden_states, fusion_layers))
        last_h = bout.last_hidden_state[:, -1:, :]
        del bout
    nxt = model.lm_head(last_h)[:, -1].argmax(dim=-1, keepdim=True)
    del last_h
    gen = []
    for _ in range(max_new):
        gen.append(nxt)
        out = model(nxt, past_key_values=past, output_hidden_states=True,
                    use_cache=True)
        past = out.past_key_values
        fused_chunks.append(fuse_target_hidden(out.hidden_states, fusion_layers))
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    full_ids = torch.cat([ids] + gen, dim=1)            # [1, P+G]
    fused_seq = torch.cat(fused_chunks, dim=1)          # [1, P+G, 3H]
    return full_ids, fused_seq


@torch.no_grad()
def _acceptance_for_prompt(model, draft, ids, max_new, fusion_layers):
    """Walk the greedy reference as a chain-spec decoder. Returns the per-iter
    accept counts (m) and (top1_hit, top1_total) for the sanity agreement."""
    full_ids, fused_seq = _greedy_reference(model, ids, max_new, fusion_layers)
    P = ids.shape[1]
    total = full_ids.shape[1]

    accepts = []
    top1_hit = top1_tot = 0
    tip = P - 1  # absolute index of last verified token; its bonus = full_ids[tip+1]
    while True:
        bonus_idx = tip + 1
        if bonus_idx >= total:
            break
        bonus_tok = full_ids[:, bonus_idx]                 # [1]
        # fused hidden over prefix ids[:bonus_idx] (T positions) + the matching
        # context ids; propose_chain shifts internally so hidden[p] pairs with
        # ids[p+1], and the tip hidden pairs with the bonus token.
        ctx_ids = full_ids[:, :bonus_idx]                  # [1, T]
        fused = fused_seq[:, :bonus_idx, :]                # [1, T, 3H]
        chain = draft.propose_chain(fused, bonus_tok, n_draft=N_DRAFT,
                                    context_ids=ctx_ids,
                                    chunk=PREFILL_CHUNK)   # [n_draft] full vocab

        m = 0
        for j in range(N_DRAFT):
            tgt_idx = bonus_idx + 1 + j  # token that should follow draft j-1
            if tgt_idx >= total:
                break
            target_tok = full_ids[0, tgt_idx].item()
            if j == 0:
                top1_tot += 1
                if chain[0].item() == target_tok:
                    top1_hit += 1
            if chain[j].item() == target_tok:
                m += 1
            else:
                break
        accepts.append(m)
        tip = bonus_idx + m  # committed: bonus + m accepted; next bonus = tip+1
    return accepts, top1_hit, top1_tot


def _time(fn, iters=30, warmup=10):
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
def _step_ratio(model, draft, ids, fusion_layers):
    """median wall-time of ONE isolated draft step (single cached-KV forward
    through the 1-layer head) vs ONE isolated target decode step (S=1)."""
    # warm the target KV via the base model in chunks (no full-prompt logits,
    # no all-positions 29-layer hidden states held at once)
    base = model.model
    P = ids.shape[1]
    cz = 256
    past = None
    fchunks = []
    last_h = None
    for s in range(0, P, cz):
        e = min(s + cz, P)
        bout = base(ids[:, s:e], past_key_values=past, output_hidden_states=True,
                    use_cache=True)
        past = bout.past_key_values
        fchunks.append(fuse_target_hidden(bout.hidden_states, fusion_layers))
        last_h = bout.last_hidden_state[:, -1:, :]
        del bout
    fused = torch.cat(fchunks, dim=1)
    nxt = model.lm_head(last_h)[:, -1].argmax(dim=-1, keepdim=True)
    bonus = nxt[:, 0]
    del last_h, fchunks

    # prime the head once so step() runs against a warm KV
    last_hidden, dpast = draft.prefill(fused, bonus, ids, chunk=PREFILL_CHUNK)

    def target_decode():
        model(nxt, past_key_values=past, output_hidden_states=True, use_cache=True)

    def draft_step():
        # reuse the warm state each call (timing the per-step cost, not prefill)
        draft.step(last_hidden, dpast)

    t_tgt = _time(target_decode)
    t_drf = _time(draft_step)
    return t_drf, t_tgt, t_drf / t_tgt


@torch.no_grad()
def main():
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    max_new = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    dev = "cuda"
    torch.manual_seed(0)

    model, tok = load_awq_model(TARGET)
    L = len(model.model.layers)
    fusion_layers = (2, L // 2, L - 3)
    assert tuple(fusion_layers) == tuple(EAGLE3_FUSION_LAYERS), fusion_layers
    draft = load_eagle3_draft(HEAD, device=dev, dtype=torch.bfloat16,
                              embed_weight=model.model.embed_tokens.weight)

    torch.cuda.reset_peak_memory_stats()

    workloads = []
    all_accepts = []
    top1_hit = top1_tot = 0
    for name, builder in [("paulgraham_summary", _load_pg_prompt),
                          ("ruler_niah", _make_niah_prompt)]:
        ids = builder(tok, ctx).to(dev)
        accepts, h, t = _acceptance_for_prompt(model, draft, ids, max_new,
                                               fusion_layers)
        emitted = [1 + m for m in accepts]
        all_accepts.extend(accepts)
        top1_hit += h
        top1_tot += t
        workloads.append({
            "name": name,
            "prompt_tokens": int(ids.shape[1]),
            "iters": len(accepts),
            "mean_accepted_drafts": statistics.mean(accepts) if accepts else 0.0,
            "mean_emitted": statistics.mean(emitted) if emitted else 0.0,
        })
        print(f"[{name}] prompt={ids.shape[1]} iters={len(accepts)} "
              f"mean_emitted={statistics.mean(emitted):.3f} "
              f"top1={(h/max(1,t)):.1%}")

    emitted_all = [1 + m for m in all_accepts]
    mean_accepted = statistics.mean(emitted_all)
    hist = {}
    for m in all_accepts:
        hist[str(m)] = hist.get(str(m), 0) + 1

    ids = _make_niah_prompt(tok, ctx).to(dev)
    t_drf, t_tgt, ratio = _step_ratio(model, draft, ids, fusion_layers)

    peak_mib = torch.cuda.max_memory_allocated() / 2**20
    top1 = top1_hit / max(1, top1_tot)

    out = {
        "mean_accepted": mean_accepted,
        "accept_histogram": hist,
        "draft_step_ratio": ratio,
        "draft_step_ms": t_drf,
        "target_decode_ms": t_tgt,
        "top1_agreement": top1,
        "ctx": ctx,
        "n_steps": len(all_accepts),
        "peak_mib": peak_mib,
        "n_draft": N_DRAFT,
        "workloads": workloads,
        "fusion_layers": list(fusion_layers),
        "target": TARGET,
        "head": HEAD,
        "note": ("dense AWQ target; sparse-path + long-ctx acceptance is a "
                 "Part-B equivalence concern, not this gate. mean_accepted = "
                 "mean tokens emitted per iter (1 free + accepted drafts); "
                 "controller computes net = mean_accepted/(0.87+draft_step_ratio)."),
    }
    Path("benchmarks/phase12").mkdir(parents=True, exist_ok=True)
    Path("benchmarks/phase12/task1b.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("workloads", "note")}, indent=2))
    return out


if __name__ == "__main__":
    main()
