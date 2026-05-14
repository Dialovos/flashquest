"""Multi-step deterministic passkey on Llama-3.2-3B-AWQ at ctx=4096
(small enough to run in ~5 min). Seeded head_pattern. Run Phase 5 wiring
and Phase 6 wiring back-to-back; report hits.

If Phase 5 ≥4/6 and Phase 6 also ≥4/6: my code is fine; earlier 0/4 was
random pattern unluckiness in the un-seeded passkey script.
If Phase 5 ≥4/6 and Phase 6 0/6: real code regression; investigate.
If both 0/6: ctx=4096 too short for this passkey setup; need bigger.
"""
from __future__ import annotations

import importlib
import random
import torch

from flashquest.cache import PersistentInt8KVCache


def make_passkey_prompt(tok, ctx_len: int, depth: float, seed: int) -> tuple[str, str]:
    rng = random.Random(seed)
    passkey = "".join(str(rng.randint(0, 9)) for _ in range(5))
    filler = "The grass is green. The sky is blue. The sun is yellow. " * (ctx_len // 8)
    target_pos = int(depth * len(filler))
    prefix = filler[:target_pos]
    suffix = filler[target_pos:]
    prompt = (
        f"There is an important info hidden in the text. Find it.\n\n"
        f"{prefix} The pass key is {passkey}. Remember it. {passkey} is the pass key. {suffix}\n\n"
        f"What is the pass key? The pass key is "
    )
    ids = tok(prompt, return_tensors="pt").input_ids
    if ids.shape[1] > ctx_len:
        ids = ids[:, :ctx_len]
    return tok.decode(ids[0]), passkey


def install_phase5_wiring():
    from flashquest.eager.criticality import page_scores
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages
    from flashquest.kernel import flash_attn_sparse_fwd
    from flashquest.kernel.kv_quant import dequantize_k

    from flashquest.eager import llama_persistent_patch as patch_mod

    def _quest_duo_fused_with_lse(
        Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
        *, head_pattern, page_size, retention, num_sinks, window_pages,
    ):
        B, H_q, S_q, D = Q.shape
        _, H_kv, _, _ = K_uint8.shape
        n_rep = H_q // H_kv
        pattern_per_q = head_pattern.to(Q.device).repeat_interleave(n_rep)
        retention_per_q = torch.where(
            pattern_per_q,
            torch.full((H_q,), retention, device=Q.device),
            torch.zeros(H_q, device=Q.device),
        )
        K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=page_size)
        K_dq_full = K_dq.repeat_interleave(n_rep, dim=1)
        page_min, page_max = compute_page_summary(K_dq_full.float(), page_size=page_size)
        scores = page_scores(Q.float(), page_min, page_max)
        sel = select_pages(
            scores, retention=retention_per_q,
            num_sinks=num_sinks, window_pages=window_pages,
        )
        O, lse = flash_attn_sparse_fwd(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            selection_mask=sel, page_size=page_size, return_lse=True,
        )
        return O, lse

    patch_mod._quest_duo_fused_with_lse = _quest_duo_fused_with_lse


def restore_phase6_wiring():
    from flashquest.eager import llama_persistent_patch
    importlib.reload(llama_persistent_patch)


def run_passkey(label: str):
    from flashquest.eager import llama_persistent_patch as patch_mod
    from flashquest.runtime.awq_load import load_awq_model

    print(f"\n=== {label} ===")
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    CTX = 4096
    MAX_NEW = 8

    # SEED head_pattern.
    torch.manual_seed(7)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)
    print(f"  pattern hash: {hash(tuple(pattern.flatten().tolist()))}")

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=CTX + MAX_NEW + 16, page_size=64, device="cuda",
    )
    patch_mod.patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    n_hit = 0
    for depth in [0.1, 0.5, 0.9]:
        for trial in range(2):
            cache._seen_tokens = [0] * cache.num_layers
            torch.cuda.empty_cache()
            prompt, key = make_passkey_prompt(tok, CTX, depth, seed=trial * 1000 + int(depth * 100))
            ids = tok(prompt, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            hit = key in text
            n_hit += int(hit)
            print(f"  d={depth} t={trial} key={key} hit={hit} out={text!r}")
    print(f"  TOTAL: {n_hit}/6")
    return n_hit


def main():
    install_phase5_wiring()
    p5 = run_passkey("Phase 5 wiring (3B AWQ ctx=4096, seed=7)")
    restore_phase6_wiring()
    p6 = run_passkey("Phase 6 wiring (3B AWQ ctx=4096, seed=7)")

    print(f"\n=== SUMMARY ===")
    print(f"Phase 5 wiring: {p5}/6")
    print(f"Phase 6 wiring: {p6}/6")


if __name__ == "__main__":
    main()
