"""Multi-step deterministic passkey on Llama-3.2-1B, ctx=1024. Run both
Phase 5 wiring and Phase 6 wiring with identical seeds; compare outputs.

Goal: prove or disprove that Phase 6 wiring causes a quality regression
relative to Phase 5 wiring on the same inputs.
"""
from __future__ import annotations

import importlib
import random
import torch


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


def run(label: str):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from flashquest.cache import PersistentInt8KVCache
    from flashquest.eager import llama_persistent_patch as patch_mod

    print(f"\n=== {label} ===")
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    cfg = AutoConfig.from_pretrained(name)

    CTX = 1024
    MAX_NEW = 8

    torch.manual_seed(123)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq_len=CTX + MAX_NEW + 16, page_size=64, device="cuda",
    )
    patch_mod.patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    results = []
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
            results.append((depth, trial, key, text, hit))
            print(f"  d={depth} t={trial} key={key} hit={hit} out={text!r}")
    n_hit = sum(1 for _, _, _, _, h in results if h)
    print(f"  TOTAL: {n_hit}/{len(results)}")
    return n_hit, results


def main():
    install_phase5_wiring()
    p5_hit, p5_res = run("Phase 5 wiring")
    restore_phase6_wiring()
    p6_hit, p6_res = run("Phase 6 wiring")

    print(f"\n=== SUMMARY ===")
    print(f"Phase 5 wiring: {p5_hit}/{len(p5_res)}")
    print(f"Phase 6 wiring: {p6_hit}/{len(p6_res)}")


if __name__ == "__main__":
    main()
