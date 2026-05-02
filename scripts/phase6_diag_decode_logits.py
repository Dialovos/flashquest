"""Diagnostic: prefill + 4 decode steps on Llama-3.2-1B with the persistent
patch, log logits + first decode step's selection mask sum per head per layer.

Run twice: once with Phase 6 (current) wiring, once with Phase 5 wiring (by
patching imports at runtime). Compare logits + mask shapes.

This forces the decode branch to actually execute, which the persistent_e2e
test doesn't (it's a single forward pass = prefill-only).
"""
from __future__ import annotations

import sys
import torch

# Pin everything we can.
torch.manual_seed(42)
import random
random.seed(42)


def run_path(name: str, use_phase5_path: bool):
    print(f"\n=== {name} ===")
    torch.cuda.empty_cache()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from flashquest.cache import PersistentInt8KVCache

    # Patch the wiring at runtime.
    from flashquest.eager import llama_persistent_patch as patch_mod

    if use_phase5_path:
        from flashquest.eager.criticality import page_scores
        from flashquest.eager.page_summary import compute_page_summary
        from flashquest.eager.selection import select_pages
        from flashquest.kernel.kv_quant import dequantize_k

        def _quest_duo_fused_with_lse(
            Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
            *, head_pattern, page_size, retention, num_sinks, window_pages,
        ):
            from flashquest.kernel import flash_attn_sparse_fwd
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
            print(f"    P5 sel sum/head: {sel.sum(dim=(0, 2, 3)).tolist()[:8]}...")
            O, lse = flash_attn_sparse_fwd(
                Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
                selection_mask=sel, page_size=page_size, return_lse=True,
            )
            return O, lse

        patch_mod._quest_duo_fused_with_lse = _quest_duo_fused_with_lse

    name_model = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name_model)
    cfg = AutoConfig.from_pretrained(name_model)

    # Long enough to get past sinks+window so selection actually filters.
    # 1024 tokens / page_size 64 = 16 pages. sinks=4, window=2 forces 6.
    # retention=0.25 -> top-4 of 16 = 4. So 10/16 selected, 6 filtered.
    CTX = 1024
    text = "The quick brown fox jumps over the lazy dog. " * (CTX // 9 + 4)
    ids = tok(text, return_tensors="pt").input_ids[:, :CTX].cuda()

    torch.manual_seed(42)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    model = AutoModelForCausalLM.from_pretrained(
        name_model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        max_seq_len=CTX + 8, page_size=64, device="cuda",
    )
    patch_mod.patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=0.25, num_sinks=4, window_pages=2, page_size=64,
    )

    with torch.no_grad():
        out = model(ids, use_cache=True, logits_to_keep=1)
        prefill_last = out.logits[:, -1, :].clone()
        next_tok = prefill_last.argmax(dim=-1, keepdim=True)
        # Run 1 decode step to exercise the sparse path.
        print("  decode step 1:")
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        decode_logits = out.logits[:, -1, :].clone()
        next_tok = decode_logits.argmax(dim=-1, keepdim=True)
        decoded_token_id = next_tok[0, 0].item()

    print(f"  decoded next token id: {decoded_token_id}")
    print(f"  decode logits norm: {decode_logits.norm():.4f}")
    print(f"  decode logits max: {decode_logits.max():.4f}, argmax: {decode_logits.argmax(dim=-1).item()}")
    print(f"  top 5 logits: {decode_logits.topk(5, dim=-1).values.tolist()}")
    print(f"  top 5 token ids: {decode_logits.topk(5, dim=-1).indices.tolist()}")
    print(f"  top 5 tokens: {[tok.decode([i]) for i in decode_logits.topk(5, dim=-1).indices[0].tolist()]}")

    return prefill_last.cpu(), decode_logits.cpu()


def main():
    p5_pre, p5_dec = run_path("Phase 5 path", use_phase5_path=True)
    # Re-import fresh to undo our monkey-patch
    import importlib
    from flashquest.eager import llama_persistent_patch
    importlib.reload(llama_persistent_patch)

    p6_pre, p6_dec = run_path("Phase 6 path", use_phase5_path=False)

    print("\n=== Comparison ===")
    print(f"prefill last-logit max abs diff: {(p5_pre - p6_pre).abs().max():.6f}")
    print(f"decode  last-logit max abs diff: {(p5_dec - p6_dec).abs().max():.6f}")
    print(f"decode  argmax: P5={p5_dec.argmax(-1).item()}, P6={p6_dec.argmax(-1).item()}")


if __name__ == "__main__":
    main()
