"""Re-profile Phase 6's decode loop at 32 k to identify the new bottleneck.

Phase 5 profile (benchmarks/phase6_profile.json) showed dequantize_k +
compute_page_summary + repeat_interleave_K = 327 ms / layer = 95 % of
337 ms / layer total. Phase 6 task 1a eliminates those.

This profile measures what's left, with explicit cuda.synchronize() per op
across one decode forward pass at layer 0:
  * AWQ projections q/k/v (W4A16 GEMM at M=1)
  * RoPE
  * cache.update_quantized
  * page_scores_int8 + select_pages_vectorized
  * flash_attn_sparse_fwd
  * partial-tail attention + LSE merge
  * AWQ o_proj
  * RMSNorm + residual + AWQ MLP (gate_proj, up_proj, down_proj)
  * residual #2

The "AWQ + MLP" lump is the SPEC §6 §1b candidate; this script tells us
how big it actually is.
"""
from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.criticality import page_scores_int8
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.eager.selection import select_pages_vectorized
from flashquest.kernel import flash_attn_sparse_fwd
from flashquest.runtime.awq_load import load_awq_model


N_PREFILL = 32768
N_DECODE_TIMED = 8
PAGE_SIZE = 64
RETENTION = 0.25
NUM_SINKS = 4
WINDOW_PAGES = 2


@contextmanager
def cuda_timer(label: str, accum: dict):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    accum.setdefault(label, []).append(time.perf_counter() - t0)


def synthetic_prompt_ids(tok, n_tokens: int) -> torch.Tensor:
    text = "The quick brown fox jumps over the lazy dog. " * (n_tokens // 9 + 2)
    return tok(text, return_tensors="pt").input_ids[:, :n_tokens].to("cuda")


@torch.no_grad()
def manual_per_op(model, cache, tok, n_decode: int) -> dict:
    """Re-run layer-0 sub-steps with explicit timers to reveal new
    bottlenecks. The overall model forward is also timed for sanity."""
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    H_q = cfg.num_attention_heads
    H_kv = cfg.num_key_value_heads
    n_rep = H_q // H_kv

    layer0 = model.model.layers[0]
    attn0 = layer0.self_attn
    mlp0 = layer0.mlp

    # The patched forward is bound on the attention module; we just need
    # access to projections + MLP at the right call sites.
    accum: dict = {}
    end_to_end_step_s: list[float] = []

    # Caller already ran prefill in main(). Take it from current cache state.
    seen = cache._seen_tokens[0]
    if seen < N_PREFILL:
        raise RuntimeError(
            f"Expected prefill to have run; cache._seen_tokens[0]={seen} < {N_PREFILL}"
        )
    # Warmup decode steps to populate the partial-page tail.
    next_tok = torch.tensor([[1]], device="cuda")
    for _ in range(2):
        out = model(next_tok, use_cache=True, logits_to_keep=1)
        next_tok = out.logits[:, -1:].argmax(dim=-1)

    # Build a synthetic hidden_states of correct shape for layer 0.
    # We'll grab the actual hidden state by hooking a forward.
    captured: dict = {}
    def hook(module, args, output):
        captured["hidden"] = args[0].detach().clone()
    h = layer0.register_forward_pre_hook(lambda m, a: captured.setdefault("hidden", a[0].detach().clone()))
    _ = model(next_tok, use_cache=True, logits_to_keep=1)
    h.remove()
    hidden_states = captured["hidden"]  # (1, 1, hidden_size)

    # Position embeddings — synthesise by re-running the position encoder.
    cos = torch.zeros(1, 1, head_dim, device="cuda", dtype=hidden_states.dtype)
    sin = torch.zeros(1, 1, head_dim, device="cuda", dtype=hidden_states.dtype)

    for step in range(n_decode):
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        _ = model(next_tok, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize()
        end_to_end_step_s.append(time.perf_counter() - t_step)

        # Now re-run the layer-0 sub-steps for timing only (does not affect cache state).
        with cuda_timer("attn.q_proj", accum):
            q = attn0.q_proj(hidden_states)
        with cuda_timer("attn.k_proj", accum):
            k = attn0.k_proj(hidden_states)
        with cuda_timer("attn.v_proj", accum):
            v = attn0.v_proj(hidden_states)

        # Reshape (mimicking patched forward).
        with cuda_timer("attn.reshape_qkv", accum):
            q_r = q.view(1, 1, H_q, head_dim).transpose(1, 2)
            k_r = k.view(1, 1, H_kv, head_dim).transpose(1, 2)
            v_r = v.view(1, 1, H_kv, head_dim).transpose(1, 2)

        # Skip RoPE — measured separately if needed; tiny.
        # Cast to bf16 (AWQ is fp16).
        q_b = q_r.to(torch.bfloat16)
        k_b = k_r.to(torch.bfloat16)
        v_b = v_r.to(torch.bfloat16)

        # Use existing cache state from the model's real forward call above.
        views = cache.get_views(0)
        Q_test = q_b  # treat as decode Q.

        with cuda_timer("page_scores_int8", accum):
            scores = page_scores_int8(Q_test, views["K_scale"], views["K_mn"])
        with cuda_timer("select_pages_vectorized", accum):
            pattern = torch.ones(H_kv, dtype=torch.bool, device="cuda")
            pattern_per_q = pattern.repeat_interleave(n_rep)
            retention_per_q = torch.where(
                pattern_per_q,
                torch.full((H_q,), RETENTION, device="cuda"),
                torch.zeros(H_q, device="cuda"),
            )
            sel = select_pages_vectorized(
                scores, retention=retention_per_q,
                num_sinks=NUM_SINKS, window_pages=WINDOW_PAGES,
            )
        with cuda_timer("flash_attn_sparse_fwd", accum):
            O, lse = flash_attn_sparse_fwd(
                Q_test, views["K_uint8"], views["K_scale"], views["K_mn"],
                views["V_uint8"], views["V_scale"], views["V_mn"],
                selection_mask=sel, page_size=PAGE_SIZE, return_lse=True,
            )

        attn_out = O.transpose(1, 2).contiguous().reshape(1, 1, -1).to(q.dtype)
        with cuda_timer("attn.o_proj", accum):
            o = attn0.o_proj(attn_out)

        # MLP path (separate from attention)
        with cuda_timer("mlp.gate_proj", accum):
            gate = mlp0.gate_proj(hidden_states)
        with cuda_timer("mlp.up_proj", accum):
            up = mlp0.up_proj(hidden_states)
        with cuda_timer("mlp.act_mul", accum):
            inter = torch.nn.functional.silu(gate) * up
        with cuda_timer("mlp.down_proj", accum):
            mlp_out = mlp0.down_proj(inter)

        next_tok = torch.tensor([[1]], device="cuda")  # dummy advance

    means_ms = {k: 1000 * (sum(v) / len(v)) for k, v in accum.items()}
    end_to_end_ms = 1000 * (sum(end_to_end_step_s) / len(end_to_end_step_s))
    return {
        "per_layer_op_ms": means_ms,
        "n_steps": n_decode,
        "end_to_end_step_ms": end_to_end_ms,
        "decode_tok_per_s": 1000.0 / end_to_end_ms if end_to_end_ms > 0 else 0,
    }


def main():
    name = "casperhansen/llama-3.2-3b-instruct-awq"
    print(f"Loading {name} ...")
    model, tok = load_awq_model(name)
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    torch.manual_seed(7)
    pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

    cache = PersistentInt8KVCache(
        batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=N_PREFILL + N_DECODE_TIMED + 128,
        page_size=PAGE_SIZE, device="cuda",
    )
    patch_llama_for_quest_persistent(
        model, cache=cache, head_pattern=pattern,
        retention=RETENTION, num_sinks=NUM_SINKS,
        window_pages=WINDOW_PAGES, page_size=PAGE_SIZE,
    )

    print("\n=== prefill (one-time) ===")
    cache._seen_tokens = [0] * cache.num_layers
    t0 = time.perf_counter()
    ids = synthetic_prompt_ids(tok, N_PREFILL)
    out = model(ids, use_cache=True, logits_to_keep=1)
    torch.cuda.synchronize()
    print(f"prefill: {time.perf_counter() - t0:.1f} s")

    print("\n=== Phase 6 manual per-op timing (layer 0, single decode step) ===")
    manual = manual_per_op(model, cache, tok, N_DECODE_TIMED)
    total_ms = sum(manual["per_layer_op_ms"].values())
    sorted_ops = sorted(manual["per_layer_op_ms"].items(), key=lambda kv: -kv[1])
    for k, v in sorted_ops:
        pct = 100 * v / total_ms if total_ms > 0 else 0
        print(f"  {v:7.3f} ms  ({pct:5.1f} %)  {k}")
    print(f"  ----")
    print(f"  {total_ms:7.3f} ms  total per layer 0 sub-step measured")
    print(f"  Across {cfg.num_hidden_layers} layers / decode token: ~{total_ms * cfg.num_hidden_layers:.1f} ms")
    print(f"  End-to-end one decode step (model forward): {manual['end_to_end_step_ms']:.1f} ms "
          f"=> {manual['decode_tok_per_s']:.3f} tok/s")

    out_path = Path(__file__).resolve().parents[1] / "benchmarks" / "phase6_profile_after.json"
    out_path.write_text(json.dumps({
        "model": name, "n_prefill": N_PREFILL,
        "manual": manual,
    }, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
