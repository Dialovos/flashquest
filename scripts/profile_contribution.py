"""INT4 metadata, exact packed-attention, and decode-component ablations.

CPU-safe imports. CUDA work is explicit production execution. Synthetic fixtures
are microbenchmarks, not full-model quality/throughput or capacity evidence.
Model mode captures actual post-RoPE BF16 K/Q, one layer at a time, using NIAH.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import math
import time
import traceback
from pathlib import Path
from statistics import median

from bench_common import (
    DEFAULT_MODEL,
    REPO_ROOT,
    SCHEMA_VERSION,
    content_hash,
    error_code,
    export_identity,
    file_hash,
    make_identity,
    model_identity,
    provenance,
    source_identity,
    validate_export,
    write_record,
)

PROTOCOL = {
    "version": 1,
    "kind": "int4-contribution-and-component-ablation",
    "scores": "fp32-two-matmul-gqa; metadata versus stored-bf16-minmax",
    "oracle": "fp32-affine-dequantization-of-identical-packed-values-no-requantization",
    "tail": "unquantized-bf16-tail-always-included; decode-noncausal",
    "timing": "independent-warmed-cuda-event-samples; allocator-baseline-reset-per-call",
    "limits": "operator microbenchmarks; no end-to-end, competitive, quality or capacity claim",
}


def measurement_source():
    """Fingerprint measurement code/runtime, excluding unrelated competitor scripts."""
    source = source_identity()
    source["files"] = {name: digest for name, digest in source["files"].items()
                       if name.startswith("src/") or name in {
                           "scripts/profile_contribution.py", "scripts/bench_common.py",
                           "pyproject.toml", "requirements-validation.txt",
                           "data/PaulGrahamEssays.json"}}
    source["content_sha256"] = content_hash(source["files"])
    return source


def tensor_hash(tensor) -> str:
    import torch
    value = tensor.detach().contiguous().cpu().view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def export_report(report, raw_path):
    """Keep tensor digests explicitly typed; raw measured records stay immutable."""
    rows = []
    for row in report["rows"]:
        public = {k: v for k, v in row.items() if k not in {"query_sha256", "raw_key_sha256"}}
        public["tensor_fingerprints"] = [{"sha256": row[key], "tensor": tensor}
                                          for key, tensor in (("query_sha256", "post-rope-query"),
                                                              ("raw_key_sha256", "post-rope-key"))]
        rows.append(public)
    exported = {**report, "identity": export_identity(report["identity"]), "rows": rows,
                "export_format": "typed-tensor-fingerprints-v2",
                "raw_evidence": {"path": raw_path.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(raw_path)}}
    validate_export(exported)
    return exported


def exact_dequant(packed, scale, minimum, *, page_size: int | None = None):
    """Reconstruct the *kernel's FP32* affine values, without BF16 rounding.

    K metadata is page/channel-wise. V metadata is token-wise. CPU and GPU work
    identically; no import of the quantization module or its CUDA codebooks.
    """
    import torch
    if packed.ndim != 4 or packed.dtype != torch.uint8:
        raise ValueError("packed cache must be a four-dimensional uint8 tensor")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float()
    if page_size is not None:
        if page_size < 1 or scale.shape[2] != math.ceil(packed.shape[2] / page_size):
            raise ValueError("K metadata page count differs from packed tokens")
        scale = scale.repeat_interleave(page_size, dim=2)[:, :, :packed.shape[2]]
        minimum = minimum.repeat_interleave(page_size, dim=2)[:, :, :packed.shape[2]]
    return codes * scale.float() + minimum.float()


def summary_scores(q, minimum, maximum):
    """Same two-matmul identity as metadata scoring, including GQA grouping."""
    if q.ndim != 4 or q.shape[2] != 1 or minimum.shape != maximum.shape:
        raise ValueError("decode query and matching min/max summaries required")
    batch, heads, _, dim = q.shape
    kv_heads, pages = minimum.shape[1:3]
    if heads % kv_heads or minimum.shape[0] != batch or minimum.shape[-1] != dim:
        raise ValueError("GQA/summary shapes differ")
    grouped = q.float().reshape(batch, kv_heads, heads // kv_heads, dim)
    low, delta = minimum.float(), maximum.float() - minimum.float()
    scores = grouped @ low.transpose(-1, -2) + grouped.clamp_min(0) @ delta.transpose(-1, -2)
    return scores.reshape(batch, heads, 1, pages)


def token_mask(selection, completed: int, tail: int, page_size: int):
    import torch
    if selection.ndim != 4 or completed != selection.shape[-1] * page_size:
        raise ValueError("selection describes full completed pages only")
    mask = selection.repeat_interleave(page_size, dim=-1)
    suffix = torch.ones((*mask.shape[:-1], tail), device=mask.device, dtype=torch.bool)
    return torch.cat((mask, suffix), dim=-1)


def grouped_logits(q, keys):
    if q.shape[2] != 1 or q.shape[1] % keys.shape[1]:
        raise ValueError("decode GQA query/key shapes required")
    batch, heads, _, dim = q.shape
    kv_heads = keys.shape[1]
    grouped = q.float().reshape(batch, kv_heads, heads // kv_heads, dim)
    logits = (grouped @ keys.float().transpose(-1, -2)) / math.sqrt(dim)
    return logits.reshape(batch, heads, 1, keys.shape[2])


def exact_attention(q, keys, values, mask):
    """Explicit FP32 output/LSE on identical dequantized values, grouped for GQA."""
    import torch
    if keys.shape != values.shape or mask.shape != (*q.shape[:3], keys.shape[2]):
        raise ValueError("reference K/V/mask shapes differ")
    logits = grouped_logits(q, keys).masked_fill(~mask, -math.inf)
    lse = torch.logsumexp(logits, dim=-1)
    weights = torch.softmax(logits, dim=-1)
    weights = torch.where(torch.isfinite(lse).unsqueeze(-1), weights, 0.0)
    batch, heads, _, tokens = weights.shape
    kv_heads = values.shape[1]
    grouped = weights.reshape(batch, kv_heads, heads // kv_heads, tokens)
    output = grouped @ values.float()
    return output.reshape(*q.shape), lse


def sdpa_attention(q, keys, values, mask):
    import torch
    return torch.nn.functional.scaled_dot_product_attention(
        q.float(), keys.float(), values.float(), attn_mask=mask,
        is_causal=False, enable_gqa=True, scale=q.shape[-1] ** -.5,
    )


def bf16_sdpa_attention(q, keys, values, mask):
    """Separate rounded-value optimized-dispatch comparison; not the exact oracle."""
    import torch
    repeats = q.shape[1] // keys.shape[1]
    return torch.nn.functional.scaled_dot_product_attention(
        q.bfloat16(), keys.bfloat16().repeat_interleave(repeats, dim=1),
        values.bfloat16().repeat_interleave(repeats, dim=1),
        attn_mask=mask, is_causal=False, scale=q.shape[-1] ** -.5,
    )


def sdpa_dispatch(fn):
    """Observe operator dispatch outside the timing samples; do not guess from dtype."""
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as trace:
        result = fn()
        torch.cuda.synchronize()
        del result
    return sorted({event.key for event in trace.key_averages()
                   if "scaled_dot_product" in event.key})


def scalar_metrics(value, reference):
    import torch
    actual, expected = value.float(), reference.float()
    difference = (actual - expected).abs()
    return {"max_abs_error": difference.max().item(), "mean_abs_error": difference.mean().item(),
            "max_relative_error_eps_1e6": (difference / expected.abs().clamp_min(1e-6)).max().item(),
            "relative_l2_error": (difference.norm() / expected.norm().clamp_min(1e-12)).item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                actual.flatten().unsqueeze(0), expected.flatten().unsqueeze(0), dim=-1).item()}


def overlap(a, b):
    union = (a | b).sum(dim=-1)
    common = (a & b).sum(dim=-1)
    result = common.float() / union.clamp_min(1)
    return result.where(union > 0, 1.0)


def boundary_margin(scores, top_k: int):
    """Top-k boundary margin; null if there is no selected/unselected boundary."""
    if not 0 < top_k < scores.shape[-1]:
        return None
    ordered = scores.topk(top_k + 1, dim=-1).values
    return (ordered[..., top_k - 1] - ordered[..., top_k]).flatten().tolist()


def score_diagnostics(q, raw_k, views, *, page_size, retention, sinks, window,
                      score_fn=None, summary_fn=None, select_fn=None):
    import torch

    if score_fn is None or summary_fn is None or select_fn is None:
        from flashquest.eager.criticality import page_scores_int4_fast
        from flashquest.eager.page_summary import compute_page_summary
        from flashquest.eager.selection import select_pages_vectorized

        score_fn = score_fn or page_scores_int4_fast
        summary_fn = summary_fn or compute_page_summary
        select_fn = select_fn or select_pages_vectorized

    completed, tail = views["completed_len"], views["partial_len"]
    if completed < page_size or raw_k.shape[2] != completed + tail:
        raise ValueError("raw post-RoPE keys differ from persistent cache length")
    page_min, page_max = summary_fn(raw_k[:, :, :completed], page_size)
    metadata = score_fn(q, views["K_scale"], views["K_mn"])
    separate = summary_scores(q, page_min, page_max)
    pages = completed // page_size
    k = math.ceil(retention * pages)

    def select(scores, num_sinks=0, window_pages=0):
        return select_fn(scores, retention, num_sinks, window_pages, k_max_static=k)

    meta_top, summary_top = select(metadata), select(separate)
    meta_effective, summary_effective = select(metadata, sinks, window), select(separate, sinks, window)
    logits = grouped_logits(q, raw_k)
    mass = torch.softmax(logits, dim=-1)
    page_logits = logits[..., :completed].reshape(*logits.shape[:-1], pages, page_size)
    page_mass = mass[..., :completed].reshape(*mass.shape[:-1], pages, page_size).sum(dim=-1)
    true_max = page_logits.max(dim=-1).values
    oracle_top = select(true_max)
    tail_mass = mass[..., completed:].sum(dim=-1)
    max_reconstruction = views["K_mn"].float() + 15 * views["K_scale"].float()
    raw_scale = (page_max.float() - page_min.float()) / 15
    elements = raw_k[:, :, :completed].numel()
    packed_k_bytes = views["K_packed"].numel() * views["K_packed"].element_size()
    summary_bytes = sum(t.numel() * t.element_size() for t in (page_min, page_max))
    metadata_bytes = sum(views[key].numel() * views[key].element_size()
                         for key in ("K_scale", "K_mn"))
    diagnostics = {
        "score_error": scalar_metrics(metadata, separate),
        "topk_jaccard_per_query_head": overlap(meta_top, summary_top).flatten().tolist(),
        "effective_selection_jaccard_per_query_head": overlap(meta_effective, summary_effective).flatten().tolist(),
        "metadata_boundary_margin_per_query_head": boundary_margin(metadata, k),
        "summary_boundary_margin_per_query_head": boundary_margin(separate, k),
        "near_tie_tolerance": 1e-4,
        "metadata_near_tie_query_heads": (sum(x <= 1e-4 for x in boundary_margin(metadata, k))
                                          if boundary_margin(metadata, k) is not None else None),
        "metadata_oracle_topk_recall_per_query_head": ((meta_top & oracle_top).sum(-1) / k).flatten().tolist(),
        "metadata_retained_full_context_mass_per_query_head": ((page_mass * meta_effective).sum(-1) + tail_mass).flatten().tolist(),
        "summary_retained_full_context_mass_per_query_head": ((page_mass * summary_effective).sum(-1) + tail_mass).flatten().tolist(),
        "oracle_mass_normalization": "all original post-RoPE keys including always-included tail",
        "selected_pages_per_query_head": meta_effective.sum(-1).flatten().tolist(),
        "raw_scale_at_or_below_epsilon_channels": int((raw_scale <= 1e-6).sum().item()),
        "epsilon": 1e-6,
        "metadata_min_error": scalar_metrics(views["K_mn"], page_min),
        "metadata_reconstructed_max_error": scalar_metrics(max_reconstruction, page_max),
        "bytes": {"packed_k": packed_k_bytes, "shared_affine_k_metadata": metadata_bytes,
                  "incremental_metadata_for_scoring": 0, "separate_bf16_summaries": summary_bytes,
                  "analytic_separate_bf16_summaries": elements * 4 // page_size,
                  "separate_summaries_fraction_of_packed_k": summary_bytes / packed_k_bytes},
    }
    return diagnostics, meta_effective, page_min, page_max


def cuda_samples(fn, *, warmup, reps, device):
    """Independent call peaks above current live inputs; distributions, not a single minimum."""
    import torch
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize(device)
    timings, increments, baselines = [], [], []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        start.record()
        result = fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
        increments.append(max(0, torch.cuda.max_memory_allocated(device) - baseline))
        baselines.append(baseline)
        del result
    return {"cuda_ms": timings, "median_cuda_ms": median(timings),
            "min_cuda_ms": min(timings), "max_cuda_ms": max(timings),
            "incremental_peak_allocated_bytes": increments,
            "live_input_allocated_bytes": baselines,
            "max_incremental_peak_allocated_bytes": max(increments),
            "boundary": "one isolated operator call, synchronized CUDA events; excludes inputs/JIT"}


def decoded_cache(views, page_size, *, dtype=None):
    import torch
    dtype = dtype or torch.float32
    k = exact_dequant(views["K_packed"], views["K_scale"], views["K_mn"], page_size=page_size).to(dtype)
    v = exact_dequant(views["V_packed"], views["V_scale"], views["V_mn"]).to(dtype)
    return (torch.cat((k, views["K_partial"].to(dtype)), dim=2),
            torch.cat((v, views["V_partial"].to(dtype)), dim=2))


def profile_fixture(q, raw_k, views, args):
    import torch

    from flashquest.eager import llama_persistent_patch as patch
    from flashquest.eager.criticality import page_scores_int4_fast
    from flashquest.eager.page_summary import compute_page_summary
    from flashquest.eager.selection import select_pages_vectorized
    from flashquest.kernel.sparse_int4_fwd import flash_attn_sparse_int4_fwd

    diagnostics, selection, low, high = score_diagnostics(
        q, raw_k, views, page_size=args.page_size, retention=args.retention,
        sinks=args.num_sinks, window=args.window_pages)
    mask = token_mask(selection, views["completed_len"], views["partial_len"], args.page_size)

    def packed_attention():
        return flash_attn_sparse_int4_fwd(
            q, views["K_packed"], views["K_scale"], views["K_mn"],
            views["V_packed"], views["V_scale"], views["V_mn"],
            selection_mask=selection, page_size=args.page_size, return_lse=True)

    def tail_attention():
        return patch._bf16_dense_attn_with_lse(q, views["K_partial"], views["V_partial"])

    def fused_attention():
        output, lse = packed_attention()
        if views["partial_len"]:
            tail_output, tail_lse = tail_attention()
            output = patch._merge_two_attentions(output, lse, tail_output, tail_lse)
            lse = torch.logaddexp(lse, tail_lse)
        return output, lse

    def sdpa_dequantization_path():
        keys, values = decoded_cache(views, args.page_size)
        return sdpa_attention(q, keys, values, mask)

    def bf16_dequantization_path():
        # Release each FP32 reconstruction after rounding, as the native BF16
        # dequant path does; do not retain a full FP32 K/V pair during SDPA.
        keys, values = decoded_cache(views, args.page_size, dtype=torch.bfloat16)
        return bf16_sdpa_attention(q, keys, values, mask)

    keys, values = decoded_cache(views, args.page_size)
    reference_output, reference_lse = exact_attention(q, keys, values, mask)
    sdpa_output = sdpa_attention(q, keys, values, mask)
    bf16_sdpa_output = bf16_sdpa_attention(q, keys, values, mask)
    output, lse = fused_attention()
    correctness = {"fused_output_vs_fp32_oracle": scalar_metrics(output, reference_output),
                   "fused_lse_vs_fp32_oracle": scalar_metrics(lse, reference_lse),
                   "sdpa_output_vs_fp32_oracle": scalar_metrics(sdpa_output, reference_output),
                   "bf16_sdpa_output_vs_fp32_oracle": scalar_metrics(bf16_sdpa_output, reference_output),
                   "oracle_requantizes": False,
                   "sdpa_dtype": "float32 identical affine values; PyTorch dispatch",
                   "lse_reference": "explicit float32 logits/logsumexp; not SDPA or INT8 requantization",
                   "fp32_sdpa_dispatch": sdpa_dispatch(sdpa_dequantization_path),
                   "bf16_sdpa_dispatch": sdpa_dispatch(bf16_dequantization_path),
                   "bf16_comparison": "rounded dequantized values; repeated GQA heads; PyTorch optimized dispatch"}
    # Correctness temporaries must not inflate either arm's live allocation baseline.
    del keys, values, reference_output, reference_lse, sdpa_output, bf16_sdpa_output, output, lse
    torch.cuda.synchronize(q.device)
    metadata_scores = page_scores_int4_fast(q, views["K_scale"], views["K_mn"])
    k = math.ceil(args.retention * metadata_scores.shape[-1])
    operations = {
        "metadata_scoring": lambda: page_scores_int4_fast(q, views["K_scale"], views["K_mn"]),
        "separate_summary_scoring": lambda: summary_scores(q, low, high),
        "separate_summary_construction": lambda: compute_page_summary(raw_k[:, :, :views["completed_len"]], args.page_size),
        "topk_selection": lambda: select_pages_vectorized(metadata_scores, args.retention,
            args.num_sinks, args.window_pages, k_max_static=k),
        "packed_attention": packed_attention,
        "packed_attention_with_tail": fused_attention,
        "dequantization_plus_sdpa": sdpa_dequantization_path,
        "dequantization_plus_bf16_sdpa": bf16_dequantization_path,
    }
    packed_output, packed_lse = packed_attention()
    if views["partial_len"]:
        tail_output, tail_lse = tail_attention()
        operations["tail_attention"] = tail_attention
        operations["merge"] = lambda: patch._merge_two_attentions(packed_output, packed_lse,
                                                                  tail_output, tail_lse)
    timings = {name: cuda_samples(operation, warmup=args.warmup, reps=args.reps, device=q.device)
               for name, operation in operations.items()}
    return {"actual_cache_tokens": views["completed_len"] + views["partial_len"],
            "completed_tokens": views["completed_len"], "tail_tokens": views["partial_len"],
            "query_heads": q.shape[1], "kv_heads": raw_k.shape[1], "head_dim": q.shape[-1],
            "query_sha256": tensor_hash(q), "raw_key_sha256": tensor_hash(raw_k),
            "scoring": diagnostics, "attention": correctness, "components": timings,
            "tail_or_merge_unmeasured": not bool(views["partial_len"])}


def synthetic_fixture(context, args):
    import torch

    from flashquest.kernel.kv_quant import quantize_k_int4, quantize_v_int4
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    shape = (1, args.kv_heads, context + args.tail_len, args.head_dim)
    keys = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    values = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    q = torch.randn((1, args.query_heads, 1, args.head_dim), device="cuda",
                    dtype=torch.bfloat16, generator=generator)
    kp, ks, km = quantize_k_int4(keys[:, :, :context], args.page_size)
    vp, vs, vm = quantize_v_int4(values[:, :, :context])
    views = {"K_packed": kp, "K_scale": ks, "K_mn": km,
             "V_packed": vp, "V_scale": vs, "V_mn": vm,
             "K_partial": keys[:, :, context:], "V_partial": values[:, :, context:],
             "completed_len": context, "partial_len": args.tail_len}
    return q, keys, views


def model_rows(model, tokenizer, context, args):
    """Capture actual runtime BF16 K/Q; finish each layer before retaining another.

    Hooks never time whole model forwards. Their CPU capture overhead is excluded
    from the isolated operator timings and cannot establish runtime throughput.
    """
    import torch

    from flashquest.cache.persistent_int4 import PersistentInt4KVCache
    from flashquest.eager import llama_persistent_patch as patch
    from flashquest.eval.niah import make_prompt

    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    layers = args.layers or sorted({0, cfg.num_hidden_layers // 2, cfg.num_hidden_layers - 1})
    if min(layers) < 0 or max(layers) >= cfg.num_hidden_layers:
        raise ValueError("capture layer exceeds model layer count")
    prompt, _ = make_prompt(args.task, context, tokenizer, seed=args.seed)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    input_tokens = ids.shape[1]
    total_steps = max(args.decode_steps) + 1
    capacity = input_tokens + total_steps
    if capacity > cfg.max_position_embeddings:
        raise ValueError("prompt plus capture steps exceeds model context capacity")
    cache = PersistentInt4KVCache(batch_size=1, num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
        max_seq_len=capacity, page_size=args.page_size, device=model.device)
    modules = [m for m in model.modules() if isinstance(m, patch.LlamaAttention)]
    originals = [(m, m.forward) for m in modules]
    original_rope, original_update = patch.apply_rotary_pos_emb, cache.update_quantized
    pattern = torch.ones(cfg.num_hidden_layers, cfg.num_key_value_heads, dtype=torch.bool)
    state = {"layer": None, "target": None, "step": -1, "query": None, "raw_k": []}
    hooks = []
    try:
        patch.patch_llama_for_quest_persistent(model, cache=cache, head_pattern=pattern,
            retention=args.retention, num_sinks=args.num_sinks,
            window_pages=args.window_pages, page_size=args.page_size)

        def enter(module, _inputs, _kwargs):
            state["layer"] = module.layer_idx

        def rope(*positional, **keywords):
            q, k = original_rope(*positional, **keywords)
            if state["layer"] == state["target"] and q.shape[2] == 1:
                state["query"] = q.detach().to(torch.bfloat16).clone()
            return q, k

        def update(k, v, layer_idx):
            if layer_idx == state["target"]:
                state["raw_k"].append(k.detach().cpu())
            return original_update(k, v, layer_idx)

        patch.apply_rotary_pos_emb = rope
        cache.update_quantized = update
        hooks = [m.register_forward_pre_hook(enter, with_kwargs=True) for m in modules]
        positions = torch.arange(capacity, device=model.device)
        for layer in layers:
            state.update(target=layer, step=-1, query=None, raw_k=[])
            cache._seen_tokens = [0] * cache.num_layers
            with torch.inference_mode():
                output = model(input_ids=ids, cache_position=positions[:input_tokens],
                               use_cache=True, logits_to_keep=1)
                next_ids = output.logits[:, -1:].argmax(dim=-1)
                del output
                for step in range(total_steps):
                    state["step"] = step
                    output = model(input_ids=next_ids,
                        cache_position=positions[input_tokens + step:input_tokens + step + 1],
                        use_cache=True, logits_to_keep=1)
                    next_ids = output.logits[:, -1:].argmax(dim=-1)
                    del output
                    if step not in args.decode_steps:
                        continue
                    query = state["query"]
                    if query is None:
                        raise ValueError("post-RoPE query was not captured")
                    raw_k = torch.cat(state["raw_k"], dim=2).to(model.device)
                    row = profile_fixture(query, raw_k, cache.get_views(layer), args)
                    row.update(layer=layer, decode_step=step, nominal_context_budget=context,
                        input_tokens=input_tokens, input_sha256=tensor_hash(ids),
                        model_compute_dtype=str(model.dtype), capture_qk_dtype="torch.bfloat16",
                        source="actual-model-post-RoPE-bf16; all-retrieval-int4")
                    del raw_k
                    yield row
            state.update(query=None, raw_k=[])
            gc.collect()
    finally:
        patch.apply_rotary_pos_emb = original_rope
        del cache.update_quantized
        original_update = None
        for hook in hooks:
            hook.remove()
        for module, forward in originals:
            module.forward = forward
        state.clear()
        del cache
        gc.collect()
        torch.cuda.synchronize(model.device)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "model"], default="synthetic")
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[0, 1, 63])
    parser.add_argument("--task", choices=["single", "multikey", "multivalue"], default="single")
    parser.add_argument("--query-heads", type=int, default=24)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, choices=[64, 128], default=128)
    parser.add_argument("--tail-len", type=int, default=1,
                        help="synthetic BF16 tail added after completed context pages")
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--retention", type=float, default=.20)
    parser.add_argument("--num-sinks", type=int, default=4)
    parser.add_argument("--window-pages", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "validation" / "contrib")
    args = parser.parse_args(argv)
    if (min(args.contexts) < args.page_size or len(set(args.contexts)) != len(args.contexts)
            or args.page_size != 64 or not 0 < args.retention <= 1 or args.warmup < 1
            or args.reps < 1 or args.seed < 0 or args.attempt < 0
            or min(args.decode_steps) < 0 or len(set(args.decode_steps)) != len(args.decode_steps)
            or args.kv_heads < 1 or args.query_heads < 1 or args.query_heads % args.kv_heads
            or not 0 <= args.tail_len < args.page_size or min(args.num_sinks, args.window_pages) < 0):
        parser.error("valid unique contexts/steps, GQA, page=64, retention and positive sampling budgets required")
    if args.mode == "synthetic" and any(ctx % args.page_size for ctx in args.contexts):
        parser.error("synthetic contexts describe completed page tokens and must be page-aligned")
    if args.mode == "model" and not args.revision:
        parser.error("model captures require a pinned --revision")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    config = {key: value for key, value in vars(args).items() if key not in {"out_dir", "model", "revision"}}
    if args.mode == "model":
        metadata = model_identity(args.model, args.revision)
    else:
        metadata = {"model": "synthetic-gaussian-operator-fixture", "revision": None,
                    "files": {}, "content_sha256": content_hash({})}
    config.update(model=metadata["model"], revision=metadata["revision"])
    run = make_identity(config, metadata, PROTOCOL, provenance(), source=measurement_source())
    output_dir = args.out_dir / run["run_identity"]
    output = output_dir / "report.json"
    if output.exists():
        raise ValueError("run already recorded; select a new --attempt to preserve earlier evidence")
    raw_dir = REPO_ROOT / "artifacts" / "contrib" / run["run_identity"]
    report = {**run, "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
              "status": "incomplete", "rows": [], "error": None}

    def save_report():
        raw_path = raw_dir / "report-raw.json"
        write_record(raw_path, report)
        exported = export_report(report, raw_path)
        validate_export(exported)
        write_record(output, exported)

    t_start = time.perf_counter()
    model = None
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("contribution measurements require CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.mode == "model":
            from flashquest.runtime.awq_load import load_awq_model
            model, tokenizer = load_awq_model(args.model, revision=metadata["revision"],
                                              cache_dir=str(REPO_ROOT / "artifacts" / "hf-cache"))
        for context in args.contexts:
            if args.mode == "model":
                iterator = model_rows(model, tokenizer, context, args)
            else:
                def synthetic_row(ctx=context):
                    q, keys, views = synthetic_fixture(ctx, args)
                    row = profile_fixture(q, keys, views, args)
                    row.update(source="synthetic-gaussian-operator-fixture", completed_context=ctx)
                    yield row
                iterator = synthetic_row()
            try:
                with torch.inference_mode():
                    for row in iterator:
                        report["rows"].append(row)
                        save_report()
            finally:
                iterator.close()
            gc.collect()
            torch.cuda.empty_cache()
        if measurement_source()["content_sha256"] != run["identity"]["source"]["content_sha256"]:
            raise RuntimeError("source changed during measurement")
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 — preserve failed operator evidence
        raw_dir.mkdir(parents=True, exist_ok=True)
        trace = raw_dir / "error.log"
        trace.write_text(traceback.format_exc(), encoding="utf-8")
        report.update(status="error", error=error_code(exc),
                      raw_error={"path": trace.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(trace)})
    finally:
        del model
        gc.collect()
    report["wall_s"] = time.perf_counter() - t_start
    save_report()
    print(f"{report['status']}: {len(report['rows'])} contribution fixtures recorded")
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
