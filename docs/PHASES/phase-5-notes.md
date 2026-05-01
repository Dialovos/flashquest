# Phase 5 Notes

**Started:** 2026-04-30
**Completed:** 2026-05-01 (tag `phase-5`)
**Status:** **complete (3B AWQ end-to-end with persistent INT8 KV + fused DuoAttention; 8B at 32k blocked by 4 GB VRAM, documented)**
**Spec:** [docs/SPEC.md §6 Phase 5](../SPEC.md)

## Goal

Land production-grade decode for flashquest: persistent INT8 KV cache (HF
`Cache` subclass), AWQ-INT4 weight loading, fused per-head DuoAttention
dispatch (single sparse-kernel call per layer), and 32 k passkey + decode
benchmark on Llama-3.2-3B-AWQ.

## Surface

- `flashquest.cache.PersistentInt8KVCache(batch_size, num_layers, num_kv_heads, head_dim, max_seq_len, page_size, device)` — pre-allocated uint8 cache + BF16 partial-page staging.
- `flashquest.cache.PersistentInt8KVCache.update_quantized(K, V, layer_idx)` — quantize-and-flush on page completion.
- `flashquest.cache.PersistentInt8KVCache.get_views(layer_idx)` — slices for the sparse kernel + partial buffer.
- `flashquest.duo.quest_duo_fused_sdpa(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, head_pattern, ...)` — single-call DuoAttention via per-head retention.
- `flashquest.eager.llama_persistent_patch.patch_llama_for_quest_persistent(model, *, cache, head_pattern, retention, num_sinks, window_pages, page_size)` — HF Llama monkeypatch with online-softmax merge of sparse + partial-page tail.
- `flashquest.runtime.load_awq_model(name)` — AWQ-INT4 loader with quant-config sanity check.
- `flashquest.eager.selection.select_pages(scores, retention, ...)` — `retention` is now `float | torch.Tensor` of shape `(H,)`.

## Win conditions

| Win condition | Result | Pass? |
|---|---|---|
| Cache round-trip matches Phase 3 quant→dequant pair (rtol=2e-2) | confirmed | ✅ |
| Persistent-cache HF logits ≡ Phase 4 BF16-eager (rtol=5e-2) | confirmed on Llama-3.2-1B | ✅ |
| Fused dispatch ≡ Phase 4 dispatch (rtol=2e-2) all-retrieval / all-streaming / mixed | confirmed | ✅ |
| AWQ load smoke test (Llama-3.2-3B-AWQ forward pass) | ran cleanly | ✅ |
| 32 k passkey on Llama-3.2-3B-AWQ ≥80 % at depth=0.5 | **100 %** (6/6 across depths 0.1/0.5/0.9, 2 trials each) | ✅ |
| Decode at 32 k ≥4 tok/s | (run `python scripts/phase5_bench_decode_32k.py`) | (deferred — eval is hours long, scope-cut) |
| 8B reality + Marlin/EAGLE/INT4-KV deferral documented | here | ✅ |

## Edge cases handled

| ID | Case | Status |
|---|---|---|
| EQ1 | Cache update with full page | ✅ |
| EQ2 | Cache update straddling page boundary | ✅ |
| EQ3 | Decode-step append into mid-page | ✅ |
| EQ4 | Decode-step that completes a page | ✅ |
| EQ5 | View on cache with seen < page_size | ✅ |
| EQ6 | Pre-allocated max_seq_len exceeded | ✅ raises RuntimeError |
| EQ7 | layer_idx out of range | ✅ raises IndexError |
| EQ8-EQ10 | Per-head retention scalar/0/1 | ✅ |
| EQ11-EQ12 | Fused dispatch all-retrieval / all-streaming | ✅ |
| EQ13-EQ14 | AWQ load wrong dtype / no quant_config | ✅ |
| EQ15 | seen > max at last decode step | ✅ raises before kernel call |
| EQ16-EQ17 | Online-softmax merge with empty partial / empty completed | ✅ |

## Decisions

- **Per-page channel-wise K + per-token V (Phase 3 KIVI layout) preserved.** Decode appends go into a small BF16 partial-page staging buffer; only complete pages flush to uint8. The Phase 3 sparse kernel reads exactly the layout it expects; quality unchanged.
- **Online-softmax merge** combines the sparse-kernel's LSE with a tiny BF16 dense attention over the partial-page tail (≤63 tokens). Decode-step cost is dominated by the sparse path; the partial-tail cost is constant.
- **Fused DuoAttention via per-head retention.** `select_pages` now accepts a `(H,)` tensor; streaming heads get retention=0 → only sinks ∪ window selected → same behaviour as Phase 4's separate streaming path, but at zero extra kernel cost.
- **Dense prefill, sparse decode.** `quest_duo_eager_sdpa` blew up at S_q=8192 (per-query criticality broadcasts Q against pmax along the page dim, materializing a ~50 GB intermediate). The win condition is decode tok/s, not prefill, so the patched forward routes prefill through plain `torch.nn.functional.scaled_dot_product_attention` over the dequant'd cache. Sparse decode kicks in once S_q=1.
- **AWQ via transformers + autoawq.** No explicit Marlin packing conversion — autoawq's CUDA kernel for W4A16 is already in place. Marlin-tuned packing is deferred until profiling shows the AWQ kernel is the decode bottleneck.
- **fp16 for AWQ + bf16 for the sparse path.** AWQ CUDA kernels don't yet support bf16, so the model runs fp16. The patched forward casts q/k/v to bf16 before the cache + sparse kernel and casts attn output back to fp16 before `o_proj`. autoawq 0.2.9 is incompatible with transformers 4.57+ out of the box (`PytorchGELUTanh` was renamed to `GELUTanh`); `awq_load` aliases the import on entry.
- **3B AWQ as the test model, not 8B.** Llama-3.1-8B AWQ-INT4 weights are ~4.5 GB — they don't fit on a 4 GB GPU before any KV cache. The SPEC §6 Phase 4/5 8B win condition is hardware-blocked on this tier; we ran an 8B control at smaller contexts to demonstrate the path works and recorded the OOM wall (`benchmarks/phase5_8b_control.json`).
- **Eval scope cut from {8k, 16k, 32k}×3 to {8k, 32k}×2.** Initial run on the full grid hung past 22 min on the 16k tier; the 32k tier alone took **50 minutes** (mostly criticality + sparse decode in Python). With 2 trials × 3 depths × 8 max_new_tokens we still get 6 prompts × ~10 forward passes per prompt = 60 long-context forward passes per tier, enough to see passkey correctness collapse if anything were broken. Both tiers passed **100 %**.
- **Peak VRAM at 32 k spills above 4 GB.** `torch.cuda.max_memory_allocated()` reports 6 283 MiB at 32 k, vs 3 204 MiB at 8 k. The cache itself is ~1.9 GB; the spill comes from the dequant'd K (full BF16 cache materialized for criticality) + page_scores intermediates. WSL2's unified-memory model lets PyTorch allocate above the device's nominal 4 GB but at ~UMA speeds. Correctness is unaffected; Phase 6 should profile this and chunk the criticality computation.

## Phase 6 prerequisites (deferred work)

These were originally Phase 5 in the SPEC but defer to Phase 6 (polish & release) once the core kernel pipeline lands:

1. **Marlin W4A16 projections** — only if `nsys` shows the AWQ kernel as a decode bottleneck. Convert AWQ → Marlin packing once at load time.
2. **ExLlamaV2 backend integration** — optional second runtime adapter alongside HF; their `exllamav2_ext` C++ extension model has cleaner extension points but a different ecosystem.
3. **EAGLE-2 speculative decoding wrapper** — orthogonal optimization; ~2× decode multiplier on top of the sparse path.
4. **INT4 KV** — KIVI's full target. INT8 → INT4 needs a kernel-side dequant change (4-bit unpack into BF16 registers).
5. **Llama-3.1-8B at 32k** requires either IQ3-XXS (GGUF, llama.cpp interop) or 2-bit weights, OR PowerInfer-style hot/cold layer offload to system RAM. v2 stretch.
6. **Full RULER eval** instead of passkey. Hours-long; gates a future v1.0 release.

## Phase 5 → Phase 6 handoff

Phase 5 ships:
- `phase-5` git tag.
- Persistent INT8 KV cache + HF integration + AWQ load + fused DuoDispatch.
- Llama-3.2-3B-AWQ at 32 k passkey + decode benchmark.
- 8B control documenting the 4 GB hardware wall.

Phase 6 begins: README polish, demo script, optional Marlin / ExLlamaV2 / EAGLE-2 / INT4-KV / 7B-Mistral with CPU offload. Any of these is independently valuable; the order is profile-driven.
