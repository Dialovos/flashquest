# flashquest — Project Doc

Single living document for the project. See `README.md` for the elevator pitch and `docs/SPEC.md` for the full design.

## Overview

Sparse-retrieval attention Triton kernel for Ampere laptop GPUs (sm_86, 4 GB VRAM). Targets long-context inference (32 k – 128 k) of 3 B – 8 B models on a 3050 Ti class GPU. Composes Quest, KIVI, DuoAttention, StreamingLLM, Marlin, FlashAttention-2, and (optionally) EAGLE-2 into one fused kernel.

## Getting started

```bash
git clone <repo>
cd flashquest
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev,bench]"
./scripts/vendor_clone.sh                # vendor reference repos to vendor/
python scripts/verify_env.py             # snapshot host env to docs/PHASES/env_snapshot.json
python scripts/verify_triton_int8.py     # confirm sm_86 INT8 mma works
```

To reproduce Phase 0 baselines:
```bash
# 1) build llama.cpp with CUDA (already cmake-configured under vendor/llama.cpp)
nice -n 19 cmake --build vendor/llama.cpp/build --config Release -j 4

# 2) pull the GGUF weights
hf download bartowski/Llama-3.2-3B-Instruct-GGUF \
  Llama-3.2-3B-Instruct-Q4_K_M.gguf --local-dir ~/models/llama-3.2-3b

# 3) bench
./scripts/bench_llamacpp.sh              # llama.cpp Q4_K_M @ 8k
python scripts/bench_vllm.py             # vLLM AWQ-INT4 @ 4k (8k OOMs on 4 GB)
python scripts/profile_fa2.py            # flash-attn fwd reference
```

To use the Phase 1 eager Quest attention on any HF Llama model:

```python
from transformers import AutoModelForCausalLM
from flashquest.eager.llama_patch import patch_llama_for_quest_eager

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct", torch_dtype="bfloat16",
    attn_implementation="sdpa",
).cuda()
patch_llama_for_quest_eager(
    model, retention=0.25, num_sinks=4, window_pages=2, page_size=64,
)
# model.generate(...) now uses Quest-eager sparse attention.
```

Reproduce Phase 1 evals:
```bash
hf download unsloth/Llama-3.2-1B-Instruct --local-dir ~/models/llama-3.2-1b-instruct
python scripts/phase1_run_perplexity.py  # Wikitext-2 retention sweep
python scripts/phase1_run_passkey.py     # synthetic long-context retrieval
```

Use the Phase 2 dense Triton kernel directly:

```python
import torch
from flashquest.kernel import flash_attn_fwd

Q = torch.randn(1, 24, 8192, 64, dtype=torch.bfloat16, device="cuda")
K = torch.randn(1,  8, 8192, 64, dtype=torch.bfloat16, device="cuda")
V = torch.randn(1,  8, 8192, 64, dtype=torch.bfloat16, device="cuda")
O, lse = flash_attn_fwd(Q, K, V, causal=True)
```

Decode-only sparse INT8 attention (Phase 3):

```python
import torch
from flashquest.eager.criticality import page_scores
from flashquest.eager.page_summary import compute_page_summary
from flashquest.eager.selection import select_pages
from flashquest.kernel import flash_attn_sparse_fwd
from flashquest.kernel.kv_quant import dequantize_k, quantize_k, quantize_v

# Quantize the KV cache.
K_uint8, K_scale, K_mn = quantize_k(K_bf16, page_size=64)
V_uint8, V_scale, V_mn = quantize_v(V_bf16)

# Decode step: build a per-head selection mask via Quest criticality.
K_dq = dequantize_k(K_uint8, K_scale, K_mn, page_size=64)
K_dq_rep = K_dq.repeat_interleave(H_q // H_kv, dim=1)
pmin, pmax = compute_page_summary(K_dq_rep.float(), page_size=64)
scores = page_scores(Q.float(), pmin, pmax)
sel = select_pages(scores, retention=0.25, num_sinks=4, window_pages=2)

O, lse = flash_attn_sparse_fwd(
    Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn,
    selection_mask=sel, page_size=64,
)
```

DuoAttention per-head dispatch on a HF Llama model (Phase 4):

```python
import torch
from transformers import AutoModelForCausalLM
from flashquest.duo import load_duo_pattern
from flashquest.eager.llama_duo_patch import patch_llama_for_quest_duo

model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct", torch_dtype="bfloat16",
    attn_implementation="sdpa",
).cuda()

# Synthetic pattern (Llama-3.2 has no upstream DuoAttention file):
pattern = (torch.rand(model.config.num_hidden_layers, model.config.num_key_value_heads) < 0.7)

patch_llama_for_quest_duo(
    model, head_pattern=pattern, retention=0.25, num_sinks=4, window_pages=2, page_size=64,
)
# Llama-3.1-8B users can load the upstream pattern instead:
# pattern = load_duo_pattern(
#     "vendor/duo-attention/attn_patterns/Meta-Llama-3.1-8B-Instruct/"
#     "lr=0.02-reg=0.05-ctx=1000_128000-multi_passkey10/full_attention_heads.tsv"
# )
```

Phase 5 persistent-cache decode on a Llama-3.2-3B-AWQ checkpoint:

```python
import torch
from flashquest.cache import PersistentInt8KVCache
from flashquest.eager.llama_persistent_patch import patch_llama_for_quest_persistent
from flashquest.runtime.awq_load import load_awq_model

model, tok = load_awq_model("casperhansen/llama-3.2-3b-instruct-awq")
cfg = model.config
head_dim = cfg.hidden_size // cfg.num_attention_heads

cache = PersistentInt8KVCache(
    batch_size=1, num_layers=cfg.num_hidden_layers,
    num_kv_heads=cfg.num_key_value_heads, head_dim=head_dim,
    max_seq_len=32_768 + 128, page_size=64, device="cuda",
)
pattern = (torch.rand(cfg.num_hidden_layers, cfg.num_key_value_heads) < 0.7)

patch_llama_for_quest_persistent(
    model, cache=cache, head_pattern=pattern,
    retention=0.25, num_sinks=4, window_pages=2, page_size=64,
)
# model.generate(...) now uses persistent INT8 KV + fused DuoAttention.
```

Reproduce Phase 5 / 6 evals:
```bash
hf download casperhansen/llama-3.2-3b-instruct-awq --local-dir ~/models/llama-3.2-3b-awq
python scripts/phase5_run_passkey_32k.py     # 32k passkey (un-seeded; results brittle)
python scripts/phase5_bench_decode_32k.py    # 32k decode tok/s, Phase 5 chain
python scripts/phase5_run_8b_control.py      # 8B AWQ control (largest fitting ctx)
python scripts/phase6_bench_decode_32k.py    # 32k decode tok/s, Phase 6 algebraic chain
python scripts/phase6_diag_passkey_3b.py     # seeded P5/P6 wiring head-to-head
python scripts/phase6_profile_decode.py      # per-op breakdown of decode time
```

## Architecture

See `docs/SPEC.md §4`. Single Triton kernel per attention layer; sparse outer loop over Quest-selected KV blocks; INT8 KV with KIVI-style scales; per-head pattern dispatch (DuoAttention).

## Phases

- **Phase 0 — Setup & baselines** ✅ **complete (2026-04-30, tag `phase-0`)**. Env verified, INT8 mma confirmed on sm_86 (SPEC OQ1 = yes), baselines captured (llama.cpp 39.6 tok/s decode @ 8 k Q4_K_M; vLLM cannot fit 8 k on 4 GB at all — fell back to 4 k @ 17.3 tok/s; flash-attn 22.76 ms / fwd at S = 8192 BF16). WSL2 profiler verdict: nsys captures, importer needs upgrade; ncu blocked by perf-counter perms on consumer drivers (use PyTorch profiler instead). See `docs/PHASES/phase-0-notes.md` and `benchmarks/baselines.json`.
- **Phase 1 — Eager Quest reference** ✅ **complete (tag `phase-1`)**. Pure-PyTorch composition: page summary → criticality → top-k ∪ sinks ∪ window → sparse SDPA. HF LlamaAttention monkeypatch. Validated on `unsloth/Llama-3.2-1B-Instruct`: passkey retrieval **25/25 at retention=0.10** across depths 0.1 / 0.5 / 0.9 (Quest's claim metric). Wikitext perplexity loose at 1B + page_size=64 (gap acknowledged in `docs/PHASES/phase-1-notes.md`). See `src/flashquest/eager/`.
- **Phase 2 — Dense FA-2 Triton kernel** ✅ **complete (tag `phase-2`)**. `flashquest.kernel.flash_attn_fwd(Q, K, V, *, causal, sm_scale=None, return_lse=True)`. BLOCK_M=BLOCK_N=64 on sm_86. 14 catalogued edge cases pass (E1–E14, `tests/test_kernel_flash_fwd_edges.py`); equivalence with Phase 1 eager at retention=1.0 confirmed; perf at the Llama-3.2-3B reference shape is 27.37 ms / fwd vs FA-2's 25.51 ms (**1.073×**, well within the 30 % SPEC target). See `docs/PHASES/phase-2-notes.md`.
- **Phase 3 — Sparse retrieval + INT8 KV** ✅ **complete (tag `phase-3`)**. `flashquest.kernel.flash_attn_sparse_fwd(Q, K_uint8, K_scale, K_mn, V_uint8, V_scale, V_mn, *, selection_mask, page_size, sm_scale, return_lse)`. Decode-only (S_q=1). KIVI-style asymmetric uint8 KV (per-page channel-wise K, per-token V); dequant fused inside the kernel. 11 catalogued edge cases pass + 15 hypothesis-fuzzed shapes. Decode at 8 k context is **2.48×** faster than Phase 2 dense. See `docs/PHASES/phase-3-notes.md`.
- **Phase 4 — DuoAttention head split + HF integration** ✅ **complete (tag `phase-4`)**. `flashquest.duo.{load_duo_pattern, quest_duo_eager_sdpa}` + `flashquest.eager.llama_duo_patch.patch_llama_for_quest_duo`. Per-layer per-KV-head dispatch (retrieval ∪ streaming) wired through HF Llama. Validated on Llama-3.2-1B with a synthetic 70/30 split: passkey 25/25 across depths, matches Phase 1 all-retrieval baseline. Llama-3.1-8B AWQ end-to-end + persistent INT8 KV cache + Marlin projections deferred to Phase 5 (documented in `docs/PHASES/phase-4-notes.md`).
- **Phase 5 — Persistent INT8 KV cache + AWQ + fused DuoAttention** ✅ **complete (tag `phase-5`)**. `flashquest.cache.PersistentInt8KVCache` (HF `Cache` subclass, KIVI-style per-page channel K + per-token V + BF16 partial-page staging) + `flashquest.runtime.load_awq_model` (autoawq 0.2.9 + transformers 4.57 compat shim, fp16 model + bf16 sparse path) + `flashquest.duo.quest_duo_fused_sdpa` (single sparse-kernel call replacing Phase 4's torch.where) + `flashquest.eager.llama_persistent_patch.patch_llama_for_quest_persistent` (dense prefill, sparse decode, online-softmax merge of completed-page sparse + partial-page tail). Validated on Llama-3.2-3B-AWQ at 32 k context with passkey + decode tok/s benchmarks. 8B at 32k blocked by 4 GB VRAM (Llama-3.1-8B AWQ alone is ~4.5 GB) — documented + 8B control script in `docs/PHASES/phase-5-notes.md`.
- **Phase 6 task 1 — Criticality + top-k fix** ✅ **complete (tag `phase-6-task-1`)**. Three sub-tasks: (1a) algebraic `page_scores_int8` + `select_pages_vectorized` skip the dequant chain via `K_mn ≡ page_min`, `K_mn + 255*K_scale ≡ page_max` — decode 0.092 → 2.03 tok/s; (1b) re-profile reveals `page_scores_int8` is 49% of per-layer time; EAGLE-2 deferred (cache-incompatible) and Marlin deferred (M=1 ≈ AWQ); (1c) two-matmul `page_scores_int8_fast` exploits `K_scale ≥ 0` to reduce `max(Q·K_mn, Q·K_mx) = Q·K_mn + 255·relu(Q)·K_scale` — decode 2.03 → **5.14 tok/s** at 32 k context, peak VRAM 6 379 MiB. **56× over Phase 5; SPEC ≥4 tok/s gate cleared.** See `docs/PHASES/phase-6-notes.md`. **Methodology note:** Phase 5 passkey "6/6" was un-seeded `torch.rand` lucky pattern; both Phase 5 and Phase 6 wiring produce bit-identical outputs at ctx=4096 / seed=7 (`scripts/phase6_diag_passkey_3b.py`). RULER 4 k subset (task 2) replaces passkey as the SPEC's quality gate.
- **Phase 6 task 2 — RULER NIAH 4 k subset eval** ✅ **complete (tag `phase-6-task-2`)**. `flashquest.eval.{niah, runner}` with three task generators (`single`, `multikey`, `multivalue`) and a substring-match scorer following the RULER NIAH protocol verbatim (`vendor/RULER/scripts/data/synthetic/niah.py`). Replaces the broken un-seeded passkey eval. Run on Llama-3.2-3B-AWQ at ctx=4 k, retention=0.25, all-retrieval head_pattern, n=20: `niah_single` 20/20 vs 20/20 (100 %), `niah_multikey` 20/20 vs 20/20 (100 %), `niah_multivalue` 19/20 vs 20/20 (95 %) — all ≥85 % vs dense SDPA. Wall 40 min total. Corpus committed at `data/PaulGrahamEssays.json` (gkamradt-pre-extracted PG essays, no html2text dep). See `docs/PHASES/phase-6-notes.md`.
- **Phase 6 task 3 — `flashquest chat` CLI** ✅ **complete (tag `phase-6-task-3`)**. Console script `flashquest` (entry point in `pyproject.toml`) wraps `load_awq_model` + `PersistentInt8KVCache` + `patch_llama_for_quest_persistent` into a streaming chat driver. Single-shot default + interactive REPL (`-i`); `--context-file PATH` (or `-` for stdin) loads up to 32 k of context; greedy default with opt-in `--sample`. Streaming via `TextIteratorStreamer` in a thread; cache resets between REPL turns; oldest-pair truncation. Matches SPEC §11 invocation `flashquest --model casperhansen/llama-3.2-3b-instruct-awq --context 32768 --interactive` literally. 14 unit tests + 1 slow smoke; manual gates on Llama-3.2-3B-AWQ at ctx=32 768 (single-shot 29.6 s) and ctx=8 192 with 4 k context-file (PG-essay summarize, 39.3 s) both produced coherent output. See `docs/PHASES/phase-6-notes.md`.
- **Phase 6 task 4 — head-to-head benchmark** ✅ **complete (tag `phase-6-task-4`); SPEC §11.4 ≥5× gate did NOT clear on raw tok/s — annotated**. `scripts/bench_flashquest.py` + parametrized `bench_llamacpp.sh` + parametrized `bench_vllm.py` + `scripts/phase6_run_headtohead.py` orchestrator (30 min/cell hard cap, `nice -n 19`, `--skip-existing` resume). Three backends × three contexts (8 k / 32 k / 128 k) on Llama-3.2-3B-Instruct, RTX 3050 Ti Laptop, 4 GB VRAM, WSL2. Result: at 32 k+, **none** of flashquest / llama.cpp / vLLM produce a working decode within budget on this hardware. flashquest 8 k = 2.29 tok/s (all-retrieval head_pattern, what `flashquest chat` ships with); llama.cpp 8 k = 39.88 tok/s; vLLM OOMs at all three contexts. SPEC §11.4 gate gated on Phase 6 task 5+ (INT4 KV + kernel-fused criticality + TurboQuant per the post-§11 roadmap). Per-cell JSONs in `benchmarks/phase6_cells/`; aggregated in `benchmarks/phase6_headtohead.{json,md}`. See `docs/PHASES/phase-6-notes.md` for the verdict + capability-axis frame.
- **Phase 6 task 5 — INT4 KV cache** ✅ **complete (tag `phase-6-task-5`); RULER NIAH 4k @ INT4 cleared 100/100/100; SPEC §11.4 still gated on kernel-fused unpack (queued v2)**. KIVI-style asymmetric INT4 (range 0-15), per-page channel-wise K, per-token V; packed 2-per-byte uint8 along `head_dim` (2× storage shrink vs INT8). `page_scores_int4_fast` is a one-constant change (15 vs 255) from `page_scores_int8_fast`. `flashquest.kernel.sparse_int4_fwd` ships as a reference path (INT4 → BF16 → INT8 → existing kernel) that validates plumbing + correctness; the kernel-fused inline INT4 unpack is queued v2 (sketch in the module docstring). `PersistentInt4KVCache` lives alongside `PersistentInt8KVCache`; dispatcher in `llama_persistent_patch.py` branches on `cache.kv_bits`. CLI flag `--kv-bits {4,8}` defaults to 4 after the RULER gate cleared. 8 k decode INT4 = 1.81 tok/s (vs INT8's 2.29 — reference-path round-trip cost dominates the 2× storage shrink at this ctx). 32 k OOMs in the BF16 dequant intermediate; SPEC §11.4 closing axis is the kernel-fused unpack. See `docs/PHASES/phase-6-notes.md`.
- Phase 6 task 6+ — kernel-fused INT4 unpack, TurboQuant, EAGLE-2, Marlin, ExLlamaV2 (post-§11 follow-up).

## Configuration

Pinned versions in `pyproject.toml`; full host snapshot in `docs/PHASES/env_snapshot.json` (gitignored, regenerate via `python scripts/verify_env.py`).

Working stack (2026-04-30):
- Python 3.12.3 (system)
- torch 2.5.1+cu121, triton 3.1.0, numpy < 2
- vllm 0.7.3, transformers 4.57.6 (vLLM ≥ 0.20 forces torch 2.11/cu13 — incompatible with our CUDA 12.5 driver)
- flash-attn 2.7.4.post1 (built from source; no prebuilt wheel matched torch 2.5.1)

Model weights live in `~/models/`; vendored upstream repos in `vendor/` (gitignored, repopulate via `scripts/vendor_clone.sh`).

### WSL2 caveats

- WSL2 default RAM cap is 7.6 GiB on this host. For Phase 4 CPU offload of 8 B models, bump via `~/.wslconfig`:
  ```ini
  [wsl2]
  memory=12GB
  ```
  and `wsl --shutdown` from PowerShell.
- nsys 2022.4 captures `.qdstrm` traces but lacks the importer that converts to `.nsys-rep`. Install a newer nsys, or open the `.qdstrm` directly in nsight-systems on Windows.
- ncu is blocked by `ERR_NVGPUCTRPERM` on consumer drivers without Windows-side `NVreg_RestrictProfilingToAdminUsers=0` or root. Use PyTorch profiler / Triton's built-in metrics for Phase 2 kernel tuning.

## References

- `docs/SPEC.md` — full design
- `docs/REFERENCES.md` — papers + code paths index
- `docs/PHASES/phase-N-notes.md` — per-phase journals
- `docs/superpowers/plans/` — implementation plans (one per phase)
- `benchmarks/baselines.json` — machine-readable benchmark history
- `vendor/README.md` — what we vendor and why
