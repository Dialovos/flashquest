# Vendored reference repositories

Read-only references for the techniques composed in flashquest. See `docs/SPEC.md §5` and `docs/REFERENCES.md` for what we crib from each.

| Path | Repo | Phase | Why we vendor |
|---|---|---|---|
| `quest/` | mit-han-lab/Quest | P3 | Top-k page-retrieval algorithm + criticality scoring |
| `kivi/` | jy-yuan/KIVI | P3 | Per-channel-K / per-token-V quant; Triton quant kernels |
| `duo-attention/` | mit-han-lab/duo-attention | P4 | Pre-trained head classifications; dispatch pattern |
| `streaming-llm/` | mit-han-lab/streaming-llm | P3 | Sink-token + RoPE-shift trick |
| `marlin/` | IST-DASLab/marlin | P4 | W4A16 GEMM, used as-is for projections |
| `eagle/` | SafeAILab/EAGLE | P5 | Speculative decoding wrapper |
| `triton/` | triton-lang/triton | P2+ | `python/tutorials/06-fused-attention.py` is our FA-2 scaffold |
| `flash-attention/` | Dao-AILab/flash-attention | P0 | Reference numbers + profiling target |
| `block-sparse-attention/` | mit-han-lab/Block-Sparse-Attention | P4 | Per-head pattern API reference |
| `llama.cpp/` | ggerganov/llama.cpp | P0 | Baseline GGUF inference for `bench_llamacpp.sh` |

Contents are gitignored. Re-clone with `scripts/vendor_clone.sh`.

Pin SHAs once Phase 1 begins to avoid upstream drift breaking phase reproducibility.
