# Scripts

Run every script from the repository root, for example `python scripts/run_competitors.py --help`.
Script names are kept stable because the saved evidence records them. Some current tools
therefore still carry their original `phase6_` names.

## Current validation (2026-10)

| Script | Purpose |
| --- | --- |
| `phase6_run_ruler_4k_int4.py` | Retrieval quality runs. Dense, all-pages INT4 and sparse INT4 answer identical prompts; used for the pilots and the frozen confirmation (any context, despite the name). |
| `validation_stats.py` | Statistics for the frozen confirmation: paired, simultaneous exact binomial bounds. |
| `run_validation_ablation.py` | Balanced sparse vs. all-pages INT4 timing blocks; calls `bench_flashquest.py` for each cell. |
| `bench_flashquest.py` | One warmed FlashQuest request with separate prefill and decode timing. |
| `summarize_validation.py` | Verifies a saved timing block and reports paired per-seed ratios. |
| `run_competitors.py` | llama.cpp and vLLM performance or quality cells, run in sequence. |
| `bench_competitor.py`, `competitor_niah.py` | Child processes for one competitor performance cell or quality cell. |
| `competitor_backend.py`, `vllm_validation_worker.py` | Server adapters for llama.cpp and vLLM. |
| `gpu_memory.py` | Samples GPU, process and system memory during a run. |
| `profile_contribution.py` | Operator reports: metadata scoring, packed attention, decode components. |
| `diagnose_selection_flips.py` | Explains pages that metadata scoring and exact summaries select differently. |
| `bench_common.py` | Shared identities, provenance and record conventions. |

Setup details and full commands are in [docs/competitor-validation.md](../docs/competitor-validation.md)
and the [README](../README.md#reproducing-the-validation).

## Superseded comparison runner

`phase6_run_headtohead.py`, `bench_llamacpp.sh` and `bench_vllm.py` make up the first corrected
comparison matrix. Its vLLM path needs the old V0 engine. The `run_competitors.py` workflow
replaces it. These scripts are kept because their tests document the timing fixes.

## Setup and checks

| Script | Purpose |
| --- | --- |
| `fetch_ruler_corpus.sh` | Rebuilds the committed filler corpus `data/PaulGrahamEssays.json` from RULER's URL list; run `vendor_clone.sh` first. |
| `vendor_clone.sh` | Clones reference repositories into the ignored `vendor/` folder, including the DuoAttention pattern used by tests. |
| `verify_env.py` | Prints the Python, CUDA and GPU environment. |
| `verify_triton_int8.py` | Checks that Triton accepts INT8 matrix operands on the GPU. |

## Historical v1.0 experiments

The `phase1_*` through `phase10_*` scripts not listed above, plus `profile_fa2.py`, produced the
v1.0 results in [`benchmarks/`](../benchmarks/README.md#historical-v10-results-top-level-files).
They are kept for provenance. Some expect the hardware and file layout of that time.
