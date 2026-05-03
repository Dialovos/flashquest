# Phase 6 task 4 — head-to-head benchmark table

**Date:** 2026-05-02
**Status:** Design approved; ready for implementation plan
**SPEC reference:** `docs/SPEC.md` §6 task 4 ("Head-to-head benchmark table vs `llama.cpp -ngl 999` and vLLM at the same context, on the same machine. README §11 acceptance criterion.") and §11 acceptance bullet 4 ("≥5× capability gain over `llama.cpp -ngl 999` on the same hardware").

## Problem

Phase 0 captured single-context baselines:
- llama.cpp Q4_K_M @ 8 k = **39.6 tok/s** decode (faster than us at 8 k).
- vLLM AWQ @ 4 k = 17.3 tok/s decode (vLLM **OOMs above ~3.9 k** on 4 GB).
- flashquest at 32 k = **5.14 tok/s** (Phase 6 task 1).

A naive "tok/s at the same context" comparison at 8 k makes flashquest look slow. The interesting story is **what each backend can fit**: flashquest decodes at 32 k where vLLM cliffs at ~4 k. The README §11 "≥5× capability gain" line is most honestly read as *capability axis* — the largest context that decodes at all — not raw tok/s at a fixed context.

We need a reproducible table that captures both: (a) tok/s where they all fit; (b) max ctx that fits per backend. If raw tok/s gate fails, that's an honest result we publish + annotate with the post-§11 path (Phase 6 task 5+).

## Decisions

| Question | Decision |
|---|---|
| Headline metric | **tok/s at 32 k decode** (Q1 option C) — same hardware, same context, what `flashquest chat --context 32k` actually runs. |
| If gate fails (llama.cpp faster at 32 k) | **Report honestly + lead with capability axis** (Q2 option A+D). flashquest's value is "decodes at 32 k+" not "fastest decode". Annotate SPEC §11.4 with the post-§11 path (INT4 KV / kernel-fused criticality / TurboQuant). Don't redefine the gate. |
| Model + quant variants | **Each backend's native-strongest** (Q3 option A). flashquest = AWQ-INT4 + INT8 paged KV; llama.cpp = Q4_K_M GGUF + FP16 KV; vLLM = AWQ-INT4 + FP16 KV. ~4-bit weights across the board; KV format differs (this is part of the comparison). Footnote in the table. |
| Context curve | **3 points: 8 k, 32 k, 128 k** (Q4 option B). 8 k anchors against Phase 0; 32 k is the headline; 128 k is the v1 ceiling — finding which backends survive 128 k *is* the capability gain. |
| Wall-time budget per cell | **Hard cap 10 min**, then record timeout. Total run ≈ 1-2 hr. |
| Concurrency | **One backend at a time.** No parallel torch processes (4 GB VRAM + WSL host). `nice -n 19` for shell-driven cells. Background-run via Bash + Monitor so the user can keep working. |

## Architecture

```
scripts/
├── bench_llamacpp.sh          # MODIFY — accept CTX env, drive 8k/32k/128k
├── bench_vllm.py              # MODIFY — accept --max-model-len; clean OOM record
├── bench_flashquest.py        # NEW — single-request decode bench, parametric ctx
└── phase6_run_headtohead.py   # NEW — orchestrator: 3 backends × 3 ctxs, JSON + table

benchmarks/
├── phase6_headtohead.json     # NEW — committed at end of run
└── phase6_headtohead.md       # NEW — generated markdown table
```

No new `src/flashquest/` code — this is benchmarking glue. Each per-backend script is independently runnable for debugging. The orchestrator sequences them and merges JSONs.

## Components

### `bench_flashquest.py`

Mirrors `phase6_bench_decode_32k_v2.py` but parametric:

```python
parser.add_argument("--ctx-len", type=int, required=True)
parser.add_argument("--n-decode", type=int, default=32)
parser.add_argument("--max-new-tokens", type=int, default=128)
parser.add_argument("--out", type=str, required=True)  # writes per-cell JSON
```

Per-cell record:
```json
{
  "backend": "flashquest",
  "quant": "AWQ-INT4 + INT8 paged KV + Quest top-k retention=0.25",
  "ctx_len": 32768,
  "prefill_tok_s": ..., "decode_tok_s": ..., "peak_vram_mib": ...,
  "wall_s": ..., "oom": false, "error": null
}
```

Catches `torch.cuda.OutOfMemoryError` and writes `oom: true` instead of crashing the orchestrator.

### `bench_llamacpp.sh` (modified)

Existing script hardcodes `-p 8192 -n 128`. Change to `-p ${CTX:-8192} -n 128`. Output continues to go to `benchmarks/llamacpp_${CTX}.txt`. The orchestrator parses `llama-bench`'s markdown output to extract `prefill_tok_s` / `decode_tok_s` / `peak_vram_mib` and produce the JSON cell. Returns non-zero on OOM (caught by orchestrator).

### `bench_vllm.py` (modified)

Existing script hardcodes `max_model_len=4096`. Add `--max-model-len` arg. Wrap `LLM(...)` and `llm.generate(...)` in try/except `RuntimeError`/`torch.OutOfMemoryError`; write `oom: true` to JSON instead of raising.

### `phase6_run_headtohead.py`

```python
BACKENDS = ["flashquest", "llamacpp", "vllm"]
CTXS = [8192, 32768, 131072]

def main():
    args = parse()                      # --dry-run, --skip-existing
    matrix = [(b, c) for b in BACKENDS for c in CTXS]
    if args.dry_run:
        print_matrix(matrix); return

    cells = []
    for backend, ctx in matrix:
        cell = run_one(backend, ctx)    # subprocess.run + json load + nvidia-smi
        cells.append(cell)
        free_gpu()                       # 2s sleep + cuda.empty_cache via stub
    out = build_results(cells)
    Path("benchmarks/phase6_headtohead.json").write_text(json.dumps(out, indent=2))
    Path("benchmarks/phase6_headtohead.md").write_text(render_markdown(out))
```

Runs sequentially. `free_gpu()` between cells: 2 s sleep + GPU idle check. Each subprocess is killed at 600 s (10 min hard cap).

## Data flow / results JSON

```json
{
  "host": {"gpu": "RTX 3050 Ti Laptop", "vram_mib": 4095, "cuda": "12.5", "wsl2": true},
  "model": "Llama-3.2-3B-Instruct",
  "date": "2026-05-02",
  "headline_metric": "decode tok/s at ctx=32768",
  "capability_axis": "max ctx that decodes (≥1 tok/s, no OOM)",
  "results": [
    {"backend": "flashquest", "quant": "AWQ-INT4 + INT8 paged KV + Quest", "ctx": 8192,   "decode_tok_s": ..., "peak_vram_mib": ..., "oom": false},
    {"backend": "flashquest", "quant": "AWQ-INT4 + INT8 paged KV + Quest", "ctx": 32768,  "decode_tok_s": 5.14, "peak_vram_mib": 6379, "oom": false},
    {"backend": "flashquest", "quant": "AWQ-INT4 + INT8 paged KV + Quest", "ctx": 131072, "decode_tok_s": ..., "peak_vram_mib": ..., "oom": ...},
    {"backend": "llama.cpp",  "quant": "Q4_K_M, FP16 KV",                  "ctx": 8192,   "decode_tok_s": 39.60, ...},
    {"backend": "llama.cpp",  "quant": "Q4_K_M, FP16 KV",                  "ctx": 32768,  ...},
    {"backend": "llama.cpp",  "quant": "Q4_K_M, FP16 KV",                  "ctx": 131072, ...},
    {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4, FP16 KV",                "ctx": 8192,   "oom": true, "decode_tok_s": null},
    {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4, FP16 KV",                "ctx": 32768,  "oom": true},
    {"backend": "vLLM 0.7.3", "quant": "AWQ-INT4, FP16 KV",                "ctx": 131072, "oom": true}
  ]
}
```

## Generated markdown table

```markdown
| Backend | Quant | 8 k decode tok/s | 32 k decode tok/s | 128 k fits? | Peak VRAM @ max ctx |
|---|---|---|---|---|---|
| **flashquest** | AWQ-INT4 + INT8 paged KV + Quest top-k | A | **5.14** | ✓/✗ | M MiB |
| llama.cpp -ngl 999 | Q4_K_M, FP16 KV | 39.60 | B | ✓/✗ | M MiB |
| vLLM 0.7.3 (FA-2) | AWQ-INT4, FP16 KV | OOM† | OOM† | OOM† | n/a |

† vLLM cannot fit ≥4 k context on 4 GB VRAM (Phase 0 finding; see baselines).
```

The README narrative around the table:
- "Of the three, only flashquest decodes at 32 k+ on this hardware."
- "Capability ratio: flashquest fits ∞× vLLM (OOMs at 4 k+) and X× llama.cpp."
- If llama.cpp survives 128 k → "tied on capability; raw tok/s gap targeted by Phase 6 task 5+ (INT4 KV)."
- If llama.cpp OOMs at ≤64 k → "flashquest is the only backend that runs at the v1 ceiling."

## Edge cases / tests

| ID | Case | Handling |
|---|---|---|
| ER1 | vLLM init OOMs at `max_model_len=8192` | catch, record `oom: true`, continue |
| ER2 | llama.cpp `-c 131072` exits non-zero | record `error: "<stderr tail>"`, continue |
| ER3 | flashquest at 128 k OOMs during prefill | catch `torch.OutOfMemoryError`, record `oom: true` |
| ER4 | Cell exceeds 10 min wall | kill subprocess, record `error: "timeout (>600s)"` |
| ER5 | nvidia-smi races with next launch | run after `cuda.empty_cache` + 2 s sleep + process exit |
| ER6 | vLLM 0.7.3 broken by env update since Phase 0 | document version pin; if install fails, record vLLM cells as `error: "vllm install failed (<sha>)"` |
| ER7 | `--dry-run` prints planned matrix without launching | unit-test via importing orchestrator |
| ER8 | `--skip-existing` skips cells already in JSON | resume after partial run |

No unit tests for the orchestrator beyond `--dry-run` smoke (it's I/O glue + table rendering). Each per-backend script gets a smoke pass with a tiny ctx (256) to confirm arg parsing.

## Validation gates

| Gate | Target |
|---|---|
| Each per-backend script smoke at ctx=256 | exits 0, writes JSON cell |
| Orchestrator `--dry-run` | prints all 9 cells planned |
| Full run completes within ~2 hr wall | JSON + markdown written |
| flashquest at 32 k matches Phase 6 task 1 | 5.14 ± 1 tok/s |
| Table renders into README | manual inspection |

## Non-goals

- Pareto across model sizes (just 3B).
- Multi-request throughput (single-request).
- Latency percentiles (mean tok/s only).
- ExLlamaV2 (separate SPEC task 8).
- Energy / power.
- INT4 KV variant of flashquest (Phase 6 task 5).

## File touch list

| File | Action |
|---|---|
| `scripts/bench_flashquest.py` | Create |
| `scripts/bench_llamacpp.sh` | Modify (accept `CTX` env) |
| `scripts/bench_vllm.py` | Modify (accept `--max-model-len`) |
| `scripts/phase6_run_headtohead.py` | Create |
| `benchmarks/phase6_headtohead.json` | Create at end of run |
| `benchmarks/phase6_headtohead.md` | Create at end of run |
| `docs/PHASES/phase-6-notes.md` | Append task 4 section |
| `DOC.md` / `README.md` / `docs/SPEC.md` | Tick task 4, embed table |

## Open questions

None — every dimension is locked: metric, OOM-handling story, quant, contexts, wall budget, concurrency. The result will be honest either way.
