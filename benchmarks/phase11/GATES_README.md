# Phase 11 — eval gates (Tasks 9 & 10): how to run & read

Implementation (Tasks 1–7) is committed. These are the quality/speed gates that
decide whether K3-V3 *calibrated* ships as the default (Task 11). Each command
writes a JSON you can read directly — no need to trust terminal scrollback.

## Run (one at a time; ~60–80 min RULER + ~20 min decode)

```bash
cd /home/hoang/code/personal/active/flashquest

# Task 9 — RULER NIAH 4k quality gate (THE gate)
nice -n 19 .venv/bin/python scripts/phase11_run_ruler_4k_calibrated.py \
  --codebook calibrated --n-samples 20 --ctx-len 4096 --retention 0.20 \
  --out benchmarks/phase11/ruler_4k_calibrated.json
nice -n 19 .venv/bin/python scripts/phase11_run_ruler_4k_calibrated.py \
  --codebook paper --n-samples 20 --ctx-len 4096 --retention 0.20 \
  --out benchmarks/phase11/ruler_4k_paper.json

# Task 10 — decode tok/s @ 32k speed gate (run only if Task 9 passes)
nice -n 19 .venv/bin/python scripts/phase11_bench_decode_32k.py \
  --codebook paper --ctx 32768 --retention 0.20 \
  --out benchmarks/phase11/decode_32k_paper.json
nice -n 19 .venv/bin/python scripts/phase11_bench_decode_32k.py \
  --codebook calibrated --ctx 32768 --retention 0.20 \
  --out benchmarks/phase11/decode_32k_calibrated.json
```

> The RULER script exits 1 when the gate fails — that's an intentional signal,
> NOT a crash. Judge by the JSON's `gate_all_ge_95`, not the exit code.

## Read the results

```bash
.venv/bin/python - <<'PY'
import json, glob
for f in sorted(glob.glob("benchmarks/phase11/ruler_4k_*.json")):
    d = json.load(open(f))
    print(f"\n{d['codebook']}  (n={d['n_samples']}, ctx={d['ctx_len']}, ret={d['retention']})")
    for t, v in d["results"].items():
        print(f"  {t:11s} {v['hits']}/{v['total']} = {100*v['rate']:.0f}%")
    if d.get("gate_all_ge_95") is not None:
        print(f"  GATE (all >=95%): {'PASS' if d['gate_all_ge_95'] else 'FAIL'}")
for f in sorted(glob.glob("benchmarks/phase11/decode_32k_*.json")):
    d = json.load(open(f))
    print(f"{d['codebook']:10s} decode {d['decode_tok_per_s']} tok/s @32k, peak {d['peak_vram_mib']} MiB")
PY
```

## Gate criteria → Task 11 decision

- **Quality (Task 9):** calibrated `single/multikey/multivalue` all **≥95%**.
  Pass → calibrated ships as default. Fail → keep `--kv-bits 4` default;
  consider Phase 11b (per-head). (Smoke at n=1 already hit 100% on all three.)
- **Speed (Task 10):** calibrated decode within **−5%** of paper (≥ ~2.49 tok/s
  if paper ≈ 2.62). Pass → proceed to ship.
- **Task 11 (only if both pass):** update `DOC.md`, `README.md`, `CHANGELOG.md`,
  write `docs/PHASES/phase-11-notes.md`, tag `phase-11`. This changes the
  user-facing default — do it deliberately, after confirming the JSON numbers.
