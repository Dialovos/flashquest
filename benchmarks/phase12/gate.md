# Phase 12 Gate Verdicts

## Gate 1a — Verifier Economics

**Rule:** PASS iff `ratio_q4 ≤ 1.6` OR `ratio_q8 ≤ 2.4`

**Verdict: PASS** — both conditions cleared.

### Results (Llama-3.2-3B shape, 32k ctx, retention=0.20, k_pages=109)

| Measurement         | Value (ms) | Ratio vs q=1 |
|---------------------|------------|--------------|
| T_decode_q1         | 1.27        | 1.00×        |
| T_verify_q2         | 1.10        | 0.865×       |
| T_verify_q4         | 1.09        | 0.854×       |
| T_verify_q8         | 1.11        | 0.872×       |

**Peak VRAM:** ~35 MiB

### Diagnostic

**First attempt** — fp32 `input_precision="ieee"` `tl.dot` → T_verify ~8.3 ms, ratios ~6.8×. Kernel
was compute-bound using fp32 emulation; sm_86 has no fp32 tensor cores, so all dot products ran
through CUDA cores. Gate 1a FAIL.

**Fix** — switched to bf16 inputs + fp32 accumulation in `tl.dot`. Tensor cores engaged; T_verify
dropped to ~1.1 ms (~8× speedup). Parity unchanged: atol/rtol 2e-2 holds across all batch sizes.
Gate 1a PASS.

---

## Gate 1b

*(placeholder — next task)*
