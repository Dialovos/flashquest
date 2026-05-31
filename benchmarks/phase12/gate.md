# Phase 12 Gate Verdicts

## Gate 1a — Verifier Economics

**Rule:** PASS iff `ratio_q4 ≤ 1.6` OR `ratio_q8 ≤ 2.4`

**Verdict: FAIL** — neither condition cleared.

### Results (Llama-3.2-3B shape, 32k ctx, retention=0.20, k_pages=109)

| Measurement         | Value (ms) | Ratio vs q=1 |
|---------------------|------------|--------------|
| T_decode_q1         | 1.213       | 1.00×        |
| T_verify_q2         | 8.159       | 6.73×        |
| T_verify_q4         | 8.327       | 6.87×        |
| T_verify_q8         | 8.274       | 6.82×        |

**Peak VRAM:** 35.1 MiB

### Notes

- The S_q>1 kernel (`flash_attn_sparse_int4_fwd_compact_sq`) is register-bound:
  Task 1 measured 255 regs / ~30 B spill / 44 KB shared at SQ_MAX=16, num_warps=8, num_stages=1.
- The flat ratio across q=2/4/8 (~6.7–6.9×) confirms the kernel is not amortizing KV page
  loads across query rows — overhead dominates at all tested q values.
- The S_q=1 baseline kernel achieves 1.21 ms; the verify kernel costs 8.2–8.3 ms regardless
  of q, suggesting the bottleneck is fixed launch overhead or register/spill pressure rather
  than memory bandwidth.
- Gate 1a thresholds (1.6× at q=4, 2.4× at q=8) are not close to being met.

---

## Gate 1b

*(placeholder — next task)*
