# Phase 11 — Tighter Entry-Gate Probe Addendum

Tests whether per-layer codebook specificity moves more samples than uniform proxies.
Baseline (paper) generations are compared against each proxy's generation; diff count is samples (of N=3) where generated tokens differ.

| proxy | diffs | sample diffs |
|---|---|---|
| paper | 0/3 | [False, False, False] |
| mean | 1/3 | [True, False, False] |
| layer-0 | 3/3 | [True, True, True] |
| layer-14 | 3/3 | [True, True, True] |
| layer-27 | 2/3 | [True, False, True] |
| per-layer-proper | 3/3 | [True, True, True] |

**Reading: per-layer specificity moves additional samples vs uniform mean → per-layer scope is JUSTIFIED. Proceed with Tasks 2-11 as planned.**
