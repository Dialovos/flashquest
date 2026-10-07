# Contribution and prior-work map

Reviewed 2026-10-03 against primary papers, author repositories, and the local
INT4 implementation. The defensible present description is an engineering
integration of established sparse selection and KV quantization. A narrower
metadata-sharing optimization remains a candidate contribution whose practical
value and differentiation need evidence.

## Established mechanisms

| Primary work | Mechanism relevant here | Overlap and distinction |
| --- | --- | --- |
| [Quest, ICML 2024, paper v2](https://arxiv.org/html/2406.10774v2), [author code](https://github.com/mit-han-lab/Quest) | Current queries score pages using channel-wise key minima/maxima, then select important pages. The full cache remains available for later queries. | FlashQuest uses this page criticality rule and reversible read selection. These are established mechanisms; a sparse read budget does not itself reduce persistent cache capacity. |
| [KIVI, ICML 2024](https://arxiv.org/html/2402.02750v2), [author code](https://github.com/jy-yuan/KIVI) | Asymmetric low-bit keys use channel grouping; values use token grouping; a recent residual cache stays in higher precision. | FlashQuest's completed-page channel-wise K, token-wise V, affine minimum/scale, and BF16 tail follow this design family. Its INT4 page grouping and packing are implementation choices, rather than a new quantizer. |
| [Q-Hitter, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/file/bbb7506579431a85861a05fff048d3e1-Paper-Conference.pdf), [author code](https://github.com/VITA-Group/Q-Hitter) | Token selection combines accumulated attention importance with quantization error to retain important, quantization-friendly tokens. | Joint sparse/quantized KV optimization predates this project. Q-Hitter's historical token eviction differs from current-query page reads with all history retained. |
| [QServe, 2024/MLSys 2025](https://arxiv.org/abs/2405.04532), [author system](https://github.com/mit-han-lab/omniserve) | W4A8KV4 co-design includes optimized fused attention over quantized KV. | Packed low-bit KV loads with dequantization inside attention already have a strong precedent. Porting this principle to Triton or a laptop GPU does not by itself establish algorithmic novelty. |
| [LeanKV, December 2024](https://arxiv.org/html/2412.03131v1) | Mixed-precision K/V storage and token pruning share an importance policy, with per-head budgets and flexible paging. | The broad compression-plus-sparsity idea is established. Its heterogeneous precision and physical pruning differ from the present uniform INT4 persistent store. |
| [LServe, February 2025/MLSys 2025](https://arxiv.org/html/2502.14866v1), [author code](https://github.com/mit-han-lab/omniserve) | Static/streaming heads, query-aware dynamic page selection, and KV quantization run in unified sparse kernels. Logical subpages use min/max scores; physical pages are selected hierarchically, and selections can be reused across steps. | This is a closer systems precedent than citing Quest and KIVI separately. The broad fused sparse-plus-quantized attention claim is already covered. |
| [Twilight, NeurIPS 2025](https://openreview.net/pdf/690f5f6390bd74e24fcac72c3bcf39e1a480e996.pdf), [author code](https://github.com/tsinghua-ideal/Twilight) | A selected candidate set is pruned adaptively by attention mass. Its SpGEMV uses INT4 packed keys with on-the-fly dequantization and per-head scale/zero metadata. | Low-bit keys used during sparse selection are established. Its token-level adaptive pruning differs from reusing page/channel affine metadata for a Quest upper-bound score. |
| [Self-Indexing KVCache, AAAI 2026](https://arxiv.org/html/2603.14224v1), [author code](https://github.com/LfieLike/selfindexingkv) | Sign-pattern compressed keys and centroid lookup tables directly approximate query/key similarities for top-k retrieval; custom kernels fuse sparse access and dequantization. | Reusing a compressed representation as a retrieval index is established. Its token-level sign/VQ lookup differs from reading only affine page/channel minimum and scale. |
| [Self-Indexing Attention, 2026 preprint](https://arxiv.org/html/2609.13205v1) | Stored transformed-key signs and existing compression norms provide a token-level index for sparse prefill and decode, compatible with external low-bit KV compression. | This is another close conceptual challenge to metadata serving both compression and selection without a separate index. Its sign/magnitude representation and norm-weighted agreement differ from affine page endpoints. |
| [TurboQuant, April 2025 paper](https://arxiv.org/html/2504.19874v1), [authors' March 2026 account](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | Random rotation followed by scalar quantization; its inner-product variant adds a QJL residual correction. | The validated INT4 path uses unrotated affine quantization. This repository's separate experimental codebook path does not inherit TurboQuant's guarantees or the INT4 results. No author-maintained TurboQuant implementation was verified in this review; similarly named independent repositories were not treated as authoritative. |
| [Minima-KV, August 2026 preprint](https://arxiv.org/html/2608.23834v1) | Recent/anchor pages stay in FP8 and stale pages use packed TQ3, while every live-request page remains addressable. | Retention-preserving low-bit paged storage is also an active design space. This adjacent preprint does not establish the specific affine scoring identity below. |

GQA head mapping and online normalization are established separately: multiple
query heads share KV heads in [GQA](https://arxiv.org/abs/2305.13245), and tiled
online softmax is central to [FlashAttention](https://arxiv.org/abs/2205.14135).
They should be credited as implementation foundations.

## Exact local mechanism

[Quantization](../src/flashquest/kernel/kv_quant.py) packs two unsigned INT4 codes
into each byte. For completed K pages, each channel has a BF16 minimum `m`
and nonnegative BF16 scale `a`; V has token-wise scalar metadata. Stored codes
are in `0,...,15`, and reconstruction uses `k_hat = c*a + m`.
[Persistent storage](../src/flashquest/cache/persistent_int4.py) retains the full
packed history plus an unquantized BF16 partial-page tail.

[Metadata scoring](../src/flashquest/eager/criticality.py) applies

```text
score(p,q) = sum_d max(q[d]*m[p,d], q[d]*(m[p,d] + 15*a[p,d]))
           = q @ m[p] + 15*relu(q) @ a[p]
```

This is a direct algebraic combination of Quest's interval score and KIVI's
affine range parameters. In exact arithmetic, KIVI's
`a = (max(k)-min(k))/(2**b-1)` already determines both range endpoints;
recovering a maximum from a minimum and scale is a known property of affine
quantization. The two-matmul rewrite is an inference from that algebra, rather
than evidence of a new retrieval algorithm.
[KIVI's parameter definition](https://arxiv.org/html/2402.02750v2),
[Quest's criticality rule](https://arxiv.org/html/2406.10774v2).

In real arithmetic, the interval encloses keys reconstructed from the same
stored affine parameters and clipped INT4 codes. It need not reproduce the
original unquantized page extrema exactly: BF16 scale/range arithmetic, the
epsilon clamp, and clipping can change endpoints and rankings. Top-k agreement
with separately computed original-key summaries is therefore an empirical
question. An interval bound on reconstructed dot products also does not imply
an attention-output or answer-quality guarantee.

The GQA scoring reshape avoids replicating the large cache. Each query head
keeps its own page selection; sinks and recent pages are included.
[The packed kernel](../src/flashquest/kernel/sparse_int4_fwd.py) reconstructs
selected K/V in floating point inside the attention loop. The
[compact alternative](../src/flashquest/kernel/sparse_int4_fwd_compact.py) walks
selected page IDs. These kernels map query heads to KV heads, but do not make
all query heads sharing a KV head jointly consume one packed load. The
[model patch](../src/flashquest/eager/llama_persistent_patch.py) merges the packed
attention result with the BF16 tail using log-sum-exp normalization; prefill
dequantizes the full history. Decode fusion alone establishes no prefill
capacity benefit.

## Narrow possible delta

The candidate is **aligning channel-wise affine K quantization groups with
selection pages, then using the existing minimum/scale tensors as the page
index**, with no additional persistent min/max summary arrays. This should be
described as a metadata-sharing systems optimization until comparative evidence
supports more.

There is a concrete difference from the inspected LServe implementation:
its pinned [context-pooling kernel](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/fused_attention/sparse_utils/ContextPool/context_pool_kernel.cu)
writes explicit FP16 pooled minima/maxima into a statistics region, and its
[page selector](https://github.com/mit-han-lab/omniserve/blob/02b2925aa6fa3b92b06316a1524b7f38922cd9c8/kernels/csrc/fused_attention/sparse_utils/KVPageSelector/KVPageSelectorTemplate.hpp)
loads those extrema. This inspected path is not the same representation as
FlashQuest's shared per-page/channel minimum and scale. LServe's logical/physical
page hierarchy also means that eliminating summaries by sharing quantization
metadata would require choosing compatible grouping. This code-level distinction
does not demonstrate that the shared representation is unprecedented.

KIVI's pinned [quantizer](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/quant/new_pack.py)
already computes minimum, maximum, and scale for packed groups; its
[Llama implementation](https://github.com/jy-yuan/KIVI/blob/876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6/models/llama_kivi.py)
uses packed K/V with a full-precision residual. Self-Indexing KVCache and
[Self-Indexing Attention](https://arxiv.org/html/2609.13205v1) both challenge a
broad “compression doubles as an index” claim, despite their different
representations. No author implementation of the latter was verified here.
The searched primary works
did not establish an identical affine page-parameter implementation, but absence
from this bounded review cannot certify novelty.

For page size 64, an extra pair of BF16 min/max vectors costs `4D` bytes per
page/head, versus `32D` bytes of packed K payload. Sharing avoids that
incremental 12.5% of the K payload, or 6.25% of combined packed K/V payload
before other metadata, tails, padding, and weights. These are analytical
accounting ratios, not total-device memory savings or latency measurements.

## Evidence needed for a contribution claim

1. Compare identical INT4 caches and queries with shared affine metadata versus
   independently stored BF16 min/max summaries. Keep the score algebra,
   grouping, selection budget, attention kernel, and sink/tail policy matched.
   Measure summary construction/update cost as well as score time and bytes.
2. Use real post-RoPE K/Q from every layer/head of held-out retrieval prompts,
   rather than only Gaussian tensors. Report endpoint errors, epsilon-clamped
   channels, top-k and effective-selection agreement, boundary margins,
   true-score recall, and retained attention mass. Investigate disagreements
   without changing the pre-specified confirmatory settings.
3. Validate fused attention against explicit FP32 reconstruction of the
   **same packed INT4 values** and the **same selected tokens**, including tail
   and normalization. An INT4-to-INT8 requantized path changes the values and
   is unsuitable as an exact oracle. Compare with an optimized dense attention
   backend where possible; a slow materialized reference only proves a limited
   fusion benefit.
4. Measure scoring, selection, packed attention, tail/merge, and full generation
   separately at 8k and 32k with warmed repeated trials. Attribute a speed gain
   to metadata sharing only when its own ablation supports that attribution.
   Sparse versus all-pages timing changes pages read and cannot isolate it.
5. Retain the frozen paired retrieval confirmation and matching competitor
   quality. A narrow generator/model result is useful, but extending a general
   language-quality claim requires further tasks and models.
6. Review the exact candidate representation against LServe, Twilight,
   Self-Indexing KVCache, Self-Indexing Attention, KIVI, and later primary work
   before publication.
   A first-of-its-kind claim needs stronger literature/code evidence than
   missing search matches, plus a useful measured result.

The current roadmap can justify a reproducible engineering reference even if
this optimization saves little time or fails a quality gate. A research pivot
should be conditional on a measured advantage of the precise shared metadata
representation or a materially better fused implementation; the broad hybrid
description is insufficient.
