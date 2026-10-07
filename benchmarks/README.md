# Benchmarks and evidence

Start with the [final comparison report](validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md)
and the [research decision](../docs/research-decision.md).

## `validation/` — 2026-10 validation

Each run folder is named by the SHA-256 identity of its configuration: model, source, environment
and protocol. A changed input therefore creates a new folder instead of overwriting earlier
evidence. Every record names the commit it was measured at. Raw prompts, model answers and memory
traces were kept out of the repository; the records hold their hashes and the derived results.
The scripts that built and audited the comparison report run only beside that raw data, so they
are kept with it. The report records their SHA-256 hashes.

| Folder | Contents | Key records |
| --- | --- | --- |
| `comparison/` | Final descriptive comparison of FlashQuest, llama.cpp and vLLM | [summary.md](validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.md), [summary.json](validation/comparison/652ab6a300eb34fedfc0a5ccb2769032aef8d1947f8a7561523e04cd8e752b6b/summary.json) |
| `quality/` | Retrieval runs comparing dense, all-pages INT4 and sparse INT4 on identical prompts | See [quality runs](#quality-runs) |
| `confirmation/` | Nine-endpoint non-inferiority summaries | [Final family](validation/confirmation/9a279d05ca13c7038902ebd26fa9244b3b566240d91407c3a6261efef8d724d6/summary.json); [earlier partial group](validation/confirmation/699b3aecbca4be9cf65f52e08c2f3adc49fe6ea4b4a9eeb043109f79b3e7e2c7/summary.json) |
| `protocols/` | Protocols frozen before data collection | [Confirmation](validation/protocols/9f2f55af018bf2dc27efc97cf0d90940acf638fc08aeb2ae23330318b63226aa.json), [pilot](validation/protocols/845e356c713ead1ff7abf8d2273ef9a968e69e850287368cf6e18f9190a7bf9e.json), plus four timing-block protocols |
| `ablation/` | Sparse vs. all-pages INT4 timing blocks | See [timing blocks](#timing-blocks) |
| `competitors/` | llama.cpp and vLLM runs | Final [performance matrix](validation/competitors/3a5fa9de623a7d2a49b853f6588bc72821ce36a76542b6791a08abba677a0653/schedule.json) (32 cells) and [quality matrix](validation/competitors/ee7c318de630e7cdbd3e9b7859134df60a611e761c750bd2fb32fa7cf3133ed9/schedule.json) (8 cells). The other folders are setup smokes and failed or interrupted attempts, kept as evidence. |
| `contrib/` | Operator-level reports: metadata scoring, packed attention, decode components | [summary.md](validation/contrib/summary.md) |
| `selection-flips/` | Diagnostic of pages where metadata scoring and exact summaries select differently | [summary.md](validation/selection-flips/summary.md) |
| `environment/` | Backend environment records | `backends.json` |

### Quality runs

| Run | Context | Seeds, examples | Record |
| --- | --- | --- | --- |
| Pilot, retention 0.20 | 4k | 0, 20 per task | [quality.json](validation/quality/dd8f60de24c2957a2120473d9ff3a194b64b22d85b2adfecca3aab485092f104/quality.json) |
| Pilot, retention 0.20 | 8k | 0, 20 per task | [quality.json](validation/quality/85c9abbb694ffb21ba5c03336188658ca2d0e9dff0790a9a9b62cb6b28c2ef07/quality.json) |
| Pilot, retention 0.20 (failed screen) | 32k | 0, 20 per task | [quality.json](validation/quality/52b2599c93490bdc948431f43bf4fd2294273a00fac583461fe10bd5b0ba9f93/quality.json) |
| Pilot, retention 0.25 | 32k | 0, 20 per task | [quality.json](validation/quality/2baea9556c7ecdb9bb4213e0c02caf7444820468d36d912879d1a63190903f97/quality.json) |
| Confirmation, final group | 4k | 1–5, 100 per task | [quality.json](validation/quality/7d9921cb0d5b6167c5303932475ccb71402ece5ca9066d7e067f23dab4d9cdef/quality.json) |
| Confirmation, final group | 8k | 1–5, 100 per task | [quality.json](validation/quality/6c938a6b4515ffaf0de9ed8bf833e521584887705e4665e43b51e9e94c3aafb1/quality.json) |
| Confirmation, final group | 32k | 1–5, 100 per task | [quality.json](validation/quality/07ce7aa457aa9070cae386a86c9bcd00fb501e2d29c99aca56410183f74c98c4/quality.json) |
| Confirmation, earlier group (before an OS update) | 4k | 1–5, 100 per task | [quality.json](validation/quality/0526a7546dbef2cb2e78c919423d65ab0b55e3457ea1aa845edc2cf5502bdb69/quality.json) |
| Confirmation, earlier group (before an OS update) | 8k | 1–5, 100 per task | [quality.json](validation/quality/7cec2647f00356ca86a526ff3bb2a16c28a20661d6dcf1fd8ba5d6e7c1427384/quality.json) |

The two confirmation groups are kept separate and not pooled. The 32k runs use retention 0.25;
the others use 0.20. Every run also includes dense and all-pages arms.

### Timing blocks

| Block | Context | Commit | Summary |
| --- | --- | --- | --- |
| Final | 8k, retention 0.20 | `5e22bf5` | [summary.json](validation/ablation/bd5a7ab68ed2ec69e8e916c27968c57c68783cb3f4514293621fa0c542960f40/summary.json) |
| Final | 32k, retention 0.25 | `5e22bf5` | [summary.json](validation/ablation/efd8bfe82549542f6cc7994414e6af11b0943c09404647a9457cf0e42ca4e559/summary.json) |
| Original | 8k, retention 0.20 | `925eaab` | [summary.json](validation/ablation/d86279b66c726c5697f408aabfd346170f8f072990bf95f7e0155416643a29a8/summary.json) |
| Original | 32k, retention 0.25 | `35c56db` | [summary.json](validation/ablation/c921467929e57dad7293c868610d5dc39cdd0c01275c515a91d9d3d63060d66b/summary.json) |
| Repeat | 8k, retention 0.20 | `76f00fe` | [summary.json](validation/ablation/f22ba06c8c74dd6aa294fc69707404ead02002f9c3958ab4b0015af32acaf7cd/summary.json) |
| Repeat | 32k, retention 0.25 | `76f00fe` | [summary.json](validation/ablation/0b19179a04ffe455f53d821d4cb827c38958c6d2375e578db1c8f51ce5a78087/summary.json) |

All six blocks agree: about 1.8× at 32k and no gain at 8k. The final blocks are the ones in the
comparison report.

## Historical v1.0 results (top-level files)

Everything outside `validation/` dates from v1.0 development (May 2026). That covers the
`phase*` files and folders, `baselines.json`, `fa2_profile.json`, `llamacpp_8k.txt`,
`vllm_4k.json`, and the two `phase8a`/`phase9` helper scripts. These files are kept for
provenance only:

- the competitor comparisons used mismatched timing;
- the 32k runs allocated more memory than the labelled 4 GB GPU;
- the hardware metadata was hardcoded.

See [Historical results](../README.md#historical-results-v10-may-2026) in the README.
