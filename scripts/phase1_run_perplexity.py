"""Sweep retention on Wikitext-2 (raw); compare ppl against dense baseline."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashquest.eager.llama_patch import patch_llama_for_quest_eager
from flashquest.eval.perplexity import perplexity


def main() -> None:
    name = "unsloth/Llama-3.2-1B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    ids = ids[: 8192]

    results: dict[str, dict] = {"input_tokens": ids.numel()}

    def fresh_model() -> torch.nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).cuda().eval()

    # Window 2048 / stride 1024: keeps the explicit per-token attention mask
    # under ~256 MB at 32 heads BF16. Dense uses the same window for parity.
    WIN, STRIDE = 2048, 1024

    import gc

    def _free():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("dense baseline ...")
    m = fresh_model()
    t0 = time.perf_counter()
    ppl = perplexity(m, ids, window=WIN, stride=STRIDE)
    dt = time.perf_counter() - t0
    results["dense"] = {"ppl": ppl, "elapsed_s": dt, "window": WIN, "stride": STRIDE}
    print(f"  ppl={ppl:.4f} ({dt:.1f}s)")
    del m
    _free()

    PAGE_SIZE = int(__import__("os").environ.get("FLASHQUEST_PAGE_SIZE", "64"))
    results["page_size"] = PAGE_SIZE

    for r in [1.0, 0.5, 0.25, 0.10]:
        print(f"retention={r} (page_size={PAGE_SIZE}) ...")
        m = fresh_model()
        # window_pages scales inversely with page size to keep the recency
        # window at ~128 tokens regardless of page granularity.
        win_pages = max(1, 128 // PAGE_SIZE)
        patch_llama_for_quest_eager(
            m, retention=r, num_sinks=4, window_pages=win_pages, page_size=PAGE_SIZE
        )
        t0 = time.perf_counter()
        ppl = perplexity(m, ids, window=WIN, stride=STRIDE)
        dt = time.perf_counter() - t0
        delta_pct = 100.0 * (ppl - results["dense"]["ppl"]) / results["dense"]["ppl"]
        results[f"retention_{r}"] = {"ppl": ppl, "elapsed_s": dt, "delta_pct": delta_pct}
        print(f"  ppl={ppl:.4f}  delta={delta_pct:+.2f}%  ({dt:.1f}s)")
        del m
        _free()

    out = Path(__file__).resolve().parents[1] / "benchmarks" / "phase1_perplexity.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
