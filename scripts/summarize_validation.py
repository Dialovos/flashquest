"""Verify saved ablation evidence and summarize paired per-seed pilot ratios."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from bench_common import (
    REPO_ROOT,
    canonical_identity,
    content_hash,
    file_hash,
    validate_export,
    write_record,
)
from gpu_memory import checked_memory
from run_validation_ablation import checked_cell, schedule


def summarize(path: Path) -> dict:
    path = path.resolve()
    saved = json.loads(path.read_text())
    identity = canonical_identity(saved["identity"])
    if content_hash(identity) != saved["run_identity"] or content_hash(saved["protocol"]) != identity["protocol_sha256"]:
        raise ValueError("schedule identity or protocol changed")
    config = identity["config"]
    expected = schedule(config["contexts"], config["seeds"], config["retention"])
    if saved["protocol"]["cells"] != expected or [c["cell"] for c in saved["cells"]] != expected[:len(saved["cells"])]:
        raise ValueError("schedule does not preserve its balanced order")
    if saved["status"] == "complete" and (len(saved["cells"]) != len(expected) or
                                            any(c["status"] != "complete" for c in saved["cells"])):
        raise ValueError("schedule completion differs from its cells")
    records = {}
    for entry in saved["cells"]:
        if entry["status"] != "complete":
            continue
        result_path = REPO_ROOT / entry["result"]["path"]
        if file_hash(result_path) != entry["result"]["sha256"]:
            raise ValueError("cell file changed after observation")
        record = checked_cell(result_path, entry["cell"], {"identity": identity}, config["reps"], config["n_decode"])
        memory = entry["memory"]
        checked_memory(memory, REPO_ROOT, config["reps"])
        records[entry["cell"]["cell_id"]] = (record, memory)
    contexts = []
    for context in config["contexts"]:
        pairs = []
        for seed in config["seeds"]:
            sparse = records.get(f"c{context}-s{seed}-sparse")
            dense = records.get(f"c{context}-s{seed}-all-pages")
            if sparse is None or dense is None:
                continue
            first, second = sparse[0]["identity"], dense[0]["identity"]
            first_cfg = {k: v for k, v in first["config"].items() if k != "retention"}
            second_cfg = {k: v for k, v in second["config"].items() if k != "retention"}
            if first_cfg != second_cfg:
                raise ValueError("paired arms have different inputs/runtime/cache settings")
            pair = {"seed": seed, "decode_ratio": sparse[0]["decode_tok_s"] / dense[0]["decode_tok_s"]}
            for arm, (record, memory) in (("sparse", sparse), ("all-pages", dense)):
                pair[arm] = {"decode_tok_s": record["decode_tok_s"],
                             "decode_sample_min_tok_s": min(s["decode_tok_s"] for s in record["samples"]),
                             "decode_sample_max_tok_s": max(s["decode_tok_s"] for s in record["samples"]),
                             "prefill_tok_s": record["prefill_tok_s"],
                             "end_to_end_tok_s": record["end_to_end_tok_s"],
                             "peak_allocated_mib": record["peak_allocated_mib"],
                             "peak_reserved_mib": record["peak_reserved_mib"],
                             "device_sampled_peak_mib": memory["device_sampled_peak_mib"],
                             "device_baseline_adjusted_peak_mib": memory["device_baseline_adjusted_peak_mib"],
                             "process_tree_rss_sampled_peak_mib": memory["process_tree_rss_sampled_peak_mib"],
                             "actual_interval_median_ms": memory["actual_interval_median_ms"],
                             "actual_interval_max_ms": memory["actual_interval_max_ms"],
                             "device_dropouts": memory["device_dropouts"],
                             "phases": memory["phases"], "runtime": record["config"]["runtime"]}
            pairs.append(pair)
        complete = len(pairs) == len(config["seeds"])
        ratios = [pair["decode_ratio"] for pair in pairs]
        median_ratio = median(ratios) if ratios else None
        context_record = {"ctx_len": context, "pairs": pairs, "paired_seed_count": len(pairs),
                          "median_decode_ratio": median_ratio,
                          "min_decode_ratio": min(ratios) if ratios else None,
                          "max_decode_ratio": max(ratios) if ratios else None,
                          "practical_screen_pass": None}
        if complete and len(pairs) >= 4:
            context_record["practical_screen_pass"] = (
                median_ratio >= saved["protocol"]["minimum_median_ratio"] and
                all(ratio > saved["protocol"]["minimum_each_seed_ratio"] for ratio in ratios))
        contexts.append(context_record)
    result = {"schema_version": saved["schema_version"], "run_identity": saved["run_identity"],
              "schedule": {"path": path.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(path)},
              "status": saved["status"], "contexts": contexts,
              "control": "all-pages INT4 uses the same scoring/top-k/attention path",
              "claims": "pilot ablation only; competitor performance, novelty and 4 GB capacity unproven"}
    validate_export(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("schedule", type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.schedule)
    write_record(args.schedule.parent / "summary.json", result)
    for context in result["contexts"]:
        ratio = context["median_decode_ratio"]
        print(f"{context['ctx_len']}: {context['paired_seed_count']} pairs, median ratio={ratio}, practical screen={context['practical_screen_pass']}")


if __name__ == "__main__":
    main()
