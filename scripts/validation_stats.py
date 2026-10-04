"""Frozen, paired retrieval confirmation with conservative exact binomial bounds.

The confirmatory family has nine task/context endpoints. Each difference lower
bound subtracts an exact loss upper bound from an exact gain lower bound, using
alpha=0.05/18 per bound. No independence across endpoints is needed by the union
bound. Within an endpoint, binomial coverage assumes independent Bernoulli trials
from this fixed synthetic prompt generator. Observed accuracy floors are screens,
not confidence guarantees about population accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from bench_common import (
    REPO_ROOT,
    SCHEMA_VERSION,
    canonical_identity,
    content_hash,
    file_hash,
    validate_export,
    write_record,
)

TASKS = ["single", "multikey", "multivalue"]
CONTEXT_RETENTIONS = [(4096, .20), (8192, .20), (32768, .25)]
SEEDS = [1, 2, 3, 4, 5]


def load_protocol(path: Path) -> dict:
    """Accept the named, hash-addressed protocol; reject changed decision rules."""
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if path.stem != content_hash(protocol):
        raise ValueError("confirmation protocol filename differs from its hash")
    expected = {
        "version": 1, "kind": "paired-retrieval-confirmation",
        "tasks": TASKS, "seeds": SEEDS, "examples_per_task_seed": 20,
        "examples_per_endpoint": 100, "excluded_tuning_seeds": [0],
        "contexts": [{"ctx_len": c, "retention": r} for c, r in CONTEXT_RETENTIONS],
        "scorer": "all-expected-substrings", "generation": "greedy-eos-or-limit",
        "max_new_tokens": 128, "page_size": 64, "num_sinks": 4,
        "window_pages": 2, "kv_bits": 4,
        "family_alpha": .05, "confirmatory_endpoints": 9,
        "bound_alpha": .05 / 18, "noninferiority_margin": .10,
        "observed_sparse_accuracy_floor": .80, "observed_dense_accuracy_floor": .80,
        "diagnostic_bound_alpha": .025,
        "minimum_lower_difference_exclusive": -.10,
        "source_policy": "one-clean-identical-source-snapshot",
        "stopping": "fixed-count-no-favorable-early-stop-or-post-hoc-expansion",
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("confirmation protocol differs from frozen decisions")
    model = protocol.get("model", {})
    if (not isinstance(model.get("model"), str) or
            not isinstance(model.get("revision"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", model["revision"]) or
            not isinstance(model.get("content_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", model["content_sha256"])):
        raise ValueError("confirmation protocol lacks immutable model identity")
    inputs = protocol.get("input_sources", [])
    if ({row.get("path") for row in inputs} !=
            {"src/flashquest/eval/niah.py", "data/PaulGrahamEssays.json"} or
            len(inputs) != 2 or any(not isinstance(row.get("sha256"), str) or
                                   not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                                   for row in inputs)):
        raise ValueError("confirmation protocol lacks generator/corpus identity")
    validate_export(protocol)
    return protocol


def validate_run_settings(args, model: dict, protocol: dict) -> None:
    """Check the frozen choices before model loading or fresh prompt generation."""
    retention = dict(CONTEXT_RETENTIONS).get(args.ctx_len)
    if (args.tasks != protocol["tasks"] or args.seeds != protocol["seeds"] or
            args.n_samples != protocol["examples_per_task_seed"] or
            args.retentions != [retention, 1.0] or retention is None or
            any(getattr(args, key) != protocol[key]
                for key in ("max_new_tokens", "page_size", "num_sinks", "window_pages")) or
            any(model.get(key) != value for key, value in protocol["model"].items())):
        raise ValueError("quality run settings differ from frozen confirmation")
    for row in protocol["input_sources"]:
        if file_hash(REPO_ROOT / row["path"]) != row["sha256"]:
            raise ValueError("generator/corpus differs from frozen confirmation")


@lru_cache(maxsize=32)
def _log_combinations(n: int) -> tuple[float, ...]:
    base = math.lgamma(n + 1)
    return tuple(base - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1))


def _binomial_range_probability(n: int, p: float, start: int, end: int) -> float:
    """Stable log-sum of a binomial tail, including p=0/1 endpoints."""
    if p == 0:
        return float(start == 0)
    if p == 1:
        return float(end == n)
    log_p, log_q = math.log(p), math.log1p(-p)
    combinations = _log_combinations(n)
    terms = [combinations[k] + k * log_p + (n - k) * log_q
             for k in range(start, end + 1)]
    largest = max(terms)
    return min(1.0, math.exp(largest) * math.fsum(math.exp(t - largest) for t in terms))


def exact_bounds(k: int, n: int, alpha: float) -> tuple[float, float]:
    """One-sided Clopper-Pearson lower/upper bounds, each with error alpha.

    Invert P_p(X>=k)=alpha for the lower bound and P_p(X<=k)=alpha
    for the upper. These are numerical inversions of exact binomial tails,
    not a normal approximation. Zero/all success bounds remain non-degenerate.
    """
    if (type(k) is not int or type(n) is not int or n < 1 or not 0 <= k <= n or
            not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("invalid binomial count or tail probability")
    lower = 0.0
    if k:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if _binomial_range_probability(n, mid, k, n) < alpha:
                lo = mid
            else:
                hi = mid
        lower = lo  # Outward rounding preserves a conservative lower bound.
    upper = 1.0
    if k != n:
        lo, hi = 0.0, 1.0
        for _ in range(70):
            mid = (lo + hi) / 2
            if _binomial_range_probability(n, mid, 0, k) > alpha:
                lo = mid
            else:
                hi = mid
        upper = hi
    return lower, upper


def paired_difference(reference: list[bool], selected: list[bool], alpha: float) -> dict:
    """Return paired gains/losses and an at-least 1-2*alpha lower bound."""
    if (not reference or len(reference) != len(selected) or
            any(type(hit) is not bool for hit in [*reference, *selected])):
        raise ValueError("paired outcomes must be complete binary observations")
    n = len(reference)
    gains = sum(not a and b for a, b in zip(reference, selected, strict=True))
    losses = sum(a and not b for a, b in zip(reference, selected, strict=True))
    gain_lower, gain_upper = exact_bounds(gains, n, alpha)
    loss_lower, loss_upper = exact_bounds(losses, n, alpha)
    return {"n": n, "reference_hits": sum(reference), "selected_hits": sum(selected),
            "gains": gains, "losses": losses,
            "difference": (gains - losses) / n, "bound_alpha": alpha,
            "gain_lower": gain_lower, "gain_upper": gain_upper,
            "loss_lower": loss_lower, "loss_upper": loss_upper,
            "difference_lower": gain_lower - loss_upper}


def _local_evidence_path(value: str, root: Path) -> Path:
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("raw evidence path is not relative")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("raw evidence escapes the project")
    return path


def checked_quality(path: Path, protocol: dict, root: Path = REPO_ROOT) -> dict:
    """Rebuild hits and public samples from the hash-verified local evidence."""
    from phase6_run_ruler_4k_int4 import screen_verdict, validate_cell

    path = path.resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    validate_export(record)
    identity = canonical_identity(record["identity"])
    if (content_hash(identity) != record["run_identity"] or
            record["schema_version"] != SCHEMA_VERSION or
            record["protocol"] != protocol or
            identity["protocol_sha256"] != content_hash(protocol)):
        raise ValueError("quality identity/protocol mismatch")
    cfg = identity["config"]
    args = SimpleNamespace(**cfg)
    retention = dict(CONTEXT_RETENTIONS).get(cfg["ctx_len"])
    if (cfg["tasks"] != protocol["tasks"] or cfg["seeds"] != protocol["seeds"] or
            cfg["n_samples"] != protocol["examples_per_task_seed"] or
            cfg["retentions"] != [retention, 1.0] or retention is None or
            cfg["model"] != identity["model"]["model"] or
            cfg["revision"] != identity["model"]["revision"] or
            any(cfg[key] != protocol[key] for key in
                ("max_new_tokens", "page_size", "num_sinks", "window_pages", "kv_bits")) or
            any(identity["model"].get(key) != value for key, value in protocol["model"].items()) or
            identity["source"].get("dirty") is not False or
            not isinstance(identity["source"].get("commit"), str) or
            not re.fullmatch(r"[0-9a-f]{40}", identity["source"]["commit"])):
        raise ValueError("quality evidence differs from frozen confirmation settings")
    for row in protocol["input_sources"]:
        if identity["source"]["files"].get(row["path"]) != row["sha256"]:
            raise ValueError("quality generator/corpus differs from protocol")
    if record["raw_evidence"]["path"] != f"artifacts/quality/{record['run_identity']}/raw.json":
        raise ValueError("raw quality evidence is not in its run directory")
    raw_path = _local_evidence_path(record["raw_evidence"]["path"], root)
    if file_hash(raw_path) != record["raw_evidence"]["sha256"]:
        raise ValueError("raw quality evidence hash differs")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    manifest_file = json.loads((raw_path.parent / "manifest.json").read_text(encoding="utf-8"))
    examples = manifest_file["examples"]
    if (raw.get("identity") != identity or raw.get("run_identity") != record["run_identity"] or
            manifest_file["run_identity"] != record["run_identity"] or
            any(raw.get(key) != record.get(key) for key in
                ("schema_version", "protocol", "status", "error", "manifest", "screen"))):
        raise ValueError("raw/public quality identity or status differs")
    expected_ids = [f"{task}:{seed}:{i}" for task in TASKS for seed in SEEDS for i in range(20)]
    if (len(examples) != 300 or [e["example_id"] for e in examples] != expected_ids or
            content_hash(examples) != record["manifest"]["sha256"] or
            cfg["manifest_sha256"] != record["manifest"]["sha256"] or
            record["manifest"]["count"] != 300):
        raise ValueError("quality manifest is incomplete or differs from its hash")
    for example in examples:
        ids = example["input_ids"]
        if (not ids or any(type(token) is not int or token < 0 for token in ids) or
                content_hash(ids) != example["input_sha256"] or
                example["example_id"] != f"{example['task']}:{example['seed']}:{example['index']}" or
                not example["expected"] or
                any(not isinstance(value, str) or not value for value in example["expected"])):
            raise ValueError("quality manifest token IDs or expected values differ")
    lengths = [len(e["input_ids"]) for e in examples]
    if (record["manifest"]["min_prompt_tokens"] != min(lengths) or
            record["manifest"]["max_prompt_tokens"] != max(lengths) or
            cfg["actual_max_prompt_tokens"] != max(lengths) or
            cfg["cache_capacity"] != max(lengths) + protocol["max_new_tokens"]):
        raise ValueError("quality prompt lengths or cache capacity differ")
    groups = {(task, seed): [e for e in examples if e["task"] == task and e["seed"] == seed]
              for task in TASKS for seed in SEEDS}
    arm_names = ["dense", f"int4-r{retention:g}", "int4-r1"]
    expected_cells = [(arm, task, seed) for arm in arm_names for task in TASKS for seed in SEEDS]
    if (len(raw["cells"]) != len(record["cells"]) or
            [(c["arm"], c["task"], c["seed"]) for c in raw["cells"]] !=
            expected_cells[:len(raw["cells"])]):
        raise ValueError("quality cells are duplicated, incomplete within a cell, or out of order")
    for raw_cell, public_cell in zip(raw["cells"], record["cells"], strict=True):
        if (not isinstance(raw_cell.get("wall_s"), (int, float)) or
                not math.isfinite(raw_cell["wall_s"]) or raw_cell["wall_s"] < 0):
            raise ValueError("invalid quality cell duration")
        if raw_cell["retention"] != (None if raw_cell["arm"] == "dense" else
                                     1.0 if raw_cell["arm"] == "int4-r1" else retention):
            raise ValueError("quality cell arm/retention mismatch")
        validate_cell(raw_cell, groups[(raw_cell["task"], raw_cell["seed"])],
                      protocol["max_new_tokens"])
        exported = {key: raw_cell[key] for key in
                    ("task", "seed", "arm", "retention", "hits", "total", "wall_s")}
        exported["samples"] = []
        for sample in raw_cell["samples"]:
            if (sample["termination"] == "limit" and
                    sample["output_tokens"] != protocol["max_new_tokens"]):
                raise ValueError("quality limit termination has a different output count")
            exported["samples"].append({key: sample[key] for key in
                                        ("example_id", "input_sha256", "prompt_tokens", "hit",
                                         "output_tokens", "termination")} |
                                       {"generated_sha256": content_hash(sample["generated"])})
        if exported != public_cell:
            raise ValueError("quality public samples differ from raw/scored evidence")
    if screen_verdict(record["cells"], args) != record["screen"]:
        raise ValueError("quality screen differs from reconstructed cells")
    if record["status"] == "complete" and (len(raw["cells"]) != 45 or record["error"] is not None):
        raise ValueError("complete quality evidence is missing cells or reports an error")
    if record["status"] not in {"complete", "incomplete", "execution_error"}:
        raise ValueError("unknown quality completion status")
    return {"record": record, "identity": identity, "path": path,
            "complete": record["status"] == "complete", "retention": retention}


def summarize(paths: list[Path], protocol: dict, *, root: Path = REPO_ROOT) -> dict:
    """Apply the frozen family to complete records; absent/incomplete stays inconclusive."""
    checked = [checked_quality(path, protocol, root) for path in paths]
    contexts = [row["identity"]["config"]["ctx_len"] for row in checked]
    if len(contexts) != len(set(contexts)):
        raise ValueError("confirmation context appears more than once")
    for section in ("model", "source", "environment"):
        if checked and any(row["identity"][section] != checked[0]["identity"][section]
                           for row in checked[1:]):
            raise ValueError("confirmation records differ in model/source/environment")
    if checked and any(row["identity"]["config"]["runtime"] !=
                       checked[0]["identity"]["config"]["runtime"] for row in checked[1:]):
        raise ValueError("confirmation records differ in realized runtime")
    endpoints = []
    for row in checked:
        if not row["complete"]:
            continue
        record = row["record"]
        context = row["identity"]["config"]["ctx_len"]
        for task in TASKS:
            arms = {}
            paired_ids = None
            for arm in ["dense", f"int4-r{row['retention']:g}", "int4-r1"]:
                samples = [sample for cell in record["cells"]
                           if cell["task"] == task and cell["arm"] == arm
                           for sample in cell["samples"]]
                ids = [(s["example_id"], s["input_sha256"], s["prompt_tokens"]) for s in samples]
                if paired_ids is not None and ids != paired_ids:
                    raise ValueError("confirmation arms do not have exact paired inputs")
                paired_ids = ids
                arms[arm] = [s["hit"] for s in samples]
            sparse = arms[f"int4-r{row['retention']:g}"]
            primary = paired_difference(arms["dense"], sparse, protocol["bound_alpha"])
            dense_rate = primary["reference_hits"] / primary["n"]
            sparse_rate = primary["selected_hits"] / primary["n"]
            if dense_rate < protocol["observed_dense_accuracy_floor"]:
                status = "inconclusive-weak-dense-baseline"
            elif (sparse_rate < protocol["observed_sparse_accuracy_floor"] or
                  primary["difference_lower"] <= protocol["minimum_lower_difference_exclusive"]):
                status = "fail"
            else:
                status = "pass"
            endpoints.append({"ctx_len": context, "task": task, "retention": row["retention"],
                              "status": status, "dense_rate": dense_rate, "sparse_rate": sparse_rate,
                              "all_pages_rate": sum(arms["int4-r1"]) / 100,
                              "sparse_minus_dense": primary,
                              "diagnostic_all_pages_minus_dense": paired_difference(
                                  arms["dense"], arms["int4-r1"], protocol["diagnostic_bound_alpha"]),
                              "diagnostic_sparse_minus_all_pages": paired_difference(
                                  arms["int4-r1"], sparse, protocol["diagnostic_bound_alpha"])})
    complete = set(contexts) == {c for c, _ in CONTEXT_RETENTIONS} and all(
        row["complete"] for row in checked)
    if not complete or any(e["status"].startswith("inconclusive") for e in endpoints):
        status = "inconclusive"
    else:
        status = "pass" if all(e["status"] == "pass" for e in endpoints) else "fail"
    evidence = [{"path": row["path"].relative_to(root).as_posix(),
                 "sha256": file_hash(row["path"]), "run_identity": row["record"]["run_identity"],
                 "ctx_len": row["identity"]["config"]["ctx_len"],
                 "status": row["record"]["status"]} for row in checked]
    result = {"version": 1, "kind": "paired-retrieval-confirmation-summary",
              "protocol_sha256": content_hash(protocol), "status": status,
              "complete": complete, "family_confidence_at_least": .95,
              "confirmatory_endpoints": 9, "completed_endpoints": len(endpoints),
              "evidence": sorted(evidence, key=lambda e: e["ctx_len"]),
              "endpoints": sorted(endpoints, key=lambda e: (e["ctx_len"], TASKS.index(e["task"]))),
              "limitations": ["fixed synthetic retrieval generator and pinned model only",
                              "binomial coverage assumes independent within-endpoint Bernoulli trials",
                              "observed accuracy floors are screening criteria, not confidence guarantees",
                              "diagnostic all-pages bounds are not simultaneous confirmatory endpoints",
                              "noninferiority failure does not prove inferiority",
                              "no general language quality, competitive speed, or capacity claim"]}
    result["run_identity"] = content_hash(result)
    validate_export(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--quality", nargs="+", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    result = summarize(args.quality, protocol)
    out = args.out or (REPO_ROOT / "benchmarks" / "validation" / "confirmation" /
                       result["run_identity"] / "summary.json")
    write_record(out, result)
    print(f"Confirmation: {result['status']}; {result['completed_endpoints']}/9 endpoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
