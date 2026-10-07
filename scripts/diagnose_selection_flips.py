"""Bounded model recapture explaining affine-metadata page-selection flips.

This producer records scores and selected page IDs, with no timed operators.
One seed-0 prompt and representative layers/steps explain these observations;
they establish no general quality, novelty, speed or capacity claim.
"""
from __future__ import annotations

import argparse
import gc
import math
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import profile_contribution as capture
from bench_common import (
    DEFAULT_MODEL,
    REPO_ROOT,
    SCHEMA_VERSION,
    content_hash,
    error_code,
    export_identity,
    file_hash,
    make_identity,
    model_identity,
    provenance,
    source_identity,
    validate_export,
    write_record,
)

PROTOCOL = {
    "version": 1,
    "kind": "bounded-affine-metadata-selection-flip-diagnostic",
    "capture": "existing model_rows: actual post-RoPE BF16 Q/K; one layer at a time",
    "scores": "actual FP32 two-matmul metadata and separate-BF16-minmax implementations",
    "selection": "actual torch topk; sorted page IDs; sink/window union reported separately",
    "error_explanation": "score perturbation versus boundary gap and affine-endpoint perturbation",
    "page_ids": "zero-based completed-cache pages; page_id*64 starts its token range",
    "limits": "one tuning prompt; no retuning; no general quality, novelty, performance or capacity claim",
}


def measurement_source():
    """Bind this producer, the existing capture implementation and its runtime."""
    source = source_identity()
    names = {
        "scripts/diagnose_selection_flips.py", "scripts/profile_contribution.py",
        "scripts/bench_common.py", "pyproject.toml", "requirements-validation.txt",
        "data/PaulGrahamEssays.json",
    }
    source["files"] = {name: digest for name, digest in source["files"].items()
                       if name.startswith("src/") or name in names}
    source["files"]["scripts/diagnose_selection_flips.py"] = file_hash(Path(__file__))
    source["content_sha256"] = content_hash(source["files"])
    return source


def _selected(scores, selected, k):
    if not scores or any(not math.isfinite(x) for x in scores):
        raise ValueError("finite nonempty score vector required")
    if (len(selected) != k or len(set(selected)) != k
            or any(type(p) is not int or not 0 <= p < len(scores) for p in selected)):
        raise ValueError("unique in-range top-k selected page IDs required")
    chosen = set(selected)
    if (chosen and len(chosen) < len(scores)
            and min(scores[p] for p in chosen) < max(scores[p] for p in range(len(scores)) if p not in chosen)):
        raise ValueError("selected pages do not represent a score top-k")
    return chosen


def _boundary(scores, chosen):
    low = min((scores[p] for p in chosen), default=None)
    high = max((scores[p] for p in range(len(scores)) if p not in chosen), default=None)
    return {"minimum_selected_score": low, "maximum_unselected_score": high,
            "margin": None if low is None or high is None else low - high,
            "exact_boundary_tie": low is not None and high is not None and low == high}


def explain_head(metadata, separate, meta_ids, summary_ids, *, k, sinks=0, window=0):
    """Explain actual selected sets; do not silently replace tied top-k decisions."""
    if (len(metadata) != len(separate) or type(k) is not int or not 0 <= k <= len(metadata)
            or min(sinks, window) < 0):
        raise ValueError("matching score vectors, valid top-k and forced-page counts required")
    meta = _selected(metadata, meta_ids, k)
    summary = _selected(separate, summary_ids, k)
    pages = len(metadata)
    forced = set(range(min(sinks, pages))) | set(range(max(0, pages - window), pages))
    meta_effective, summary_effective = meta | forced, summary | forced
    added, removed = sorted(meta - summary), sorted(summary - meta)
    eps = max(abs(a - b) for a, b in zip(metadata, separate))
    meta_boundary, summary_boundary = _boundary(metadata, meta), _boundary(separate, summary)
    gap = summary_boundary["margin"]
    changed = [
        {"page_id": p, "direction": "metadata-only" if p in meta else "summary-only",
         "metadata_score": metadata[p], "summary_score": separate[p],
         "score_delta_metadata_minus_summary": metadata[p] - separate[p],
         "forced_sink_or_window": p in forced}
        for p in sorted(set(added + removed))
    ]
    exchanges = []
    for lost in removed:
        for gained in added:
            summary_gap = separate[lost] - separate[gained]
            perturbation_advantage = (metadata[gained] - separate[gained]) - (metadata[lost] - separate[lost])
            exchanges.append({
                "summary_only_page": lost, "metadata_only_page": gained,
                "summary_score_advantage": summary_gap,
                "metadata_perturbation_advantage_for_gained_page": perturbation_advantage,
                "metadata_score_advantage_for_gained_page": metadata[gained] - metadata[lost],
                "exact_summary_pair_tie": summary_gap == 0,
                "exact_metadata_pair_tie": metadata[gained] == metadata[lost],
            })
    if not changed:
        explanation = "unchanged"
    elif summary_boundary["exact_boundary_tie"] and meta_boundary["exact_boundary_tie"]:
        explanation = "both-boundaries-tied-allow-ambiguous-membership"
    elif summary_boundary["exact_boundary_tie"]:
        explanation = "summary-boundary-tie-allows-ambiguous-membership"
    elif meta_boundary["exact_boundary_tie"]:
        explanation = "metadata-boundary-tie-after-perturbation-allows-ambiguous-membership"
    else:
        explanation = "observed-score-perturbation-crosses-summary-boundary"
    return {
        "completed_pages": pages, "topk": k,
        "metadata_boundary": meta_boundary, "summary_boundary": summary_boundary,
        "max_abs_score_perturbation": eps,
        "summary_margin_gt_twice_max_score_perturbation": gap is not None and gap > 2 * eps,
        "changed_membership_within_twice_error_bound": None if not changed or gap is None else gap <= 2 * eps,
        "explanation": explanation,
        "metadata_only_page_ids": added, "summary_only_page_ids": removed,
        "effective_metadata_only_page_ids": sorted(meta_effective - summary_effective),
        "effective_summary_only_page_ids": sorted(summary_effective - meta_effective),
        "forced_page_ids": sorted(forced),
        "topk_jaccard": len(meta & summary) / len(meta | summary) if meta | summary else 1.0,
        "effective_jaccard": len(meta_effective & summary_effective) / len(meta_effective | summary_effective)
        if meta_effective | summary_effective else 1.0,
        "changed_pages": changed, "exchanges": exchanges,
    }


def score_fixture(q, raw_k, views, args, *, score_fn=None, summary_fn=None, select_fn=None):
    """Capture score arrays and endpoint errors, excluding attention and timings."""
    import torch

    if score_fn is None or summary_fn is None or select_fn is None:
        from flashquest.eager.criticality import page_scores_int4_fast
        from flashquest.eager.page_summary import compute_page_summary
        from flashquest.eager.selection import select_pages_vectorized

        score_fn = score_fn or page_scores_int4_fast
        summary_fn = summary_fn or compute_page_summary
        select_fn = select_fn or select_pages_vectorized

    completed, tail = views["completed_len"], views["partial_len"]
    if q.shape[0] != 1 or q.shape[2] != 1 or completed < args.page_size:
        raise ValueError("single-example decode and at least one completed page required")
    if raw_k.shape[2] != completed + tail:
        raise ValueError("raw key and cache lengths differ")
    low, high = summary_fn(raw_k[:, :, :completed], args.page_size)
    metadata = score_fn(q, views["K_scale"], views["K_mn"])
    separate = capture.summary_scores(q, low, high)
    pages, heads = metadata.shape[-1], q.shape[1]
    # Match the selector's float32 retention tensor arithmetic exactly. Python
    # float multiplication can land on the other side of an integer boundary.
    k = int((torch.tensor(args.retention, dtype=torch.float32, device=metadata.device)
             * pages).ceil().long().clamp(min=0, max=pages).item())
    meta_top = select_fn(metadata, args.retention, 0, 0, k_max_static=k)
    summary_top = select_fn(separate, args.retention, 0, 0, k_max_static=k)
    if any(count != k for mask in (meta_top, summary_top) for count in mask.sum(-1).flatten().tolist()):
        raise ValueError("both selector masks must have the same uniform float32 top-k count")
    minimum_error = views["K_mn"].float() - low.float()
    maximum_error = views["K_mn"].float() + 15 * views["K_scale"].float() - high.float()
    grouped = q.float().reshape(1, raw_k.shape[1], heads // raw_k.shape[1], q.shape[-1])
    endpoint_perturbation = (grouped.clamp_max(0) @ minimum_error.transpose(-1, -2)
                            + grouped.clamp_min(0) @ maximum_error.transpose(-1, -2))
    endpoint_perturbation = endpoint_perturbation.reshape_as(metadata)
    residual = metadata - separate - endpoint_perturbation
    raw_scale = (high.float() - low.float()) / 15
    per_page = {
        "minimum_max_abs_error": minimum_error.abs().amax(-1)[0].cpu().tolist(),
        "maximum_max_abs_error": maximum_error.abs().amax(-1)[0].cpu().tolist(),
        "maximum_mean_abs_error": maximum_error.abs().mean(-1)[0].cpu().tolist(),
        "fp32_endpoint_scale_at_or_below_epsilon_channels": (raw_scale <= 1e-6).sum(-1)[0].cpu().tolist(),
    }
    meta_scores = metadata[0, :, 0].cpu().tolist()
    summary_scores = separate[0, :, 0].cpu().tolist()
    selected_meta = [row.nonzero().flatten().cpu().tolist() for row in meta_top[0, :, 0]]
    selected_summary = [row.nonzero().flatten().cpu().tolist() for row in summary_top[0, :, 0]]
    forced = set(range(min(args.num_sinks, pages))) | set(range(max(0, pages - args.window_pages), pages))
    explanations = []
    for h in range(heads):
        diagnostic = explain_head(meta_scores[h], summary_scores[h], selected_meta[h], selected_summary[h],
                                  k=k, sinks=args.num_sinks, window=args.window_pages)
        kv = h // (heads // raw_k.shape[1])
        diagnostic.update(query_head=h, kv_head=kv,
                          max_abs_score_residual_after_endpoint_perturbation=residual[0, h, 0].abs().max().item())
        for page in diagnostic["changed_pages"]:
            p = page["page_id"]
            page["token_range_start"] = p * args.page_size
            page["token_range_end_exclusive"] = (p + 1) * args.page_size
            page["affine_endpoint_perturbation"] = endpoint_perturbation[0, h, 0, p].item()
            page["score_residual_after_endpoint_perturbation"] = residual[0, h, 0, p].item()
            page["reconstruction"] = {name: values[kv][p] for name, values in per_page.items()}
        explanations.append(diagnostic)
    return {
        "actual_cache_tokens": completed + tail, "completed_tokens": completed, "tail_tokens": tail,
        "query_heads": heads, "kv_heads": raw_k.shape[1], "head_dim": q.shape[-1],
        "tensor_fingerprints": [
            {"sha256": capture.tensor_hash(q), "tensor": "post-rope-query"},
            {"sha256": capture.tensor_hash(raw_k), "tensor": "post-rope-key"},
            {"sha256": capture.tensor_hash(views["K_scale"]), "tensor": "affine-k-scale"},
            {"sha256": capture.tensor_hash(views["K_mn"]), "tensor": "affine-k-minimum"},
        ],
        "score_error": capture.scalar_metrics(metadata, separate),
        "minimum_reconstruction_error": capture.scalar_metrics(views["K_mn"], low),
        "maximum_reconstruction_error": capture.scalar_metrics(
            views["K_mn"].float() + 15 * views["K_scale"].float(), high),
        "endpoint_epsilon_screen": {
            "epsilon": 1e-6,
            "arithmetic": "float32(original BF16 maximum - minimum) / 15",
            "interpretation": "endpoint-range screen; not the quantizer's BF16 clamp decision",
        },
        "query_heads_with_topk_changes": sum(bool(x["changed_pages"]) for x in explanations),
        "query_heads_with_effective_changes": sum(
            bool(x["effective_metadata_only_page_ids"] or x["effective_summary_only_page_ids"]) for x in explanations),
        "heads": explanations,
        "private_scores": {
            "metadata": meta_scores, "separate_summary": summary_scores,
            "metadata_selected_page_ids": selected_meta, "summary_selected_page_ids": selected_summary,
            "effective_metadata_selected_page_ids": [sorted(set(ids) | forced) for ids in selected_meta],
            "effective_summary_selected_page_ids": [sorted(set(ids) | forced) for ids in selected_summary],
            "endpoint_perturbation": endpoint_perturbation[0, :, 0].cpu().tolist(),
            "score_residual_after_endpoint_perturbation": residual[0, :, 0].cpu().tolist(),
            "per_kv_head_page_reconstruction": per_page,
        },
    }


@contextmanager
def score_only_capture():
    """Reuse existing model capture hooks while restoring its fixture callback."""
    original = capture.profile_fixture
    capture.profile_fixture = score_fixture
    try:
        yield
    finally:
        capture.profile_fixture = original


def export_report(report, raw_path):
    rows = []
    for row in report["rows"]:
        public = {key: value for key, value in row.items() if key not in {"private_scores", "input_sha256"}}
        public["input_fingerprint"] = {"sha256": row["input_sha256"], "tensor": "exact-prompt-token-ids"}
        public["score_evidence_fingerprint"] = {"sha256": content_hash(row["private_scores"]),
                                                "kind": "canonical-private-score-arrays-and-selected-page-ids"}
        rows.append(public)
    exported = {**report, "identity": export_identity(report["identity"]), "rows": rows,
                "raw_evidence": {"path": raw_path.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(raw_path)}}
    validate_export(exported)
    return exported


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--retentions", type=float, nargs="+", default=[.20, .25])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 13, 27])
    parser.add_argument("--decode-steps", type=int, nargs="+", default=[0, 1, 63])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-sinks", type=int, default=4)
    parser.add_argument("--window-pages", type=int, default=2)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "benchmarks" / "validation" / "selection-flips")
    args = parser.parse_args(argv)
    if (len(args.contexts) != len(args.retentions) or len(set(args.contexts)) != len(args.contexts)
            or min(args.contexts) < args.page_size or args.page_size != 64
            or any(not math.isfinite(r) or not 0 < r <= 1 for r in args.retentions)
            or len(set(args.layers)) != len(args.layers) or min(args.layers) < 0
            or len(set(args.decode_steps)) != len(args.decode_steps) or min(args.decode_steps) < 0
            or min(args.seed, args.num_sinks, args.window_pages, args.attempt) < 0):
        parser.error("matching contexts/retentions, unique layers/steps and nonnegative budgets required")
    return args


def main(argv=None):
    args = parse_args(argv)
    if Path(__file__).resolve() != REPO_ROOT / "scripts" / "diagnose_selection_flips.py":
        raise ValueError("apply this reviewed producer to scripts before recording research evidence")
    metadata = model_identity(args.model, args.revision)
    config = {key: value for key, value in vars(args).items() if key not in {"model", "revision", "out_dir"}}
    config.update(task="single", model=metadata["model"], revision=metadata["revision"])
    run = make_identity(config, metadata, PROTOCOL, provenance(), source=measurement_source())
    output = args.out_dir / run["run_identity"] / "report.json"
    if output.exists():
        raise ValueError("run already recorded; choose a new attempt to preserve earlier evidence")
    raw_dir = REPO_ROOT / "artifacts" / "selection-flips" / run["run_identity"]
    report = {**run, "schema_version": SCHEMA_VERSION, "protocol": PROTOCOL,
              "status": "incomplete", "rows": [], "error": None}

    def save():
        raw_path = raw_dir / "report-raw.json"
        write_record(raw_path, report)
        write_record(output, export_report(report, raw_path))

    model = None
    started = time.perf_counter()
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("selection recaptures require CUDA")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from flashquest.runtime.awq_load import load_awq_model
        model, tokenizer = load_awq_model(args.model, revision=metadata["revision"],
                                          cache_dir=str(REPO_ROOT / "artifacts" / "hf-cache"))
        with score_only_capture(), torch.inference_mode():
            for context, retention in zip(args.contexts, args.retentions):
                capture_args = SimpleNamespace(**vars(args), task="single", retention=retention)
                iterator = capture.model_rows(model, tokenizer, context, capture_args)
                try:
                    for row in iterator:
                        row["retention"] = retention
                        report["rows"].append(row)
                        save()
                finally:
                    iterator.close()
                gc.collect()
                torch.cuda.empty_cache()
        if measurement_source()["content_sha256"] != run["identity"]["source"]["content_sha256"]:
            raise RuntimeError("measurement source changed during recapture")
        expected = len(args.contexts) * len(args.layers) * len(args.decode_steps)
        if len(report["rows"]) != expected:
            raise RuntimeError("incomplete capture coverage")
        report["status"] = "complete"
    except Exception as exc:  # noqa: BLE001 — preserve bounded diagnostic failures
        raw_dir.mkdir(parents=True, exist_ok=True)
        trace = raw_dir / "error.log"
        trace.write_text(traceback.format_exc(), encoding="utf-8")
        report.update(status="error", error=error_code(exc),
                      raw_error={"path": trace.relative_to(REPO_ROOT).as_posix(), "sha256": file_hash(trace)})
    finally:
        del model
        gc.collect()
    report["wall_s"] = time.perf_counter() - started
    save()
    print(f"{report['status']}: {len(report['rows'])} bounded selection snapshots")
    return int(report["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
