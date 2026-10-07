"""Focused CPU checks of selection differences, ties and perturbation attribution."""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = next(parent for parent in Path(__file__).resolve().parents
            if (parent / "scripts" / "profile_contribution.py").is_file())
sys.path.insert(0, str(REPO / "scripts"))
PRODUCER = Path(__file__).with_name("diagnose_selection_flips.py")
if not PRODUCER.is_file():
    PRODUCER = REPO / "scripts" / "diagnose_selection_flips.py"
spec = importlib.util.spec_from_file_location("selection_diagnostic_draft", PRODUCER)
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def test_import_does_not_load_gpu_capture_modules():
    import subprocess
    code = (
        "import importlib.util, sys; sys.path.insert(0, 'scripts'); "
        "spec=importlib.util.spec_from_file_location('selection', "
        f"{str(PRODUCER.relative_to(REPO))!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "assert 'torch' not in sys.modules; assert 'flashquest.cache.persistent_int4' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO, check=True)


def test_flip_explains_changed_ids_gap_and_forced_mask():
    row = S.explain_head([10, 7.9, 8.1, 1], [10, 8.1, 7.9, 1], [0, 2], [0, 1], k=2, sinks=2)
    assert row["metadata_only_page_ids"] == [2]
    assert row["summary_only_page_ids"] == [1]
    assert row["effective_metadata_only_page_ids"] == [2]
    assert row["effective_summary_only_page_ids"] == []
    assert row["summary_boundary"]["margin"] == pytest.approx(.2)
    assert row["max_abs_score_perturbation"] == pytest.approx(.2)
    assert row["changed_membership_within_twice_error_bound"] is True
    assert row["summary_margin_gt_twice_max_score_perturbation"] is False
    assert row["topk_jaccard"] == pytest.approx(1 / 3)
    assert row["effective_jaccard"] == pytest.approx(2 / 3)
    assert row["exchanges"] == [pytest.approx({
        "summary_only_page": 1, "metadata_only_page": 2,
        "summary_score_advantage": .2,
        "metadata_perturbation_advantage_for_gained_page": .4,
        "metadata_score_advantage_for_gained_page": .2,
        "exact_summary_pair_tie": False, "exact_metadata_pair_tie": False,
    })]


def test_large_margin_certifies_unchanged_membership():
    row = S.explain_head([10.1, 8.0, 2.1], [10, 8, 2], [0, 1], [0, 1], k=2)
    assert row["summary_margin_gt_twice_max_score_perturbation"] is True
    assert row["explanation"] == "unchanged"
    assert row["changed_membership_within_twice_error_bound"] is None
    assert row["changed_pages"] == []


def test_zero_error_tie_preserves_actual_ambiguous_topk_ids():
    row = S.explain_head([3, 3, 1], [3, 3, 1], [0], [1], k=1)
    assert row["metadata_only_page_ids"] == [0]
    assert row["summary_only_page_ids"] == [1]
    assert row["max_abs_score_perturbation"] == 0
    assert row["summary_boundary"]["exact_boundary_tie"] is True
    assert row["explanation"] == "both-boundaries-tied-allow-ambiguous-membership"
    assert row["changed_membership_within_twice_error_bound"] is True
    assert row["exchanges"][0]["exact_summary_pair_tie"] is True


def test_metadata_boundary_tie_does_not_claim_strict_rank_crossing():
    row = S.explain_head([2, 2], [3, 2], [1], [0], k=1)
    assert row["metadata_boundary"]["exact_boundary_tie"] is True
    assert row["summary_boundary"]["exact_boundary_tie"] is False
    assert row["explanation"] == "metadata-boundary-tie-after-perturbation-allows-ambiguous-membership"
    assert row["exchanges"][0]["metadata_score_advantage_for_gained_page"] == 0
    assert row["exchanges"][0]["metadata_perturbation_advantage_for_gained_page"] == 1


def test_summary_boundary_tie_is_distinct_from_metadata_boundary_tie():
    row = S.explain_head([2, 3], [2, 2], [1], [0], k=1)
    assert row["metadata_boundary"]["exact_boundary_tie"] is False
    assert row["summary_boundary"]["exact_boundary_tie"] is True
    assert row["explanation"] == "summary-boundary-tie-allows-ambiguous-membership"


@pytest.mark.parametrize("k,ids", [(0, []), (3, [0, 1, 2])])
def test_no_boundary_when_none_or_all_pages_selected(k, ids):
    row = S.explain_head([3, 2, 1], [3, 2, 1], ids, ids, k=k)
    assert row["summary_boundary"]["margin"] is None
    assert row["summary_margin_gt_twice_max_score_perturbation"] is False


@pytest.mark.parametrize("scores,ids", [([2, 1], [1]), ([2, 1], [0, 0]), ([2, 1], [2]), ([float("nan"), 1], [0])])
def test_bad_selected_sets_and_scores_are_rejected(scores, ids):
    with pytest.raises(ValueError):
        S.explain_head(scores, [2, 1], ids, [0], k=1)


def test_scoped_capture_callback_restores_after_failure():
    original = S.capture.profile_fixture
    with pytest.raises(RuntimeError), S.score_only_capture():
        assert S.capture.profile_fixture is S.score_fixture
        raise RuntimeError("fixture capture failed")
    assert S.capture.profile_fixture is original


def test_score_fixture_attributes_affine_perturbation_without_attention():
    from types import SimpleNamespace

    import torch
    keys = torch.zeros(1, 1, 128, 2, dtype=torch.bfloat16)
    keys[:, :, 63, :] = torch.tensor([1, 1], dtype=torch.bfloat16)
    keys[:, :, 127, :] = torch.tensor([1.0078125, 1.0078125], dtype=torch.bfloat16)
    # Deliberately perturbed metadata tests attribution independently of the quantizer.
    scale = torch.tensor([[[[.07, .07], [.066, .066]]]], dtype=torch.bfloat16)
    views = {"completed_len": 128, "partial_len": 0,
             "K_mn": torch.zeros(1, 1, 2, 2, dtype=torch.bfloat16), "K_scale": scale}
    q = torch.tensor([[[[1, 1]], [[2, .5]]]], dtype=torch.bfloat16)
    def summarize(tensor, page_size):
        grouped = tensor.reshape(1, 1, 2, page_size, 2)
        return grouped.amin(3), grouped.amax(3)

    def select(scores, retention, sinks, window, k_max_static):
        return torch.zeros_like(scores, dtype=torch.bool).scatter(-1, scores.topk(k_max_static).indices, True)

    row = S.score_fixture(q, keys, views, SimpleNamespace(page_size=64, retention=.5, num_sinks=0, window_pages=0),
                          score_fn=lambda query, scale, low: S.capture.summary_scores(query, low, low.float() + 15 * scale.float()),
                          summary_fn=summarize, select_fn=select)
    assert row["query_heads_with_topk_changes"] == 2
    assert row["query_heads_with_effective_changes"] == 2
    assert all(list(entry) == ["sha256", "tensor"] for entry in row["tensor_fingerprints"])
    for h in row["heads"]:
        assert h["metadata_only_page_ids"] == [0]
        assert h["summary_only_page_ids"] == [1]
        assert h["max_abs_score_residual_after_endpoint_perturbation"] < 1e-6
        assert all(page["reconstruction"]["maximum_max_abs_error"] > 0 for page in h["changed_pages"])
    raw = row["private_scores"]
    assert raw["metadata_selected_page_ids"] == [[0], [0]]
    assert raw["summary_selected_page_ids"] == [[1], [1]]
    assert len(raw["metadata"][0]) == 2
    assert row["maximum_reconstruction_error"]["max_abs_error"] > 0


def _cpu_score_row(keys, retention, *, select_fn=None):
    from types import SimpleNamespace

    import torch
    if select_fn is None:
        # Load the CPU-safe leaf without its parent package's CUDA eager imports.
        spec = importlib.util.spec_from_file_location("cpu_page_selector", REPO / "src/flashquest/eager/selection.py")
        selection = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selection)
        select_fn = selection.select_pages_vectorized

    def summarize(tensor, page_size):
        grouped = tensor.reshape(1, 1, tensor.shape[2] // page_size, page_size, tensor.shape[-1])
        return grouped.amin(3), grouped.amax(3)

    low, high = summarize(keys, 64)
    views = {"completed_len": keys.shape[2], "partial_len": 0, "K_mn": low,
             "K_scale": ((high - low) / 15).clamp_min(1e-6)}
    return S.score_fixture(torch.ones(1, 1, 1, keys.shape[-1], dtype=torch.bfloat16), keys, views,
                           SimpleNamespace(page_size=64, retention=retention, num_sinks=0, window_pages=0),
                           score_fn=lambda query, scale, minimum: S.capture.summary_scores(
                               query, minimum, minimum.float() + 15 * scale.float()),
                           summary_fn=summarize, select_fn=select_fn)


def test_topk_matches_actual_float32_selector_at_rounding_boundary():
    import math

    import torch
    keys = torch.zeros(1, 1, 6400, 2, dtype=torch.bfloat16)
    keys[:, :, 63::64] = torch.arange(1, 101).reshape(1, 1, 100, 1)
    assert math.ceil(.07 * 100) == 8
    row = _cpu_score_row(keys, .07)
    assert row["heads"][0]["topk"] == 7
    assert len(row["private_scores"]["metadata_selected_page_ids"][0]) == 7
    assert len(row["private_scores"]["summary_selected_page_ids"][0]) == 7


def test_endpoint_epsilon_screen_is_not_labeled_as_bf16_quantizer_clamping():
    import torch
    keys = torch.zeros(1, 1, 64, 2, dtype=torch.bfloat16)
    keys[:, :, 63] = 1.5e-5
    high = keys.amax(2)
    assert bool(((high - 0) / 15 <= 1e-6).all())  # BF16 intermediate arithmetic.
    assert bool((high.float() / 15 > 1e-6).all())  # FP32 endpoint-range screen differs.
    row = _cpu_score_row(keys, 1)
    errors = row["private_scores"]["per_kv_head_page_reconstruction"]
    assert errors["fp32_endpoint_scale_at_or_below_epsilon_channels"] == [[0]]
    assert "clamped_scale_channels" not in errors
    assert row["endpoint_epsilon_screen"]["interpretation"].endswith("BF16 clamp decision")


def test_manifest_binds_the_new_producer():
    source = S.measurement_source()
    name = "scripts/diagnose_selection_flips.py"
    assert source["files"][name] == S.file_hash(Path(S.__file__))
    assert source["content_sha256"] == S.content_hash(source["files"])
    assert "scripts/profile_contribution.py" in source["files"]


def test_public_export_keeps_exact_changed_pages_and_binds_private_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "REPO_ROOT", tmp_path)
    source = {"files": {}, "content_sha256": S.content_hash({})}
    model = {"files": {}, "content_sha256": S.content_hash({})}
    run = S.make_identity({}, model, S.PROTOCOL, {}, source=source)
    private = {"metadata": [[4, 3]], "metadata_selected_page_ids": [[0]]}
    report = {**run, "rows": [{"private_scores": private, "input_sha256": "a" * 64,
                              "heads": [{"metadata_only_page_ids": [0], "summary_only_page_ids": [1]}]}]}
    raw = tmp_path / "artifacts" / "evidence.json"
    S.write_record(raw, report)
    public = S.export_report(report, raw)
    assert public["rows"][0]["heads"] == report["rows"][0]["heads"]
    assert "private_scores" not in public["rows"][0]
    assert public["rows"][0]["score_evidence_fingerprint"]["sha256"] == S.content_hash(private)
    assert public["raw_evidence"] == {"path": "artifacts/evidence.json", "sha256": S.file_hash(raw)}
    S.validate_export(public)


@pytest.mark.parametrize("extra", [
    ["--contexts", "8192"], ["--retentions", "nan", ".25"],
    ["--layers", "0", "0"], ["--decode-steps", "-1"], ["--page-size", "128"],
])
def test_rejects_mismatched_or_invalid_capture_settings(extra):
    with pytest.raises(SystemExit):
        S.parse_args(["--revision", "pinned", *extra])
