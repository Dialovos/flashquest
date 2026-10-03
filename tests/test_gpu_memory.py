"""CPU-only checks for device selection, attribution, windows and child cleanup."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import gpu_memory as M

DEVICE = {"index": 0, "uuid": "test-device", "name": "test GPU", "total_mib": 1000, "driver": "test"}


def test_physical_device_selection_and_visibility(monkeypatch):
    calls = []
    def query(args):
        calls.append(args)
        return [["2", "test-device", "test GPU", "1000", "driver"]]
    monkeypatch.setattr(M, "query", query)
    assert M.resolve_device(2)["index"] == 2
    assert "--id=2" in calls[0]
    with pytest.raises(ValueError, match="ambiguous"):
        M.resolve_device(1)


def test_unsupported_device_counters_remain_unmeasured(monkeypatch):
    monkeypatch.setattr(M, "query", lambda args: [["100", "[N/A]", "[Not Supported]"]])
    assert M.gpu_sample(DEVICE) == {"device_used_mib": 100, "sm_clock_mhz": None, "temperature_c": None}
    assert M.number("nan") is None and M.number("inf") is None


def proc_info(ppid, start, rss, group):
    return {"ppid": ppid, "start_ticks": start, "rss_mib": rss, "pgrp": group}


def test_worker_tree_tracks_orphans_and_rejects_reused_pid():
    known = {10: 100}
    tree = {10: proc_info(1, 100, 10, 10), 20: proc_info(10, 200, 20, 20),
            30: proc_info(20, 300, 30, 20), 40: proc_info(1, 400, 500, 40)}
    owned = M.owned_processes(tree, 10, 100, known)
    assert sum(p["rss_mib"] for p in owned.values()) == 60
    tree = {20: proc_info(1, 200, 20, 20), 30: proc_info(20, 999, 300, 99)}
    # PID 30 is reused, but its new parent belongs to this workload: a new owned child.
    assert set(M.owned_processes(tree, 10, 100, known)) == {20, 30}
    tree = {30: proc_info(1, 1000, 900, 99)}
    assert M.owned_processes(tree, 10, 100, known) == {}


def test_proc_stat_parser_handles_spaces_in_process_name(tmp_path):
    (tmp_path / "5").mkdir()
    fields = ["S", "1", "5", *["0"] * 16, "123", "0", "10"]
    (tmp_path / "5" / "stat").write_text("5 (worker (spaces)) " + " ".join(fields))
    parsed = M.processes(tmp_path)[5]
    assert parsed["start_ticks"] == 123 and parsed["ppid"] == 1 and parsed["rss_mib"] > 0


def row(start, end, used, rss=10):
    return {"query_start_ns": start, "query_end_ns": end, "device_used_mib": used,
            "process_tree_rss_mib": rss, "mem_available_mib": 500,
            "sm_clock_mhz": 1000, "temperature_c": 50, "ownership_checked": True}


def test_window_peaks_exclude_boundary_queries_and_record_dropouts():
    samples = [row(1, 2, 200), row(3, 7, 900), row(8, 9, None), row(10, 11, 300)]
    markers = [{"event": "decode_start", "monotonic_ns": 5, "repetition": 0},
               {"event": "decode_end", "monotonic_ns": 12, "repetition": 0}]
    result = M.summarize(samples, {"device_used_mib": 100}, .05, markers, 0, 15, False)
    assert result["device_sampled_peak_mib"] == 900
    assert result["device_baseline_adjusted_peak_mib"] == 800
    assert result["phases"]["decode"]["device_sampled_peak_mib"] == 300
    assert result["device_dropouts"] == 1 and result["actual_interval_max_ms"] is not None
    assert result["phases"]["load"] is None


@pytest.mark.parametrize("markers", [
    [{"event": "load_start", "monotonic_ns": 1}],
    [{"event": "load_end", "monotonic_ns": 1}],
    [{"event": "load_start", "monotonic_ns": -1}],
    [{"event": "unknown_start", "monotonic_ns": 1}],
])
def test_invalid_markers_leave_phase_peaks_unmeasured(markers):
    result = M.summarize([row(0, 2, 100)], {}, .05, markers, 0, 10, False)
    assert result["phase_status"] == "invalid-markers" and all(v is None for v in result["phases"].values())


def test_rss_is_per_cell_after_larger_previous_run():
    large = M.summarize([row(1, 2, 100, 500)], {}, .05, [], 0, 3, False)
    small = M.summarize([row(1, 2, 100, 10)], {}, .05, [], 0, 3, False)
    assert large["process_tree_rss_sampled_peak_mib"] == 500
    assert small["process_tree_rss_sampled_peak_mib"] == 10


@pytest.fixture
def fake_gpu(monkeypatch):
    monkeypatch.setattr(M, "compute_pids", lambda device: set())
    monkeypatch.setattr(M, "gpu_sample", lambda device: {"device_used_mib": 100, "sm_clock_mhz": 1000,
                                                       "temperature_c": 50})


def test_busy_device_is_rejected_without_starting_child(tmp_path, fake_gpu, monkeypatch):
    monkeypatch.setattr(M, "compute_pids", lambda device: {99999})
    with pytest.raises(RuntimeError, match="another compute workload"):
        M.observe_command(["unavailable-command"], tmp_path, DEVICE)
    assert not (tmp_path / "cell.log").exists()


def test_owned_cpu_worker_rss_and_visible_device(tmp_path, fake_gpu):
    code = ("import os, subprocess, sys, time; "
            "assert os.environ['CUDA_VISIBLE_DEVICES']=='test-device'; "
            "p=subprocess.Popen([sys.executable,'-c','import time; a=bytearray(20000000); time.sleep(.4)']); "
            "p.wait(); time.sleep(.1)")
    result = M.observe_command([sys.executable, "-c", code], tmp_path, DEVICE, interval_s=.02)
    assert result["returncode"] == 0 and result["status"] == "complete"
    series = json.loads((tmp_path / "memory-series.json").read_text())
    assert max(s["process_count"] for s in series["samples"]) >= 2
    assert result["memory"]["process_tree_rss_sampled_peak_mib"] > 20
    assert "uuid" not in result["memory"]["physical_device"]


def test_timeout_stops_only_owned_group(tmp_path, fake_gpu):
    result = M.observe_command([sys.executable, "-c", "import time; time.sleep(30)"],
                               tmp_path, DEVICE, timeout_s=.1, interval_s=.02)
    assert result["status"] == "timeout" and result["returncode"] != 0
    assert result["memory"]["sample_count"] > 0


def test_new_foreign_workload_stops_our_child_and_preserves_foreign_process(tmp_path, fake_gpu, monkeypatch):
    with subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True) as foreign:
        calls = [0]
        def active(device):
            calls[0] += 1
            return set() if calls[0] == 1 else {foreign.pid}
        monkeypatch.setattr(M, "compute_pids", active)
        try:
            result = M.observe_command([sys.executable, "-c", "import time; time.sleep(30)"],
                                       tmp_path, DEVICE, interval_s=.02)
            assert result["status"] == "concurrent-workload" and result["returncode"] != 0
            assert result["memory"]["concurrent_compute_workload"] is True
            assert foreign.poll() is None
        finally:
            foreign.terminate()
