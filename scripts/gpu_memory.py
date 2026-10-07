"""Sample one physical GPU and an owned Linux process tree without importing CUDA.

Device samples include display/idle allocations. RSS sums can double-count shared
pages and miss short-lived children. All windows use Linux's monotonic clock;
queries overlapping a window boundary are excluded from that window's peak.
"""
from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import threading
import time
from itertools import pairwise
from pathlib import Path
from statistics import median

from bench_common import file_hash, validate_export, write_record


def query(arguments: list[str]) -> list[list[str]]:
    result = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=3)
    return [[item.strip() for item in row] for row in csv.reader(result.stdout.splitlines()) if row]


def resolve_device(index: int) -> dict:
    rows = query(["--query-gpu=index,uuid,name,memory.total,driver_version", f"--id={index}"])
    if len(rows) != 1 or len(rows[0]) != 5 or int(rows[0][0]) != index:
        raise ValueError("physical device selection is ambiguous")
    idx, uuid, name, total, driver = rows[0]
    return {"index": int(idx), "uuid": uuid, "name": name,
            "total_mib": float(total), "driver": driver}


def compute_pids(device: dict) -> set[int]:
    rows = query(["--query-compute-apps=pid", f"--id={device['uuid']}"])
    return {int(row[0]) for row in rows}


def number(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except ValueError:
        return None


def gpu_sample(device: dict) -> dict:
    rows = query(["--query-gpu=memory.used,clocks.sm,temperature.gpu", f"--id={device['uuid']}"])
    if len(rows) != 1 or len(rows[0]) != 3:
        raise ValueError("invalid device sample")
    return dict(zip(("device_used_mib", "sm_clock_mhz", "temperature_c"),
                    (number(value) for value in rows[0]), strict=True))


def processes(proc: Path = Path("/proc")) -> dict[int, dict]:
    """PID plus start ticks guards ownership against PID reuse."""
    snapshot = {}
    page_mib = os.sysconf("SC_PAGE_SIZE") / 2**20
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
            snapshot[int(directory.name)] = {
                "ppid": int(fields[1]), "pgrp": int(fields[2]), "start_ticks": int(fields[19]),
                "rss_mib": max(0, int(fields[21])) * page_mib,
            }
        except (OSError, ValueError, IndexError):
            continue
    return snapshot


def owned_processes(snapshot: dict, root_pid: int, root_start: int,
                    known: dict[int, int]) -> dict[int, dict]:
    owned = {pid: info for pid, info in snapshot.items()
             if known.get(pid) == info["start_ticks"] or
             (info["pgrp"] == root_pid and info["start_ticks"] >= root_start)}
    if root_pid in snapshot and snapshot[root_pid]["start_ticks"] == root_start:
        owned[root_pid] = snapshot[root_pid]
    changed = True
    while changed:
        changed = False
        for pid, info in snapshot.items():
            if pid not in owned and info["ppid"] in owned:
                owned[pid] = info
                changed = True
    known.update({pid: info["start_ticks"] for pid, info in owned.items()})
    return owned


def available_memory(proc: Path = Path("/proc")) -> float | None:
    try:
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


def peak(samples: list[dict], key: str) -> float | None:
    values = [row[key] for row in samples if row.get(key) is not None]
    return max(values) if values else None


def phase_windows(markers: list[dict], start_ns: int, end_ns: int) -> dict[str, list[tuple[int, int]]]:
    windows = {}
    opened = {}
    completed = set()
    previous = start_ns
    for marker in markers:
        timestamp = marker["monotonic_ns"]
        if not isinstance(timestamp, int) or not previous <= timestamp <= end_ns:
            raise ValueError("markers do not share the observation clock/window")
        previous = timestamp
        phase, edge = marker["event"].rsplit("_", 1)
        if phase not in {"load", "warmup", "prefill", "decode", "request"} or edge not in {"start", "end"}:
            raise ValueError("unknown phase marker")
        key = (phase, marker.get("repetition"))
        if edge == "start":
            if key in opened or key in completed:
                raise ValueError("duplicate phase start")
            opened[key] = timestamp
        else:
            begin = opened.pop(key, None)
            if begin is None or begin >= timestamp:
                raise ValueError("invalid phase window")
            windows.setdefault(phase, []).append((begin, timestamp))
            completed.add(key)
    if opened:
        raise ValueError("incomplete phase markers")
    return windows


def summarize(samples: list[dict], baseline: dict, interval_s: float,
              markers: list[dict], start_ns: int, end_ns: int, contaminated: bool) -> dict:
    gaps = [(b["query_start_ns"] - a["query_start_ns"]) / 1e6
            for a, b in pairwise(samples)]
    used = peak(samples, "device_used_mib")
    base = baseline.get("device_used_mib")
    result = {
        "clock": "linux-monotonic-ns", "requested_interval_ms": interval_s * 1000,
        "actual_interval_median_ms": median(gaps) if gaps else None,
        "actual_interval_max_ms": max(gaps) if gaps else None,
        "sample_count": len(samples),
        "device_sample_count": sum(s.get("device_used_mib") is not None for s in samples),
        "device_dropouts": sum(s.get("device_used_mib") is None for s in samples),
        "ownership_checks": sum(s.get("ownership_checked", False) for s in samples),
        "ownership_check_dropouts": sum(s.get("ownership_check_failed", False) for s in samples),
        "concurrent_compute_workload": contaminated,
        "device_baseline_mib": base, "device_sampled_peak_mib": used,
        "device_baseline_adjusted_peak_mib": max(0, used - base) if used is not None and base is not None else None,
        "process_tree_rss_sampled_peak_mib": peak(samples, "process_tree_rss_mib"),
        "mem_available_before_mib": baseline.get("mem_available_mib"),
        "mem_available_after_mib": samples[-1].get("mem_available_mib") if samples else None,
        "sm_clock_min_mhz": min((s["sm_clock_mhz"] for s in samples if s.get("sm_clock_mhz") is not None), default=None),
        "sm_clock_max_mhz": peak(samples, "sm_clock_mhz"),
        "temperature_max_c": peak(samples, "temperature_c"),
        "phase_status": "unmeasured", "phases": {phase: None for phase in ("load", "warmup", "prefill", "decode")},
        "limits": ["sampled-peaks", "rss-shared-pages-may-double-count",
                   "short-lived-workers-may-be-missed", "device-includes-idle-and-display",
                   "configured-placement-does-not-prove-no-os-fallback"],
    }
    if markers:
        try:
            windows = phase_windows(markers, start_ns, end_ns)
        except (KeyError, TypeError, ValueError):
            result["phase_status"] = "invalid-markers"
        else:
            result["phase_status"] = "validated-windows"
            for phase, bounds in windows.items():
                selected = [s for s in samples if any(begin <= s["query_start_ns"] <= s["query_end_ns"] <= end
                                                       for begin, end in bounds)]
                result["phases"][phase] = {
                    "window_count": len(bounds), "sample_count": len(selected),
                    "device_sampled_peak_mib": peak(selected, "device_used_mib"),
                    "process_tree_rss_sampled_peak_mib": peak(selected, "process_tree_rss_mib"),
                }
    return result


def checked_memory(memory: dict, root: Path, repetitions: int) -> bool:
    """Validate complete FlashQuest observation; verify local raw series if present."""
    validate_export(memory)
    if (memory["sampler_failed"] or memory["concurrent_compute_workload"] or
            memory["ownership_check_dropouts"] or memory["phase_status"] != "validated-windows" or
            memory["device_sample_count"] <= 0 or memory["ownership_checks"] <= 0 or
            memory["device_sample_count"] + memory["device_dropouts"] != memory["sample_count"]):
        raise ValueError("invalid memory observation coverage")
    for field in ("device_baseline_mib", "device_sampled_peak_mib",
                  "device_baseline_adjusted_peak_mib", "process_tree_rss_sampled_peak_mib"):
        value = memory[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid memory peak")
    if not math.isclose(memory["device_baseline_adjusted_peak_mib"],
                        max(0, memory["device_sampled_peak_mib"] - memory["device_baseline_mib"])):
        raise ValueError("memory baseline adjustment differs")
    for phase in ("load", "warmup", "prefill", "decode"):
        window = memory["phases"][phase]
        if window is None or window["window_count"] != (1 if phase in {"load", "warmup"} else repetitions):
            raise ValueError("incomplete memory phase windows")
    series = root / memory["raw_series"]["path"]
    if series.exists() and file_hash(series) != memory["raw_series"]["sha256"]:
        raise ValueError("raw memory series changed")
    return series.exists()


class MemorySampler:
    def __init__(self, device: dict, root_pid: int, root_start: int, *, interval_s: float = 0.05):
        self.device, self.root_pid, self.root_start = device, root_pid, root_start
        self.interval_s = interval_s
        self.known = {root_pid: root_start}
        self.samples = []
        self.contaminated = False
        self.failed = False
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def sample(self, check_ownership: bool) -> dict:
        row = {"query_start_ns": time.monotonic_ns()}
        try:
            row.update(gpu_sample(self.device))
        except (OSError, ValueError, subprocess.SubprocessError):
            row.update(device_used_mib=None, sm_clock_mhz=None, temperature_c=None)
        snapshot = processes()
        owned = owned_processes(snapshot, self.root_pid, self.root_start, self.known)
        row.update(process_tree_rss_mib=sum(info["rss_mib"] for info in owned.values()),
                   process_count=len(owned), mem_available_mib=available_memory(),
                   ownership_checked=check_ownership, ownership_check_failed=False)
        if check_ownership:
            try:
                active = compute_pids(self.device)
                foreign = {pid for pid in active if pid not in owned}
                self.contaminated |= bool(foreign)
                row["foreign_compute_pids"] = sorted(foreign)
            except (OSError, ValueError, subprocess.SubprocessError):
                row["ownership_check_failed"] = True
        row["query_end_ns"] = time.monotonic_ns()
        return row

    def run(self):
        last_check = 0
        try:
            while not self.stop_event.is_set():
                start = time.monotonic()
                check = start - last_check >= 1
                self.samples.append(self.sample(check))
                if check:
                    last_check = start
                self.stop_event.wait(max(0, self.interval_s - (time.monotonic() - start)))
        except Exception:  # noqa: BLE001 — preserve the cell and flag missing telemetry
            self.failed = True

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.failed = True


def stop_owned_group(process: subprocess.Popen, root_start: int, known=None):
    """Terminate creation-checked descendants, including tracked escaped workers."""
    known = known if known is not None else {}
    for sig in (signal.SIGTERM, signal.SIGKILL):
        owned = owned_processes(processes(), process.pid, root_start, known)
        for pid, info in owned.items():
            # Recheck immediately before signaling to guard reuse between snapshots.
            if processes().get(pid, {}).get("start_ticks") == info["start_ticks"]:
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
        if sig == signal.SIGTERM:
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                process.poll()
                if not owned_processes(processes(), process.pid, root_start, known):
                    return
                time.sleep(0.05)


def observe_command(command: list[str], directory: Path, device: dict,
                    *, timeout_s: float = 1800, interval_s: float = 0.05) -> dict:
    """Run only our child group; never terminate a foreign GPU workload."""
    if compute_pids(device):
        raise RuntimeError("selected GPU has another compute workload")
    baseline = {**gpu_sample(device), "mem_available_mib": available_memory()}
    directory.mkdir(parents=True, exist_ok=True)
    marker_path = directory / "markers.jsonl"
    if any((directory / name).exists() for name in ("markers.jsonl", "cell.log", "memory-series.json")):
        raise ValueError("observation directory already contains evidence")
    start_ns = time.monotonic_ns()
    status = "complete"
    with (directory / "cell.log").open("w", encoding="utf-8") as log, \
        subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": device["uuid"],
                                   "FLASHQUEST_MARKERS": str(marker_path),
                                   "FLASHQUEST_MEMORY_PROTOCOL": json.dumps({"version": 1,
                                       "method": "nvidia-smi-query-linux-proc-monotonic",
                                       "interval_s": interval_s, "physical_device_index": device["index"]})}) as process:
        root_start = processes().get(process.pid, {}).get("start_ticks")
        if root_start is None:
            raise RuntimeError("child identity unavailable")
        sampler = MemorySampler(device, process.pid, root_start, interval_s=interval_s)
        sampler.thread.start()
        try:
            while process.poll() is None:
                if sampler.contaminated:
                    status = "concurrent-workload"
                    break
                if (time.monotonic_ns() - start_ns) / 1e9 > timeout_s:
                    status = "timeout"
                    break
                time.sleep(0.05)
        finally:
            stop_owned_group(process, root_start, sampler.known)
            process.wait()
            sampler.stop()
    end_ns = time.monotonic_ns()
    try:
        markers = [json.loads(line) for line in marker_path.read_text().splitlines()] if marker_path.exists() else []
    except (OSError, ValueError):
        markers = [{"invalid": True}]
    memory = summarize(sampler.samples, baseline, interval_s, markers, start_ns, end_ns, sampler.contaminated)
    memory["mem_available_after_mib"] = available_memory()
    memory["sampler_failed"] = sampler.failed
    memory["physical_device"] = {key: value for key, value in device.items() if key != "uuid"}
    memory["backend_visible_device"] = "cuda:0"
    series_path = directory / "memory-series.json"
    write_record(series_path, {"baseline": baseline, "samples": sampler.samples,
                               "markers": markers, "start_ns": start_ns, "end_ns": end_ns})
    memory["raw_series"] = {"sha256": file_hash(series_path)}
    return {"returncode": process.returncode, "status": status, "memory": memory}
