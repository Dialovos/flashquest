#!/usr/bin/env python3
"""Phase 0 environment introspection. Re-runnable, idempotent, no side effects."""
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return (out.stdout + out.stderr).strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"<not available: {e}>"


def main() -> None:
    info: dict[str, object] = {
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "cpu": _run(["lscpu"]).split("\n")[:20],
        "memory": _run(["free", "-h"]),
        "nvidia_smi": _run(["nvidia-smi"]),
        "nvcc": _run(["nvcc", "--version"]),
        "ncu": _run(["ncu", "--version"]) if shutil.which("ncu") else "<absent>",
        "nsys": _run(["nsys", "--version"]) if shutil.which("nsys") else "<absent>",
        "gcc": _run(["gcc", "--version"]).split("\n")[0],
        "cmake": _run(["cmake", "--version"]).split("\n")[0],
        "git": _run(["git", "--version"]),
    }

    try:
        import torch
        info["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        info["torch"] = "<not installed>"

    try:
        import triton
        info["triton"] = {"version": triton.__version__}
    except ImportError:
        info["triton"] = "<not installed>"

    out_path = Path(__file__).resolve().parents[1] / "docs" / "PHASES" / "env_snapshot.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps(info, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
