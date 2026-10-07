#!/usr/bin/env bash
# --check collects tests only; --run executes them with local logs and an exit receipt.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mode="${1:---check}"
if [[ $# -gt 1 || "$mode" != --check && "$mode" != --run ]]; then
    echo 'Usage: bash scripts/run_slow_tests.sh [--check|--run]' >&2
    exit 2
fi

export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HUB_CACHE=artifacts/hf-cache PYTHONPATH=src
export HYPOTHESIS_STORAGE_DIRECTORY=artifacts/hypothesis

if [[ "$mode" == --run ]]; then
    mkdir -p artifacts/tests
    run_dir="$(mktemp -d "artifacts/tests/slow-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
    echo "Test results: $run_dir"
    trap 'printf "%s\n" "$?" > "$run_dir/exit-code"' EXIT
    exec > "$run_dir/pytest.log" 2>&1
    git rev-parse HEAD > "$run_dir/source-commit.txt"
    git status --porcelain > "$run_dir/source-status.txt"
fi

.venv/bin/python - <<'PY'
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

record = json.loads(Path("artifacts/setup/slow-test-models.json").read_text())
assert record["status"] == "ready" and len(record["models"]) == 2
for model in record["models"]:
    assert Path(model["offline_main_reference"]).read_text().strip() == model["revision"]
    assert any(entry["path"].endswith(".safetensors") for entry in model["files"])
    for entry in model["files"]:
        assert Path(entry["path"]).is_file(), entry["path"]
        assert Path(entry["path"]).stat().st_size == entry["bytes"], entry["path"]
    AutoConfig.from_pretrained(model["model"], token=False, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model["model"], token=False, local_files_only=True)
    assert len(tokenizer) > 0
    print(f"Offline model ready: {model['model']} at {model['revision']}", flush=True)
PY

# Collection imports CUDA constants too. Check the GPU before either mode.
gpu_status="$(nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version \
    --format=csv,noheader,nounits)"
if [[ "$mode" == --run ]]; then
    printf '%s\n' "$gpu_status" > "$run_dir/gpu-before.csv"
else
    printf '%s\n' "$gpu_status"
fi
# Refuse to compete with existing compute jobs; never stop another process.
compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "$compute_pids" ]]; then
    echo 'GPU compute jobs are already running; no collection or tests started.' >&2
    exit 1
fi
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'
if [[ "$mode" == --check ]]; then
    .venv/bin/python -m pytest -m slow --collect-only -q
    exit 0
fi

echo 'Starting the slow test suite.'
systemd-inhibit --what=sleep:idle:handle-lid-switch --why='FlashQuest slow integration tests' \
    .venv/bin/python -m pytest -m slow -ra --junitxml="$run_dir/results.xml"
