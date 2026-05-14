#!/usr/bin/env bash
# Idempotent vendor sync. Re-running pulls latest on each repo's default branch.
# Pin SHAs once Phase 1 begins to avoid upstream drift breaking phase reproducibility.
set -euo pipefail

VENDOR_DIR="$(git rev-parse --show-toplevel)/vendor"
mkdir -p "$VENDOR_DIR"
cd "$VENDOR_DIR"

clone_or_pull() {
  local url="$1"
  local dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "==> Updating $dir"
    git -C "$dir" fetch --depth=1 origin
    git -C "$dir" reset --hard origin/HEAD
  else
    echo "==> Cloning $dir"
    git clone --depth=1 "$url" "$dir"
  fi
}

clone_or_pull https://github.com/mit-han-lab/Quest.git                  quest
clone_or_pull https://github.com/jy-yuan/KIVI.git                       kivi
clone_or_pull https://github.com/mit-han-lab/duo-attention.git          duo-attention
clone_or_pull https://github.com/mit-han-lab/streaming-llm.git          streaming-llm
clone_or_pull https://github.com/IST-DASLab/marlin.git                  marlin
clone_or_pull https://github.com/SafeAILab/EAGLE.git                    eagle
clone_or_pull https://github.com/triton-lang/triton.git                 triton
clone_or_pull https://github.com/Dao-AILab/flash-attention.git          flash-attention
clone_or_pull https://github.com/mit-han-lab/Block-Sparse-Attention.git block-sparse-attention
clone_or_pull https://github.com/ggerganov/llama.cpp.git                llama.cpp
clone_or_pull https://github.com/NVIDIA/RULER.git                       RULER

echo "Done. Vendored repos in $VENDOR_DIR"
