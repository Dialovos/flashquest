#!/usr/bin/env bash
# Run llama.cpp benchmark at 8k context. Records tok/s + peak VRAM.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LLAMA_BIN="$REPO_ROOT/vendor/llama.cpp/build/bin/llama-bench"
MODEL="${MODEL:-$HOME/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"
CTX="${CTX:-8192}"
OUT="${OUT:-$REPO_ROOT/benchmarks/llamacpp_${CTX}.txt}"
NGL="${NGL:-999}"

mkdir -p "$REPO_ROOT/benchmarks"

if [ ! -x "$LLAMA_BIN" ]; then
  echo "ERROR: $LLAMA_BIN not built. Run cmake build in vendor/llama.cpp first." >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

# -p $CTX prefill, -n 128 decode tokens, -ngl 999 = all layers on GPU.
"$LLAMA_BIN" \
  -m "$MODEL" \
  -p "$CTX" -n 128 \
  -ngl "$NGL" \
  -t 6 \
  -r "${REPS:-1}" \
  -o md \
  | tee "$OUT"

echo "---" >> "$OUT"
echo "VRAM at run-tail (nvidia-smi):" >> "$OUT"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv >> "$OUT"
