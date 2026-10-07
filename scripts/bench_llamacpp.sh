#!/usr/bin/env bash
# Separate prefill and decode at the requested cache depth. Requires llama-bench -d.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
LLAMA_BIN="${LLAMA_BIN:-$REPO_ROOT/vendor/llama.cpp/build/bin/llama-bench}"
MODEL="${MODEL:-$HOME/models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf}"
CTX="${CTX:-8192}"
OUT="${OUT:-$REPO_ROOT/benchmarks/validation/llamacpp_${CTX}.llamacpp.json}"
NGL="${NGL:-999}"
N_DECODE="${N_DECODE:-128}"
KV_K="${KV_K:-q4_0}"
KV_V="${KV_V:-q4_0}"
REPS="${REPS:-3}"

mkdir -p "$(dirname "$OUT")"

if [ ! -x "$LLAMA_BIN" ]; then
  echo "ERROR: set LLAMA_BIN to an executable llama-bench with context-depth support (-d)." >&2
  exit 1
fi
if [ ! -f "$MODEL" ]; then
  echo "ERROR: model not found at $MODEL" >&2
  exit 1
fi

if (( CTX < 1 || N_DECODE < 2 || REPS < 1 )); then
  echo "ERROR: CTX >= 1, N_DECODE >= 2, and REPS >= 1 are required." >&2
  exit 1
fi

common=(-m "$MODEL" -ngl "$NGL" -t "${THREADS:-6}" -r "$REPS"
        -ctk "$KV_K" -ctv "$KV_V" -fa on -o json)
# -p and -n describe independent tests, not decode following that prefill.
"$LLAMA_BIN" "${common[@]}" -p "$CTX" -n 0 -d 0 > "${OUT%.json}.prefill.json"
# Match the N_DECODE-1 post-prefill forward steps in FlashQuest/vLLM.
"$LLAMA_BIN" "${common[@]}" -p 0 -n "$((N_DECODE - 1))" -d "$CTX" | tee "$OUT"
