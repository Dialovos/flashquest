#!/usr/bin/env bash
# Phase 6 task 2 — RULER NIAH filler corpus.
#
# RULER ships URLs (vendor/RULER/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt)
# and a download script that requires html2text + bs4 + tqdm to scrape paulgraham.com.
# We instead pull the gkamradt-pre-extracted .txt subset (~50 essays, ~650 KB)
# directly via curl and bundle to {"text": ...} JSON — same shape RULER's pipeline
# expects. Output is committed to data/PaulGrahamEssays.json so re-running this
# script is optional (only needed if you want to regenerate the corpus from
# upstream).
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
URLS_FILE="$REPO_ROOT/vendor/RULER/scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt"
OUT="$REPO_ROOT/data/PaulGrahamEssays.json"

if [ ! -f "$URLS_FILE" ]; then
  echo "Run scripts/vendor_clone.sh first to fetch vendor/RULER/." >&2
  exit 1
fi

TMPDIR=$(mktemp -d)
trap "rm -rf '$TMPDIR'" EXIT

cd "$TMPDIR"
grep "gkamradt" "$URLS_FILE" > urls.txt
echo "Fetching $(wc -l < urls.txt) essays from gkamradt/LLMTest_NeedleInAHaystack..."
while IFS= read -r url; do
  curl -sL -o "$(basename "$url")" "$url"
done < urls.txt

mkdir -p "$(dirname "$OUT")"
python3 - <<'EOF' "$OUT"
import glob, json, sys
out = sys.argv[1]
text = ""
for f in sorted(glob.glob("*.txt")):
    text += open(f).read()
with open(out, "w") as fh:
    json.dump({"text": text}, fh)
print(f"wrote {out} ({len(text)} chars)")
EOF
