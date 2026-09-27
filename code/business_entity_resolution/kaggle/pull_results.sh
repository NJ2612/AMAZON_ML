#!/usr/bin/env bash
# Poll the Kaggle kernel until it finishes, then pull the two result TSVs into
# output/kaggle_result/. NO submit — download only.
set -uo pipefail

K_ID="happyparsi/ber-matching-run"
OUT="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/output/kaggle_result"
mkdir -p "$OUT"

echo "=== POLL START $(date -u +%FT%TZ) ==="
FINAL=""
for i in $(seq 1 720); do          # up to ~12h at 60s cadence (Kaggle CPU kernel max)
  st="$(kaggle kernels status "$K_ID" 2>&1 | grep -vi 'outdated API' || true)"
  echo "  [$i $(date -u +%H:%M:%SZ)] $st"
  if echo "$st" | grep -qiE 'has status "complete"'; then FINAL="complete"; break; fi
  if echo "$st" | grep -qiE 'has status "error"'; then FINAL="error"; break; fi
  if echo "$st" | grep -qiE 'cancel'; then FINAL="cancelled"; break; fi
  sleep 60
done
echo "=== FINAL STATUS: ${FINAL:-timeout} ==="

if [ "$FINAL" != "complete" ] && [ "$FINAL" != "error" ]; then
  echo "=== NOT downloading (status=${FINAL:-timeout}); kernel not finished ==="
  echo "=== PULL EXIT $(date -u +%FT%TZ) ==="
  exit 0
fi

echo "=== DOWNLOAD output -> $OUT ==="
kaggle kernels output "$K_ID" -p "$OUT" 2>&1 | grep -vi "outdated API"

echo "=== RESULT FILES ==="
ls -la "$OUT"
for f in matching_results.tsv candidate_pairs.tsv; do
  if [ -f "$OUT/$f" ]; then
    echo "  $f: $(wc -l < "$OUT/$f") lines"
    head -3 "$OUT/$f"
  else
    echo "  $f: MISSING"
  fi
done
echo "=== PULL DONE $(date -u +%FT%TZ) ==="
