#!/usr/bin/env bash
# Poll a kaggle_2 kernel until it finishes, then download its /kaggle/working
# output. Usage:  pull_results_neural.sh cv|infer
# cv    -> output/kaggle_results_2/cv/    (deploy_neural.json + log => the CV F0.5)
# infer -> output/kaggle_results_2/       (matching_results.tsv + candidate_pairs.tsv)
set -uo pipefail

export KAGGLE_CONFIG_DIR="C:/Users/ACER/.kaggle_2"
REPO="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/code/business_entity_resolution"
WHICH="${1:-infer}"
if [ "$WHICH" = "cv" ]; then
  K_ID="parasrawal117/ber-neural-cv";    OUT="$REPO/output/kaggle_results_2/cv"
else
  K_ID="parasrawal117/ber-neural-infer"; OUT="$REPO/output/kaggle_results_2"
fi
mkdir -p "$OUT"

echo "=== POLL $K_ID (12h cap; checks every ~60s) ==="
DONE=0
for i in $(seq 1 720); do
  st="$(kaggle kernels status "$K_ID" 2>&1 || true)"
  ts="$(date -u +%TZ)"
  echo "  [$i $ts] $st"
  if echo "$st" | grep -qi "complete"; then DONE=1; break; fi
  if echo "$st" | grep -qiE "error|cancel"; then
    echo "!!! kernel $K_ID failed: $st"; exit 1
  fi
  sleep 60
done
[ "$DONE" = 1 ] || { echo "!!! timed out polling $K_ID"; exit 1; }

echo "=== DOWNLOAD output -> $OUT ==="
kaggle kernels output "$K_ID" -p "$OUT" 2>&1
echo "=== files ==="; ls -la "$OUT"
echo "=== DONE ($K_ID -> $OUT) ==="
