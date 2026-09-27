#!/usr/bin/env bash
# Pull the LTR kernel output (matching_results.tsv + candidate_pairs.tsv) into
# output/kaggle_result/ and validate. happyparsi account.
set -uo pipefail

export KAGGLE_CONFIG_DIR="C:/Users/ACER/.kaggle"
B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
REPO="$B/code/business_entity_resolution"
K_ID="happyparsi/ber-ltr-run"
DEST="$B/output/kaggle_result"

echo "=== download kernel output -> $DEST ==="
mkdir -p "$DEST"

echo "=== POLL $K_ID until complete (<=12h; checks ~every 60s) ==="
DONE=0
for i in $(seq 1 720); do
  st="$(kaggle kernels status "$K_ID" 2>&1 || true)"
  echo "  [$i $(date -u +%TZ)] $st"
  echo "$st" | grep -qi "complete" && { DONE=1; break; }
  if echo "$st" | grep -qiE "error|cancel"; then echo "!!! kernel failed: $st"; exit 1; fi
  sleep 60
done
[ "$DONE" = 1 ] || { echo "!!! timed out polling $K_ID"; exit 1; }

kaggle kernels output "$K_ID" -p "$DEST" 2>&1

echo "=== contents ==="; ls -lh "$DEST" | awk '{print $5, $9}'

MR="$DEST/matching_results.tsv"; CP="$DEST/candidate_pairs.tsv"
if [ -f "$MR" ] && [ -f "$CP" ]; then
  echo "=== VALIDATE ==="
  python "$REPO/utils/validate_submission.py" \
    --matching "$MR" --candidate "$CP" \
    --test-dir "$B/dataset/test" --check-ids 2>&1
  echo "validate exit=$?"
else
  echo "!!! expected TSVs not found in $DEST -- check kernel log"
fi
echo "=== pull_ltr done ==="
