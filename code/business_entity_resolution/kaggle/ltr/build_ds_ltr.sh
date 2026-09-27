#!/usr/bin/env bash
# Assemble the TINY ber-ltr-artifacts upload dir: the current src/ (ltr.py,
# ltr_infer.py and the 37-feature features.py), the frozen deploy_ltr.pkl + meta,
# and validate_submission.py. The 2.3 GB of test TSVs + IDF caches + wheels are
# NOT here -- the LTR kernel reuses those from the existing ber-matching-inputs
# dataset, so this upload is only a few MB. Run ltr_train.py FIRST (freezes the
# model into models/deploy_ltr.pkl + deploy_ltr_meta.json).
set -uo pipefail

B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
REPO="$B/code/business_entity_resolution"
A="$B/dataset/work/ber_ltr_artifacts"      # upload root (== /kaggle/input/ber-ltr-artifacts)

echo "=== assemble $A ==="
rm -rf "$A"
mkdir -p "$A/src" "$A/models" "$A/utils"

echo "-- current src (full, so every ltr_infer import resolves from one mount) --"
cp -r "$REPO/src/." "$A/src/"
find "$A/src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$A/src" -type f -name "*.pyc" -delete 2>/dev/null || true
# drop bulky logs that may sit in src/ (build logs) -- keep the upload lean
find "$A/src" -maxdepth 1 -type f -name "*.log" -delete 2>/dev/null || true

echo "-- frozen deploy artifacts --"
cp "$B/models/deploy_ltr.pkl"        "$A/models/" || { echo "!!! deploy_ltr.pkl missing -- run ltr_train.py"; exit 1; }
cp "$B/models/deploy_ltr_meta.json"  "$A/models/" || { echo "!!! meta missing -- run ltr_train.py"; exit 1; }

echo "-- validator (for the local check; harmless on the kernel) --"
cp "$REPO/utils/validate_submission.py" "$A/utils/" 2>/dev/null || true

echo "-- dataset metadata --"
cp "$REPO/kaggle/ltr/dataset-metadata.json" "$A/dataset-metadata.json"

echo "=== contents ==="; find "$A" -type f | sed "s#$A/##" | sort
echo "=== size ==="; du -sh "$A" 2>/dev/null
echo "=== assembled -> $A ==="
