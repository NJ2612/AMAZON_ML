#!/usr/bin/env bash
# Assemble the ber-neural-assets upload dir, mirroring the repo layout so that on
# Kaggle common.py resolves ROOT=<mount> and every read path is correct with no
# code change. Run download_assets.sh FIRST (populates models/ + wheels/).
set -uo pipefail

B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
REPO="$B/code/business_entity_resolution"
A="$B/dataset/work/ber_neural_assets"     # upload root (== /kaggle/input/ber-neural-assets)

echo "=== assemble $A ==="
mkdir -p "$A/dataset/clean/train" "$A/dataset/clean/test" \
         "$A/dataset/train" "$A/dataset/work" "$A/models/neural" "$A/wheels"

echo "-- src (code) — 3 levels below the mount so common.py's ROOT=<mount> --"
# common.py computes ROOT = abspath(HERE/../../..). Placing src at
# code/business_entity_resolution/src makes ROOT resolve to the dataset mount, so
# CLEAN/WORK/GROUND_TRUTH/DEVSET/OOF all land on the mount with no env hacks
# (GROUND_TRUTH has NO env override, so only the correct layout fixes it).
mkdir -p "$A/code/business_entity_resolution"
cp -r "$REPO/src" "$A/code/business_entity_resolution/"
find "$A/code/business_entity_resolution/src" -type d -name "__pycache__" \
     -exec rm -rf {} + 2>/dev/null || true

echo "-- clean TSVs (train + test) --"
cp "$B/dataset/clean/train/train_source1.tsv" "$A/dataset/clean/train/"
cp "$B/dataset/clean/train/train_source2.tsv" "$A/dataset/clean/train/"
cp "$B/dataset/clean/train/train_source3.tsv" "$A/dataset/clean/train/"
cp "$B/dataset/clean/test/test_source1.tsv"   "$A/dataset/clean/test/"
cp "$B/dataset/clean/test/test_source2.tsv"   "$A/dataset/clean/test/"
cp "$B/dataset/clean/test/test_source3.tsv"   "$A/dataset/clean/test/"

echo "-- ground truth + devset/oof parquets --"
cp "$B/dataset/train/train_ground_truth.tsv"                 "$A/dataset/train/"
cp "$B/dataset/work/devset_india_n20000.parquet"            "$A/dataset/work/"
cp "$B/dataset/work/devset_india_n20000_oof.parquet"        "$A/dataset/work/"

echo "-- model weights are NOT shipped: the local box could not finish the HF"
echo "   download (WinError 10054), so run_cv.py / run_infer.py fetch LaBSE + the"
echo "   gte reranker ON KAGGLE (internet ON) into /kaggle/working. This keeps the"
echo "   uploaded dataset small (~tens of MB) so the upload survives a flaky link. --"
# (models/neural left empty on purpose)

echo "-- dataset metadata --"
cp "$REPO/kaggle/neural/dataset-metadata.json" "$A/dataset-metadata.json"

echo "=== sizes ==="; du -sh "$A" "$A/dataset/clean" "$A/models" "$A/wheels" 2>/dev/null
echo "=== assembled -> $A ==="
