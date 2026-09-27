#!/usr/bin/env bash
# Upload the (already-zipped) private inputs dataset + push the inference kernel.
# Reuses the pre-built, integrity-checked ber_inputs.zip. NO competition submit,
# internet OFF in the kernel (wheels shipped). Exits right after the kernel push
# so the caller can tell the user it's safe to close the laptop.
set -uo pipefail

HERE="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/code/business_entity_resolution/kaggle"
UP="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work/kaggle_upload"
DS_ID="happyparsi/ber-matching-inputs"
K_ID="happyparsi/ber-matching-run"

echo "=== START $(date -u +%FT%TZ) ==="

# --- Windows kaggle 1.6.17 resumable-upload sidecar-dir bug workaround ----------
KUP="$HOME/AppData/Local/Temp/.kaggle/uploads"
mkdir -p "$KUP"
mkdir -p "$KUP/C_/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work/kaggle_upload"
mkdir -p "$KUP/C_/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work"

echo "=== UPLOAD dataset (create) from $UP ==="
OUT="$(kaggle datasets create -p "$UP" 2>&1)"; RC=$?
echo "$OUT"
if [ $RC -ne 0 ] && echo "$OUT" | grep -qiE "already exists|409|version"; then
  echo "=== dataset exists -> version instead ==="
  kaggle datasets version -p "$UP" -m "inputs $(date -u +%FT%TZ)" 2>&1
  RC=$?
fi
if [ $RC -ne 0 ] && ! echo "$OUT" | grep -qi "your dataset is being created"; then
  echo "!!! UPLOAD_FAILED rc=$RC"
fi

echo "=== WAIT for dataset processing (<=30 min) ==="
READY=0
for i in $(seq 1 90); do
  st="$(kaggle datasets status "$DS_ID" 2>&1 || true)"
  echo "  [$i] status: $st"
  echo "$st" | grep -qi "ready" && { READY=1; break; }
  sleep 20
done
echo "=== dataset ready=$READY ==="

echo "=== PUSH kernel (private, internet off) ==="
kaggle kernels push -p "$HERE" 2>&1
echo "=== PUSHED $(date -u +%FT%TZ) — kernel now runs on Kaggle servers ==="
sleep 15
echo "=== kernel status: $(kaggle kernels status "$K_ID" 2>&1) ==="
echo "=== DONE — safe to close the laptop ==="
