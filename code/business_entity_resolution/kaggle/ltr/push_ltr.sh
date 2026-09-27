#!/usr/bin/env bash
# kaggle_1 LTR deploy: auth-test (NO upload) -> upload the tiny ber-ltr-artifacts
# dataset -> push the CPU inference kernel (internet OFF, reuses ber-matching-inputs
# for the test TSVs). Uses the happyparsi account (KAGGLE_CONFIG_DIR default).
# Never prints the token. NO competition submit.
set -uo pipefail

export KAGGLE_CONFIG_DIR="C:/Users/ACER/.kaggle"
B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
REPO="$B/code/business_entity_resolution"
A="$B/dataset/work/ber_ltr_artifacts"
KDIR="$REPO/kaggle/ltr"
DS_ID="happyparsi/ber-ltr-artifacts"
K_ID="happyparsi/ber-ltr-run"

echo "=== ensure kaggle CLI ==="
python -m pip install -q -U kaggle || true

echo "=== AUTH TEST (no upload) ==="
if ! kaggle datasets list -m --page-size 3 >/tmp/ltr_auth.txt 2>&1; then
  echo "!!! AUTH FAILED -- check $KAGGLE_CONFIG_DIR/kaggle.json (creds NOT printed)"
  sed -n '1,5p' /tmp/ltr_auth.txt; exit 1
fi
echo "auth OK ($(wc -l </tmp/ltr_auth.txt) lines)"

# --- Windows kaggle resumable-upload sidecar-dir bug workaround -----------------
KUP="$HOME/AppData/Local/Temp/.kaggle/uploads"
mkdir -p "$KUP/C_/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work/ber_ltr_artifacts" 2>/dev/null || true

echo "=== UPLOAD dataset (create) from $A ==="
OUT="$(kaggle datasets create -p "$A" --dir-mode zip 2>&1)"; RC=$?
echo "$OUT"
if [ $RC -ne 0 ] && echo "$OUT" | grep -qiE "already exists|409|version"; then
  echo "=== exists -> new version ==="
  kaggle datasets version -p "$A" --dir-mode zip -m "ltr artifacts $(date -u +%FT%TZ)" 2>&1; RC=$?
fi

echo "=== WAIT dataset ready (<=20 min) ==="
for i in $(seq 1 60); do
  st="$(kaggle datasets status "$DS_ID" 2>&1 || true)"
  echo "  [$i] $st"
  echo "$st" | grep -qi "ready" && { echo "dataset READY"; break; }
  sleep 20
done

echo "=== PUSH kernel from $KDIR ==="
kaggle kernels push -p "$KDIR" 2>&1
sleep 15
echo "=== kernel status: $(kaggle kernels status "$K_ID" 2>&1) ==="
echo "=== push_ltr done — kernel running on Kaggle. Poll+pull with pull_ltr.sh ==="
