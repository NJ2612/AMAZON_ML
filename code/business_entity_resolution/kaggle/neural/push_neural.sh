#!/usr/bin/env bash
# kaggle_2 deploy: auth-test (NO upload) -> upload private assets dataset ->
# push the GPU CV kernel. Uses the SECOND account via KAGGLE_CONFIG_DIR; never
# prints the token. NO competition submit. Internet OFF in the kernel.
set -uo pipefail

export KAGGLE_CONFIG_DIR="C:/Users/ACER/.kaggle_2"
B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
REPO="$B/code/business_entity_resolution"
A="$B/dataset/work/ber_neural_assets"
CVDIR="$REPO/kaggle/neural/cv"
DS_ID="parasrawal117/ber-neural-assets"
K_ID="parasrawal117/ber-neural-cv"

echo "=== ensure kaggle CLI ==="
python -m pip install -q -U kaggle || true

echo "=== AUTH TEST (no upload): list my datasets ==="
if ! kaggle datasets list -m --page-size 3 >/tmp/authtest.txt 2>&1; then
  echo "!!! AUTH FAILED — check $KAGGLE_CONFIG_DIR/kaggle.json (creds NOT printed)"
  sed -n '1,5p' /tmp/authtest.txt; exit 1
fi
echo "auth OK ($(wc -l </tmp/authtest.txt) lines)"

# --- Windows kaggle resumable-upload sidecar-dir bug workaround ------------------
KUP="$HOME/AppData/Local/Temp/.kaggle/uploads"
mkdir -p "$KUP/C_/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work/ber_neural_assets"

echo "=== UPLOAD dataset (create) from $A ==="
OUT="$(kaggle datasets create -p "$A" --dir-mode zip 2>&1)"; RC=$?
echo "$OUT"
if [ $RC -ne 0 ] && echo "$OUT" | grep -qiE "already exists|409|version"; then
  echo "=== exists -> version ==="
  kaggle datasets version -p "$A" --dir-mode zip -m "assets $(date -u +%FT%TZ)" 2>&1; RC=$?
fi

echo "=== WAIT dataset ready (<=40 min) ==="
for i in $(seq 1 120); do
  st="$(kaggle datasets status "$DS_ID" 2>&1 || true)"
  echo "  [$i] $st"; echo "$st" | grep -qi "ready" && break; sleep 20
done

echo "=== PUSH CV kernel (GPU, private, internet off) ==="
kaggle kernels push -p "$CVDIR" 2>&1
sleep 15
echo "=== CV kernel status: $(kaggle kernels status "$K_ID" 2>&1) ==="
echo "=== DONE — CV kernel running on Kaggle. Poll with pull_results_neural.sh cv ==="
