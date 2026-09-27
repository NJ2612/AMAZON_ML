#!/usr/bin/env bash
# kaggle_2 step 2: push the GPU inference kernel. Run ONLY after the CV kernel
# (parasrawal117/ber-neural-cv) has COMPLETE status — the infer kernel mounts its
# output (reranker_final + deploy_neural.json) via kernel_sources.
set -uo pipefail

export KAGGLE_CONFIG_DIR="C:/Users/ACER/.kaggle_2"
REPO="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/code/business_entity_resolution"
INFDIR="$REPO/kaggle/neural/infer"
CV_ID="parasrawal117/ber-neural-cv"
K_ID="parasrawal117/ber-neural-infer"

echo "=== check CV kernel status (must be complete) ==="
st="$(kaggle kernels status "$CV_ID" 2>&1 || true)"; echo "  $st"
if ! echo "$st" | grep -qi "complete"; then
  echo "!!! CV kernel not complete yet — wait, then re-run. (infer needs its output)"; exit 1
fi

echo "=== PUSH infer kernel (GPU, private, internet off) ==="
kaggle kernels push -p "$INFDIR" 2>&1
sleep 15
echo "=== infer kernel status: $(kaggle kernels status "$K_ID" 2>&1) ==="
echo "=== DONE — poll + pull with: pull_results_neural.sh infer ==="
