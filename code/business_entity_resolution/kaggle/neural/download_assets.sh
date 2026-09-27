#!/usr/bin/env bash
# LOCAL prep (INTERNET ON): download model weights + Linux wheels for the offline
# Kaggle kernels. Weights land under models/neural/<name> (the dir names nconfig
# expects); wheels are cross-platform manylinux builds for the Kaggle GPU image.
# Idempotent: re-running skips already-present files.
set -uo pipefail

B="C:/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge"
MODELS="$B/models/neural"
WHEELS="$B/dataset/work/ber_neural_assets/wheels"
mkdir -p "$MODELS" "$WHEELS"

echo "=== pip: huggingface_hub (local, online) ==="
python -m pip install -q -U "huggingface_hub>=0.23" || { echo "!!! hub install failed"; exit 1; }

echo "=== download LaBSE (bi-encoder) ==="
python - "$MODELS/LaBSE" <<'PY'
import sys; from huggingface_hub import snapshot_download
snapshot_download("sentence-transformers/LaBSE", local_dir=sys.argv[1],
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.h5","*.ot","*.msgpack","tf_model*","onnx/*","openvino/*"])
print("LaBSE ->", sys.argv[1])
PY

echo "=== download gte-multilingual-reranker-base (cross-encoder, incl. custom code) ==="
python - "$MODELS/gte-multilingual-reranker-base" <<'PY'
import sys; from huggingface_hub import snapshot_download
snapshot_download("Alibaba-NLP/gte-multilingual-reranker-base", local_dir=sys.argv[1],
                  local_dir_use_symlinks=False,
                  ignore_patterns=["*.h5","*.ot","*.msgpack","tf_model*","onnx/*"])
print("gte-reranker ->", sys.argv[1])
PY

echo "=== download Linux wheels (jellyfish, rapidfuzz) for cp310 + cp311 ==="
for PYV in 3.10 3.11; do
  python -m pip download jellyfish rapidfuzz --only-binary=:all: \
    --platform manylinux2014_x86_64 --implementation cp --python-version "$PYV" \
    -d "$WHEELS" || echo "  (some wheels for $PYV skipped)"
done
echo "=== wheels present ==="; ls -1 "$WHEELS"
echo "=== DONE (weights: $MODELS ; wheels: $WHEELS) ==="
