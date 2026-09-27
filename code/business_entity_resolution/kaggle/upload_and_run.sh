#!/usr/bin/env bash
# Upload the private inputs dataset and push+run the inference kernel on Kaggle.
#   - Internet OFF inside the kernel; NO competition submit (compute + storage only).
#   - The user is NOT the team lead: this NEVER touches the leaderboard.
# The 2.5 GB staging tree is packed into ONE zip (Kaggle auto-extracts it, so the
# src/ models/ work/ clean/ wheels/ utils/ layout is preserved under /kaggle/input).
# Prereqs: `pip install kaggle`, valid ~/.kaggle/kaggle.json.
# Run from this kaggle/ directory:  bash upload_and_run.sh
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(cd "$HERE/../../../dataset/work/kaggle_ds" && pwd)"     # staged inputs (2.5 GB)
UP="$HERE/../../../dataset/work/kaggle_upload"                    # single-zip upload dir
DS_ID="happyparsi/ber-matching-inputs"
K_ID="happyparsi/ber-matching-run"

echo "== 1/6 auth test (read-only, NO upload) =="
# CLI 1.6.17 has no --page-size; kernels list --mine is a real auth-required check.
kaggle kernels list --mine | head -n 5

echo "== 2/6 pack staging tree into one zip (Kaggle auto-extracts) =="
mkdir -p "$UP"
rm -f "$UP"/*.zip
python - "$STAGE" "$UP/ber_inputs.zip" <<'PY'
import os, sys, zipfile
stage, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
    for root, _, files in os.walk(stage):
        for fn in files:
            if fn == "dataset-metadata.json":
                continue
            p = os.path.join(root, fn)
            z.write(p, os.path.relpath(p, stage))   # arcname keeps src/, clean/test/, ...
print("zip bytes:", os.path.getsize(out))
PY
cp "$STAGE/dataset-metadata.json" "$UP/dataset-metadata.json"

# --- workaround for a kaggle 1.6.17 bug on Windows -----------------------------
# Its resumable (large-file) upload writes a progress sidecar to
#   %TEMP%\.kaggle\uploads\<drive-mangled absolute path of the zip>.json
# but never mkdir's that nested dir, so the upload crashes with FileNotFoundError.
# Pre-create the exact dir. Python's open() collapses the .. lexically on Windows,
# so the *resolved* target dir is what must exist. %TEMP% = C:\Users\ACER\AppData\
# Local\Temp (Python tempfile.gettempdir(), per the observed traceback).
KUP="$HOME/AppData/Local/Temp/.kaggle/uploads"
UPABS="$(cd "$UP" && pwd)"                        # resolved, no .. components
mkdir -p "$KUP"
mkdir -p "$KUP/C_${UPABS#/c}"                      # resolved upload dir mirror
mkdir -p "$KUP/C_/Users/ACER/OneDrive/Desktop/AmazonML/buisness_entity_challenge/dataset/work"
# ------------------------------------------------------------------------------

echo "== 3/6 upload private dataset from $UP =="
if kaggle datasets status "$DS_ID" >/dev/null 2>&1; then
  kaggle datasets version -p "$UP" -m "refresh $(date -u +%FT%TZ)"
else
  kaggle datasets create -p "$UP"
fi

echo "== 4/6 wait for dataset processing =="
for i in $(seq 1 60); do
  st=$(kaggle datasets status "$DS_ID" 2>/dev/null || true)
  echo "  dataset status: $st"
  echo "$st" | grep -qi "ready" && break
  sleep 20
done

echo "== 5/6 push + run kernel (private, internet off) =="
kaggle kernels push -p "$HERE"
echo "  polling (Ctrl-C stops polling only; the run continues on Kaggle)"
while true; do
  st=$(kaggle kernels status "$K_ID" 2>/dev/null || true)
  echo "  kernel status: $st"
  echo "$st" | grep -qiE "complete|error|cancelAcknowledged" && break
  sleep 60
done

echo "== 6/6 download kernel output (matching_results.tsv + candidate_pairs.tsv) =="
mkdir -p "$HERE/out"
kaggle kernels output "$K_ID" -p "$HERE/out"
ls -la "$HERE/out"
echo "Validate:  python ../../../utils/validate_submission.py \\"
echo "  --matching out/matching_results.tsv --candidate out/candidate_pairs.tsv \\"
echo "  --test-dir ../../../dataset/clean/test --check-ids"
