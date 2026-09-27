#!/usr/bin/env python3
"""Kaggle LTR inference kernel bootstrap (CPU, internet OFF).

Attaches two private datasets:
  ber-matching-inputs -> cleaned test TSVs, IDF caches (tokdf/numidf), wheels
                         (reused from the GBDT deploy; 2.3 GB not re-uploaded)
  ber-ltr-artifacts   -> current src/ (ltr.py, ltr_infer.py, 37-feat features.py)
                         + models/deploy_ltr.pkl + deploy_ltr_meta.json

Rather than hard-code /kaggle/input/<slug> paths (Kaggle's mount name can differ
from what you expect), we DISCOVER every needed file by name under /kaggle/input
and wire common.py's env overrides from what we find. Robust to any mount layout.
"""
import glob
import os
import subprocess
import sys

INPUT = "/kaggle/input"


def find(name):
    hits = glob.glob(f"{INPUT}/**/{name}", recursive=True)
    return hits[0] if hits else None


print("=== /kaggle/input layout ===", flush=True)
for d in sorted(glob.glob(f"{INPUT}/*")):
    print(" ", d, flush=True)
    for sub in sorted(glob.glob(f"{d}/*"))[:25]:
        print("     ", os.path.basename(sub), flush=True)

li = find("ltr_infer.py")
if not li:
    raise SystemExit("!!! ltr_infer.py not found under /kaggle/input")
SRC = os.path.dirname(li)
sys.path.insert(0, SRC)

dep = find("deploy_ltr.pkl")
tok = find("tokdf_india.pkl")
ts1 = find("test_source1.tsv")
whl = glob.glob(f"{INPUT}/**/*.whl", recursive=True)
if not (dep and tok and ts1):
    raise SystemExit(f"!!! missing inputs: deploy={dep} tokdf={tok} test_s1={ts1}")

MODELS = os.path.dirname(dep)
WORK = os.path.dirname(tok)
CLEAN = os.path.dirname(os.path.dirname(ts1))   # .../<CLEAN>/test/test_source1.tsv
WHEELS = os.path.dirname(whl[0]) if whl else None
print(f"SRC={SRC}\nMODELS={MODELS}\nWORK={WORK}\nCLEAN={CLEAN}\nWHEELS={WHEELS}",
      flush=True)

os.environ["BER_CLEAN"] = CLEAN
os.environ["BER_WORK"] = WORK
os.environ["BER_MODELS"] = MODELS
os.environ["BER_OUTPUT"] = "/kaggle/working"

# offline deps from shipped wheels (lightgbm/pandas/numpy preinstalled on Kaggle)
if WHEELS:
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links", WHEELS, "jellyfish", "rapidfuzz"], check=False)

print("=== [ltr-infer] full-test inference (variable-k cutoff + global 1-to-1) ===",
      flush=True)
# Import as a NORMAL module (not runpy __main__): the Pool workers pickle
# _wfeat by qualified name, so it must live in module 'ltr_infer', not '__main__'
# (runpy.run_module(run_name="__main__") reparents it -> PicklingError).
sys.argv = ["ltr_infer.py", "--topk", "200", "--batch-s1", "4000", "--procs", "4"]
import ltr_infer  # noqa: E402
ltr_infer.main()

print("=== [ltr-infer] DONE -> matching_results.tsv + candidate_pairs.tsv "
      "in /kaggle/working ===", flush=True)
