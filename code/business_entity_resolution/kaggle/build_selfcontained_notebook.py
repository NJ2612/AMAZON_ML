#!/usr/bin/env python3
"""Generate kaggle/matching_full.ipynb — a FULLY self-contained, offline-capable
Kaggle notebook that runs the ENTIRE pipeline from the RAW competition data:
    preprocess -> block -> 20k-India dev set -> 37 features -> train LGB+XGB
    -> score every test S1 (India/US/France) at t*=0.98 -> validate.

No uploaded private dataset is needed: the current src/ modules are read, lightly
patched (license-clean romanization), base64-embedded into the notebook, written
to /tmp/ber/... at runtime and driven via subprocess (clean multiprocessing).

NOT a competition submission: no submit; Internet is used only to pip-install the
similarity LIBRARIES (never external entity data).

Run:  python build_selfcontained_notebook.py
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


def read(name: str) -> str:
    with open(os.path.join(SRC, name), "r", encoding="utf-8") as f:
        return f.read()


# --- embed the current modules verbatim (with two targeted patches) -----------
MODULE_NAMES = ["common.py", "blocking.py", "features.py", "make_devset.py",
                "build_numidf.py", "cv_eval.py", "train.py", "infer.py"]
mods = {m: read(m) for m in MODULE_NAMES}

# patch preprocessing.py: robust, license-clean romanization (no hard GPL unidecode dep)
_pre = read("preprocessing.py")
assert "from unidecode import unidecode" in _pre, "unidecode import not found to patch"
_pre = _pre.replace(
    "from unidecode import unidecode",
    "try:\n"
    "    from unidecode import unidecode\n"
    "except Exception:\n"
    "    try:\n"
    "        from anyascii import anyascii as unidecode\n"
    "    except Exception:\n"
    "        import unicodedata\n"
    "        def unidecode(s):\n"
    "            return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')",
)
mods["preprocessing.py"] = _pre

# extra helper module: build + cache the India blocking index (make_devset loads it)
mods["build_index.py"] = (
    "import os, pickle, time\n"
    "import common as c, blocking as bl\n"
    "CACHE = os.path.join(c.WORK, 'idx_india_cap3000.pkl')\n"
    "t0 = time.time()\n"
    "ci = bl.load_country_index(c.TRAIN_FILES, 'India', df_cap=3000)\n"
    "with open(CACHE, 'wb') as f:\n"
    "    pickle.dump(ci, f, protocol=4)\n"
    "print('india index %d recs cached in %ds -> %s' % (ci.n_records(), time.time()-t0, CACHE), flush=True)\n"
)

FILES_B64 = {name: base64.b64encode(src.encode("utf-8")).decode("ascii")
             for name, src in mods.items()}
FILES_JSON = json.dumps(FILES_B64)

MD = r'''# Business Entity Resolution — full self-contained pipeline (Kaggle, Run-All)

**What this does.** From the RAW competition data (attached under `/kaggle/input`) this single notebook runs the whole matching pipeline end-to-end and writes the two deliverable TSVs to `/kaggle/working`: `matching_results.tsv` and `candidate_pairs.tsv`.

Stages: clean/normalize -> per-country blocking index -> 20k-India labeled dev set -> 37 pairwise features -> train LightGBM+XGBoost blend -> score every test S1 (India/US/France) at the frozen operating threshold **t\*=0.98** -> validate format.

**How to run.** (1) Add the competition data as an input (the six `*_source*.tsv` files + `train_ground_truth.tsv`). (2) Settings -> Accelerator: **None (CPU)**; **Internet: On** (only to `pip install` the similarity libraries — no external entity data is ever fetched). (3) **Run All**, or *Save Version* to run headless (CPU kernels allow up to ~12 h).

**This is NOT a competition submission.** It only writes local TSVs to the kernel output; `kaggle competitions submit` is never called. No external data lookup resolves entities (challenge fair-play); romanization uses a permissively-licensed fallback (anyascii / stdlib), not GPL unidecode.

**Honest accuracy note.** The measured leakage-safe CV of this exact blend is macro-F0.5 ~ **0.85** on the 20k-India dev set (trained on India, applied to US/France out-of-distribution). This notebook reproduces that config; it does **not** claim 0.956/0.98, which sit above the blocking recall ceiling.
'''

CELL_DEPS = r'''# --- deps: LIBRARIES only (never external entity data) ----------------------
# If Internet is OFF and any import below is missing, turn Internet ON (Settings)
# and re-run this cell. Installing PyPI wheels is not an "external data lookup".
import sys, subprocess
for _pkg in ("rapidfuzz", "jellyfish", "anyascii", "unidecode"):
    try:
        __import__(_pkg)
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg], check=False)
import numpy, pandas, sklearn, lightgbm, xgboost   # already in the Kaggle image
print("deps ok:", numpy.__version__, pandas.__version__,
      lightgbm.__version__, xgboost.__version__)
'''

CELL_LAYOUT = r'''# --- lay out /tmp/ber (ROOT), discover raw inputs, write the pipeline modules
import os, sys, glob, json, base64, shutil
ROOT = "/tmp/ber"                       # ROOT -> src/../../.. ; big intermediates stay out of the 20GB output
SRC_DIR = ROOT + "/code/business_entity_resolution/src"
for d in (ROOT + "/dataset/train", ROOT + "/dataset/test", SRC_DIR):
    os.makedirs(d, exist_ok=True)

def find_raw(*pats):
    for p in pats:
        hits = [h for h in sorted(glob.glob("/kaggle/input/**/" + p, recursive=True))
                if os.path.isfile(h)]
        if hits:
            return hits[0]
    return None

RAW = {
    "train/train_source1.tsv": find_raw("train_source1.tsv", "*train*source*1*.tsv"),
    "train/train_source2.tsv": find_raw("train_source2.tsv", "*train*source*2*.tsv"),
    "train/train_source3.tsv": find_raw("train_source3.tsv", "*train*source*3*.tsv"),
    "test/test_source1.tsv":   find_raw("test_source1.tsv", "*test*source*1*.tsv"),
    "test/test_source2.tsv":   find_raw("test_source2.tsv", "*test*source*2*.tsv"),
    "test/test_source3.tsv":   find_raw("test_source3.tsv", "*test*source*3*.tsv"),
    "train/train_ground_truth.tsv": find_raw("train_ground_truth.tsv", "*ground_truth*.tsv"),
}
missing = [k for k, v in RAW.items() if not v]
assert not missing, ("attach the competition data; not found: " + str(missing) +
                     "  |  /kaggle/input/* = " + str(glob.glob("/kaggle/input/*")))
for rel, srcpath in RAW.items():
    dst = ROOT + "/dataset/" + rel
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.symlink(srcpath, dst)
    except Exception:
        shutil.copy(srcpath, dst)
    print(rel, "<-", srcpath, flush=True)

FILES = json.loads(r"""__FILES_JSON__""")
for name, b64 in FILES.items():
    with open(os.path.join(SRC_DIR, name), "wb") as f:
        f.write(base64.b64decode(b64))
print("wrote modules:", sorted(FILES))
'''

# PLACEHOLDER_CELLS2

CELL_RUNNER = r'''# --- subprocess runner (cwd=SRC_DIR so `import common` resolves; live logs) --
import os, sys, subprocess, time
ENV = dict(os.environ)
ENV["BER_OUTPUT"] = "/kaggle/working"     # the two deliverable TSVs land here (downloadable)
ENV["PYTHONUNBUFFERED"] = "1"

def run(script, *a):
    cmd = [sys.executable, os.path.join(SRC_DIR, script), *map(str, a)]
    print(">>", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=SRC_DIR, env=ENV)
    if r.returncode != 0:
        raise RuntimeError(script + " FAILED rc=" + str(r.returncode))
    print("   [%s] done in %.1f min" % (script, (time.time() - t0) / 60), flush=True)
'''

CELL_PREPROCESS = r'''# --- Stage 1: clean/normalize all six raw source files ----------------------
run("preprocessing.py")
'''

CELL_DEVSET = r'''# --- Stage 2: India blocking index + IDFs + 20k labeled dev set + features ---
run("build_index.py")                 # -> work/idx_india_cap3000.pkl
run("make_devset.py", 20000)          # -> work/devset_india_n20000.parquet (+ tokdf_india.pkl)
run("build_numidf.py")                # -> work/numidf_india.pkl (rare-numeric address signal)
DEV = ROOT + "/dataset/work/devset_india_n20000.parquet"
run("features.py", DEV)               # -> work/devset_india_n20000_feats.parquet
'''

CELL_TRAIN = r'''# --- Stage 3: fit the deploy LGB+XGB blend (no OOF file -> frozen t*=0.98) ----
FEATS = ROOT + "/dataset/work/devset_india_n20000_feats.parquet"
run("train.py", FEATS, "--neg", "3", "--prob", "blend")
'''

CELL_INFER = r'''# --- Stage 4: score EVERY test S1 (all countries) -> the two TSVs ------------
import os
NCPU = os.cpu_count() or 4
run("infer.py", "--topk", "200", "--batch-s1", "5000", "--procs", str(NCPU))
'''

# PLACEHOLDER_CELLS3

CELL_VALIDATE = r'''# --- Stage 5: validate the two deliverables (format + coverage, no GT needed) -
import os, pandas as pd
MRES = "/kaggle/working/matching_results.tsv"
CRES = "/kaggle/working/candidate_pairs.tsv"

s1 = set()
for ch in pd.read_csv(ROOT + "/dataset/clean/test/test_source1.tsv", sep="\t",
                      dtype=str, keep_default_na=False, na_values=[], quoting=3,
                      usecols=["entity_id"], chunksize=400000):
    s1.update(ch["entity_id"])

m = pd.read_csv(MRES, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)
cd = pd.read_csv(CRES, sep="\t", dtype=str, keep_default_na=False, na_values=[], quoting=3)
assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"], m.columns.tolist()
assert list(cd.columns) == ["source1_entity_id", "candidate_entity_ids"], cd.columns.tolist()

mset, cset = set(m["source1_entity_id"]), set(cd["source1_entity_id"])
cand_map = dict(zip(cd["source1_entity_id"], cd["candidate_entity_ids"]))
notsub = 0
for sid, mm in zip(m["source1_entity_id"], m["matched_entity_ids"]):
    pm = {x for x in mm.split(",") if x}
    pc = {x for x in cand_map.get(sid, "").split(",") if x}
    if not pm.issubset(pc):
        notsub += 1
nonempty = sum(1 for v in m["matched_entity_ids"] if v)

print("test S1 entities          :", len(s1))
print("matching rows / candidate rows:", len(m), "/", len(cd))
print("every test S1 in matching :", s1.issubset(mset), "(missing %d)" % len(s1 - mset))
print("every test S1 in candidates:", s1.issubset(cset), "(missing %d)" % len(s1 - cset))
print("rows w/ match NOT subset of candidates:", notsub)
print("non-empty matched rows    : %d (%.1f%%)" % (nonempty, 100 * nonempty / max(len(m), 1)))
ok = (s1.issubset(mset) and notsub == 0 and
      list(m.columns) == ["source1_entity_id", "matched_entity_ids"])
print("\nVALIDATION:", "PASS" if ok else "CHECK ABOVE")
print("OUTPUTS in /kaggle/working:", [f for f in os.listdir("/kaggle/working") if f.endswith(".tsv")])
'''


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


cells = [
    md(MD),
    code(CELL_DEPS),
    code(CELL_LAYOUT.replace("__FILES_JSON__", FILES_JSON)),
    code(CELL_RUNNER),
    code(CELL_PREPROCESS),
    code(CELL_DEVSET),
    code(CELL_TRAIN),
    code(CELL_INFER),
    code(CELL_VALIDATE),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

OUT = os.path.join(HERE, "matching_full.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
kb = os.path.getsize(OUT) / 1024
print("wrote %s (%.0f KB, %d cells)" % (OUT, kb, len(cells)))

