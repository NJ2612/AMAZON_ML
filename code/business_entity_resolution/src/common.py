#!/usr/bin/env python3
"""Shared utilities for the Business Entity Resolution matching pipeline.

Paths, schema constants, chunked TSV IO, the macro-F0.5 scorer, leakage-safe
GroupKFold assignment (grouped by Source-1 entity), candidate-pair IO, and
lightweight tokenization helpers. Offline only.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Every path can be overridden by an env var so the SAME code runs unchanged both
# locally (repo layout) and on Kaggle (read-only /kaggle/input for CLEAN/MODELS,
# writable /kaggle/working for WORK/OUTPUT). Defaults reproduce the repo layout.
CLEAN = os.environ.get("BER_CLEAN") or os.path.join(ROOT, "dataset", "clean")
WORK = os.environ.get("BER_WORK") or os.path.join(ROOT, "dataset", "work")  # large intermediates
CAND_DIR = os.path.join(WORK, "candidates")
FEAT_DIR = os.path.join(WORK, "features")
OOF_DIR = os.path.join(WORK, "oof")
MODELS = os.environ.get("BER_MODELS") or os.path.join(ROOT, "models")       # gitignored
OUTPUT = os.environ.get("BER_OUTPUT") or os.path.join(ROOT, "output")
REPORT_DIR = os.path.join(OUTPUT, "matching")

# Only the writable dirs are auto-created; CLEAN/MODELS are read-only inputs on Kaggle.
for _d in (WORK, CAND_DIR, FEAT_DIR, OOF_DIR, OUTPUT, REPORT_DIR):
    try:
        os.makedirs(_d, exist_ok=True)
    except OSError:
        pass
try:
    os.makedirs(MODELS, exist_ok=True)
except OSError:
    pass

CLEAN_COLS = ["entity_id", "country", "business_name", "business_address",
              "name_norm", "name_core", "name_suffix", "addr_norm"]

# cleaned source file -> role
TRAIN_FILES = {"S1": os.path.join(CLEAN, "train", "train_source1.tsv"),
               "S2": os.path.join(CLEAN, "train", "train_source2.tsv"),
               "S3": os.path.join(CLEAN, "train", "train_source3.tsv")}
TEST_FILES = {"S1": os.path.join(CLEAN, "test", "test_source1.tsv"),
              "S2": os.path.join(CLEAN, "test", "test_source2.tsv"),
              "S3": os.path.join(CLEAN, "test", "test_source3.tsv")}
GROUND_TRUTH = os.path.join(ROOT, "dataset", "train", "train_ground_truth.tsv")

# PLACEHOLDER_IO


def read_clean(path: str, usecols: List[str] | None = None,
               nrows: int | None = None) -> pd.DataFrame:
    """Load a cleaned TSV (all string dtype, NA-safe)."""
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                     quoting=3, encoding="utf-8", encoding_errors="replace",
                     usecols=usecols, nrows=nrows)
    return df.fillna("")


def iter_clean(path: str, chunksize: int = 400_000, usecols: List[str] | None = None):
    """Stream a cleaned TSV in chunks (memory-safe)."""
    reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                         quoting=3, encoding="utf-8", encoding_errors="replace",
                         usecols=usecols, chunksize=chunksize)
    for chunk in reader:
        yield chunk.fillna("")


def load_ground_truth() -> Dict[str, Set[str]]:
    """S1 entity_id -> set of matched S2/S3 ids ({} means singleton)."""
    gt: Dict[str, Set[str]] = {}
    reader = pd.read_csv(GROUND_TRUTH, sep="\t", dtype=str, keep_default_na=False,
                         na_values=[], quoting=3, encoding="utf-8",
                         encoding_errors="replace", chunksize=400_000)
    for chunk in reader:
        chunk = chunk.fillna("")
        for s1, matched in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            gt[s1] = {x for x in str(matched).split(",") if x}
    return gt


# --- Candidate-pair IO: one row per S1 entity, comma-joined candidate ids -----
def write_candidate_file(path: str, cands: Dict[str, List[str]],
                         header_col: str = "candidate_entity_ids") -> None:
    """Write {s1_id: [ids]} to a submission-style TSV (empty list -> empty cell)."""
    rows = [(s1, ",".join(ids)) for s1, ids in cands.items()]
    df = pd.DataFrame(rows, columns=["source1_entity_id", header_col])
    df.to_csv(path, sep="\t", index=False, encoding="utf-8", lineterminator="\n")


def read_candidate_file(path: str) -> Dict[str, List[str]]:
    """Read a submission-style TSV back into {s1_id: [ids]}."""
    out: Dict[str, List[str]] = {}
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
                     quoting=3, encoding="utf-8", encoding_errors="replace")
    val_col = df.columns[1]
    for s1, v in zip(df["source1_entity_id"], df[val_col]):
        out[s1] = [x for x in str(v).split(",") if x]
    return out

# PLACEHOLDER_SCORE


def fbeta_entity(pred: Set[str], truth: Set[str], beta: float = 0.5) -> float:
    """F_beta for one S1 entity. Empty-truth: 1.0 if pred empty else 0.0."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    b2 = beta * beta
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom else 0.0


def macro_fbeta(preds: Dict[str, Set[str]], truths: Dict[str, Set[str]],
                beta: float = 0.5) -> Tuple[float, float, float]:
    """Macro F_beta over all S1 entities in `truths`. Returns (F, meanP, meanR).

    Any S1 in `truths` missing from `preds` is treated as predicting empty.
    """
    fs, ps, rs = [], [], []
    for s1, truth in truths.items():
        pred = preds.get(s1, set())
        fs.append(fbeta_entity(pred, truth, beta))
        if truth:
            tp = len(pred & truth)
            ps.append(tp / len(pred) if pred else 0.0)
            rs.append(tp / len(truth))
    return (float(np.mean(fs)) if fs else 0.0,
            float(np.mean(ps)) if ps else 0.0,
            float(np.mean(rs)) if rs else 0.0)


def assign_folds(s1_ids: Iterable[str], k: int = 5, seed: int = 42) -> Dict[str, int]:
    """Deterministic fold id per S1 entity (leakage-safe grouping unit).

    Uses a STABLE hash (md5) so the SAME s1_id always lands in the same fold
    across separate process runs — Python's builtin hash() is salted per process
    (PYTHONHASHSEED) and would give different folds each run, silently breaking
    the two-level OOF stacking (base-model OOF and the fusion meta-learner MUST
    share one identical fold assignment).
    """
    import hashlib
    folds: Dict[str, int] = {}
    for s1 in s1_ids:
        h = hashlib.md5(f"{seed}:{s1}".encode("utf-8")).hexdigest()
        folds[s1] = int(h, 16) % k
    return folds


# --- Tokenization helpers -----------------------------------------------------
def tokens(s: str) -> List[str]:
    return s.split() if s else []


def numeric_tokens(s: str) -> Set[str]:
    """Digit-bearing tokens (street numbers, PIN codes) — strong address signal."""
    return {t for t in s.split() if any(c.isdigit() for c in t)}


def char_ngrams(s: str, n: int = 3) -> Set[str]:
    s = s.replace(" ", "")
    return {s[i:i + n] for i in range(len(s) - n + 1)} if len(s) >= n else ({s} if s else set())


