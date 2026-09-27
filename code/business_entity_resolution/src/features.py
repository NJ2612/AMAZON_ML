#!/usr/bin/env python3
"""Stage 2 — Pairwise features for (S1, candidate) pairs.

Consumes a devset/candidate parquet with the four text fields inline
(s1_core, s1_addr, c_core, c_addr) plus s1_id/cand_id/label/fold, and emits a
feature matrix parquet. All features are pairwise TEXT similarities + a
label-free generic-name IDF signal — no feature uses match structure or record
identity, so fitting the IDF on the full corpus is leakage-safe (transductive).

Feature families:
  name  : rapidfuzz ratios, token/char-ngram Jaccard, jaro-winkler, exact,
          concatenation (space-stripped containment), length/token-count ratios
  ident : shared-token count + rarest-shared-token IDF, s1 name generic-ness
  addr  : rapidfuzz ratios, numeric-token (street#/PIN) overlap, locality
          Jaccard, presence flags, char-ngram Jaccard
  cross : name x address interaction (generic names need address agreement)

Run:  python features.py <devset.parquet> [--out path] [--chunk 500000]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import jellyfish

import common as c

IDF_CACHE = os.path.join(c.WORK, "tokdf_india.pkl")
NUMIDF_CACHE = os.path.join(c.WORK, "numidf_india.pkl")
FEATURES = [
    "n_ratio", "n_tsort", "n_tset", "n_partial", "n_wratio", "n_jw",
    "n_tok_jac", "n_g3_jac", "n_g4_jac", "n_exact", "n_concat",
    "n_len_ratio", "n_ntok_ratio",
    "id_shared_tok", "id_share_maxidf", "id_sum_idf", "s1_min_idf", "s1_max_idf",
    "a_tset", "a_tsort", "a_g4_jac", "a_num_ov", "a_num_jac", "a_loc_jac",
    "a_longnum_ov", "a_num_maxidf", "a_loc_ov",
    "a_both", "a_s1_empty", "a_c_empty",
    "a_num_conflict", "a_longnum_conflict", "n_uniq_maxidf", "n_uniq_sumidf",
    "x_name_addr", "x_name_num", "is_s3",
]


def _idf():
    with open(IDF_CACHE, "rb") as f:
        return pickle.load(f)


def _numidf():
    """Numeric-address-token IDF (rare building/PIN match). {} if not built yet."""
    if os.path.exists(NUMIDF_CACHE):
        with open(NUMIDF_CACHE, "rb") as f:
            return pickle.load(f)
    return {}


def _jac(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _region(addr: str) -> set:
    return {t for t in addr.split() if len(t) >= 4 and not any(ch.isdigit() for ch in t)}


def compute_chunk(df: pd.DataFrame, idf: dict, numidf: dict | None = None) -> pd.DataFrame:
    """Compute the FEATURES matrix for one chunk; returns a float32 DataFrame."""
    numidf = numidf or {}
    n = len(df)
    out = {f: np.zeros(n, dtype=np.float32) for f in FEATURES}
    s1c = df["s1_core"].to_numpy(); s1a = df["s1_addr"].to_numpy()
    cc = df["c_core"].to_numpy();  ca = df["c_addr"].to_numpy()
    cid = (df["cand_id"].to_numpy() if "cand_id" in df.columns
           else np.array([""] * n, dtype=object))
    med_idf = np.log(4_133_346 / 50.0)          # fallback idf for unseen name tokens
    med_numidf = np.log(4_133_346 / 20.0)       # fallback idf for unseen numeric tokens
    for i in range(n):
        a, b = s1c[i], cc[i]
        # --- name ---
        out["n_ratio"][i] = fuzz.ratio(a, b)
        out["n_tsort"][i] = fuzz.token_sort_ratio(a, b)
        out["n_tset"][i] = fuzz.token_set_ratio(a, b)
        out["n_partial"][i] = fuzz.partial_ratio(a, b)
        out["n_wratio"][i] = fuzz.WRatio(a, b)
        out["n_jw"][i] = jellyfish.jaro_winkler_similarity(a, b) * 100.0
        ta, tb = set(a.split()), set(b.split())
        out["n_tok_jac"][i] = _jac(ta, tb)
        out["n_g3_jac"][i] = _jac(c.char_ngrams(a, 3), c.char_ngrams(b, 3))
        out["n_g4_jac"][i] = _jac(c.char_ngrams(a, 4), c.char_ngrams(b, 4))
        out["n_exact"][i] = 1.0 if a == b and a else 0.0
        sa, sb = a.replace(" ", ""), b.replace(" ", "")
        out["n_concat"][i] = 1.0 if sa and sb and (sa in sb or sb in sa) else 0.0
        la, lb = len(a), len(b)
        out["n_len_ratio"][i] = min(la, lb) / max(la, lb) if max(la, lb) else 0.0
        na, nb = len(ta), len(tb)
        out["n_ntok_ratio"][i] = min(na, nb) / max(na, nb) if max(na, nb) else 0.0
        # --- identity / genericness ---
        shared = ta & tb
        out["id_shared_tok"][i] = len(shared)
        shared_idfs = [idf.get(t, med_idf) for t in shared]
        out["id_share_maxidf"][i] = max(shared_idfs, default=0.0)
        out["id_sum_idf"][i] = float(sum(shared_idfs))     # total distinctive evidence
        s1idfs = [idf.get(t, med_idf) for t in ta]
        out["s1_min_idf"][i] = min(s1idfs) if s1idfs else 0.0
        out["s1_max_idf"][i] = max(s1idfs) if s1idfs else 0.0
        # --- address ---
        aa, ab = s1a[i], ca[i]
        out["a_tset"][i] = fuzz.token_set_ratio(aa, ab)
        out["a_tsort"][i] = fuzz.token_sort_ratio(aa, ab)
        out["a_g4_jac"][i] = _jac(c.char_ngrams(aa, 4), c.char_ngrams(ab, 4))
        numa, numb = c.numeric_tokens(aa), c.numeric_tokens(ab)
        shared_num = numa & numb
        out["a_num_ov"][i] = len(shared_num)
        out["a_num_jac"][i] = _jac(numa, numb)
        # rare-numeric match: a shared building/flat/PIN number is highly diagnostic
        out["a_longnum_ov"][i] = sum(1 for t in shared_num if len(t) >= 4)
        out["a_num_maxidf"][i] = max((numidf.get(t, med_numidf) for t in shared_num),
                                     default=0.0)
        rega, regb = _region(aa), _region(ab)
        out["a_loc_jac"][i] = _jac(rega, regb)
        out["a_loc_ov"][i] = len(rega & regb)
        out["a_both"][i] = 1.0 if aa and ab else 0.0
        out["a_s1_empty"][i] = 1.0 if not aa else 0.0
        out["a_c_empty"][i] = 1.0 if not ab else 0.0
        # --- conflict / veto signals (attack saturated-agreement false merges) ---
        # These are DISAGREEMENT features: the 33 baseline feats are almost all
        # agreement signals, but the residual OOF false-positives are hard near-
        # duplicates already saturated on agreement. A conflicting street/PIN number
        # or a distinctive name token present on only one side is strong evidence of
        # two DIFFERENT entities -> a veto axis the GBDT lacked.
        out["a_num_conflict"][i] = 1.0 if (numa and numb and not shared_num) else 0.0
        longa = {t for t in numa if len(t) >= 4}
        longb = {t for t in numb if len(t) >= 4}
        out["a_longnum_conflict"][i] = 1.0 if (longa and longb and not (longa & longb)) else 0.0
        uniq = ta ^ tb                                     # tokens on exactly one side
        uniq_idfs = [idf.get(t, med_idf) for t in uniq]
        out["n_uniq_maxidf"][i] = max(uniq_idfs, default=0.0)
        out["n_uniq_sumidf"][i] = float(sum(uniq_idfs))
        # --- cross / source ---
        out["x_name_addr"][i] = (out["n_tset"][i] / 100.0) * (out["a_tset"][i] / 100.0)
        out["x_name_num"][i] = (out["n_tset"][i] / 100.0) * (min(out["a_num_ov"][i], 2) / 2.0)
        out["is_s3"][i] = 1.0 if str(cid[i]).startswith("S3") else 0.0
    res = pd.DataFrame(out)
    for keep in ("s1_id", "cand_id", "label", "fold"):
        if keep in df.columns:
            res[keep] = df[keep].to_numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("devset")
    ap.add_argument("--out", default=None)
    ap.add_argument("--chunk", type=int, default=500_000)
    args = ap.parse_args()

    idf = _idf()
    numidf = _numidf()
    if not numidf:
        print("  [warn] numidf_india.pkl missing — a_num_maxidf uses fallback idf; "
              "run build_numidf.py for the rare-numeric signal", flush=True)
    df = pd.read_parquet(args.devset)
    out = args.out or args.devset.replace(".parquet", "_feats.parquet")
    t0 = time.time()
    parts = []
    for start in range(0, len(df), args.chunk):
        sub = df.iloc[start:start + args.chunk]
        parts.append(compute_chunk(sub, idf, numidf))
        done = min(start + args.chunk, len(df))
        el = time.time() - t0
        print(f"  {done:,}/{len(df):,} pairs  ({done/el:,.0f}/s)", flush=True)
    feats = pd.concat(parts, ignore_index=True)
    feats.to_parquet(out, index=False)
    print(f"features: {len(feats):,} rows x {len(FEATURES)} feats in {time.time()-t0:.0f}s "
          f"({len(feats)/(time.time()-t0):,.0f} pairs/s) -> {out}", flush=True)


if __name__ == "__main__":
    main()
