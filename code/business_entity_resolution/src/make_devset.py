#!/usr/bin/env python3
"""Build a self-contained labeled dev set for the matching model (India shard).

Reuses the cached India blocking index (idx_india_cap3000.pkl) so we DON'T
re-stream the 24M cleaned rows. For the first N India S1 entities:
  - generate candidates (cheap-only ranking, locked FAM_W, topk),
  - pull each candidate's name_core/addr_norm straight from the index arrays,
  - label 1 if the candidate id is in ground truth for that S1 else 0,
  - assign a leakage-safe GroupKFold id (grouped by S1 entity),
and write one parquet with everything inline. Also emits a global name-token
IDF (document frequency over ALL S2+S3 India name_core, incl. generic tokens
the blocking index drops) for the generic-name feature. Singleton S1 (empty
truth) ARE included — the model must learn to emit empty for them.

Run:  python make_devset.py [N]   (default N=20000)
"""
import os
import sys
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as c
import blocking as bl

BUILD_CAP = 3000
CACHE = os.path.join(c.WORK, f"idx_india_cap{BUILD_CAP}.pkl")
IDF_CACHE = os.path.join(c.WORK, "tokdf_india.pkl")
TOPK = 200                      # lennorm ceiling ~0.943 @ 20k (was 150 raw-union ~0.934)
COUNTRY = "India"


def load_index():
    t0 = time.time()
    with open(CACHE, "rb") as f:
        ci = pickle.load(f)
    print(f"loaded cached {COUNTRY} index ({ci.n_records():,} recs) in {time.time()-t0:.0f}s",
          flush=True)
    return ci


def token_idf(ci) -> dict:
    """Global name-token IDF over the full index cores (incl. generic tokens)."""
    if os.path.exists(IDF_CACHE):
        with open(IDF_CACHE, "rb") as f:
            return pickle.load(f)
    t0 = time.time()
    df = Counter()
    for core in ci.cores:
        for t in set(core.split()):
            df[t] += 1
    n = ci.n_records()
    idf = {t: float(np.log(n / d)) for t, d in df.items()}
    with open(IDF_CACHE, "wb") as f:
        pickle.dump(idf, f, protocol=4)
    print(f"built token idf ({len(idf):,} tokens) in {time.time()-t0:.0f}s", flush=True)
    return idf


def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    ci = load_index()
    ci.ensure_norm()                        # cosine-approx cheap ranker (recall win)
    _ = token_idf(ci)                       # side-effect: cache IDF for features.py

    gt = c.load_ground_truth()
    t0 = time.time()
    rows = []                               # (s1_id, cand_id, label, s1_core, s1_addr, c_core, c_addr)
    s1_seen = []                            # ordered unique s1 ids for fold assignment
    n = 0
    for chunk in c.iter_clean(c.TRAIN_FILES["S1"],
                              usecols=["entity_id", "country", "name_core", "addr_norm"]):
        sub = chunk[chunk["country"].values == COUNTRY]
        for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
            pos = bl.candidates_for(ci, core, addr, TOPK, eff_cap=BUILD_CAP,
                                    detail_cap=0, return_pos=True)
            truth = gt.get(rid, set())
            s1_seen.append(rid)
            for j in pos:
                cid = ci.ids[j]
                rows.append((rid, cid, 1 if cid in truth else 0,
                             core, addr, ci.cores[j], ci.addrs[j]))
            n += 1
            if n >= n_target:
                break
        if n >= n_target:
            break

    folds = c.assign_folds(s1_seen, k=5, seed=42)
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "label",
                                     "s1_core", "s1_addr", "c_core", "c_addr"])
    df["fold"] = df["s1_id"].map(folds).astype("int8")
    df["label"] = df["label"].astype("int8")
    out = os.path.join(c.WORK, f"devset_india_n{n_target}.parquet")
    df.to_parquet(out, index=False)

    npos = int(df["label"].sum())
    nsing = sum(1 for s in s1_seen if not gt.get(s))
    print(f"devset: {len(s1_seen):,} S1 ({nsing:,} singletons), {len(df):,} pairs, "
          f"{npos:,} positives ({npos/len(df):.2%}), "
          f"avg {len(df)/len(s1_seen):.1f} cand/S1, built in {time.time()-t0:.0f}s", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
