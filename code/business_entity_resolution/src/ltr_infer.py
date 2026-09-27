#!/usr/bin/env python3
"""Full-test inference for the LTR (LambdaMART + global 1-to-1) architecture.

Per country: build the S2+S3 blocking index, stream S1 in batches, generate the
same lexical candidates as make_devset (topk=200, cap=3000), compute the pairwise
features AND the within-S1 relative features, score with the frozen LGBMRanker,
map score->P(match) with the frozen logistic calibration, apply the per-S1
variable-k cutoff (t*, rho*), then run a GLOBAL greedy 1-to-1 assignment over the
whole country so every S2/S3 target is claimed by at most one S1 (the strict
degree-1 constraint the data satisfies) — the precision lever. Writes:
  output/matching_results.tsv   source1_entity_id \t matched_entity_ids
  output/candidate_pairs.tsv    source1_entity_id \t candidate_entity_ids
Every test S1 gets exactly one row (empty cell = predicted singleton).

Run:  python ltr_infer.py [--topk 200] [--batch-s1 4000] [--procs 10] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common as c
import blocking as bl
import features as F
import ltr

BUILD_CAP = 3000

_IDF = None
_NUMIDF = None


def _winit(idf_path: str, numidf_path: str) -> None:
    global _IDF, _NUMIDF
    with open(idf_path, "rb") as f:
        _IDF = pickle.load(f)
    try:
        with open(numidf_path, "rb") as f:
            _NUMIDF = pickle.load(f)
    except Exception:
        _NUMIDF = {}


def _wfeat(chunk: pd.DataFrame) -> pd.DataFrame:
    """Base pairwise features for one chunk (relative feats are added in main,
    where whole-S1 groups are contiguous)."""
    return F.compute_chunk(chunk, _IDF, _NUMIDF)


def _iter_country_s1(country: str, limit: int | None):
    seen = 0
    for chunk in c.iter_clean(c.TEST_FILES["S1"],
                              usecols=["entity_id", "country", "name_core", "addr_norm"]):
        sub = chunk[chunk["country"].values == country]
        for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
            yield rid, core, addr
            seen += 1
            if limit and seen >= limit:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--batch-s1", type=int, default=4000)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="cap S1 PER COUNTRY (smoke)")
    ap.add_argument("--countries", default=None)
    args = ap.parse_args()

    import lightgbm  # noqa: F401

    with open(os.path.join(c.MODELS, "deploy_ltr.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(c.MODELS, "deploy_ltr_meta.json")) as f:
        meta = json.load(f)
    feat_cols = meta["feat_cols"]
    t_star = float(meta["t_star"]); rho_star = float(meta["rho_star"])
    use_assign = bool(meta.get("use_assign", True))
    cw = float(meta["calib"]["w"]); cb = float(meta["calib"]["b"])
    print(f"[ltr-infer] model loaded  t*={t_star:.3f} rho*={rho_star:.2f} "
          f"assign={use_assign}  {len(feat_cols)} feats  topk={args.topk}", flush=True)

    countries = (args.countries.split(",") if args.countries
                 else bl.distinct_countries(c.TEST_FILES))
    print(f"[ltr-infer] countries={countries}", flush=True)

    os.makedirs(c.OUTPUT, exist_ok=True)
    match_path = os.path.join(c.OUTPUT, "matching_results.tsv")
    cand_path = os.path.join(c.OUTPUT, "candidate_pairs.tsv")
    t_all = time.time()
    n_s1 = n_pairs = n_match = n_nonempty = 0

    pool = Pool(args.procs, initializer=_winit, initargs=(F.IDF_CACHE, F.NUMIDF_CACHE))
    try:
        with open(match_path, "w", encoding="utf-8", newline="\n") as mf, \
             open(cand_path, "w", encoding="utf-8", newline="\n") as cf:
            mf.write("source1_entity_id\tmatched_entity_ids\n")
            cf.write("source1_entity_id\tcandidate_entity_ids\n")

            for country in countries:
                t0 = time.time()
                ci = bl.load_country_index(c.TEST_FILES, country, df_cap=BUILD_CAP)
                print(f"  [{country}] index: {ci.n_records():,} recs in "
                      f"{time.time()-t0:.0f}s", flush=True)
                q0 = time.time()
                order_rids, kept_of = [], {}
                cn_s1 = cn_pairs = 0
                batch = []

                def flush(batch):
                    nonlocal cn_s1, cn_pairs
                    if not batch:
                        return
                    rids, cids_per, rows, row_s1 = [], [], [], []
                    for rid, core, addr in batch:
                        pos = bl.candidates_for(ci, core, addr, args.topk,
                                                eff_cap=BUILD_CAP, detail_cap=0,
                                                return_pos=True)
                        cids = [ci.ids[j] for j in pos]
                        rids.append(rid); cids_per.append(cids)
                        for j, cid in zip(pos, cids):
                            rows.append((cid, core, addr, ci.cores[j], ci.addrs[j]))
                            row_s1.append(rid)
                    cn_s1 += len(rids)
                    if rows:
                        bdf = pd.DataFrame(rows, columns=["cand_id", "s1_core",
                                                          "s1_addr", "c_core", "c_addr"])
                        cn_pairs += len(bdf)
                        nsp = max(1, len(bdf) // args.procs)
                        chunks = [bdf.iloc[i:i + nsp] for i in range(0, len(bdf), nsp)]
                        parts = pool.map(_wfeat, chunks)
                        fdf = pd.concat(parts, ignore_index=True)
                        fdf["s1_id"] = row_s1
                        fdf, _ = ltr.rel_features(fdf)      # groups by s1_id, keeps row order
                        X = fdf[feat_cols].to_numpy(np.float32)
                        score = model.predict(X)
                        prob = 1.0 / (1.0 + np.exp(-(cw * score + cb)))
                    else:
                        prob = np.empty(0, dtype=np.float32)
                    off = 0
                    for rid, cids in zip(rids, cids_per):
                        k = len(cids)
                        p = prob[off:off + k]; off += k
                        cf.write(f"{rid}\t{','.join(cids)}\n")
                        order_rids.append(rid)
                        if k == 0 or float(p.max()) < t_star:
                            kept_of[rid] = []
                            continue
                        thr = max(t_star, rho_star * float(p.max()))
                        kept_of[rid] = [(cid, float(pp)) for cid, pp in zip(cids, p)
                                        if pp >= thr]

                for rid, core, addr in _iter_country_s1(country, args.limit):
                    batch.append((rid, core, addr))
                    if len(batch) >= args.batch_s1:
                        flush(batch); batch = []
                        el = time.time() - q0
                        print(f"    [{country}] {cn_s1:,} S1 | {cn_pairs:,} pairs | "
                              f"{cn_s1/el:,.0f} S1/s", flush=True)
                flush(batch)

                # --- global 1-to-1 decode over the whole country, then write ------
                final = ltr.assign_1to1(kept_of) if use_assign \
                    else ltr.to_sets({r: kept_of[r] for r in order_rids})
                cn_match = cn_nonempty = 0
                for rid in order_rids:
                    s = final.get(rid, set())
                    if s:
                        cn_nonempty += 1; cn_match += len(s)
                    mf.write(f"{rid}\t{','.join(s)}\n")
                n_s1 += cn_s1; n_pairs += cn_pairs
                n_match += cn_match; n_nonempty += cn_nonempty
                del ci, kept_of, order_rids
                print(f"  [{country}] DONE {cn_s1:,} S1, {cn_pairs:,} pairs, "
                      f"{cn_match:,} matches, {cn_nonempty:,} non-empty in "
                      f"{time.time()-q0:.0f}s", flush=True)
    finally:
        pool.close(); pool.join()

    print(f"[ltr-infer] {n_s1:,} S1 | {n_pairs:,} pairs | {n_match:,} matches | "
          f"{n_nonempty:,} non-empty | {time.time()-t_all:.0f}s", flush=True)
    print(f"[ltr-infer] -> {match_path}\n[ltr-infer] -> {cand_path}", flush=True)


if __name__ == "__main__":
    main()
