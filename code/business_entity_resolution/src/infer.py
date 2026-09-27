#!/usr/bin/env python3
"""Stage 4 — full-test inference → the two submission TSVs.

Scores EVERY test S1 (all countries) with the frozen deploy models and the single
global operating threshold t* baked by train.py, then writes:
  output/matching_results.tsv   header: source1_entity_id \t matched_entity_ids
  output/candidate_pairs.tsv    header: source1_entity_id \t candidate_entity_ids
Every test S1 gets exactly one row; an S1 with no accepted match writes an empty
cell (a correctly-empty singleton earns full F0.5 credit).

Memory plan (16 GB box, ~3 GB free):
  - ONE country at a time: build its S2+S3 inverted index, free it before the next.
  - stream that country's S1 in batches; per batch, generate candidates (cheap
    numpy union ranking), compute the 33 pairwise features in a process pool
    (idf/numidf loaded once per worker), blend-predict LGB+XGB, threshold at t*,
    and stream-write both TSV rows. Nothing country-wide is held in RAM.

Blocking MATCHES make_devset (topk=200, eff_cap=BUILD_CAP=3000, detail_cap=0) so
inference sees the same candidate distribution the models were trained on.

NOTE (honest): the deploy models + idf/numidf were fit on the India dev shard.
US and France are scored out-of-distribution with the same model and India idf
(unseen tokens fall back to the median idf). This is the "best current config"
applied uniformly; per-country models are a later round.

Run:  python infer.py [--topk 200] [--batch-s1 4000] [--procs 10] [--limit N]
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

BUILD_CAP = 3000                      # df_cap used to build the dev index (must match)
IDF_PATH = F.IDF_CACHE                 # tokdf_india.pkl
NUMIDF_PATH = F.NUMIDF_CACHE           # numidf_india.pkl

# --- worker (feature computation only; no models loaded here) -----------------
_IDF = None
_NUMIDF = None


def _winit(idf_path: str, numidf_path: str) -> None:
    """Pool initializer: load the idf dicts once per worker process."""
    global _IDF, _NUMIDF
    with open(idf_path, "rb") as f:
        _IDF = pickle.load(f)
    try:
        with open(numidf_path, "rb") as f:
            _NUMIDF = pickle.load(f)
    except Exception:
        _NUMIDF = {}


def _wfeat(chunk: pd.DataFrame) -> np.ndarray:
    """Compute the FEATURES matrix for one pair chunk; return float32 (n, 33)."""
    res = F.compute_chunk(chunk, _IDF, _NUMIDF)
    return res[F.FEATURES].to_numpy(np.float32)


# --- helpers ------------------------------------------------------------------
def _predict_blend(lm, xm, X: np.ndarray, prob: str) -> np.ndarray:
    pl = lm.predict_proba(X)[:, 1].astype(np.float32)
    px = xm.predict_proba(X)[:, 1].astype(np.float32)
    if prob == "lgb":
        return pl
    if prob == "xgb":
        return px
    return 0.5 * (pl + px)


def _iter_country_s1(country: str, limit: int | None):
    """Yield (rid, core, addr) for test S1 rows of one country (streamed)."""
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
    ap.add_argument("--batch-s1", type=int, default=4000,
                    help="S1 entities per feature/predict batch (memory knob)")
    ap.add_argument("--procs", type=int, default=10,
                    help="feature-compute worker processes")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap S1 PER COUNTRY (smoke test)")
    ap.add_argument("--countries", default=None,
                    help="comma list to restrict (default: all in test S1)")
    args = ap.parse_args()

    import lightgbm  # noqa: F401  (imported here so workers don't pay for it)
    import xgboost   # noqa: F401

    with open(os.path.join(c.MODELS, "deploy_lgb.pkl"), "rb") as f:
        lm = pickle.load(f)
    with open(os.path.join(c.MODELS, "deploy_xgb.pkl"), "rb") as f:
        xm = pickle.load(f)
    with open(os.path.join(c.MODELS, "deploy_meta.json")) as f:
        meta = json.load(f)
    t_star = float(meta["threshold"])
    prob = meta.get("prob", "blend")
    assert meta["features"] == F.FEATURES, "deploy feature order != features.FEATURES"
    print(f"[infer] models loaded  t*={t_star:.3f}  prob={prob}  "
          f"topk={args.topk} batch={args.batch_s1} procs={args.procs}", flush=True)

    countries = (args.countries.split(",") if args.countries
                 else bl.distinct_countries(c.TEST_FILES))
    print(f"[infer] countries={countries}", flush=True)

    os.makedirs(c.OUTPUT, exist_ok=True)
    match_path = os.path.join(c.OUTPUT, "matching_results.tsv")
    cand_path = os.path.join(c.OUTPUT, "candidate_pairs.tsv")
    t_all = time.time()
    n_s1 = n_pairs = n_pred_pairs = n_nonempty = 0

    pool = Pool(args.procs, initializer=_winit, initargs=(IDF_PATH, NUMIDF_PATH))
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
                cn_s1 = cn_pairs = cn_pred = 0
                batch = []          # (rid, core, addr)

                def flush(batch):
                    nonlocal cn_s1, cn_pairs, cn_pred, n_nonempty
                    if not batch:
                        return
                    rids, cids_per, rows = [], [], []
                    for rid, core, addr in batch:
                        pos = bl.candidates_for(ci, core, addr, args.topk,
                                                eff_cap=BUILD_CAP, detail_cap=0,
                                                return_pos=True)
                        cids = [ci.ids[j] for j in pos]
                        rids.append(rid)
                        cids_per.append(cids)
                        for j, cid in zip(pos, cids):
                            rows.append((cid, core, addr, ci.cores[j], ci.addrs[j]))
                    cn_s1 += len(rids)

                    if rows:
                        df = pd.DataFrame(rows, columns=["cand_id", "s1_core",
                                                         "s1_addr", "c_core", "c_addr"])
                        cn_pairs += len(df)
                        nsp = max(1, len(df) // args.procs)
                        chunks = [df.iloc[i:i + nsp] for i in range(0, len(df), nsp)]
                        parts = pool.map(_wfeat, chunks)
                        X = np.concatenate(parts) if len(parts) > 1 else parts[0]
                        blend = _predict_blend(lm, xm, X, prob)
                    else:
                        blend = np.empty(0, dtype=np.float32)

                    off = 0
                    for rid, cids in zip(rids, cids_per):
                        k = len(cids)
                        p = blend[off:off + k]
                        off += k
                        kept = [cid for cid, pp in zip(cids, p) if pp >= t_star]
                        if kept:
                            n_nonempty += 1
                            cn_pred += len(kept)
                        mf.write(f"{rid}\t{','.join(kept)}\n")
                        cf.write(f"{rid}\t{','.join(cids)}\n")

                for rid, core, addr in _iter_country_s1(country, args.limit):
                    batch.append((rid, core, addr))
                    if len(batch) >= args.batch_s1:
                        flush(batch)
                        batch = []
                        el = time.time() - q0
                        print(f"    [{country}] {cn_s1:,} S1 | {cn_pairs:,} pairs | "
                              f"{cn_pred:,} matched | {cn_s1/el:,.0f} S1/s", flush=True)
                flush(batch)

                n_s1 += cn_s1
                n_pairs += cn_pairs
                n_pred_pairs += cn_pred
                del ci
                print(f"  [{country}] DONE {cn_s1:,} S1, {cn_pairs:,} pairs, "
                      f"{cn_pred:,} predicted matches in {time.time()-q0:.0f}s", flush=True)
    finally:
        pool.close()
        pool.join()

    print(f"[infer] {n_s1:,} S1 | {n_pairs:,} scored pairs | {n_pred_pairs:,} matches "
          f"| {n_nonempty:,} non-empty S1 | {time.time()-t_all:.0f}s", flush=True)
    print(f"[infer] -> {match_path}", flush=True)
    print(f"[infer] -> {cand_path}", flush=True)


if __name__ == "__main__":
    main()
