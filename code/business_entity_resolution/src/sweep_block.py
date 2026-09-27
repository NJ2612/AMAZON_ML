#!/usr/bin/env python3
"""Sweep blocking on the India shard (cheap-only ranker) without rebuilding.

Builds/caches the India index ONCE at a high DF cap, then evaluates recall-ceiling
for several (topk, family-weight) combos against ground truth on the first N India S1.
Cheap-only ranking (detail_cap=0) is fast enough to sweep many combos.
"""
import os
import pickle
import time
import numpy as np
import common as c
import blocking as bl

SAMPLE = 5000
BUILD_CAP = 3000
EFF_CAP = 3000
CACHE = os.path.join(c.WORK, f"idx_india_cap{BUILD_CAP}.pkl")

FAM_VARIANTS = {
    "default":  bl.FAM_W,
    "ng_up":    {**bl.FAM_W, "k_ng": (0.30, 0.0)},
    "ng_up2":   {**bl.FAM_W, "k_ng": (0.45, 0.0)},
}
TOPKS = [100, 150, 200, 300]


def get_index():
    if os.path.exists(CACHE):
        t0 = time.time()
        with open(CACHE, "rb") as f:
            ci = pickle.load(f)
        print(f"loaded cached India index in {time.time()-t0:.0f}s", flush=True)
        return ci
    t0 = time.time()
    ci = bl.load_country_index(c.TRAIN_FILES, "India", df_cap=BUILD_CAP)
    print(f"built India records={ci.n_records():,} in {time.time()-t0:.0f}s", flush=True)
    with open(CACHE, "wb") as f:
        pickle.dump(ci, f, protocol=4)
    return ci


def main():
    ci = get_index()
    s1, n = [], 0
    for chunk in c.iter_clean(c.TRAIN_FILES["S1"],
                              usecols=["entity_id", "country", "name_core", "addr_norm"]):
        sub = chunk[chunk["country"].values == "India"]
        for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
            s1.append((rid, core, addr)); n += 1
            if n >= SAMPLE: break
        if n >= SAMPLE: break
    gt = c.load_ground_truth()

    print(f"{'variant':>10} {'topk':>5} {'macroR':>8} {'microR':>8} {'full%':>7} "
          f"{'f05ns':>7} {'f05all':>7} {'avgC':>6} {'q_s':>6}", flush=True)
    for vname, fam_w in FAM_VARIANTS.items():
        for topk in TOPKS:
            q0 = time.time()
            recalls, full, cov, miss, ncand, nns = [], 0, 0, 0, 0, 0
            f05 = []
            for rid, core, addr in s1:
                cand = set(bl.candidates_for(ci, core, addr, topk, eff_cap=EFF_CAP,
                                             detail_cap=0, fam_w=fam_w))
                ncand += len(cand)
                truth = gt.get(rid, set())
                if not truth:
                    continue
                nns += 1
                hit = len(cand & truth)
                r = hit / len(truth)
                recalls.append(r)
                cov += hit; miss += len(truth) - hit
                if hit == len(truth):
                    full += 1
                f05.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
            # non-singleton F0.5 ceiling, and overall (singletons=5.6% @ 1.0)
            f05ns = np.mean(f05)
            f05all = 0.944 * f05ns + 0.056 * 1.0
            print(f"{vname:>10} {topk:>5} {np.mean(recalls):>8.4f} "
                  f"{cov/(cov+miss):>8.4f} {full/nns:>7.1%} {f05ns:>7.4f} {f05all:>7.4f} "
                  f"{ncand/len(s1):>6.1f} {time.time()-q0:>6.0f}", flush=True)


if __name__ == "__main__":
    main()
