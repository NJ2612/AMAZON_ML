#!/usr/bin/env python3
"""Recall-ceiling audit over the cached India index — memory-safe, no dense ANN.

Reuses idx_india_cap3000.pkl and the dev S1 list. For each ranking variant it
recomputes candidates for the first N India S1 (matching make_devset's order) and
reports the macro-F0.5 CEILING (perfect-precision upper bound) + macro/micro recall.
This isolates whether the recall gap is a RANKING loss (fixable cheaply) or a
posting-list DF-cap loss (needs a rebuild).

Variants (selected by --variants):
  cheap0       : current dev setting — weighted union, cheap-only (detail_cap=0)
  detail300    : weighted union top-300 -> address-aware score_pair re-rank
  lennorm      : weighted union divided by sqrt(candidate name+addr token count)
  addrprimary  : union of name-cheap top-k and address-only top-k

Run:  python audit_block.py [N] [--topks 150,200,300] [--variants cheap0,detail300,...]
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from collections import Counter

import numpy as np
import pandas as pd

import common as c
import blocking as bl

BUILD_CAP = 3000
CACHE = os.path.join(c.WORK, f"idx_india_cap{BUILD_CAP}.pkl")
COUNTRY = "India"


def f05_ceiling(cand_pos_by_s1, id_at, gt, s1_list):
    """Macro-F0.5 ceiling (perfect precision) + macro/micro recall over s1_list."""
    f05, recalls = [], []
    cov = miss = 0
    for s1 in s1_list:
        truth = gt.get(s1, set())
        if not truth:
            f05.append(1.0)                       # singleton -> predict empty -> 1.0
            continue
        cset = {id_at(j) for j in cand_pos_by_s1.get(s1, ())}
        hit = len(cset & truth)
        r = hit / len(truth)
        recalls.append(r)
        cov += hit
        miss += len(truth) - hit
        f05.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
    return (float(np.mean(f05)), float(np.mean(recalls)) if recalls else 0.0,
            cov / (cov + miss) if (cov + miss) else 0.0)


def rank_cheap0(ci, core, addr, topk):
    return bl.candidates_for(ci, core, addr, topk, eff_cap=BUILD_CAP,
                             detail_cap=0, return_pos=True)


def rank_detail(ci, core, addr, topk, detail_cap=300):
    return bl.candidates_for(ci, core, addr, topk, eff_cap=BUILD_CAP,
                             detail_cap=detail_cap, return_pos=True)


def _union_pos_weights(ci, core, addr, fams):
    """Weighted-union positions+weights over the requested key families only."""
    arrs, ws = [], []
    idf = ci.idf
    if "name" in fams:
        for t in set(core.split()):
            p = ci.k_tok.get(t)
            if p is not None and len(p) <= BUILD_CAP:
                arrs.append(p); ws.append(idf.get(("k_tok", t), 0.0) * 1.0)
            m = bl._mph(t)
            if m:
                p = ci.k_mph.get(m)
                if p is not None and len(p) <= BUILD_CAP:
                    arrs.append(p); ws.append(idf.get(("k_mph", m), 0.0) * 0.7)
        for g in c.char_ngrams(core, bl.NGRAM):
            p = ci.k_ng.get(g)
            if p is not None and len(p) <= BUILD_CAP:
                arrs.append(p); ws.append(idf.get(("k_ng", g), 0.0) * 0.45)
    if "addr" in fams:
        for t in c.numeric_tokens(addr):
            p = ci.k_num.get(t)
            if p is not None and len(p) <= BUILD_CAP:
                arrs.append(p); ws.append(idf.get(("k_num", t), 0.0) * 1.0 + 1.0)
        for t in bl.region_tokens(addr):
            p = ci.k_aw.get(t)
            if p is not None and len(p) <= BUILD_CAP:
                arrs.append(p); ws.append(idf.get(("k_aw", t), 0.0) * 1.0 + 0.3)
    if not arrs:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    all_idx = np.concatenate(arrs)
    lens = np.fromiter((len(a) for a in arrs), np.int64, len(arrs))
    all_w = np.repeat(np.asarray(ws, np.float64), lens)
    uniq, inv = np.unique(all_idx, return_inverse=True)
    wsum = np.bincount(inv, weights=all_w)
    return uniq, wsum


def _topk_from(uniq, score, topk):
    if len(uniq) > topk:
        sel = np.argpartition(score, -topk)[-topk:]
        sel = sel[np.argsort(score[sel])[::-1]]
    else:
        sel = np.argsort(score)[::-1]
    return [int(uniq[s]) for s in sel]


def rank_lennorm(ci, core, addr, topk, norm):
    uniq, wsum = _union_pos_weights(ci, core, addr, ("name", "addr"))
    if len(uniq) == 0:
        return []
    return _topk_from(uniq, wsum / norm[uniq], topk)


def rank_addrprimary(ci, core, addr, topk):
    """Half budget to name-weighted union, half to address-weighted union."""
    un, wn = _union_pos_weights(ci, core, addr, ("name",))
    ua, wa = _union_pos_weights(ci, core, addr, ("addr",))
    kn = topk - topk // 2
    ka = topk // 2
    name_top = _topk_from(un, wn, kn) if len(un) else []
    addr_top = _topk_from(ua, wa, ka) if len(ua) else []
    seen, out = set(), []
    for j in name_top + addr_top:
        if j not in seen:
            seen.add(j); out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=20000)
    ap.add_argument("--topks", default="150,200,300")
    ap.add_argument("--variants", default="cheap0,detail300,lennorm,addrprimary")
    args = ap.parse_args()
    topks = [int(x) for x in args.topks.split(",")]
    variants = args.variants.split(",")

    t0 = time.time()
    with open(CACHE, "rb") as f:
        ci = pickle.load(f)
    print(f"index {ci.n_records():,} recs loaded {time.time()-t0:.0f}s", flush=True)
    id_at = lambda j: ci.ids[j]

    # per-record length norm for lennorm (name+addr token counts, sqrt)
    norm = None
    if "lennorm" in variants:
        cnt = np.fromiter(((len(str(cr).split()) + len(str(ad).split()) + 1)
                           for cr, ad in zip(ci.cores, ci.addrs)),
                          np.float64, len(ci.cores))
        norm = np.sqrt(cnt)
        print(f"length norms built {time.time()-t0:.0f}s", flush=True)

    gt = c.load_ground_truth()
    # replay make_devset order: first N India S1
    s1_rows = []
    n = 0
    for chunk in c.iter_clean(c.TRAIN_FILES["S1"],
                              usecols=["entity_id", "country", "name_core", "addr_norm"]):
        sub = chunk[chunk["country"].values == COUNTRY]
        for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
            s1_rows.append((rid, core, addr)); n += 1
            if n >= args.n:
                break
        if n >= args.n:
            break
    s1_list = [r[0] for r in s1_rows]
    print(f"dev S1 {len(s1_list):,}", flush=True)

    funcs = {
        "cheap0": lambda cr, ad, k: rank_cheap0(ci, cr, ad, k),
        "detail300": lambda cr, ad, k: rank_detail(ci, cr, ad, k, 300),
        "detail600": lambda cr, ad, k: rank_detail(ci, cr, ad, k, 600),
        "lennorm": lambda cr, ad, k: rank_lennorm(ci, cr, ad, k, norm),
        "addrprimary": lambda cr, ad, k: rank_addrprimary(ci, cr, ad, k),
    }
    for v in variants:
        fn = funcs[v]
        for topk in topks:
            tt = time.time()
            cand = {}
            for rid, core, addr in s1_rows:
                cand[rid] = fn(core, addr, topk)
            f, mr, mir = f05_ceiling(cand, id_at, gt, s1_list)
            avg = np.mean([len(v_) for v_ in cand.values()])
            print(f"  [{v:>11} topk={topk:>3}] f05_ceiling={f:.4f}  macroR={mr:.4f}  "
                  f"microR={mir:.4f}  avg_cand={avg:.1f}  ({time.time()-tt:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
