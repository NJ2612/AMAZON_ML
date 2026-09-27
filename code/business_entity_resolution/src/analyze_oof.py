#!/usr/bin/env python3
"""Leakage-safe OOF error analysis — decompose the macro-F0.5 gap.

Given the base-model OOF parquet (+ the devset text fields), reproduce the
leakage-safe per-fold threshold, then at that operating point break the loss into:
  * precision loss  (false-positive predicted pairs),
  * recall loss IN candidates (true pair present but scored below threshold),
  * recall CEILING loss (true pair never blocked — unrecoverable here),
and characterize FPs / in-candidate FNs (name genericness, address agreement) so
the entity-decision policy is built on evidence.

Run:  python analyze_oof.py <oof.parquet> [--devset devset.parquet] [--prob blend]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import common as c
from cv_eval import build_groups, macro_f05_at, THRESH_GRID


def tuned_threshold(grp, folds_of, prob, f):
    """Best global threshold for fold f, tuned on the OTHER folds (leakage-safe)."""
    tune_s1 = [s for ff in range(5) if ff != f for s in folds_of[ff]]
    best_t, best = 0.5, -1.0
    for t in THRESH_GRID:
        v = macro_f05_at(grp, tune_s1, prob, t)
        if v > best:
            best, best_t = v, t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--devset", default=None)
    ap.add_argument("--prob", default="blend", choices=["blend", "lgb", "xgb"])
    args = ap.parse_args()

    oof = pd.read_parquet(args.oof).reset_index(drop=True)
    gt = c.load_ground_truth()
    prob = {"blend": 0.5 * (oof["p_lgb"] + oof["p_xgb"]),
            "lgb": oof["p_lgb"], "xgb": oof["p_xgb"]}[args.prob].to_numpy(np.float32)

    df = oof[["s1_id", "cand_id", "label", "fold"]].copy()
    grp = build_groups(df, gt)
    folds_of = {f: [s1 for s1, g in grp.items() if g["fold"] == f] for f in range(5)}

    # per-fold tuned threshold, then held-out predictions at that t
    thr = {f: tuned_threshold(grp, folds_of, prob, f) for f in range(5)}
    print(f"[analyze] prob={args.prob}  per-fold thresholds: "
          f"{{{', '.join(f'{f}:{thr[f]:.3f}' for f in range(5))}}}", flush=True)

    # entity-level confusion + loss decomposition
    n_pred = n_tp = n_fp = 0
    fn_in_cand = fn_ceiling = 0
    f05s = []
    # aggregate ceiling (perfect precision on the candidate set)
    ceil_f05 = []
    for s1, g in grp.items():
        truth = g["truth"]
        rows = g["rows"]
        p = prob[rows]
        t = thr[g["fold"]]
        cand_ids = g["cands"]
        pred_mask = p >= t
        pred = set(cand_ids[pred_mask])
        f05s.append(c.fbeta_entity(pred, truth, 0.5))
        # ceiling for this entity
        cand_set = set(cand_ids)
        if not truth:
            ceil_f05.append(1.0)
        else:
            r = len(cand_set & truth) / len(truth)
            ceil_f05.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
        # confusion
        n_pred += len(pred)
        tp = pred & truth
        n_tp += len(tp)
        n_fp += len(pred - truth)
        # false negatives: true pairs not predicted
        for tid in truth:
            if tid in pred:
                continue
            if tid in cand_set:
                fn_in_cand += 1     # was a candidate, scored below threshold
            else:
                fn_ceiling += 1     # never blocked
    macro = float(np.mean(f05s))
    ceil = float(np.mean(ceil_f05))
    tot_truth = sum(len(g["truth"]) for g in grp.values())
    print(f"[analyze] entities={len(grp):,}  true pairs={tot_truth:,}", flush=True)
    print(f"  macro-F0.5      : {macro:.4f}", flush=True)
    print(f"  macro-F0.5 CEIL : {ceil:.4f}  (gap to ceiling = {ceil-macro:.4f})", flush=True)
    prec = n_tp / max(n_pred, 1)
    print(f"  predicted pairs : {n_pred:,}  (TP={n_tp:,}  FP={n_fp:,}  micro-P={prec:.4f})", flush=True)
    print(f"  false negatives : in-candidate={fn_in_cand:,}  ceiling(unblocked)={fn_ceiling:,}", flush=True)
    print(f"  -> of {tot_truth:,} true pairs: recovered {n_tp:,} ({n_tp/tot_truth:.1%}), "
          f"missed-in-cand {fn_in_cand:,} ({fn_in_cand/tot_truth:.1%}), "
          f"missed-ceiling {fn_ceiling:,} ({fn_ceiling/tot_truth:.1%})", flush=True)

    # characterize errors using devset text (optional)
    if args.devset and os.path.exists(args.devset):
        dev = pd.read_parquet(args.devset,
                              columns=["s1_id", "cand_id", "s1_core", "c_core",
                                       "s1_addr", "c_addr"])
        key = dev["s1_id"].astype(str) + "|" + dev["cand_id"].astype(str)
        txt = dict(zip(key, zip(dev["s1_core"], dev["c_core"], dev["s1_addr"], dev["c_addr"])))
        from rapidfuzz import fuzz
        # sample FPs and in-cand FNs
        fp_addr = []; fp_name = []; fn_prob = []; fn_name = []; fn_addr = []
        rng = np.random.default_rng(0)
        for s1, g in grp.items():
            truth = g["truth"]; rows = g["rows"]; p = prob[rows]; t = thr[g["fold"]]
            cand_ids = g["cands"]
            for k in range(len(cand_ids)):
                cid = cand_ids[k]; pr = p[k]; is_true = cid in truth
                kk = f"{s1}|{cid}"
                tup = txt.get(kk)
                if tup is None:
                    continue
                sc, cc, sa, ca = tup
                if pr >= t and not is_true:                 # false positive
                    fp_name.append(fuzz.token_set_ratio(sc, cc))
                    fp_addr.append(fuzz.token_set_ratio(sa, ca) if sa and ca else 0)
                elif pr < t and is_true:                    # in-candidate FN
                    fn_prob.append(float(pr))
                    fn_name.append(fuzz.token_set_ratio(sc, cc))
                    fn_addr.append(fuzz.token_set_ratio(sa, ca) if sa and ca else 0)
        def desc(a, lbl):
            a = np.array(a) if len(a) else np.array([0.0])
            print(f"    {lbl}: n~{len(a)} mean {a.mean():.0f} p25 {np.percentile(a,25):.0f} "
                  f"p50 {np.percentile(a,50):.0f} p75 {np.percentile(a,75):.0f}", flush=True)
        print("  [FP characteristics] (predicted but wrong):", flush=True)
        desc(fp_name, "name tset"); desc(fp_addr, "addr tset")
        print("  [in-cand FN characteristics] (true, scored too low):", flush=True)
        desc(fn_name, "name tset"); desc(fn_addr, "addr tset"); desc(fn_prob, "prob")


if __name__ == "__main__":
    main()
