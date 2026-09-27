#!/usr/bin/env python3
"""Freeze the deploy artifacts for the LTR (LambdaMART + 1-to-1) architecture.

ltr.py measures leakage-safe OOF per fold; this fits ONE LGBMRanker on ALL dev
folds (max signal for inference) and freezes the full decision recipe from the
leakage-safe OOF:
  * a logistic calibration  ltr_score -> P(match)   (fit on the OOF scores)
  * the global operating point (t*, rho*) that maximizes macro-F0.5 over ALL dev
    S1, evaluated WITH the 1-to-1 assignment decode (the shipped decision).

Saves everything ltr_infer.py needs into models/ :
  deploy_ltr.pkl        the fitted LGBMRanker
  deploy_ltr_meta.json  {feat_cols, rel_base, calib:{w,b}, t_star, rho_star,
                         use_assign, cv_mean}

Run:  python ltr_train.py [--feats devset..._feats37.parquet] [--oof ..._ltr_oof.parquet]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import common as c
import cv_eval as cv
from features import FEATURES
import ltr


def freeze_decision(oof: pd.DataFrame, gt: dict, use_assign: bool):
    """Fit calibration on ALL OOF rows, then grid-search (t, rho) over ALL dev S1
    (this is the single global operating point the inference kernel ships)."""
    y = oof["label"].to_numpy(np.int8)
    score = oof["ltr_score"].to_numpy(np.float32)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000).fit(score.reshape(-1, 1), y)
    w = float(lr.coef_[0][0]); b = float(lr.intercept_[0])
    prob = lr.predict_proba(score.reshape(-1, 1))[:, 1].astype(np.float32)

    grp = cv.build_groups(oof, gt)
    pre = ltr.presort(grp, score)
    all_s1 = list(grp)
    best = (-1.0, 0.5, 0.0)
    for t in ltr.T_GRID:
        for rho in ltr.RHO_GRID:
            preds = ltr.decide_sets(pre, prob, all_s1, t, rho)
            sets = ltr.assign_1to1(preds) if use_assign else ltr.to_sets(preds)
            v = ltr.score_preds(grp, all_s1, sets)
            if v > best[0]:
                best = (v, float(t), float(rho))
    f05, t_star, rho_star = best
    print(f"  frozen operating point: t*={t_star:.3f} rho*={rho_star:.2f} "
          f"assign={use_assign}  (dev macro-F0.5={f05:.4f})", flush=True)
    return dict(w=w, b=b), t_star, rho_star, f05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats",
                    default=os.path.join(c.WORK, "devset_india_n20000_feats37.parquet"))
    ap.add_argument("--oof",
                    default=os.path.join(c.WORK, "devset_india_n20000_ltr_oof.parquet"))
    ap.add_argument("--no-assign", action="store_true",
                    help="freeze WITHOUT the 1-to-1 decode (A/B comparison)")
    args = ap.parse_args()
    use_assign = not args.no_assign

    if not os.path.exists(args.feats):
        args.feats = os.path.join(c.WORK, "devset_india_n20000_feats.parquet")
        print(f"[ltr-train] feats37 absent; using {os.path.basename(args.feats)}", flush=True)

    t0 = time.time()
    df = pd.read_parquet(args.feats).reset_index(drop=True)
    df, rel_cols = ltr.rel_features(df)
    base_cols = [f for f in FEATURES if f in df.columns]
    feat_cols = base_cols + rel_cols
    print(f"[ltr-train] {df['s1_id'].nunique():,} S1, {len(df):,} pairs, "
          f"{len(feat_cols)} feats", flush=True)

    # final ranker on ALL dev rows (grouped by S1)
    model = ltr.train_ranker(df, feat_cols, seed=777)
    print(f"[ltr-train] final ranker fit in {time.time()-t0:.0f}s", flush=True)

    gt = c.load_ground_truth()
    if os.path.exists(args.oof):
        oof = pd.read_parquet(args.oof).reset_index(drop=True)
    else:
        print("[ltr-train] no OOF parquet -> using in-sample scores for calibration "
              "(run ltr.py first for the leakage-safe operating point)", flush=True)
        oof = df[["s1_id", "cand_id", "label", "fold"]].copy()
        oof["ltr_score"] = model.predict(df[feat_cols].to_numpy(np.float32))
    calib, t_star, rho_star, cv_mean = freeze_decision(oof, gt, use_assign)

    with open(os.path.join(c.MODELS, "deploy_ltr.pkl"), "wb") as f:
        pickle.dump(model, f)
    meta = dict(feat_cols=feat_cols, base_cols=base_cols, rel_cols=rel_cols,
                rel_base=ltr.REL_BASE, calib=calib, t_star=t_star,
                rho_star=rho_star, use_assign=use_assign,
                dev_macro_f05=cv_mean, trained_on=os.path.basename(args.feats))
    with open(os.path.join(c.MODELS, "deploy_ltr_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[ltr-train] saved deploy_ltr.pkl + meta (t*={t_star:.3f} "
          f"rho*={rho_star:.2f}) -> {c.MODELS}", flush=True)


if __name__ == "__main__":
    main()
