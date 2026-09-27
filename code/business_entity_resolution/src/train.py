#!/usr/bin/env python3
"""Train the FINAL deploy models (LGB + XGB) on the full dev feature matrix.

cv_eval.py trains per-fold for leakage-safe OOF; this fits the same LGB/XGB on ALL
rows (every fold) so inference gets maximum training signal, and freezes a single
global operating threshold t* read from the leakage-safe OOF. Saves everything the
inference pass needs into models/ :
  deploy_lgb.pkl, deploy_xgb.pkl, deploy_meta.json {features, threshold, neg, prob}

Run:  python train.py <feats.parquet> --oof <oof.parquet> [--neg 3] [--prob blend]
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
from features import FEATURES
from cv_eval import lgb_params, xgb_params, _downsample, build_groups, macro_f05_at, THRESH_GRID


def deploy_threshold(oof_path: str, prob_name: str) -> float:
    """Operating t*: maximize macro-F0.5 on the (leakage-safe) OOF over ALL dev S1."""
    oof = pd.read_parquet(oof_path).reset_index(drop=True)
    gt = c.load_ground_truth()
    grp = build_groups(oof, gt)
    prob = {"blend": 0.5 * (oof["p_lgb"] + oof["p_xgb"]),
            "lgb": oof["p_lgb"], "xgb": oof["p_xgb"]}[prob_name].to_numpy(np.float32)
    all_s1 = list(grp)
    best_t, best = 0.5, -1.0
    for t in THRESH_GRID:
        v = macro_f05_at(grp, all_s1, prob, t)
        if v > best:
            best, best_t = v, float(t)
    print(f"  deploy threshold t*={best_t:.3f}  (dev macro-F0.5={best:.4f})", flush=True)
    return best_t


def main():
    import lightgbm as lgb
    import xgboost as xgb

    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--oof", default=None)
    ap.add_argument("--neg", type=float, default=3.0)
    ap.add_argument("--prob", default="blend", choices=["blend", "lgb", "xgb"])
    args = ap.parse_args()

    df = pd.read_parquet(args.feats).reset_index(drop=True)
    X = df[FEATURES].to_numpy(np.float32)
    y = df["label"].to_numpy(np.int8)
    t0 = time.time()
    idx = _downsample(np.arange(len(df)), y, args.neg, seed=7)
    print(f"[train] fit on {len(idx):,} rows "
          f"({int(y[idx].sum()):,} pos, neg_ratio={args.neg}) of {len(df):,}", flush=True)

    lm = lgb.LGBMClassifier(**lgb_params()); lm.fit(X[idx], y[idx])
    xm = xgb.XGBClassifier(**xgb_params()); xm.fit(X[idx], y[idx])
    print(f"[train] models fit in {time.time()-t0:.0f}s", flush=True)

    oof_path = args.oof or args.feats.replace("_feats.parquet", "_oof.parquet")
    thr = deploy_threshold(oof_path, args.prob) if os.path.exists(oof_path) else 0.98

    with open(os.path.join(c.MODELS, "deploy_lgb.pkl"), "wb") as f:
        pickle.dump(lm, f)
    with open(os.path.join(c.MODELS, "deploy_xgb.pkl"), "wb") as f:
        pickle.dump(xm, f)
    meta = dict(features=FEATURES, threshold=thr, neg=args.neg, prob=args.prob,
                n_train=int(len(idx)), trained_on=os.path.basename(args.feats))
    with open(os.path.join(c.MODELS, "deploy_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] saved deploy_lgb/xgb + meta (t*={thr:.3f}) -> {c.MODELS}", flush=True)


if __name__ == "__main__":
    main()
