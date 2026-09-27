#!/usr/bin/env python3
"""Stage 3+5 — Base models (LightGBM || XGBoost) with leakage-safe GroupKFold CV.

Trains LightGBM and XGBoost IN PARALLEL on the same pairwise feature matrix,
grouped by S1 entity (the macro-F0.5 unit) so no entity's pairs straddle the
train/val split. Produces out-of-fold (OOF) probabilities for every candidate
pair — the honest inputs both to the metric and to the Stage-4 fusion net.

Leakage controls:
  * fold = precomputed per-S1 GroupKFold id (all pairs of an S1 share a fold).
  * negatives are downsampled on the TRAIN side only; the val fold is scored on
    its FULL candidate set.
  * the decision threshold for a fold is tuned on the OTHER folds' OOF, never on
    the fold being scored.
  * macro-F0.5 is computed against the FULL ground truth (incl. matches blocking
    never surfaced), so the number reflects the real recall ceiling — not an
    inflated in-candidate recall.

Emits oof parquet (s1_id, cand_id, label, fold, p_lgb, p_xgb) for fusion.py and
prints a per-fold + mean+/-std macro-F0.5/P/R table for LGB / XGB / blend.

Run:  python cv_eval.py <feats.parquet> [--neg 3] [--out oof.parquet]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import common as c
from features import FEATURES

THRESH_GRID = np.concatenate([np.linspace(0.02, 0.98, 49), [0.985, 0.99, 0.995]])


def lgb_params():
    return dict(n_estimators=400, learning_rate=0.05, num_leaves=63,
                min_child_samples=50, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1, verbosity=-1)


def xgb_params():
    return dict(n_estimators=400, learning_rate=0.05, max_depth=7,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=1.0, tree_method="hist", n_jobs=-1, verbosity=0,
                eval_metric="logloss")


def _downsample(idx: np.ndarray, y: np.ndarray, neg_ratio: float,
                seed: int) -> np.ndarray:
    """Keep all positives + neg_ratio x that many negatives (train side only)."""
    rng = np.random.default_rng(seed)
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k = min(len(neg), int(len(pos) * neg_ratio))
    neg = rng.choice(neg, size=k, replace=False)
    out = np.concatenate([pos, neg])
    rng.shuffle(out)
    return out


def fit_predict(Xtr, ytr, Xval):
    """Train LGB and XGB on the same rows; return (p_lgb_val, p_xgb_val)."""
    import lightgbm as lgb
    import xgboost as xgb
    lm = lgb.LGBMClassifier(**lgb_params())
    lm.fit(Xtr, ytr)
    p_lgb = lm.predict_proba(Xval)[:, 1]
    xm = xgb.XGBClassifier(**xgb_params())
    xm.fit(Xtr, ytr)
    p_xgb = xm.predict_proba(Xval)[:, 1]
    return p_lgb, p_xgb


def build_groups(df: pd.DataFrame, gt: dict):
    """Per-S1: candidate ids, their row indices, full truth set, fold."""
    grp = {}
    for s1, sub in df.groupby("s1_id", sort=False):
        grp[s1] = dict(rows=sub.index.to_numpy(),
                       cands=sub["cand_id"].to_numpy(),
                       truth=gt.get(s1, set()),
                       fold=int(sub["fold"].iloc[0]))
    return grp


def macro_f05_at(grp, s1_list, prob, t):
    """Mean F0.5 over s1_list at threshold t (full-truth; empty pred allowed)."""
    fs = []
    for s1 in s1_list:
        g = grp[s1]
        pred = set(g["cands"][prob[g["rows"]] >= t])
        fs.append(c.fbeta_entity(pred, g["truth"], 0.5))
    return float(np.mean(fs)) if fs else 0.0


def eval_model(grp, folds_of, prob, name):
    """Per-fold macro-F0.5 with threshold tuned on the OTHER folds' OOF."""
    per_fold = []
    for f in sorted(folds_of):
        val_s1 = folds_of[f]
        tune_s1 = [s for ff in folds_of if ff != f for s in folds_of[ff]]
        best_t, best = 0.5, -1.0
        for t in THRESH_GRID:
            v = macro_f05_at(grp, tune_s1, prob, t)
            if v > best:
                best, best_t = v, t
        per_fold.append((macro_f05_at(grp, val_s1, prob, best_t), best_t))
    fs = np.array([p[0] for p in per_fold])
    print(f"  {name:>7}: per-fold F0.5 = [{', '.join(f'{x:.4f}' for x in fs)}]  "
          f"mean={fs.mean():.4f} +/- {fs.std():.4f}  (t~{np.mean([p[1] for p in per_fold]):.2f})",
          flush=True)
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("--neg", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.feats).reset_index(drop=True)
    gt = c.load_ground_truth()
    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    fold = df["fold"].to_numpy(dtype=np.int8)

    p_lgb = np.zeros(len(df), dtype=np.float32)
    p_xgb = np.zeros(len(df), dtype=np.float32)
    t0 = time.time()
    for f in range(5):
        tr_mask = fold != f
        va_idx = np.where(~tr_mask)[0]
        tr_idx = _downsample(np.where(tr_mask)[0], y, args.neg, seed=100 + f)
        pl, px = fit_predict(X[tr_idx], y[tr_idx], X[va_idx])
        p_lgb[va_idx] = pl
        p_xgb[va_idx] = px
        print(f"  fold {f}: train={len(tr_idx):,} val={len(va_idx):,} "
              f"({time.time()-t0:.0f}s)", flush=True)

    df["p_lgb"] = p_lgb
    df["p_xgb"] = p_xgb
    p_blend = 0.5 * (p_lgb + p_xgb)
    out = args.out or args.feats.replace("_feats.parquet", "_oof.parquet")
    df[["s1_id", "cand_id", "label", "fold", "p_lgb", "p_xgb"]].to_parquet(out, index=False)

    grp = build_groups(df, gt)
    folds_of = {f: [s1 for s1, g in grp.items() if g["fold"] == f] for f in range(5)}
    print(f"\n[cv] {len(grp):,} S1 entities, macro-F0.5 (full-truth, leakage-safe threshold):",
          flush=True)
    eval_model(grp, folds_of, p_lgb, "LGB")
    eval_model(grp, folds_of, p_xgb, "XGB")
    eval_model(grp, folds_of, p_blend, "blend")
    print(f"[cv] oof -> {out}", flush=True)


if __name__ == "__main__":
    main()
