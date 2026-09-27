#!/usr/bin/env python3
"""Listwise Learning-to-Rank (LambdaMART) + global 1-to-1 assignment decode.

The genuinely-different third architecture (distinct from the pointwise LGB+XGB
blend AND from the LaBSE+cross-encoder neural stack). Instead of scoring each
(S1, candidate) pair as an ABSOLUTE probability, this trains an LGBMRanker with
objective=lambdarank whose gradient is defined only over pairs WITHIN one S1's
candidate pool — so it learns to ORDER a candidate against its siblings. On top
of the 37 pointwise features it consumes WITHIN-GROUP relative features (rank,
z-score, gap-to-best) that a pointwise model cannot express.

Decision is a per-S1 variable-k cutoff (not a flat threshold): a leakage-safe
logistic calibration maps ranker score -> P(match) on the held-out folds, then a
tuned (t, rho) rule keeps the top candidate when P>=t and every other candidate
within rho*P_top1 — explicitly allowing k=0 (reject-all -> bank the singleton
1.0). Finally a GLOBAL greedy 1-to-1 assignment enforces the strict target-degree
constraint measured in the data (every matched id has degree exactly 1), killing
duplicate/false merges — a precision lever neither existing model uses.

Everything is measured leakage-safe on the identical GroupKFold(K=5) split (md5
seed=42, grouped by S1) so the macro-F0.5 lines up with the GBDT OOF.

Run:  python ltr.py [--feats devset_india_n20000_feats.parquet]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import common as c
import cv_eval as cv
from features import FEATURES

# Within-group relative features are built from these base pointwise signals.
# Each yields a z-score (vs group mean), a gap-to-best, and a fractional rank —
# all label-free and computed from the candidate pool alone, so they exist
# identically at inference time (leakage-safe, transductive).
REL_BASE = ["n_tset", "n_wratio", "n_jw", "x_name_addr", "id_sum_idf",
            "a_tset", "n_g4_jac"]


def rel_features(df: pd.DataFrame) -> tuple:
    """Append within-S1 relative features; return (df, list_of_new_cols)."""
    df = df.reset_index(drop=True)
    g = df.groupby("s1_id", sort=False)
    new_cols = []
    for b in REL_BASE:
        mean = g[b].transform("mean")
        std = g[b].transform("std").replace(0.0, 1.0).fillna(1.0)
        df[f"{b}_z"] = ((df[b] - mean) / std).astype(np.float32)
        df[f"{b}_gap"] = (df[b] - g[b].transform("max")).astype(np.float32)
        rk = g[b].rank(ascending=False, method="average", pct=True)
        df[f"{b}_rk"] = rk.astype(np.float32)
        new_cols += [f"{b}_z", f"{b}_gap", f"{b}_rk"]
    df["grp_log_size"] = np.log1p(g["s1_id"].transform("size")).astype(np.float32)
    new_cols.append("grp_log_size")
    return df, new_cols


# --- ranker ------------------------------------------------------------------
def ranker_params() -> dict:
    return dict(objective="lambdarank", n_estimators=500, learning_rate=0.05,
                num_leaves=63, min_child_samples=50, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                n_jobs=-1, verbosity=-1, label_gain=[0, 1],
                lambdarank_truncation_level=30)


def train_ranker(tr_df: pd.DataFrame, feat_cols: list, seed: int):
    """Fit an LGBMRanker; query group = one S1 (rows sorted contiguously)."""
    import lightgbm as lgb
    tr_df = tr_df.sort_values("s1_id", kind="stable")
    groups = tr_df.groupby("s1_id", sort=False).size().to_numpy()
    X = tr_df[feat_cols].to_numpy(np.float32)
    y = tr_df["label"].to_numpy(np.int8)
    m = lgb.LGBMRanker(random_state=seed, **ranker_params())
    m.fit(X, y, group=groups)
    return m


def build_oof(df: pd.DataFrame, feat_cols: list, folds: list) -> np.ndarray:
    """Per-fold fine-tune on the OTHER folds, score the held-out fold -> raw
    lambdarank scores aligned to df rows (leakage-safe)."""
    score = np.zeros(len(df), dtype=np.float32)
    fold = df["fold"].to_numpy()
    for f in folds:
        va_idx = np.where(fold == f)[0]
        if len(va_idx) == 0:
            continue
        t0 = time.time()
        m = train_ranker(df[fold != f], feat_cols, seed=200 + f)
        score[va_idx] = m.predict(df.iloc[va_idx][feat_cols].to_numpy(np.float32))
        print(f"  [fold {f}] train={int((fold!=f).sum()):,} "
              f"val={len(va_idx):,} in {time.time()-t0:.0f}s", flush=True)
    return score


# --- decision: leakage-safe calibration + variable-k cutoff + 1-to-1 ---------
def presort(grp: dict, score: np.ndarray) -> dict:
    """s1 -> (cand_ids, row_idx) both sorted DESC by raw score. Order is fixed by
    RAW score; logistic calibration is monotonic, so calibrated probs stay
    descending too -> the kept set is always a contiguous prefix (no re-sort in
    the tuning grid)."""
    pre = {}
    for s1, g in grp.items():
        order = np.argsort(-score[g["rows"]], kind="stable")
        pre[s1] = (g["cands"][order], g["rows"][order])
    return pre


def calibrate(score: np.ndarray, tr_mask: np.ndarray, y: np.ndarray):
    """Fit score->P(match) logistic on the tuning rows only; return a mapper."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(score[tr_mask].reshape(-1, 1), y[tr_mask])
    return lr.predict_proba(score.reshape(-1, 1))[:, 1].astype(np.float32)


T_GRID = np.concatenate([np.linspace(0.05, 0.95, 19), [0.97, 0.98, 0.99]])
RHO_GRID = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]


def decide_sets(pre: dict, prob: np.ndarray, s1_list, t: float, rho: float) -> dict:
    """Per-S1 variable-k cutoff. Keep top1 iff P>=t (else reject-all/empty), then
    every candidate within rho*P_top1. Returns {s1: [(cand, prob), ...]}."""
    preds = {}
    for s1 in s1_list:
        cands, rows = pre[s1]
        p = prob[rows]                       # already descending (monotone calib)
        if p.size == 0 or p[0] < t:
            preds[s1] = []
            continue
        thr = max(t, rho * p[0])
        k = int(np.count_nonzero(p >= thr))  # contiguous prefix
        preds[s1] = list(zip(cands[:k].tolist(), p[:k].tolist()))
    return preds


def assign_1to1(preds: dict) -> dict:
    """Global greedy mutual-exclusion decode: each target id goes to its highest-
    prob S1 claimant only (the strict degree-1 constraint the data satisfies).
    Drops the lower-confidence duplicate claims -> precision lever."""
    claims = []
    for s1, lst in preds.items():
        for cand, p in lst:
            claims.append((p, s1, cand))
    claims.sort(key=lambda z: z[0], reverse=True)
    taken = set()
    out = {s1: set() for s1 in preds}
    for p, s1, cand in claims:
        if cand not in taken:
            taken.add(cand)
            out[s1].add(cand)
    return out


def to_sets(preds: dict) -> dict:
    return {s1: {cand for cand, _ in lst} for s1, lst in preds.items()}


def score_preds(grp: dict, s1_list, pred_sets: dict) -> float:
    fs = [c.fbeta_entity(pred_sets.get(s1, set()), grp[s1]["truth"], 0.5)
          for s1 in s1_list]
    return float(np.mean(fs)) if fs else 0.0


def eval_ltr(df, grp, folds_of, score, use_assign: bool, tag: str) -> np.ndarray:
    """Leakage-safe per-fold: calibrate + tune (t,rho) on the OTHER folds, apply
    to the held-out fold. macro-F0.5 vs FULL ground truth."""
    pre = presort(grp, score)
    y = df["label"].to_numpy(np.int8)
    fold = df["fold"].to_numpy()
    per_fold, params = [], []
    for f in range(5):
        val_s1 = folds_of[f]
        if not val_s1:
            continue
        tune_s1 = [s for ff in range(5) if ff != f for s in folds_of[ff]]
        prob = calibrate(score, fold != f, y)      # fit on tuning rows, map all
        best = (-1.0, 0.5, 0.0)
        for t in T_GRID:
            for rho in RHO_GRID:
                preds = decide_sets(pre, prob, tune_s1, t, rho)
                sets = assign_1to1(preds) if use_assign else to_sets(preds)
                v = score_preds(grp, tune_s1, sets)
                if v > best[0]:
                    best = (v, float(t), float(rho))
        _, t, rho = best
        preds = decide_sets(pre, prob, val_s1, t, rho)
        sets = assign_1to1(preds) if use_assign else to_sets(preds)
        per_fold.append(score_preds(grp, val_s1, sets))
        params.append((t, rho))
    fs = np.array(per_fold)
    tt = np.mean([p[0] for p in params]); rr = np.mean([p[1] for p in params])
    print(f"  {tag:>16}: per-fold F0.5 = [{', '.join(f'{x:.4f}' for x in fs)}]  "
          f"mean={fs.mean():.4f} +/- {fs.std():.4f}  (t~{tt:.2f} rho~{rr:.2f})",
          flush=True)
    return fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats",
                    default=os.path.join(c.WORK, "devset_india_n20000_feats.parquet"))
    ap.add_argument("--folds", default="0,1,2,3,4")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x != ""]

    t0 = time.time()
    df = pd.read_parquet(args.feats).reset_index(drop=True)
    df, rel_cols = rel_features(df)
    # The cached feats parquet may predate some features.py columns; use whatever
    # base features are actually present (the relative features are the LTR-specific
    # signal regardless) and warn about any gap.
    base_cols = [f for f in FEATURES if f in df.columns]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        print(f"[ltr] WARN {len(missing)} base feats absent from parquet "
              f"(using {len(base_cols)}/{len(FEATURES)}): {missing}", flush=True)
    feat_cols = base_cols + rel_cols
    print(f"[ltr] {df['s1_id'].nunique():,} S1, {len(df):,} pairs, "
          f"{len(feat_cols)} feats ({len(rel_cols)} relative)", flush=True)

    gt = c.load_ground_truth()
    score = build_oof(df, feat_cols, folds)
    df["ltr_score"] = score
    oof_out = os.path.join(c.WORK, "devset_india_n20000_ltr_oof.parquet")
    df[["s1_id", "cand_id", "label", "fold", "ltr_score"]].to_parquet(oof_out, index=False)

    grp = cv.build_groups(df, gt)
    folds_of = {f: [s1 for s1, g in grp.items() if g["fold"] == f] for f in range(5)}
    print(f"\n[ltr] {len(grp):,} S1 — macro-F0.5 (full-truth, leakage-safe t,rho):",
          flush=True)
    r_flat = eval_ltr(df, grp, folds_of, score, False, "ranker+cutoff")
    r_asg = eval_ltr(df, grp, folds_of, score, True, "ranker+cut+1to1")
    print(f"\n[ltr] blend baseline (GBDT) reference = 0.8528", flush=True)
    print(f"[ltr] done in {time.time()-t0:.0f}s -> {oof_out}", flush=True)


if __name__ == "__main__":
    main()
