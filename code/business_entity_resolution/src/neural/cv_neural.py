#!/usr/bin/env python3
"""Leakage-safe 5-fold CV for the neural reranker + honest macro-F0.5 report.

For each fold: fine-tune a FRESH reranker on the other folds' pairs, score the
held-out fold -> out-of-fold match probs (p_ce). Threshold is tuned on the other
folds only (reuses cv_eval.eval_model). Reports, on the identical fold split as
the GBDT OOF:
  * reranker alone (union candidates)
  * reranker + GBDT logistic ensemble (leakage-safe per-fold meta)
  * GBDT blend baseline (its own lexical candidates) — measured, not quoted
Freezes a single global deploy threshold t* and writes deploy_neural.json.

Run (GPU):  python cv_neural.py [--devset PATH] [--folds 0,1,2,3,4]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import nconfig as nc
import common as c
import cv_eval as cv
import train_reranker as tr


def _ckpt_path(ckpt_dir, f):
    return os.path.join(ckpt_dir, f"pce_fold{f}.npy")


def _load_fold_ckpt(ckpt_dir, f, n_expected):
    """Return a completed fold's held-out p_ce slice, or None if absent/stale."""
    if not ckpt_dir:
        return None
    p = _ckpt_path(ckpt_dir, f)
    if not os.path.exists(p):
        return None
    try:
        arr = np.load(p)
    except Exception as e:  # noqa: BLE001 — corrupt/half-written on a crash -> retrain
        print(f"  [fold {f}] checkpoint unreadable ({e}) -> will retrain", flush=True)
        return None
    if len(arr) != n_expected:
        print(f"  [fold {f}] checkpoint size {len(arr)} != {n_expected} (devset "
              f"changed?) -> will retrain", flush=True)
        return None
    return arr.astype(np.float32)


def _save_fold_ckpt(ckpt_dir, f, arr):
    """Atomically persist a fold's held-out p_ce so a later crash keeps it."""
    if not ckpt_dir:
        return
    os.makedirs(ckpt_dir, exist_ok=True)
    p = _ckpt_path(ckpt_dir, f)
    tmp = p + ".partial"
    with open(tmp, "wb") as fh:
        np.save(fh, arr)          # file-handle form: no surprise .npy suffix
    os.replace(tmp, p)            # atomic on POSIX -> never a torn checkpoint
    print(f"  [fold {f}] checkpoint saved -> {p} ({len(arr):,} rows)", flush=True)


def build_oof(df: pd.DataFrame, folds, device, ckpt_dir=None) -> np.ndarray:
    """Per-fold fine-tune + held-out score -> p_ce aligned to df rows.

    Checkpoint rolling: after each fold's held-out scores are computed they are
    written to ckpt_dir/pce_fold{f}.npy (atomically). On a re-run these files are
    mounted from the prior kernel output (see run_cv.py resume seeding); a fold
    whose checkpoint is present and size-matches is RELOADED instead of retrained,
    so a crash never costs more than the one fold in flight. The reconstructed
    p_ce is identical whether a fold was trained fresh or reloaded, so the honest
    CV number is unchanged.
    """
    p_ce = np.zeros(len(df), dtype=np.float32)
    fold_arr = df["fold"].to_numpy()
    for f in folds:
        va_idx = np.where(fold_arr == f)[0]
        if len(va_idx) == 0:
            continue
        cached = _load_fold_ckpt(ckpt_dir, f, len(va_idx))
        if cached is not None:
            p_ce[va_idx] = cached
            print(f"  [fold {f}] RESUMED from checkpoint "
                  f"({len(va_idx):,} rows) — skipping train", flush=True)
            continue
        tr_df = df[df["fold"] != f]
        t0 = time.time()
        model, tok = tr.load_reranker(device=device)          # fresh per fold
        tr.train_model(model, tok, tr_df, device=device, seed=100 + f)
        p_ce[va_idx] = tr.score_df(model, tok, df.iloc[va_idx], device=device)
        _save_fold_ckpt(ckpt_dir, f, p_ce[va_idx])            # persist before next fold
        print(f"  [fold {f}] train={len(tr_df):,} val={len(va_idx):,} "
              f"in {time.time()-t0:.0f}s", flush=True)
        del model
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    return p_ce


def ensemble_oof(df: pd.DataFrame, p_ce: np.ndarray, folds):
    """Leakage-safe logistic meta over [p_ce, p_lgb, p_xgb]; returns (p_ens, coef_all)."""
    from sklearn.linear_model import LogisticRegression
    Z = np.column_stack([p_ce,
                         df["p_lgb"].to_numpy(np.float32),
                         df["p_xgb"].to_numpy(np.float32)])
    y = df["label"].to_numpy(np.int8)
    fold = df["fold"].to_numpy()
    p_ens = np.zeros(len(df), dtype=np.float32)
    for f in folds:
        tr_mask = fold != f
        va = fold == f
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(Z[tr_mask], y[tr_mask])
        p_ens[va] = lr.predict_proba(Z[va])[:, 1]
    lr_all = LogisticRegression(max_iter=1000, C=1.0).fit(Z, y)
    coef = dict(intercept=float(lr_all.intercept_[0]),
                w_ce=float(lr_all.coef_[0][0]),
                w_lgb=float(lr_all.coef_[0][1]),
                w_xgb=float(lr_all.coef_[0][2]))
    return p_ens, coef


def global_threshold(grp, prob) -> tuple:
    """Single t* maximizing macro-F0.5 over ALL S1 (deploy operating point)."""
    all_s1 = list(grp.keys())
    best_t, best = 0.5, -1.0
    for t in cv.THRESH_GRID:
        v = cv.macro_f05_at(grp, all_s1, prob, t)
        if v > best:
            best, best_t = v, float(t)
    return best_t, best


def train_final_and_save(df: pd.DataFrame, device: str) -> str:
    """Fine-tune ONE reranker on ALL dev folds and save it for the infer kernel."""
    t0 = time.time()
    model, tok = tr.load_reranker(device=device)
    tr.train_model(model, tok, df, device=device, seed=777)
    out = f"{nc.NEURAL_WORK}/reranker_final"
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[cv-neural] final reranker (all dev) saved -> {out} "
          f"in {time.time()-t0:.0f}s", flush=True)
    del model
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devset", default=f"{nc.NEURAL_WORK}/devset_neural_india.parquet")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--no-ensemble", action="store_true")
    ap.add_argument("--save-final", action="store_true",
                    help="after CV, fine-tune one reranker on ALL dev + save "
                         "(for the inference kernel)")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x != ""]
    device = nc.get_device()
    print(f"[cv-neural] device={device} reranker={nc.RERANKER_NAME} "
          f"devset={args.devset}", flush=True)

    df = pd.read_parquet(args.devset).reset_index(drop=True)
    gt = c.load_ground_truth()

    # attach GBDT OOF (for the ensemble); dense-only pairs -> 0 (GBDT never saw them)
    have_oof = os.path.exists(nc.OOF_PARQUET)
    if have_oof:
        oof = pd.read_parquet(nc.OOF_PARQUET)[["s1_id", "cand_id", "p_lgb", "p_xgb"]]
        df = df.merge(oof, on=["s1_id", "cand_id"], how="left")
        df["p_lgb"] = df["p_lgb"].fillna(0.0).astype(np.float32)
        df["p_xgb"] = df["p_xgb"].fillna(0.0).astype(np.float32)

    # --- reranker OOF ---------------------------------------------------------
    # ckpt_dir persists each completed fold so a crashed/re-pushed run resumes
    # (checkpoint rolling) instead of retraining from scratch. It lives under
    # NEURAL_WORK, so it is committed as kernel output and mountable next run.
    t0 = time.time()
    ckpt_dir = os.path.join(nc.NEURAL_WORK, "ckpt")
    p_ce = build_oof(df, folds, device, ckpt_dir=ckpt_dir)
    df["p_ce"] = p_ce
    oof_out = f"{nc.NEURAL_WORK}/neural_oof.parquet"
    keep = ["s1_id", "cand_id", "label", "fold", "src", "p_ce"]
    if have_oof:
        keep += ["p_lgb", "p_xgb"]
    df[keep].to_parquet(oof_out, index=False)
    print(f"[cv-neural] reranker OOF in {time.time()-t0:.0f}s -> {oof_out}", flush=True)

    # --- scoring (reuses the GBDT leakage-safe evaluator) ---------------------
    grp = cv.build_groups(df, gt)
    # Subset run (e.g. 3-fold): report + tune ONLY on the folds actually scored,
    # so held-out folds still carrying p_ce=0 can never contaminate the mean or
    # the frozen threshold. Each scored fold's model still trained on 80% of dev.
    _run = set(folds)
    grp = {s1: g for s1, g in grp.items() if g["fold"] in _run}
    folds_of = {f: [s1 for s1, g in grp.items() if g["fold"] == f] for f in folds}
    print(f"\n[cv-neural] {len(grp):,} S1 — macro-F0.5 (full-truth, leakage-safe t):",
          flush=True)
    fs_ce = cv.eval_model(grp, folds_of, p_ce, "reranker")
    results = {"reranker": [float(x) for x in fs_ce]}

    # The ensemble is measured for the record ONLY. infer_neural always ships
    # reranker-alone (it reads reranker_threshold and never scores the GBDTs), so
    # the deployed decision/cv_mean below must describe the reranker — quoting the
    # (chosen-because-higher) ensemble would overstate the actual deliverable.
    coef = None
    if have_oof and not args.no_ensemble:
        p_ens, coef = ensemble_oof(df, p_ce, folds)
        fs_ens = cv.eval_model(grp, folds_of, p_ens, "rr+gbdt")
        results["ensemble"] = [float(x) for x in fs_ens]

    # --- GBDT blend baseline on ITS OWN lexical candidates (measured) ---------
    if have_oof:
        base = pd.read_parquet(nc.OOF_PARQUET).reset_index(drop=True)
        gb = cv.build_groups(base, gt)
        gb = {s1: g for s1, g in gb.items() if g["fold"] in _run}   # same folds as reranker
        gfo = {f: [s1 for s1, g in gb.items() if g["fold"] == f] for f in folds}
        p_blend = 0.5 * (base["p_lgb"].to_numpy() + base["p_xgb"].to_numpy())
        fs_b = cv.eval_model(gb, gfo, p_blend, "gbdt-base")
        results["gbdt_baseline"] = [float(x) for x in fs_b]

    # --- freeze deploy operating point (the reranker-alone path infer ships) --
    t_ce, _ = global_threshold(grp, p_ce)          # reranker-alone t* (infer path)
    deploy = dict(reranker=nc.RERANKER_NAME, biencoder=nc.BIENCODER_NAME,
                  decision="reranker", threshold=t_ce,
                  reranker_threshold=t_ce,
                  ann_topm=nc.ANN_TOPM, lex_topk=nc.LEX_TOPK,
                  ensemble_coef=coef,
                  ensemble_cv_mean=(float(np.mean(results["ensemble"]))
                                    if "ensemble" in results else None),
                  cv_mean=float(np.mean(results["reranker"])),
                  cv_results=results)
    dpath = f"{nc.NEURAL_WORK}/deploy_neural.json"
    with open(dpath, "w", encoding="utf-8") as f:
        json.dump(deploy, f, indent=2)
    print(f"\n[cv-neural] DEPLOY=reranker (shipped)  t*={t_ce:.3f}  "
          f"cv_mean={np.mean(results['reranker']):.4f}"
          f"+/-{np.std(results['reranker']):.4f}  -> {dpath}", flush=True)
    print(f"[cv-neural] summary: " + "  ".join(
        f"{k}={np.mean(v):.4f}+/-{np.std(v):.4f}" for k, v in results.items()),
        flush=True)

    if args.save_final:
        train_final_and_save(df, device)


if __name__ == "__main__":
    main()

