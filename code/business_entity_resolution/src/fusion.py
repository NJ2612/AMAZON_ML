#!/usr/bin/env python3
"""Stage 4 — Cross-attention fusion of the LightGBM & XGBoost experts (PyTorch).

Consumes the OOF parquet from cv_eval.py (p_lgb, p_xgb per pair) joined back to
the feature matrix, and trains a SMALL bidirectional cross-attention net that
lets the two experts exchange information before a fused sigmoid prediction:

  each expert -> T tokens (learned projections of feature groups, gated by that
  expert's probability) -> multi-head cross-attention (LGB tokens attend to XGB
  tokens and vice-versa) -> residual+LayerNorm -> mean-pool each stream ->
  fuse [lgb', xgb', p_lgb, p_xgb] -> MLP -> sigmoid.

It is cross-validated on the SAME GroupKFold folds as the base models (two-level
OOF stacking: for fold f the fusion trains on OOF rows of folds != f and
predicts fold f), so no row is ever scored by a model that saw it. A logistic-
regression stacker on [p_lgb, p_xgb] is trained the same way as an honest A/B
baseline; we only prefer the attention net if it beats it by > 1 fold std.

Run:  python fusion.py <feats.parquet> <oof.parquet> [--epochs 8]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as c
from features import FEATURES
from cv_eval import build_groups, eval_model

NAME_FEATS = ["n_ratio", "n_tsort", "n_tset", "n_partial", "n_wratio", "n_jw",
              "n_tok_jac", "n_g3_jac", "n_g4_jac", "n_exact", "n_concat",
              "n_len_ratio", "n_ntok_ratio"]
IDENT_FEATS = ["id_shared_tok", "id_share_maxidf", "s1_min_idf", "s1_max_idf"]
ADDR_FEATS = ["a_tset", "a_tsort", "a_g4_jac", "a_num_ov", "a_num_jac",
              "a_loc_jac", "a_both", "a_s1_empty", "a_c_empty", "x_name_addr"]
GROUPS = [NAME_FEATS, IDENT_FEATS, ADDR_FEATS]     # one token per group, per expert


class CrossAttnFusion(nn.Module):
    """Bidirectional multi-head cross-attention over per-expert feature tokens."""

    def __init__(self, group_dims, d=32, heads=4):
        super().__init__()
        self.G = len(group_dims)
        # expert-specific token projections (LGB and XGB see the same feature
        # groups through DIFFERENT learned maps, so their reps can differ)
        self.lgb_proj = nn.ModuleList([nn.Linear(g, d) for g in group_dims])
        self.xgb_proj = nn.ModuleList([nn.Linear(g, d) for g in group_dims])
        self.lgb_pemb = nn.Linear(1, d)            # prob gating per expert
        self.xgb_pemb = nn.Linear(1, d)
        self.l2x = nn.MultiheadAttention(d, heads, batch_first=True)
        self.x2l = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln_l = nn.LayerNorm(d)
        self.ln_x = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d + 2, 64), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(64, 1))

    def _tokens(self, groups, proj, pemb, p):
        pe = pemb(p)                                # (B, d)
        toks = [proj[g](groups[g]) + pe for g in range(self.G)]
        return torch.stack(toks, dim=1)             # (B, G, d)

    def forward(self, groups, p_lgb, p_xgb):
        tl = self._tokens(groups, self.lgb_proj, self.lgb_pemb, p_lgb)
        tx = self._tokens(groups, self.xgb_proj, self.xgb_pemb, p_xgb)
        al, _ = self.l2x(tl, tx, tx)                # LGB queries attend to XGB
        ax, _ = self.x2l(tx, tl, tl)                # XGB queries attend to LGB
        tl = self.ln_l(tl + al)
        tx = self.ln_x(tx + ax)
        fused = torch.cat([tl.mean(1), tx.mean(1), p_lgb, p_xgb], dim=1)
        return self.head(fused).squeeze(1)          # logit


def _standardize(Xtr, Xall):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True) + 1e-6
    return (Xall - mu) / sd


def _split_groups(X):
    """Slice a standardized (N,F) array into the per-group tensors."""
    cols = {f: i for i, f in enumerate(FEATURES)}
    return [torch.tensor(X[:, [cols[f] for f in grp]], dtype=torch.float32)
            for grp in GROUPS]


def _predict(model, groups, pl, px, bs=131072):
    model.eval()
    out = np.zeros(pl.shape[0], dtype=np.float32)
    with torch.no_grad():
        for s in range(0, pl.shape[0], bs):
            e = min(s + bs, pl.shape[0])
            g = [t[s:e] for t in groups]
            logit = model(g, pl[s:e], px[s:e])
            out[s:e] = torch.sigmoid(logit).numpy()
    return out


def train_fusion(Xg, pl, px, y, epochs, seed=0):
    torch.manual_seed(seed)
    dims = [len(g) for g in GROUPS]
    model = CrossAttnFusion(dims)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pw = torch.tensor([(len(y) - y.sum()) / max(y.sum(), 1)], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    n = y.shape[0]
    bs = 16384
    yt = torch.tensor(y, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            g = [t[idx] for t in Xg]
            logit = model(g, pl[idx], px[idx])
            loss = lossf(logit, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feats")
    ap.add_argument("oof")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--neg", type=float, default=6.0)
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    from cv_eval import _downsample
    from sklearn.linear_model import LogisticRegression

    feats = pd.read_parquet(args.feats).reset_index(drop=True)
    oof = pd.read_parquet(args.oof).reset_index(drop=True)
    # align by (s1_id, cand_id) — oof is a column subset of the same rows/order
    df = feats.copy()
    df["p_lgb"] = oof["p_lgb"].to_numpy()
    df["p_xgb"] = oof["p_xgb"].to_numpy()
    gt = c.load_ground_truth()

    Xraw = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int8)
    fold = df["fold"].to_numpy(dtype=np.int8)
    pl_all = df["p_lgb"].to_numpy(dtype=np.float32)
    px_all = df["p_xgb"].to_numpy(dtype=np.float32)

    p_fus = np.zeros(len(df), dtype=np.float32)
    p_lr = np.zeros(len(df), dtype=np.float32)
    t0 = time.time()
    for f in range(5):
        tr = np.where(fold != f)[0]
        va = np.where(fold == f)[0]
        tr_ds = _downsample(tr, y, args.neg, seed=200 + f)
        Xs = _standardize(Xraw[tr_ds], Xraw)          # fit scaler on train subset
        Xg = _split_groups(Xs)
        pl_t = torch.tensor(pl_all).unsqueeze(1)
        px_t = torch.tensor(px_all).unsqueeze(1)
        model = train_fusion([t[tr_ds] for t in Xg], pl_t[tr_ds], px_t[tr_ds],
                             y[tr_ds], args.epochs, seed=f)
        p_fus[va] = _predict(model, [t[va] for t in Xg], pl_t[va], px_t[va])
        # logistic stacker baseline on the two expert probs (same folds)
        lr = LogisticRegression(max_iter=200)
        lr.fit(np.c_[pl_all[tr_ds], px_all[tr_ds]], y[tr_ds])
        p_lr[va] = lr.predict_proba(np.c_[pl_all[va], px_all[va]])[:, 1]
        print(f"  fold {f}: fusion+LR trained ({time.time()-t0:.0f}s)", flush=True)

    grp = build_groups(df, gt)
    folds_of = {f: [s1 for s1, g in grp.items() if g["fold"] == f] for f in range(5)}
    print(f"\n[fusion] {len(grp):,} S1 entities, macro-F0.5 (leakage-safe):", flush=True)
    eval_model(grp, folds_of, pl_all, "LGB")
    eval_model(grp, folds_of, px_all, "XGB")
    eval_model(grp, folds_of, 0.5 * (pl_all + px_all), "blend")
    eval_model(grp, folds_of, p_lr, "blendLR")
    eval_model(grp, folds_of, p_fus, "fusion")


if __name__ == "__main__":
    main()

