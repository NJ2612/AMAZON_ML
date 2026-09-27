#!/usr/bin/env python3
"""Round 2 — entity-level decision policy (leakage-safe, tuned on held-out folds).

The base models emit a per-pair probability; a flat global threshold treats every
(S1, candidate) pair independently. But the metric is per-ENTITY macro-F0.5, which
is precision-weighted and punishes a single false merge (one FP on a 4-match entity
~0.75; ANY false positive on a true singleton = 0.0). This module makes the
accept/reject decision at the entity level with four evidence-based knobs:

  t_base   : minimum prob to accept a candidate at all.
  t_empty  : if the BEST candidate for an S1 scores below this (>= t_base) bar,
             predict EMPTY — bank the 1.0 singleton credit rather than risk a
             false merge on a likely-singleton entity.
  margin   : accept a candidate only if prob >= margin * best_prob for that S1 —
             trims the low-confidence tail once a strong match exists (precision).
  generic gate : a match resting only on generic (low-IDF) shared name tokens is
             accepted ONLY if the address agrees (a shared numeric address token or
             a high addr token-set ratio). Kills generic-name false merges.

Every knob is tuned by coordinate ascent to maximize macro-F0.5 on the OTHER folds,
then applied to the held-out fold, so the reported number is leakage-safe. The flat
per-fold-threshold baseline is reported alongside for the A/B decision.

Run:  python policy.py <oof.parquet> --feats <feats.parquet> [--prob blend]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import common as c

def load_groups(oof_path: str, feats_path: str, prob_name: str):
    """Per-S1 arrays (sorted by prob desc) + full-truth size, aligned oof+feats.

    Returns list of dicts: p, truth_mask, maxidf, num_ov, atset, n_truth, fold.
    n_truth is the FULL ground-truth size (incl. matches blocking never surfaced),
    so recall — and thus F0.5 — is not inflated by an in-candidate-only denominator.
    """
    oof = pd.read_parquet(oof_path).reset_index(drop=True)
    fe = pd.read_parquet(feats_path,
                         columns=["s1_id", "cand_id", "id_share_maxidf",
                                  "a_num_ov", "a_tset"]).reset_index(drop=True)
    # both derive from the same devset in the same row order — verify, don't merge
    assert len(oof) == len(fe), "oof/feats length mismatch"
    assert (oof["s1_id"].to_numpy() == fe["s1_id"].to_numpy()).all() and \
           (oof["cand_id"].to_numpy() == fe["cand_id"].to_numpy()).all(), \
           "oof/feats row order mismatch — rebuild both from the same devset"
    prob = {"blend": 0.5 * (oof["p_lgb"] + oof["p_xgb"]),
            "lgb": oof["p_lgb"], "xgb": oof["p_xgb"]}[prob_name].to_numpy(np.float32)
    gt = c.load_ground_truth()
    maxidf = fe["id_share_maxidf"].to_numpy(np.float32)
    num_ov = fe["a_num_ov"].to_numpy(np.float32)
    atset = fe["a_tset"].to_numpy(np.float32)

    groups = []
    for s1, sub in oof.groupby("s1_id", sort=False):
        idx = sub.index.to_numpy()
        p = prob[idx]
        order = np.argsort(p)[::-1]           # sort every array by prob desc
        idx = idx[order]
        cids = sub["cand_id"].to_numpy()[order]
        truth = gt.get(s1, set())
        groups.append(dict(
            p=p[order], truth_mask=np.array([cid in truth for cid in cids]),
            maxidf=maxidf[idx], num_ov=num_ov[idx], atset=atset[idx],
            n_truth=len(truth), fold=int(sub["fold"].iloc[0])))
    return groups


def _f05(tp: int, fp: int, n_truth: int) -> float:
    """F0.5 for one entity from prediction counts + FULL truth size."""
    if n_truth == 0:
        return 1.0 if (tp + fp) == 0 else 0.0
    if tp == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / n_truth
    return 1.25 * p * r / (0.25 * p + r)


def score_entity(g: dict, prm: dict) -> float:
    """Apply the policy to one entity's sorted candidates and return its F0.5."""
    p = g["p"]
    if len(p) == 0 or p[0] < prm["t_empty"]:
        return _f05(0, 0, g["n_truth"])                       # predict EMPTY
    keep = (p >= prm["t_base"]) & (p >= prm["margin"] * p[0])
    if prm["gen_idf"] > 0:                                    # generic-name gate
        generic = g["maxidf"] < prm["gen_idf"]
        addr_ok = (g["num_ov"] >= 1) | (g["atset"] >= prm["addr_min"])
        keep = keep & ~(generic & ~addr_ok)
    tm = g["truth_mask"]
    tp = int(np.count_nonzero(keep & tm))
    fp = int(np.count_nonzero(keep & ~tm))
    return _f05(tp, fp, g["n_truth"])


def macro_f05(groups: list, prm: dict) -> float:
    return float(np.mean([score_entity(g, prm) for g in groups])) if groups else 0.0


GRIDS = {
    # optimum flat threshold sits ~0.98 (precision-weighted metric on 200:4 imbalance),
    # so the accept/empty grids are centered in the high-precision regime.
    "t_base":   [0.5, 0.7, 0.85, 0.92, 0.95, 0.97, 0.98, 0.99],
    "t_empty":  [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995],
    "margin":   [0.0, 0.3, 0.5, 0.7, 0.85, 0.92, 0.97],
    "gen_idf":  [0.0, 4.0, 6.0, 8.0, 10.0],   # id_share_maxidf below this = generic
    "addr_min": [0.0, 50.0, 70.0, 85.0],      # a_tset needed when name is generic
}
INIT = {"t_base": 0.95, "t_empty": 0.95, "margin": 0.0, "gen_idf": 0.0, "addr_min": 0.0}


def coordinate_ascent(groups: list, passes: int = 3) -> tuple[dict, float]:
    """Greedy per-knob search; 3 passes converges on this landscape."""
    prm = dict(INIT)
    best = macro_f05(groups, prm)
    for _ in range(passes):
        improved = False
        for knob, grid in GRIDS.items():
            cur = prm[knob]
            for v in grid:
                if v == prm[knob]:
                    continue
                trial = dict(prm); trial[knob] = v
                s = macro_f05(groups, trial)
                if s > best + 1e-6:
                    best, prm[knob], improved = s, v, True
        if not improved:
            break
    return prm, best


def flat_threshold(groups: list) -> tuple[float, float]:
    """Best single global threshold (the cv_eval baseline) on `groups`."""
    best_t, best = 0.5, -1.0
    for t in np.linspace(0.5, 0.995, 50):
        prm = {"t_base": t, "t_empty": t, "margin": 0.0, "gen_idf": 0.0, "addr_min": 0.0}
        s = macro_f05(groups, prm)
        if s > best:
            best, best_t = s, float(t)
    return best_t, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("oof")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--prob", default="blend", choices=["blend", "lgb", "xgb"])
    args = ap.parse_args()

    groups = load_groups(args.oof, args.feats, args.prob)
    by_fold = {f: [g for g in groups if g["fold"] == f] for f in range(5)}
    print(f"[policy] prob={args.prob}  entities={len(groups):,}  "
          f"singletons={sum(1 for g in groups if g['n_truth']==0):,}", flush=True)

    flat_fs, pol_fs = [], []
    for f in range(5):
        val = by_fold[f]
        tune = [g for ff in range(5) if ff != f for g in by_fold[ff]]
        # flat baseline: tune single threshold on other folds
        t, _ = flat_threshold(tune)
        flat_prm = {"t_base": t, "t_empty": t, "margin": 0.0, "gen_idf": 0.0, "addr_min": 0.0}
        flat_fs.append(macro_f05(val, flat_prm))
        # policy: coordinate ascent on other folds
        prm, _ = coordinate_ascent(tune)
        pol_fs.append(macro_f05(val, prm))
        print(f"  fold {f}: flat(t={t:.2f})={flat_fs[-1]:.4f}  "
              f"policy={pol_fs[-1]:.4f}  "
              f"prm={{tb:{prm['t_base']:.2f} te:{prm['t_empty']:.2f} "
              f"mg:{prm['margin']:.2f} gi:{prm['gen_idf']:.0f} am:{prm['addr_min']:.0f}}}",
              flush=True)

    flat_fs, pol_fs = np.array(flat_fs), np.array(pol_fs)
    print(f"\n[policy] leakage-safe macro-F0.5 (mean +/- std over 5 folds):", flush=True)
    print(f"  flat threshold : {flat_fs.mean():.4f} +/- {flat_fs.std():.4f}", flush=True)
    print(f"  entity policy  : {pol_fs.mean():.4f} +/- {pol_fs.std():.4f}  "
          f"(delta {pol_fs.mean()-flat_fs.mean():+.4f}, "
          f"{'>' if pol_fs.mean()-flat_fs.mean() > flat_fs.std() else '<='} 1 fold std)",
          flush=True)


if __name__ == "__main__":
    main()



