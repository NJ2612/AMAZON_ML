#!/usr/bin/env python3
"""Round 1 — ANN blocking recall-ceiling audit (char n-gram TF-IDF).

Adds a label-free char-ngram TF-IDF ANN candidate source (name view, optional
address view) and UNIONS it with the existing inverted-index candidates (the
`lennorm` variant), then re-audits the macro-F0.5 recall CEILING on the India
dev S1 set. TF-IDF is fit transductively over the FULL cached India index (all
S2+S3 cores/addrs) -> label-free -> leakage-safe. Reuses idx_india_cap3000.pkl.

Gate (plan Round 1): union recall-ceiling f05 >= 0.96.

Run:
  python ann_block.py --probe                       # fit tfidf, print stats, exit
  python ann_block.py --n 3000 --ms 100 --topks 200 --addr   # quick check
  python ann_block.py --n 20000 --ms 100,200 --topks 200 --addr
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

import common as c
import blocking as bl  # noqa: F401  (kept for parity / interactive use)
import audit_block as ab

BUILD_CAP = 3000
CACHE = os.path.join(c.WORK, f"idx_india_cap{BUILD_CAP}.pkl")
COUNTRY = "India"


def _mask_cols(C, keep):
    """Zero every entry of csr `C` whose column is not kept (in place)."""
    drop = ~keep[C.indices]
    if drop.any():
        C.data[drop] = 0.0
        C.eliminate_zeros()


class HashedTfidf:
    """Stateless char-ngram HashingVectorizer + TF-IDF, with a DF cap.

    HashingVectorizer keeps memory CONSTANT during fit (no vocabulary dict — the
    part that spiked to ~4 GB with TfidfVectorizer on 4.1M docs) and is trivially
    portable to the offline Kaggle kernel (nothing fitted to serialize but the
    tiny idf_ vector + column-keep mask). Ultra-frequent n-grams (DF > cap) are
    dropped — they don't discriminate and they blow up query density.
    """

    def __init__(self, hv, tt, keep, D):
        self.hv, self.tt, self.keep, self.D = hv, tt, keep, D

    def transform(self, texts):
        from sklearn.preprocessing import normalize
        C = self.hv.transform([str(t) for t in texts]).tocsr()
        _mask_cols(C, self.keep)
        Q = self.tt.transform(C)
        normalize(Q, norm="l2", copy=False)
        return Q.tocsr()


def build_ann(texts, name, n_features=2 ** 20, df_cap_frac=0.02):
    """Fit a HashedTfidf over `texts` (the full corpus) and return it."""
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    t0 = time.time()
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4),
                           n_features=n_features, alternate_sign=False,
                           norm=None, dtype=np.float32)
    C = hv.transform([str(t) for t in texts]).tocsr()
    n = C.shape[0]
    df = C.getnnz(axis=0)                      # per-column document frequency
    cap = max(int(df_cap_frac * n), 1000)
    keep = df <= cap
    _mask_cols(C, keep)
    tt = TfidfTransformer(norm="l2", sublinear_tf=True)
    D = tt.fit_transform(C).tocsr()            # rows already L2-normalized
    del C
    mem = (D.data.nbytes + D.indices.nbytes + D.indptr.nbytes) / 1e6
    print(f"  ann[{name}] docs={n:,} feat={n_features:,} dropped_cols={int((~keep).sum()):,} "
          f"(df>{cap:,}) nnz={D.nnz:,} ({D.nnz/max(n,1):.1f}/doc) mem={mem:.0f}MB "
          f"build {time.time()-t0:.0f}s", flush=True)
    return HashedTfidf(hv, tt, keep, D)
def ann_topm(mdl, texts, M, batch=16):
    """Return {row_idx: np.int64[<=M] doc positions} by cosine, memory-safe.

    Computes D @ Q_batch.T (N x b) from the doc matrix directly (no N x V
    transpose kept in RAM); the b columns are the queries. Ultra-frequent n-grams
    were dropped at build, so the product stays sparse. Per-column top-M.
    """
    Q = mdl.transform(texts)
    D = mdl.D
    out = {}
    n = Q.shape[0]
    for s in range(0, n, batch):
        e = min(s + batch, n)
        S = D.dot(Q[s:e].T).tocsc()          # N x (e-s), columns = queries
        for j in range(e - s):
            beg, end = S.indptr[j], S.indptr[j + 1]
            idx = S.indices[beg:end]
            val = S.data[beg:end]
            if idx.size > M:
                sel = np.argpartition(val, -M)[-M:]
                idx, val = idx[sel], val[sel]
            out[s + j] = idx[np.argsort(val)[::-1]].astype(np.int64)
    return out


def _dev_s1_rows(n_target):
    """First N India S1 (matching make_devset / audit_block order)."""
    rows, n = [], 0
    for chunk in c.iter_clean(c.TRAIN_FILES["S1"],
                              usecols=["entity_id", "country", "name_core", "addr_norm"]):
        sub = chunk[chunk["country"].values == COUNTRY]
        for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
            rows.append((rid, core, addr))
            n += 1
            if n >= n_target:
                return rows
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ms", default="100", help="ANN top-M values, comma-sep")
    ap.add_argument("--topks", default="200", help="baseline inverted-index topk, comma-sep")
    ap.add_argument("--addr", action="store_true", help="also union an address-view ANN")
    ap.add_argument("--df-cap", type=float, default=0.02,
                    help="drop n-grams with DF > this fraction of docs")
    ap.add_argument("--nfeat", type=int, default=2 ** 20, help="hashing feature dim")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--probe", action="store_true", help="build ANN, print stats, exit")
    args = ap.parse_args()

    t0 = time.time()
    with open(CACHE, "rb") as f:
        ci = pickle.load(f)
    print(f"index {ci.n_records():,} recs loaded {time.time()-t0:.0f}s", flush=True)

    mdl_n = build_ann(ci.cores, "name", args.nfeat, args.df_cap)
    mdl_a = None
    if args.addr:
        mdl_a = build_ann(ci.addrs, "addr", args.nfeat, args.df_cap)
    if args.probe:
        print(f"probe done {time.time()-t0:.0f}s", flush=True)
        return

    cnt = np.fromiter(((len(str(cr).split()) + len(str(ad).split()) + 1)
                       for cr, ad in zip(ci.cores, ci.addrs)), np.float64, len(ci.cores))
    norm = np.sqrt(cnt)

    gt = c.load_ground_truth()
    s1_rows = _dev_s1_rows(args.n)
    s1_list = [r[0] for r in s1_rows]
    id_at = lambda j: ci.ids[j]  # noqa: E731
    print(f"dev S1 {len(s1_list):,}", flush=True)

    ms = [int(x) for x in args.ms.split(",")]
    topks = [int(x) for x in args.topks.split(",")]
    mmax = max(ms)

    tt = time.time()
    ann_n = ann_topm(mdl_n, [r[1] for r in s1_rows], mmax, args.batch)
    print(f"ann name top-{mmax} in {time.time()-tt:.0f}s", flush=True)
    ann_a = None
    if args.addr:
        tt = time.time()
        ann_a = ann_topm(mdl_a, [r[2] for r in s1_rows], mmax, args.batch)
        print(f"ann addr top-{mmax} in {time.time()-tt:.0f}s", flush=True)

    for topk in topks:
        tt = time.time()
        base = {r[0]: ab.rank_lennorm(ci, r[1], r[2], topk, norm) for r in s1_rows}
        f, mr, mir = ab.f05_ceiling(base, id_at, gt, s1_list)
        avg = np.mean([len(v) for v in base.values()])
        print(f"  [base lennorm topk={topk}] f05={f:.4f} macroR={mr:.4f} "
              f"microR={mir:.4f} avg={avg:.1f} ({time.time()-tt:.0f}s)", flush=True)
        for M in ms:
            comb = {}
            for i, (rid, core, addr) in enumerate(s1_rows):
                sset = set(base[rid])
                sset.update(int(x) for x in ann_n[i][:M])
                comb[rid] = sset
            f, mr, mir = ab.f05_ceiling(comb, id_at, gt, s1_list)
            avg = np.mean([len(v) for v in comb.values()])
            print(f"    [+annN M={M}] f05={f:.4f} macroR={mr:.4f} "
                  f"microR={mir:.4f} avg={avg:.1f}", flush=True)
            if args.addr:
                for i, (rid, core, addr) in enumerate(s1_rows):
                    comb[rid].update(int(x) for x in ann_a[i][:M])
                f, mr, mir = ab.f05_ceiling(comb, id_at, gt, s1_list)
                avg = np.mean([len(v) for v in comb.values()])
                print(f"    [+annN+annA M={M}] f05={f:.4f} macroR={mr:.4f} "
                      f"microR={mir:.4f} avg={avg:.1f}", flush=True)


if __name__ == "__main__":
    main()
