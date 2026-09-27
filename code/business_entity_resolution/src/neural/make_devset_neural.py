#!/usr/bin/env python3
"""Build the neural dev set: dense∪lexical candidates for the 20k India shard.

Reuses the EXACT S1 set + fold assignment from devset_india_n20000.parquet so
the neural CV lines up with the GBDT OOF. For each dev S1 it unions the existing
lexical candidates with LaBSE dense-retrieval neighbours, labels every pair vs
ground truth, and enriches with RAW business_name/business_address (the fields
the cross-encoder reads). Writes devset_neural_india.parquet and re-audits the
recall ceiling (lexical-only vs union) — the honest test of whether dense
retrieval lifts recall above the 0.9458 lexical ceiling.

Run (GPU):  python make_devset_neural.py [--limit-s1 N] [--limit-corpus N]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd

import nconfig as nc
import common as c
import records as rc
import embed as em
import ann_faiss as ann

TRAIN_FILES = c.TRAIN_FILES
COUNTRY = "India"
OUT = f"{nc.NEURAL_WORK}/devset_neural_india.parquet"


def lexical_candidates(devset: pd.DataFrame) -> dict:
    """{s1_id: [lexical cand_id ...]} from the GBDT devset parquet, truncated to
    nc.LEX_TOPK. The parquet bakes up to 200 cheap-ranked candidates/S1, but
    inference pools only the top nc.LEX_TOPK (identical ranker/caps => a strict
    prefix), so the CV pool MUST be truncated identically or the reported
    macro-F0.5 would describe a wider pool than the submission ever builds."""
    out = {}
    for s1, sub in devset.groupby("s1_id", sort=False):
        out[s1] = sub["cand_id"].tolist()[:nc.LEX_TOPK]
    return out


def audit_ceiling(cands: dict, gt: dict, tag: str) -> None:
    """Recall-ceiling audit (macro recall + perfect-precision macro-F0.5)."""
    recalls, f05 = [], []
    cov = miss = 0
    for s1, truth in gt.items():
        if s1 not in cands:
            continue
        if not truth:
            f05.append(1.0)
            continue
        hit = len(set(cands[s1]) & truth)
        r = hit / len(truth)
        recalls.append(r); cov += hit; miss += len(truth) - hit
        f05.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
    if recalls:
        print(f"[audit:{tag}] macro-recall={np.mean(recalls):.4f} "
              f"micro-recall={cov/(cov+miss):.4f} "
              f"macro-F0.5-ceiling={np.mean(f05):.4f} "
              f"(nonsingleton={len(recalls):,})", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-s1", type=int, default=None,
                    help="cap dev S1 (CPU smoke test)")
    ap.add_argument("--limit-corpus", type=int, default=None,
                    help="cap S2+S3 corpus records (CPU smoke test)")
    ap.add_argument("--topm", type=int, default=nc.ANN_TOPM)
    args = ap.parse_args()

    t0 = time.time()
    devset = pd.read_parquet(nc.DEVSET_PARQUET)
    if args.limit_s1:
        keep = devset["s1_id"].drop_duplicates().head(args.limit_s1)
        devset = devset[devset["s1_id"].isin(set(keep))].reset_index(drop=True)
    lex = lexical_candidates(devset)
    fold_of = dict(zip(devset["s1_id"], devset["fold"]))
    dev_s1 = list(dict.fromkeys(devset["s1_id"].tolist()))
    print(f"[dev] {len(dev_s1):,} S1, {len(devset):,} lexical pairs", flush=True)

    gt = c.load_ground_truth()

    # --- raw text: S1 (queries) + S2/S3 (candidates) --------------------------
    s1df = rc.s1_records(TRAIN_FILES, COUNTRY)
    s1_txt = {r.entity_id: (r.business_name, r.business_address)
              for r in s1df.itertuples(index=False)}
    q_names = [nc.name_text(s1_txt.get(s, ("", ""))[0]) for s in dev_s1]

    # --- embed corpus + dense retrieve ---------------------------------------
    enc, tag = em.load_biencoder()
    print(f"[embed] {tag}", flush=True)
    if args.limit_corpus:
        cids, cnames, caddrs = [], [], []
        for rid, nm, ad in rc.iter_country_records(TRAIN_FILES, COUNTRY):
            cids.append(rid); cnames.append(nm); caddrs.append(ad)
            if len(cids) >= args.limit_corpus:
                break
        cvecs = em.embed_texts(enc, cnames)
        cid_txt = {i: (n, a) for i, n, a in zip(cids, cnames, caddrs)}
        cids = np.array(cids, dtype=object)
    else:
        prefix = f"{nc.NEURAL_WORK}/corpus_india"
        te = time.time()
        cids, cvecs = em.embed_corpus_to_memmap(enc, TRAIN_FILES, COUNTRY, prefix)
        print(f"[embed] corpus {len(cids):,} recs in {time.time()-te:.0f}s", flush=True)
        cid_txt = rc.build_id_text_map(TRAIN_FILES, COUNTRY)

    qvecs = em.embed_texts(enc, q_names)
    tr = time.time()
    dense = ann.retrieve(cids, cvecs, dev_s1, qvecs, topm=args.topm)
    print(f"[ann] retrieved top-{args.topm} for {len(dev_s1):,} S1 in "
          f"{time.time()-tr:.0f}s", flush=True)

    # corpus vectors are only needed for retrieval; drop the memmap + its on-disk
    # files. On Kaggle NEURAL_WORK is /kaggle/working/neural (the committed CV
    # output), and the India corpus memmap is ~10+ GB, so leaving it risks the
    # ~20 GB output cap and ships a useless file to the infer kernel.
    del cvecs
    if not args.limit_corpus:
        for _ext in (".vecs.f32", ".ids.npy"):
            try:
                os.remove(f"{nc.NEURAL_WORK}/corpus_india" + _ext)
            except OSError:
                pass

    # --- union, label, enrich -------------------------------------------------
    rows = []
    union = {}
    for s1 in dev_s1:
        lx = lex.get(s1, [])
        dn = dense.get(s1, [])
        lset = set(lx)
        merged = list(lx) + [d for d in dn if d not in lset]
        union[s1] = merged
        both = lset & set(dn)
        truth = gt.get(s1, set())
        s1n, s1a = s1_txt.get(s1, ("", ""))
        f = int(fold_of.get(s1, 0))
        for cid in merged:
            cn, ca = cid_txt.get(cid, ("", ""))
            src = ("both" if cid in both else ("lex" if cid in lset else "dense"))
            rows.append((s1, cid, 1 if cid in truth else 0, f, src,
                         s1n, s1a, cn, ca))

    df = pd.DataFrame(rows, columns=["s1_id", "cand_id", "label", "fold", "src",
                                     "s1_name", "s1_addr", "c_name", "c_addr"])
    df["label"] = df["label"].astype("int8")
    df["fold"] = df["fold"].astype("int8")
    df.to_parquet(OUT, index=False)

    npos = int(df["label"].sum())
    ndense_new = int((df["src"] == "dense").sum())
    print(f"[dev-neural] {len(df):,} pairs ({npos:,} pos), "
          f"{ndense_new:,} dense-only, avg {len(df)/len(dev_s1):.1f}/S1 -> {OUT}",
          flush=True)
    audit_ceiling(lex, {s: gt[s] for s in dev_s1 if s in gt}, "lexical")
    audit_ceiling(union, {s: gt[s] for s in dev_s1 if s in gt}, "union")
    print(f"[dev-neural] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

