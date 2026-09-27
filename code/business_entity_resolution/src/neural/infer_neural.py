#!/usr/bin/env python3
"""Full-test inference for the neural pipeline → the two submission TSVs.

Per country: embed S1 + S2/S3 RAW names (LaBSE), dense-retrieve top-M, UNION
with the lexical blocking candidates, bound the pool per S1, rerank every pooled
pair with the fine-tuned cross-encoder, threshold at the frozen t*, and stream
both rows. Writes into output/kaggle_results_2/ (distinct from the GBDT output).

Every test S1 gets exactly one row (empty cell = predicted singleton). This ships
the reranker-alone decision; the CV also measures a reranker+GBDT ensemble, but
inference keeps the single-model path (the genuinely-different deliverable).

Run (GPU):  python infer_neural.py [--limit N] [--countries India,US]
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
import blocking as bl
import records as rc
import embed as em
import ann_faiss as ann
import train_reranker as tr

INFER_LEX_TOPK = int(os.environ.get("BER_INFER_LEX_TOPK", str(nc.LEX_TOPK)))
RERANKER_FINAL = os.environ.get("BER_RERANKER_FINAL",
                                f"{nc.NEURAL_WORK}/reranker_final")
DEPLOY_JSON = os.environ.get("BER_DEPLOY_JSON", f"{nc.NEURAL_WORK}/deploy_neural.json")
BUILD_CAP = 3000                      # blocking df_cap (matches make_devset / infer)


def load_deploy() -> dict:
    if os.path.exists(DEPLOY_JSON):
        with open(DEPLOY_JSON, encoding="utf-8") as f:
            return json.load(f)
    print(f"[infer] WARN no {DEPLOY_JSON}; using t*=0.5", flush=True)
    return {"reranker_threshold": 0.5, "threshold": 0.5, "decision": "reranker"}


def load_country_s1(country: str) -> pd.DataFrame:
    """Test S1 of one country with BOTH raw (rerank/embed) and romanized (lexical)."""
    cols = ["entity_id", "country", "business_name", "business_address",
            "name_core", "addr_norm"]
    parts = []
    for chunk in c.iter_clean(c.TEST_FILES["S1"], usecols=cols):
        sub = chunk[chunk["country"].values == country]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)

def pooled_candidates(ci, s1df, dense, topk_lex):
    """{s1_id: [cand_id...]} — lexical top-K unioned with dense top-M, deduped."""
    pool = {}
    for r in s1df.itertuples(index=False):
        lex = bl.candidates_for(ci, r.name_core, r.addr_norm, topk_lex,
                                eff_cap=BUILD_CAP, detail_cap=0)
        lset = set(lex)
        dn = [d for d in dense.get(r.entity_id, []) if d not in lset]
        pool[r.entity_id] = lex + dn
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap S1 per country (smoke)")
    ap.add_argument("--countries", default=None)
    ap.add_argument("--topm", type=int, default=nc.ANN_TOPM)
    ap.add_argument("--lex-topk", type=int, default=INFER_LEX_TOPK)
    ap.add_argument("--batch-s1", type=int, default=2000)
    args = ap.parse_args()
    device = nc.get_device()

    deploy = load_deploy()
    t_star = float(deploy.get("reranker_threshold", deploy.get("threshold", 0.5)))
    model, tok = tr.load_reranker(RERANKER_FINAL, device)
    enc, tag = em.load_biencoder()
    print(f"[infer] device={device} reranker={RERANKER_FINAL} t*={t_star:.3f} "
          f"| {tag}", flush=True)

    countries = (args.countries.split(",") if args.countries
                 else bl.distinct_countries(c.TEST_FILES))
    print(f"[infer] countries={countries}", flush=True)

    match_path = os.path.join(nc.NEURAL_OUT, "matching_results.tsv")
    cand_path = os.path.join(nc.NEURAL_OUT, "candidate_pairs.tsv")
    t_all = time.time()
    n_s1 = n_pairs = n_match = n_nonempty = 0

    with open(match_path, "w", encoding="utf-8", newline="\n") as mf, \
         open(cand_path, "w", encoding="utf-8", newline="\n") as cf:
        mf.write("source1_entity_id\tmatched_entity_ids\n")
        cf.write("source1_entity_id\tcandidate_entity_ids\n")

        for country in countries:
            t0 = time.time()
            ci = bl.load_country_index(c.TEST_FILES, country, df_cap=BUILD_CAP)
            cid_txt = rc.build_id_text_map(c.TEST_FILES, country)
            prefix = f"{nc.NEURAL_WORK}/test_{country}"
            cids, cvecs = em.embed_corpus_to_memmap(enc, c.TEST_FILES, country, prefix)
            s1df = load_country_s1(country)
            if args.limit:
                s1df = s1df.head(args.limit)
            qvecs = em.embed_texts(enc, s1df["business_name"].tolist())
            dense = ann.retrieve(cids, cvecs, s1df["entity_id"].tolist(), qvecs,
                                 topm=args.topm)
            pool = pooled_candidates(ci, s1df, dense, args.lex_topk)
            print(f"  [{country}] {len(s1df):,} S1, {ci.n_records():,} corpus, "
                  f"index+embed+retrieve {time.time()-t0:.0f}s", flush=True)

            # --- batched rerank + threshold + stream write --------------------
            s1raw = {r.entity_id: (r.business_name, r.business_address)
                     for r in s1df.itertuples(index=False)}
            order = s1df["entity_id"].tolist()
            for b0 in range(0, len(order), args.batch_s1):
                bs1 = order[b0:b0 + args.batch_s1]
                rows, spans = [], []
                for s1 in bs1:
                    cids_s1 = pool.get(s1, [])
                    s1n, s1a = s1raw.get(s1, ("", ""))
                    spans.append((s1, len(cids_s1)))
                    for cid in cids_s1:
                        cn, ca = cid_txt.get(cid, ("", ""))
                        rows.append((s1n, s1a, cn, ca))
                if rows:
                    pdf = pd.DataFrame(rows, columns=["s1_name", "s1_addr",
                                                      "c_name", "c_addr"])
                    prob = tr.score_df(model, tok, pdf, device=device)
                else:
                    prob = np.empty(0, dtype=np.float32)
                off = 0
                for s1, k in spans:
                    cids_s1 = pool.get(s1, [])
                    p = prob[off:off + k]; off += k
                    kept = [cid for cid, pp in zip(cids_s1, p) if pp >= t_star]
                    if kept:
                        n_nonempty += 1; n_match += len(kept)
                    n_pairs += k
                    mf.write(f"{s1}\t{','.join(kept)}\n")
                    cf.write(f"{s1}\t{','.join(cids_s1)}\n")
                n_s1 += len(bs1)
            del ci, cid_txt, cvecs, cids
            # Drop this country's on-disk corpus memmaps. NEURAL_WORK lives under
            # /kaggle/working (the committed output dir), so leaving them would
            # accumulate ~30 GB across countries and blow the ~20 GB output cap.
            for _ext in (".vecs.f32", ".ids.npy"):
                try:
                    os.remove(prefix + _ext)
                except OSError:
                    pass
            print(f"  [{country}] DONE in {time.time()-t0:.0f}s "
                  f"(cum {n_s1:,} S1, {n_match:,} matches)", flush=True)

    print(f"[infer] {n_s1:,} S1 | {n_pairs:,} reranked pairs | {n_match:,} matches "
          f"| {n_nonempty:,} non-empty | {time.time()-t_all:.0f}s", flush=True)
    print(f"[infer] -> {match_path}\n[infer] -> {cand_path}", flush=True)


if __name__ == "__main__":
    main()

