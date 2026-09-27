#!/usr/bin/env python3
"""Dense ANN retrieval over the bi-encoder embeddings.

faiss (GPU if available) over L2-normalized vectors with inner-product metric
== cosine nearest neighbours. IVF-Flat for the large (~4M) corpora, exact Flat
for small ones. A pure-numpy chunked fallback keeps CPU smoke tests runnable
when faiss is absent (SLOW — smoke only).

Retrieval is UNIONED with the lexical candidates downstream; on its own it
raises the recall ceiling by surfacing native-script / transliteration matches
the token/metaphone blocking never keys on.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

import nconfig as nc


def _have_faiss():
    try:
        import faiss  # noqa: F401
        return True
    except Exception:
        return False


def build_index(vecs: np.ndarray, use_gpu: bool | None = None,
                nlist: int | None = None):
    """Build a cosine (IP-on-normalized) ANN index over `vecs` (n,768)."""
    import faiss
    n, d = vecs.shape
    nlist = nlist or nc.FAISS_NLIST
    # Large corpora stay on CPU: a ~4M x 768 float32 index is ~12.7 GB and would
    # crowd the reranker off the T4's 16 GB VRAM. CPU IVF search is fast enough.
    if use_gpu is None:
        use_gpu = (nc.get_device() == "cuda") and n < 2_000_000
    vecs = np.ascontiguousarray(vecs, dtype=np.float32)
    if n < 50_000:                                  # small corpus: exact flat
        index = faiss.IndexFlatIP(d)
    else:
        quant = faiss.IndexFlatIP(d)
        nlist = min(nlist, max(1, n // 39))         # faiss wants >=~39 pts/centroid
        index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
        train = vecs if n <= 500_000 else vecs[np.random.default_rng(0).choice(
            n, 500_000, replace=False)]
        index.train(train)
    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        except Exception:
            pass
    index.add(vecs)
    if hasattr(index, "nprobe"):
        index.nprobe = nc.FAISS_NPROBE
    return index


def search(index, qvecs: np.ndarray, topm: int):
    qvecs = np.ascontiguousarray(qvecs, dtype=np.float32)
    return index.search(qvecs, topm)               # (D, I)


def _numpy_search(corpus: np.ndarray, qvecs: np.ndarray, topm: int,
                  ctile: int = 100_000):
    """Chunked brute-force top-M (SMOKE ONLY — O(q*n) in BLAS tiles)."""
    nq = len(qvecs)
    best_s = np.full((nq, topm), -np.inf, dtype=np.float32)
    best_i = np.full((nq, topm), -1, dtype=np.int64)
    for c0 in range(0, len(corpus), ctile):
        blk = corpus[c0:c0 + ctile]
        s = qvecs @ blk.T                          # (nq, tile)
        cat_s = np.concatenate([best_s, s], axis=1)
        cat_i = np.concatenate([best_i, np.arange(c0, c0 + len(blk))[None, :]
                                .repeat(nq, 0)], axis=1)
        sel = np.argpartition(-cat_s, topm - 1, axis=1)[:, :topm]
        best_s = np.take_along_axis(cat_s, sel, 1)
        best_i = np.take_along_axis(cat_i, sel, 1)
    order = np.argsort(-best_s, axis=1)
    return np.take_along_axis(best_s, order, 1), np.take_along_axis(best_i, order, 1)


def _torch_gpu_search(corpus: np.ndarray, qvecs: np.ndarray, topm: int,
                      qtile: int = 4096, ctile: int = 200_000):
    """Exact top-M cosine on GPU, streaming CPU corpus tiles → GPU (fp16).

    Self-contained (no faiss): keeps only a small corpus tile + the query tile
    resident, so the 471M bi-encoder and 306M reranker can stay on the T4. fp16
    matmul over L2-normalized vecs is cosine to ample precision for ranking.
    """
    import torch
    dev = "cuda"
    n = len(corpus)
    topm = min(topm, n)
    all_s = np.empty((len(qvecs), topm), np.float32)
    all_i = np.empty((len(qvecs), topm), np.int64)
    for q0 in range(0, len(qvecs), qtile):
        q = torch.from_numpy(np.ascontiguousarray(qvecs[q0:q0 + qtile], np.float32)
                             ).to(dev).half()                      # (nq, d)
        nq = q.shape[0]
        best_s = torch.full((nq, topm), -1e4, device=dev, dtype=torch.float16)
        best_i = torch.full((nq, topm), -1, device=dev, dtype=torch.int32)
        for c0 in range(0, n, ctile):
            blk = torch.from_numpy(np.ascontiguousarray(corpus[c0:c0 + ctile],
                                   np.float32)).to(dev).half()      # (tile, d)
            s = q @ blk.T                                           # (nq, tile)
            # Local top-k of THIS tile first, then merge only (nq, topm+k) tensors.
            # Cat-ing the full (nq, ctile) score+index blocks would materialize a
            # ~6.5 GB contiguous int64 index per tile (topm+ctile wide) and OOM the
            # 16 GB T4; the local-top-k merge keeps transient well under 2 GB and is
            # exact (a global top-m element is in its own tile's top-m).
            k = min(topm, s.shape[1])
            tv, ti = torch.topk(s, k, dim=1)                        # (nq, k)
            gi = (ti + c0).to(torch.int32)                          # global corpus ids
            cat_s = torch.cat([best_s, tv], dim=1)
            cat_i = torch.cat([best_i, gi], dim=1)
            best_s, sel = torch.topk(cat_s, min(topm, cat_s.shape[1]), dim=1)
            best_i = torch.gather(cat_i, 1, sel)
            del blk, s, tv, ti, gi, cat_s, cat_i
        all_s[q0:q0 + nq] = best_s.float().cpu().numpy()
        all_i[q0:q0 + nq] = best_i.cpu().numpy()
        del q, best_s, best_i
        torch.cuda.empty_cache()
    return all_s, all_i


def retrieve(corpus_ids: np.ndarray, corpus_vecs: np.ndarray,
             query_ids: List[str], query_vecs: np.ndarray,
             topm: int | None = None) -> Dict[str, List[str]]:
    """{query_id: [candidate_id ...]} — top-M dense neighbours per query.

    GPU: exact torch brute force (no faiss dep). CPU: faiss if present, else a
    numpy fallback (smoke only). Retrieval is unioned with lexical downstream.
    """
    topm = topm or nc.ANN_TOPM
    if nc.get_device() == "cuda":
        _, I = _torch_gpu_search(np.asarray(corpus_vecs, np.float32), query_vecs, topm)
    elif _have_faiss():
        index = build_index(corpus_vecs)
        _, I = search(index, query_vecs, topm)
    else:
        print("[ann] cpu + no faiss — numpy brute force (SMOKE ONLY)", flush=True)
        _, I = _numpy_search(np.asarray(corpus_vecs, np.float32), query_vecs, topm)
    out: Dict[str, List[str]] = {}
    cid = np.asarray(corpus_ids, dtype=object)
    for qi, qid in enumerate(query_ids):
        out[qid] = [cid[j] for j in I[qi] if j >= 0]
    return out
