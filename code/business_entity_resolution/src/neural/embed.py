#!/usr/bin/env python3
"""Bi-encoder embedding (LaBSE) for dense candidate retrieval.

LaBSE (sentence-transformers/LaBSE, Apache-2.0) is a 109-language sentence
encoder whose 768-d embeddings put transliterations / native-script variants of
the SAME name close in cosine space — exactly the recall the lexical blocking
misses. We embed the RAW business_name (see records.py for why).

Uses sentence-transformers (handles LaBSE's dense-pooler + L2 normalize
correctly); falls back to a transformers mean-pool encoder if s-t is absent.
Runs on GPU when available, CPU otherwise (small smoke tests).
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

import nconfig as nc


def load_biencoder(model_dir: str | None = None, device: str | None = None):
    """Load the bi-encoder. Returns (encode_fn, tag) where encode_fn(texts,bs)->np."""
    model_dir = model_dir or nc.BIENCODER_DIR
    device = device or nc.get_device()
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_dir, device=device)
        m.max_seq_length = nc.EMB_MAX_TOK

        def enc(texts: List[str], bs: int) -> np.ndarray:
            return m.encode(texts, batch_size=bs, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        return enc, f"sentence-transformers:{os.path.basename(model_dir)}"
    except Exception as e:  # pragma: no cover - fallback path
        return _load_transformers_meanpool(model_dir, device, e)


def _load_transformers_meanpool(model_dir, device, why):
    """Fallback encoder using plain transformers (no sentence-transformers dep).

    LaBSE's intended sentence vector is the dense-tanh pooler on the CLS token
    (== `pooler_output`); we prefer that when the model exposes it and fall back
    to attention-masked mean pooling otherwise. Both are L2-normalized.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir).to(device).eval()

    @torch.no_grad()
    def enc(texts: List[str], bs: int) -> np.ndarray:
        out = []
        for i in range(0, len(texts), bs):
            b = texts[i:i + bs]
            t = tok(b, padding=True, truncation=True, max_length=nc.EMB_MAX_TOK,
                    return_tensors="pt").to(device)
            o = model(**t)
            pooled = getattr(o, "pooler_output", None)
            if pooled is None:
                h = o.last_hidden_state
                mask = t["attention_mask"].unsqueeze(-1).float()
                pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            v = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(v.float().cpu().numpy().astype(np.float32))
        return np.concatenate(out) if out else np.zeros((0, nc.EMB_DIM), np.float32)
    return enc, f"transformers-pooler:{os.path.basename(model_dir)} (st fallback: {why})"


def embed_texts(enc, texts: List[str], batch: int | None = None) -> np.ndarray:
    """Embed a list of texts → (n, 768) float32, L2-normalized."""
    batch = batch or nc.EMB_BATCH
    texts = [nc.name_text(t) for t in texts]
    v = enc(texts, batch)
    return np.ascontiguousarray(v.astype(np.float32))


def embed_corpus_to_memmap(enc, files: dict, country: str, out_prefix: str,
                           roles=("S2", "S3"), batch: int | None = None):
    """Stream-embed a country's S2+S3 corpus to a float32 memmap + ids .npy.

    Returns (ids: np.ndarray[object], vecs: np.memmap (n,768)). Two passes: the
    first counts records (cheap streamed read) so the memmap is sized exactly.
    """
    import records as rc
    batch = batch or nc.EMB_BATCH
    n = sum(1 for _ in rc.iter_country_records(files, country, roles))
    vecs = np.memmap(out_prefix + ".vecs.f32", dtype=np.float32, mode="w+",
                     shape=(n, nc.EMB_DIM))
    ids = np.empty(n, dtype=object)
    pos = 0
    for bids, bnames, _ in rc.iter_batches(files, country, roles, batch=max(batch, 4096)):
        v = embed_texts(enc, bnames, batch)
        vecs[pos:pos + len(bids)] = v
        ids[pos:pos + len(bids)] = np.array(bids, dtype=object)
        pos += len(bids)
    vecs.flush()
    np.save(out_prefix + ".ids.npy", ids)
    return ids, vecs
