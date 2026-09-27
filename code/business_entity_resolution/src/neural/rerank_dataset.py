#!/usr/bin/env python3
"""Torch dataset / collation for the cross-encoder reranker.

Stores only the raw text (memory-light for millions of pairs) and tokenizes a
batch at a time in the collate_fn. Each example is the (S1, candidate) pair fed
to the cross-encoder as a text pair — query = S1 'name | address', passage =
candidate 'name | address' (raw native script; see nconfig.pair_text).
"""
from __future__ import annotations

from typing import List

import numpy as np

import nconfig as nc


class PairDataset:
    """Holds query/passage text + optional labels; tokenized lazily by collate."""

    def __init__(self, s1_name, s1_addr, c_name, c_addr, labels=None):
        self.q = [nc.pair_text(n, a) for n, a in zip(s1_name, s1_addr)]
        self.p = [nc.pair_text(n, a) for n, a in zip(c_name, c_addr)]
        self.y = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.q)


def make_collate(tokenizer, max_tok: int | None = None):
    """Return a collate_fn tokenizing a list of (idx, dataset) into model inputs."""
    import torch
    max_tok = max_tok or nc.RERANK_MAXTOK

    def collate(batch):
        q = [b[0] for b in batch]
        p = [b[1] for b in batch]
        enc = tokenizer(q, p, padding=True, truncation=True, max_length=max_tok,
                        return_tensors="pt")
        if batch[0][2] is None:
            return enc, None
        y = torch.tensor([b[2] for b in batch], dtype=torch.float32)
        return enc, y
    return collate


def iter_minibatches(ds: PairDataset, order: np.ndarray, batch: int):
    """Yield lists of (query, passage, label) tuples in `order`, `batch` at a time."""
    for i in range(0, len(order), batch):
        idx = order[i:i + batch]
        yield [(ds.q[j], ds.p[j], None if ds.y is None else float(ds.y[j]))
               for j in idx]
