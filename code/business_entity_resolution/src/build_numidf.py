#!/usr/bin/env python3
"""Build a global numeric-address-token IDF (India corpus) for features v2.

The strongest address signal in the positive pairs is a shared *rare* numeric
token — a building / flat / PIN number like "2505" that both records carry. Plain
overlap count (a_num_ov) treats a shared "1" like a shared "700053"; weighting by
inverse document frequency lets the model reward the rare, discriminative match
and ignore the ubiquitous one.

Reuses the cached India blocking index (its .addrs already hold addr_norm for all
S2+S3 India records) so we don't re-stream the corpus. Emits numidf_india.pkl:
{ numeric_token -> idf } consumed by features.compute_chunk.

Run:  python build_numidf.py
"""
import os
import pickle
import time
from collections import Counter

import numpy as np

import common as c

CACHE = os.path.join(c.WORK, "idx_india_cap3000.pkl")
OUT = os.path.join(c.WORK, "numidf_india.pkl")


def main():
    t0 = time.time()
    with open(CACHE, "rb") as f:
        ci = pickle.load(f)
    n = len(ci.addrs)
    df = Counter()
    for addr in ci.addrs:
        for t in c.numeric_tokens(str(addr)):
            df[t] += 1
    idf = {t: float(np.log(n / d)) for t, d in df.items()}
    with open(OUT, "wb") as f:
        pickle.dump(idf, f, protocol=4)
    print(f"numeric idf: {len(idf):,} tokens over {n:,} addrs "
          f"in {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
