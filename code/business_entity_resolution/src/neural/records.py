#!/usr/bin/env python3
"""Raw-record access for the neural pipeline.

The GBDT side works off the romanized `name_core`; the bi-encoder MUST see the
RAW `business_name` (native script preserved — unidecode turns "राम मार्केटिंग"
into "raam maarketting", destroying the cross-lingual signal LaBSE exists to
exploit). These helpers stream the cleaned TSVs pulling the RAW name/address.
"""
from __future__ import annotations

from typing import Dict, Iterator, List, Tuple

import pandas as pd

import nconfig as nc  # noqa: F401  (ensures src/ on path)
import common as c

RAW_COLS = ["entity_id", "country", "business_name", "business_address"]


def iter_country_records(files: dict, country: str,
                         roles=("S2", "S3")) -> Iterator[Tuple[str, str, str]]:
    """Yield (entity_id, business_name, business_address) for one country."""
    for role in roles:
        for chunk in c.iter_clean(files[role], usecols=RAW_COLS):
            sub = chunk[chunk["country"].values == country]
            if sub.empty:
                continue
            for rid, nm, ad in zip(sub["entity_id"], sub["business_name"],
                                   sub["business_address"]):
                yield rid, nm, ad


def iter_batches(files: dict, country: str, roles=("S2", "S3"),
                 batch: int = 4096) -> Iterator[Tuple[List[str], List[str], List[str]]]:
    """Stream (ids, names, addrs) in batches — for the bi-encoder encode loop."""
    ids, names, addrs = [], [], []
    for rid, nm, ad in iter_country_records(files, country, roles):
        ids.append(rid); names.append(nm); addrs.append(ad)
        if len(ids) >= batch:
            yield ids, names, addrs
            ids, names, addrs = [], [], []
    if ids:
        yield ids, names, addrs


def build_id_text_map(files: dict, country: str,
                      roles=("S2", "S3")) -> Dict[str, Tuple[str, str]]:
    """{entity_id: (business_name, business_address)} for candidate enrichment."""
    m: Dict[str, Tuple[str, str]] = {}
    for rid, nm, ad in iter_country_records(files, country, roles):
        m[rid] = (nm, ad)
    return m


def s1_records(files: dict, country: str) -> pd.DataFrame:
    """All S1 rows of one country with RAW name/address (small — fits in RAM)."""
    parts = []
    for chunk in c.iter_clean(files["S1"], usecols=RAW_COLS):
        sub = chunk[chunk["country"].values == country]
        if not sub.empty:
            parts.append(sub[["entity_id", "business_name", "business_address"]])
    if not parts:
        return pd.DataFrame(columns=["entity_id", "business_name", "business_address"])
    return pd.concat(parts, ignore_index=True)
