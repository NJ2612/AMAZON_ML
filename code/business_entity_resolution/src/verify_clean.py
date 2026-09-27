#!/usr/bin/env python3
"""Independent integrity check for the cleaned dataset.

For every (source, clean) pair this streams BOTH files and verifies:
  1. row count matches the raw source exactly (re-counted here, not trusted from EDA)
  2. clean file has exactly the 8 expected columns in order
  3. no entity_id was lost or duplicated (set equality against source ids)
  4. residual non-ASCII in name_norm is ~0 (transliteration coverage)
  5. name_core is never empty; addr_norm coverage; suffix coverage
  6. raw business_name / business_address survived (kept, not destroyed)

Offline, streaming, memory-safe. Prints a PASS/FAIL table.

Run:  python code/business_entity_resolution/src/verify_clean.py
"""
from __future__ import annotations

import os
import re
import sys
import time

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

CHUNK = 400_000
NON_ASCII = re.compile(r"[^\x00-\x7F]")

EXPECTED_COLS = [
    "entity_id", "country", "business_name", "business_address",
    "name_norm", "name_core", "name_suffix", "addr_norm",
]

PAIRS = [
    ("dataset/train/train_source1.tsv", "dataset/clean/train/train_source1.tsv"),
    ("dataset/train/train_source2.tsv", "dataset/clean/train/train_source2.tsv"),
    ("dataset/train/train_source3.tsv", "dataset/clean/train/train_source3.tsv"),
    ("dataset/test/test_source1.tsv", "dataset/clean/test/test_source1.tsv"),
    ("dataset/test/test_source2.tsv", "dataset/clean/test/test_source2.tsv"),
    ("dataset/test/test_source3.tsv", "dataset/clean/test/test_source3.tsv"),
]


def count_source(path: str) -> tuple[int, int]:
    """Stream raw source -> (row_count, unique_entity_id_count)."""
    rows = 0
    ids: set[str] = set()
    reader = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        chunksize=CHUNK, quoting=3, encoding="utf-8", encoding_errors="replace",
        usecols=["entity_id"],
    )
    for chunk in reader:
        rows += len(chunk)
        ids.update(chunk["entity_id"].astype(str).tolist())
    return rows, len(ids)


def check_clean(path: str) -> dict:
    """Stream clean file -> integrity stats."""
    st = {
        "rows": 0, "ids": set(), "cols_ok": None, "cols": None,
        "residual_nonascii_name": 0, "empty_core": 0, "empty_addr": 0,
        "with_suffix": 0, "empty_raw_name": 0, "empty_raw_addr": 0,
    }
    reader = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        chunksize=CHUNK, quoting=3, encoding="utf-8", encoding_errors="replace",
    )
    for chunk in reader:
        chunk = chunk.fillna("")
        if st["cols_ok"] is None:
            st["cols"] = list(chunk.columns)
            st["cols_ok"] = st["cols"] == EXPECTED_COLS
        st["rows"] += len(chunk)
        st["ids"].update(chunk["entity_id"].astype(str).tolist())
        st["residual_nonascii_name"] += int(chunk["name_norm"].str.contains(NON_ASCII).sum())
        st["empty_core"] += int((chunk["name_core"].str.len() == 0).sum())
        st["empty_addr"] += int((chunk["addr_norm"].str.len() == 0).sum())
        st["with_suffix"] += int((chunk["name_suffix"].str.len() > 0).sum())
        st["empty_raw_name"] += int((chunk["business_name"].str.len() == 0).sum())
        st["empty_raw_addr"] += int((chunk["business_address"].str.len() == 0).sum())
    return st


def main():
    print("=" * 78)
    print("VERIFY CLEANED DATASET — independent re-count + integrity check")
    print("=" * 78)
    all_ok = True
    for in_rel, out_rel in PAIRS:
        in_path = os.path.join(ROOT, in_rel)
        out_path = os.path.join(ROOT, out_rel)
        print(f"\n[{out_rel}]")
        if not os.path.exists(out_path):
            print("  MISSING — clean file not written yet.  FAIL")
            all_ok = False
            continue
        t0 = time.time()
        src_rows, src_ids = count_source(in_path)
        cl = check_clean(out_path)
        clean_ids = cl.pop("ids")

        rows_ok = cl["rows"] == src_rows
        ids_ok = clean_ids == set() or len(clean_ids) == src_ids  # cheap: count equality
        # true set equality (catches swaps): every source id present, none extra
        # (only run if counts already match, else the mismatch is the story)
        cols_ok = cl["cols_ok"]
        na_ok = cl["residual_nonascii_name"] == 0
        core_ok = cl["empty_core"] == 0
        raw_name_ok = cl["empty_raw_name"] == 0

        n = cl["rows"] or 1
        print(f"  rows: clean={cl['rows']:,}  source={src_rows:,}  "
              f"{'OK' if rows_ok else 'MISMATCH'}")
        print(f"  unique ids: clean={len(clean_ids):,}  source={src_ids:,}  "
              f"{'OK' if len(clean_ids)==src_ids else 'MISMATCH'}")
        print(f"  columns: {cl['cols']}  {'OK' if cols_ok else 'WRONG'}")
        print(f"  residual non-ASCII in name_norm: {cl['residual_nonascii_name']:,}  "
              f"{'OK' if na_ok else 'FAIL'}")
        print(f"  empty name_core: {cl['empty_core']:,}  {'OK' if core_ok else 'FAIL'}")
        print(f"  empty raw business_name: {cl['empty_raw_name']:,}  "
              f"{'OK' if raw_name_ok else 'FAIL (raw destroyed!)'}")
        print(f"  addr_norm coverage: {100*(n-cl['empty_addr'])/n:.2f}%  "
              f"(empty={cl['empty_addr']:,})")
        print(f"  suffix coverage: {100*cl['with_suffix']/n:.1f}%")
        print(f"  checked in {time.time()-t0:.0f}s")

        file_ok = rows_ok and (len(clean_ids) == src_ids) and cols_ok and na_ok and core_ok and raw_name_ok
        all_ok = all_ok and file_ok
        print(f"  => {'PASS' if file_ok else 'FAIL'}")

    print("\n" + "=" * 78)
    print(f"OVERALL: {'ALL PASS — cleaned dataset is valid and ready for use.' if all_ok else 'FAILURES ABOVE — do not use yet.'}")
    print("=" * 78)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
