#!/usr/bin/env python3
"""High-level EDA for the Amazon ML Business Entity Resolution challenge.

Runs memory-safe *streaming* (chunked) passes over all 7 TSV files and emits:
  - output/eda/eda_report.txt      full text profile
  - output/eda/*.png               summary plots
The numbers here drive the preprocessing plan. No external data is used
(stdlib + pandas + matplotlib only), per challenge rules.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Windows consoles default to cp1252 and choke on native-script text; force
# UTF-8 so streaming progress lines never crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DATA = os.path.join(ROOT, "dataset")
OUT = os.path.join(ROOT, "output", "eda")
os.makedirs(OUT, exist_ok=True)

CHUNK = 400_000
NON_ASCII = re.compile(r"[^\x00-\x7F]")
TOKEN = re.compile(r"[a-z0-9]+")

# Legal-suffix / boilerplate tokens we explicitly track the frequency of.
LEGAL_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "pvt", "private", "llp", "plc", "pc", "lp", "pllc",
    "opc", "nidhi", "group", "holdings", "services", "trust", "and",
    "sarl", "sas", "sasu", "sa", "eurl", "sci", "snc", "sca", "association",
}
SUFFIX_RE = re.compile(r"\b(" + "|".join(sorted(LEGAL_SUFFIXES, key=len, reverse=True)) + r")\b")
# Single pass to pull ALL suffix tokens out of a name at once (fast: findall + explode).
SUFFIX_FIND = SUFFIX_RE



SOURCES = [
    ("train_source1", "dataset/train/train_source1.tsv"),
    ("train_source2", "dataset/train/train_source2.tsv"),
    ("train_source3", "dataset/train/train_source3.tsv"),
    ("test_source1", "dataset/test/test_source1.tsv"),
    ("test_source2", "dataset/test/test_source2.tsv"),
    ("test_source3", "dataset/test/test_source3.tsv"),
]

report_lines: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    report_lines.append(msg)


def norm_key_series(name: pd.Series) -> pd.Series:
    """Vectorized normalized name key: lowercased, legal suffixes + non-alnum
    stripped, whitespace collapsed. Used to detect the 'generic name' trap."""
    s = name.str.lower()
    s = s.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    s = s.str.replace(SUFFIX_RE, " ", regex=True)
    s = s.str.replace(r"\s+", "", regex=True)
    return s


def profile_source(label: str, rel_path: str, sample_ids: set[str]):
    """One streaming pass over a source file -> aggregate stats dict."""
    path = os.path.join(ROOT, rel_path)
    is_ref = label.endswith("1")
    st = {
        "rows": 0, "size": os.path.getsize(path),
        "country": Counter(), "prefix": Counter(),
        "empty_name": 0, "empty_addr": 0,
        "name_len_sum": 0, "name_len_min": 10**9, "name_len_max": 0,
        "addr_len_sum": 0, "addr_len_min": 10**9, "addr_len_max": 0,
        "nonascii_name": 0, "nonascii_addr": 0,
        "name_digit": 0, "name_amp": 0,
        "suffix": Counter(),
        "key_hashes": set(),                 # uniqueness ratio (hashes save memory)
        "dup_probe": Counter() if is_ref else None,  # top generic names, S1 only
        "captured": [],                      # rows whose id is in sample_ids
    }
    reader = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        chunksize=CHUNK, quoting=3, encoding="utf-8", encoding_errors="replace",
    )
    for chunk in reader:
        chunk = chunk.fillna("")
        name = chunk["business_name"].astype(str)
        addr = chunk["business_address"].astype(str)
        eid = chunk["entity_id"].astype(str)

        st["rows"] += len(chunk)
        st["country"].update(chunk["country"].astype(str).tolist())
        st["prefix"].update(eid.str.slice(0, 3).tolist())

        nlen, alen = name.str.len(), addr.str.len()
        st["empty_name"] += int((nlen == 0).sum())
        st["empty_addr"] += int((alen == 0).sum())
        st["name_len_sum"] += int(nlen.sum())
        st["addr_len_sum"] += int(alen.sum())
        st["name_len_min"] = min(st["name_len_min"], int(nlen.min()))
        st["name_len_max"] = max(st["name_len_max"], int(nlen.max()))
        st["addr_len_min"] = min(st["addr_len_min"], int(alen.min()))
        st["addr_len_max"] = max(st["addr_len_max"], int(alen.max()))

        st["nonascii_name"] += int(name.str.contains(NON_ASCII).sum())
        st["nonascii_addr"] += int(addr.str.contains(NON_ASCII).sum())
        st["name_digit"] += int(name.str.contains(r"\d", regex=True).sum())
        st["name_amp"] += int(name.str.contains("&", regex=False).sum())

        # suffix frequency: ONE findall pass, then explode + value_counts (fast in C)
        found = name.str.lower().str.findall(SUFFIX_FIND).explode()
        vc = found.value_counts()
        for tok, cnt in vc.items():
            if isinstance(tok, str):
                st["suffix"][tok] += int(cnt)

        keys = norm_key_series(name)
        hs = pd.util.hash_pandas_object(keys, index=False)
        st["key_hashes"].update(hs.tolist())
        if is_ref:
            st["dup_probe"].update(keys[keys.str.len() > 0].tolist())

        if sample_ids:
            hit = chunk[eid.isin(sample_ids)]
            for _, r in hit.iterrows():
                st["captured"].append(
                    (r["entity_id"], r["business_name"], r["business_address"], r["country"])
                )
        print(f"  [{label}] {st['rows']:,} rows...", flush=True)

    st["uniq"] = len(st["key_hashes"])   # keep the count; drop the big set to free memory
    del st["key_hashes"]
    return st



def profile_ground_truth(rel_path: str, n_examples: int = 12):
    """Stream ground truth -> match-count distribution + a set of example ids."""
    path = os.path.join(ROOT, rel_path)
    gt = {
        "rows": 0, "singletons": 0, "matched_entities": 0,
        "total_links": 0, "links_s2": 0, "links_s3": 0, "links_other": 0,
        "dup_in_list": 0, "max_matches": 0, "hist": Counter(),
    }
    example_map: dict[str, list[str]] = {}   # s1_id -> matched ids (small sample)
    want_s1: set[str] = set()
    want_matches: set[str] = set()

    reader = pd.read_csv(
        path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        chunksize=CHUNK, quoting=3, encoding="utf-8", encoding_errors="replace",
    )
    for chunk in reader:
        chunk = chunk.fillna("")
        for s1id, matched in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            gt["rows"] += 1
            ids = [x for x in str(matched).split(",") if x]
            n = len(ids)
            gt["hist"][n] += 1
            gt["max_matches"] = max(gt["max_matches"], n)
            if n == 0:
                gt["singletons"] += 1
            else:
                gt["matched_entities"] += 1
                gt["total_links"] += n
                if len(set(ids)) != n:
                    gt["dup_in_list"] += 1
                for x in ids:
                    p = x[:3]
                    if p == "S2-":
                        gt["links_s2"] += 1
                    elif p == "S3-":
                        gt["links_s3"] += 1
                    else:
                        gt["links_other"] += 1
                # capture a handful of multi-source examples for noise inspection
                if len(example_map) < n_examples and 3 <= n <= 6:
                    example_map[s1id] = ids
                    want_s1.add(s1id)
                    want_matches.update(ids)
        print(f"  [ground_truth] {gt['rows']:,} rows...", flush=True)
    return gt, example_map, want_s1, want_matches


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:.2f}%" if d else "n/a"


def make_plots(stats: dict, gt: dict):
    order = [s[0] for s in SOURCES]

    # 1) Country distribution per source (stacked)
    countries = ["US", "India", "France"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(order)
    for c in countries:
        vals = [stats[s]["country"].get(c, 0) for s in order]
        ax.bar(order, vals, bottom=bottom, label=c)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Country distribution per source (France is test-only)")
    ax.set_ylabel("rows")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "country_distribution.png"), dpi=110)
    plt.close()

    # 2) Match-count histogram
    fig, ax = plt.subplots(figsize=(9, 5))
    ks = sorted(gt["hist"])
    ax.bar([str(k) for k in ks], [gt["hist"][k] for k in ks], color="#4C78A8")
    ax.set_title("Matches per Source-1 entity (0 = singleton)")
    ax.set_xlabel("number of matched IDs")
    ax.set_ylabel("Source-1 entities")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "match_count_histogram.png"), dpi=110)
    plt.close()

    # 3) Native-script (non-ASCII) name rate per source
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = [100.0 * stats[s]["nonascii_name"] / stats[s]["rows"] for s in order]
    ax.bar(order, rates, color="#E45756")
    ax.set_title("Non-ASCII (native-script) business-name rate per source")
    ax.set_ylabel("% of rows")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "nonascii_name_rate.png"), dpi=110)
    plt.close()

    # 4) Missing-address rate per source
    fig, ax = plt.subplots(figsize=(10, 5))
    rates = [100.0 * stats[s]["empty_addr"] / stats[s]["rows"] for s in order]
    ax.bar(order, rates, color="#72B7B2")
    ax.set_title("Empty business-address rate per source")
    ax.set_ylabel("% of rows")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "missing_address_rate.png"), dpi=110)
    plt.close()


def main():
    log("=" * 70)
    log("HIGH-LEVEL EDA — Amazon ML Business Entity Resolution")
    log("Streaming passes over all 7 TSV files. No external data used.")
    log("=" * 70)

    log("\n### GROUND TRUTH ###")
    gt, example_map, want_s1, want_matches = profile_ground_truth(
        "dataset/train/train_ground_truth.tsv"
    )

    stats: dict[str, dict] = {}
    for label, rel in SOURCES:
        log(f"\n### {label} ###")
        sample = want_s1 if label.endswith("source1") else want_matches
        st = profile_source(label, rel, sample)
        stats[label] = st
        n = st["rows"]
        log(f"  path={rel}  size={st['size']:,} bytes  rows={n:,}")
        log(f"  entity_id prefixes: {dict(st['prefix'])}")
        log(f"  country: {dict(st['country'])}")
        log(f"  empty_name={st['empty_name']} ({pct(st['empty_name'], n)})   "
            f"empty_address={st['empty_addr']} ({pct(st['empty_addr'], n)})")
        log(f"  name_len  min={st['name_len_min']} max={st['name_len_max']} "
            f"avg={st['name_len_sum']/n:.1f}")
        log(f"  addr_len  min={st['addr_len_min']} max={st['addr_len_max']} "
            f"avg={st['addr_len_sum']/n:.1f}")
        log(f"  non-ASCII names={st['nonascii_name']} ({pct(st['nonascii_name'], n)})   "
            f"non-ASCII addrs={st['nonascii_addr']} ({pct(st['nonascii_addr'], n)})")
        log(f"  names with digit={st['name_digit']} ({pct(st['name_digit'], n)})   "
            f"names with '&'={st['name_amp']}")
        uniq = st["uniq"]
        log(f"  normalized-name uniqueness: unique_keys={uniq:,} of {n:,} "
            f"-> collision rows={n-uniq:,} ({pct(n-uniq, n)})")
        top_suf = ", ".join(f"{k}={v:,}" for k, v in st["suffix"].most_common(12))
        log(f"  top legal/suffix tokens: {top_suf}")
        if st["dup_probe"] is not None:
            top = st["dup_probe"].most_common(6)
            log(f"  most repeated normalized-name keys (S1, likely GENERIC/distinct): {top}")

    # ground-truth summary
    log("\n### GROUND TRUTH SUMMARY ###")
    log(f"  rows (S1 entities)={gt['rows']:,}")
    log(f"  singletons (0 matches)={gt['singletons']:,} ({pct(gt['singletons'], gt['rows'])})")
    log(f"  entities with >=1 match={gt['matched_entities']:,} "
        f"({pct(gt['matched_entities'], gt['rows'])})")
    log(f"  total match links={gt['total_links']:,}   to S2={gt['links_s2']:,}   "
        f"to S3={gt['links_s3']:,}   other={gt['links_other']:,}")
    if gt["matched_entities"]:
        log(f"  avg matches per matched entity="
            f"{gt['total_links']/gt['matched_entities']:.2f}   max={gt['max_matches']}")
    log(f"  rows with duplicate ids inside list={gt['dup_in_list']}")
    log(f"  match-count histogram: "
        f"{dict(sorted(gt['hist'].items()))}")

    # matched-pair noise examples (S1 record vs its matched S2/S3 records)
    log("\n### MATCHED-PAIR NOISE EXAMPLES ###")
    lut: dict[str, tuple] = {}
    for label in ("train_source1", "train_source2", "train_source3"):
        for eid, nm, ad, co in stats[label]["captured"]:
            lut[eid] = (nm, ad, co)
    for s1id, ids in example_map.items():
        if s1id not in lut:
            continue
        nm, ad, co = lut[s1id]
        log(f"\n  S1 {s1id} [{co}]  name='{nm}'  addr='{ad}'")
        for mid in ids:
            if mid in lut:
                mnm, mad, mco = lut[mid]
                log(f"     -> {mid} [{mco}]  name='{mnm}'  addr='{mad}'")

    make_plots(stats, gt)
    report_path = os.path.join(OUT, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")
    log(f"\nReport written to {report_path}")
    log(f"Plots written to {OUT}")


if __name__ == "__main__":
    main()




