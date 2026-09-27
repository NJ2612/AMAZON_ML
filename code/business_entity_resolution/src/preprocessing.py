#!/usr/bin/env python3
"""Preprocessing / data cleaning for the Business Entity Resolution challenge.

Streaming (chunked) cleaner over all 6 source files. For every row it produces
normalized, transliterated, comparison-ready fields WITHOUT discarding the raw
text, and writes enriched TSVs to dataset/clean/.  Offline only (unidecode's
static table + regex); no external data lookup, per challenge rules.

Output columns:
    entity_id, country, business_name, business_address,
    name_norm, name_core, name_suffix, addr_norm

Run:  python code/business_entity_resolution/src/preprocessing.py
"""
from __future__ import annotations

import os
import re
import sys
import time

import pandas as pd
from unidecode import unidecode

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(ROOT, "dataset", "clean")

CHUNK = 200_000
NON_ASCII = re.compile(r"[^\x00-\x7F]")

SOURCES = [
    ("dataset/train/train_source1.tsv", "dataset/clean/train/train_source1.tsv"),
    ("dataset/train/train_source2.tsv", "dataset/clean/train/train_source2.tsv"),
    ("dataset/train/train_source3.tsv", "dataset/clean/train/train_source3.tsv"),
    ("dataset/test/test_source1.tsv", "dataset/clean/test/test_source1.tsv"),
    ("dataset/test/test_source2.tsv", "dataset/clean/test/test_source2.tsv"),
    ("dataset/test/test_source3.tsv", "dataset/clean/test/test_source3.tsv"),
]

# --- Legal suffixes: variant -> canonical token (order-agnostic, dotless) -----
LEGAL_SUFFIX_CANON = {
    "llc": "llc", "l.l.c": "llc", "llc.": "llc",
    "inc": "inc", "incorporated": "inc", "inc.": "inc",
    "corp": "corp", "corporation": "corp", "corp.": "corp",
    "co": "co", "company": "co", "co.": "co",
    "ltd": "ltd", "limited": "ltd", "ltd.": "ltd",
    "pvt": "pvt", "private": "pvt", "pvt.": "pvt",
    "llp": "llp", "l.l.p": "llp",
    "plc": "plc", "pllc": "pllc", "pc": "pc", "pa": "pa",
    "lp": "lp", "opc": "opc", "nidhi": "nidhi",
    # French
    "sarl": "sarl", "sasu": "sasu", "sas": "sas", "sa": "sa",
    "eurl": "eurl", "sci": "sci", "snc": "snc", "sca": "sca",
    "sarlu": "sarl", "association": "association", "gie": "gie",
}
# Multi-word phrases collapsed before token scan.
SUFFIX_PHRASES = [
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\.?\s*ltd\.?\b", "pvt ltd"),
    (r"\bpublic\s+limited\b", "ltd"),
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\bone\s+person\s+company\b", "opc"),
]
SUFFIX_TOKENS = set(LEGAL_SUFFIX_CANON.values()) | set(LEGAL_SUFFIX_CANON.keys())

# --- Street-type abbreviations (US + a few FR) --------------------------------
STREET_ABBR = {
    "st": "street", "st.": "street", "str": "street",
    "rd": "road", "rd.": "road", "ave": "avenue", "ave.": "avenue",
    "av": "avenue", "blvd": "boulevard", "blvd.": "boulevard", "bd": "boulevard",
    "dr": "drive", "dr.": "drive", "ln": "lane", "ct": "court",
    "pl": "place", "cir": "circle", "pkwy": "parkway", "hwy": "highway",
    "sq": "square", "ter": "terrace", "trl": "trail", "expy": "expressway",
    "hno": "house no", "h.no": "house no",
}

# --- State / region: code -> canonical full name, kept per-country to avoid
#     the US/India 2-letter collisions (ca, co, in, mp, or, ...). Applied only
#     to a comma segment that is EXACTLY a valid code for the row's country, so
#     we never rewrite a real word. -------------------------------------------
US_STATE = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas",
    "ca": "california", "co": "colorado", "ct": "connecticut", "de": "delaware",
    "fl": "florida", "ga": "georgia", "hi": "hawaii", "id": "idaho",
    "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada",
    "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico", "ny": "new york",
    "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah",
    "vt": "vermont", "va": "virginia", "wa": "washington", "wv": "west virginia",
    "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}
IN_STATE = {
    "ap": "andhra pradesh", "as": "assam", "br": "bihar", "cg": "chhattisgarh",
    "ga": "goa", "gj": "gujarat", "hr": "haryana", "hp": "himachal pradesh",
    "jh": "jharkhand", "ka": "karnataka", "kl": "kerala", "mp": "madhya pradesh",
    "mh": "maharashtra", "ml": "meghalaya", "mz": "mizoram", "nl": "nagaland",
    "od": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "tg": "telangana", "ts": "telangana", "tr": "tripura",
    "up": "uttar pradesh", "uk": "uttarakhand", "wb": "west bengal", "dl": "delhi",
    "jk": "jammu and kashmir",
}
STATE_BY_COUNTRY = {"US": US_STATE, "India": IN_STATE}

# Landmark / noise phrases stripped from addresses (do not help matching).
LANDMARK_RE = re.compile(
    r"\b(near|nr|opp|opposite|behind|beside|next to|adjacent to|above|below)\b[^,]*",
    re.IGNORECASE,
)

# --- Regex building blocks ----------------------------------------------------
# Strip protocol/www and the trailing TLD only, so "heassociates.com" keeps the
# matchable token "heassociates" instead of vanishing.
WEB_PREFIX_RE = re.compile(r"\b(?:https?://|www\.)")
TLD_RE = re.compile(r"\.(?:com|net|org|in|co|io|fr|biz|info)\b")
NULL_RE = re.compile(r"<?\bnull\b>?", re.IGNORECASE)
JUNK_CHARS_RE = re.compile(r"[\[\]{}()<>#*\"|~^`_\\]+")
DASH_RUN_RE = re.compile(r"(?:^|\s)[-–—]{1,}(?=\s|$)")
PHONE_RE = re.compile(r"\b\d{7,}\b")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9 ]+")


def _translit(s: str) -> str:
    """Romanize only when needed (non-ASCII); unidecode is the offline table."""
    if s and NON_ASCII.search(s):
        return unidecode(s)
    return s


def normalize_name(raw: str) -> tuple[str, str, str]:
    """raw business_name -> (name_norm, name_core, name_suffix).

    name_norm   : transliterated, lowercased, de-junked, '&'->'and', ws-collapsed
    name_suffix : sorted canonical legal-suffix tokens found (space-joined)
    name_core   : name_norm with those suffix tokens removed (for name comparison)
    """
    s = _translit(raw).lower()
    s = WEB_PREFIX_RE.sub(" ", s)
    s = TLD_RE.sub(" ", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"\bd\.?b\.?a\.?\b", " ", s)          # drop DBA marker
    s = JUNK_CHARS_RE.sub(" ", s)
    s = PHONE_RE.sub(" ", s)
    for pat, repl in SUFFIX_PHRASES:
        s = re.sub(pat, repl, s)
    s = NON_ALNUM_SPACE_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()

    toks = s.split()
    suffixes, core = [], []
    for t in toks:
        canon = LEGAL_SUFFIX_CANON.get(t)
        if canon is not None:
            suffixes.append(canon)
        else:
            core.append(t)
    name_core = " ".join(core).strip() or s     # never leave core empty
    name_suffix = " ".join(sorted(set(suffixes)))
    return s, name_core, name_suffix


def normalize_address(raw: str, country: str) -> str:
    """Transliterate + de-noise an address into a comparison-ready string."""
    if not raw:
        return ""
    s = _translit(raw).lower()
    s = NULL_RE.sub(" ", s)
    s = WEB_PREFIX_RE.sub(" ", s)
    s = TLD_RE.sub(" ", s)
    s = LANDMARK_RE.sub(" ", s)
    s = JUNK_CHARS_RE.sub(" ", s)

    state_map = STATE_BY_COUNTRY.get(country)
    out_segments = []
    for seg in s.split(","):
        seg = seg.strip()
        if not seg:
            continue
        # a segment that is exactly a state code -> expand to full name
        if state_map and seg in state_map:
            out_segments.append(state_map[seg])
            continue
        toks = [STREET_ABBR.get(t, t) for t in seg.split()]
        out_segments.append(" ".join(toks))
    s = " ".join(out_segments)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)          # tokens only; commas already used above
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


OUT_COLS = [
    "entity_id", "country", "business_name", "business_address",
    "name_norm", "name_core", "name_suffix", "addr_norm",
]


def clean_source_file(in_rel: str, out_rel: str) -> dict:
    """Stream one source file, add normalized columns, write enriched TSV."""
    in_path = os.path.join(ROOT, in_rel)
    out_path = os.path.join(ROOT, out_rel)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stats = {"rows": 0, "residual_nonascii_name": 0, "empty_core": 0,
             "with_suffix": 0, "empty_addr": 0}
    reader = pd.read_csv(
        in_path, sep="\t", dtype=str, keep_default_na=False, na_values=[],
        chunksize=CHUNK, quoting=3, encoding="utf-8", encoding_errors="replace",
    )
    first = True
    for chunk in reader:
        chunk = chunk.fillna("")
        names = chunk["business_name"].astype(str).tolist()
        addrs = chunk["business_address"].astype(str).tolist()
        ctry = chunk["country"].astype(str).tolist()

        nn = [normalize_name(x) for x in names]
        chunk["name_norm"] = [t[0] for t in nn]
        chunk["name_core"] = [t[1] for t in nn]
        chunk["name_suffix"] = [t[2] for t in nn]
        chunk["addr_norm"] = [normalize_address(a, c) for a, c in zip(addrs, ctry)]

        # keep the raw text but strip stray tab/newline so the TSV stays 1 row/record
        chunk["business_name"] = chunk["business_name"].str.replace(r"[\t\r\n]+", " ", regex=True)
        chunk["business_address"] = chunk["business_address"].str.replace(r"[\t\r\n]+", " ", regex=True)

        out = chunk[OUT_COLS]
        out.to_csv(
            out_path, sep="\t", index=False, mode="w" if first else "a",
            header=first, encoding="utf-8", lineterminator="\n",
        )
        first = False

        stats["rows"] += len(out)
        stats["residual_nonascii_name"] += int(
            out["name_norm"].str.contains(NON_ASCII).sum())
        stats["empty_core"] += int((out["name_core"].str.len() == 0).sum())
        stats["with_suffix"] += int((out["name_suffix"].str.len() > 0).sum())
        stats["empty_addr"] += int((out["addr_norm"].str.len() == 0).sum())
        print(f"    ...{stats['rows']:,} rows", flush=True)
    return stats


def preview(out_rel: str, n: int = 6):
    path = os.path.join(ROOT, out_rel)
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     na_values=[], nrows=n, encoding="utf-8")
    for _, r in df.iterrows():
        print(f"    RAW  name={r['business_name']!r}  addr={r['business_address']!r} [{r['country']}]")
        print(f"    ->   norm={r['name_norm']!r} core={r['name_core']!r} "
              f"suf={r['name_suffix']!r}")
        print(f"    ->   addr_norm={r['addr_norm']!r}")


def main():
    print("=" * 70)
    print("PREPROCESSING / CLEANING — writing enriched TSVs to dataset/clean/")
    print("=" * 70)
    grand = 0
    t0 = time.time()
    for in_rel, out_rel in SOURCES:
        print(f"\n[{in_rel}] -> [{out_rel}]")
        ts = time.time()
        st = clean_source_file(in_rel, out_rel)
        grand += st["rows"]
        n = st["rows"]
        print(f"  done {n:,} rows in {time.time()-ts:.0f}s | "
              f"residual non-ASCII name={st['residual_nonascii_name']} "
              f"({100*st['residual_nonascii_name']/n:.3f}%) | "
              f"empty core={st['empty_core']} | with suffix={st['with_suffix']} "
              f"({100*st['with_suffix']/n:.1f}%) | empty addr_norm={st['empty_addr']}")
        preview(out_rel)
    print(f"\nALL DONE: {grand:,} rows cleaned in {time.time()-t0:.0f}s")
    print(f"Cleaned dataset written under: {OUT_DIR}")


if __name__ == "__main__":
    main()





