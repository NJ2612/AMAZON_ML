#!/usr/bin/env python3
"""Stage 1 — Blocking / candidate generation.

Memory-light, streaming, per-country multi-key inverted index (the machine has
~3 GB free, so no dense 10M-row ANN matrices). Blocking keys, unioned:
  K1  rare name_core word tokens         (romanized names, typos share tokens)
  K2  metaphone of rare name tokens      (word-order + phonetic variants)
  K3  numeric address tokens (street#/PIN) + first region token
                                          (rescues native-script names via address)
Ultra-frequent keys (DF > cap) are dropped (they only add noise + blow up the
candidate count). Candidates are scored by a cheap name+address similarity and
capped at TOP_K per S1 entity. Emits a candidate file per split and, for train,
an honest recall-ceiling audit vs ground truth.

Run:
  python blocking.py --split train [--sample N] [--topk 40]
  python blocking.py --split test
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict

import numpy as np
import jellyfish

import common as c

DF_CAP = 400          # drop a key that points at more than this many records
TOP_K = 40            # max candidates kept per S1 entity
CAND_HARD_CAP = 4000  # safety: never score more than this many raw candidates/S1
MAX_POST = 6000       # insertion-time soft cap: stop growing a posting list past this
                      # (it will be dropped at finalize anyway) — bounds build memory
NGRAM = 4             # char n-gram size for the fuzzy name key family

# cheap-ranker per-family weighting: idf * mult + bonus. Address families and exact
# name tokens are rewarded; the fuzzy ngram family is discounted (it is noisy — it
# only needs to surface the candidate, not dominate the score).
FAM_W = {
    "k_tok": (1.0, 0.0),
    "k_mph": (0.7, 0.0),
    "k_ng":  (0.45, 0.0),   # locked from sweep: 0.45 best on recall-ceiling (0.15<0.30<0.45)
    "k_num": (1.0, 1.0),
    "k_aw":  (1.0, 0.3),
}

def _mph(tok: str) -> str:
    """Metaphone code for a token; empty string if jellyfish can't encode it."""
    try:
        return jellyfish.metaphone(tok)
    except Exception:
        return ""


def region_tokens(addr: str) -> set:
    """Alphabetic address tokens length>=4 (city / locality words) — coarse region."""
    return {t for t in addr.split() if len(t) >= 4 and not any(ch.isdigit() for ch in t)}


# --- Per-country record store + inverted indices ------------------------------
class CountryIndex:
    """Streaming multi-key inverted index over the S2+S3 records of one country."""

    def __init__(self, country: str):
        self.country = country
        self.df_cap = DF_CAP
        self.ids: list[str] = []
        self.cores: list[str] = []
        self.addrs: list[str] = []
        # key -> list[int record-idx]; capped after build
        self.k_tok: dict[str, list[int]] = defaultdict(list)     # name word tokens
        self.k_mph: dict[str, list[int]] = defaultdict(list)     # metaphone of tokens
        self.k_num: dict[str, list[int]] = defaultdict(list)     # numeric address tokens
        self.k_aw: dict[str, list[int]] = defaultdict(list)      # alpha address words (locality)
        self.k_ng: dict[str, list[int]] = defaultdict(list)      # name char n-grams (space-stripped)

    @staticmethod
    def _push(idx: dict, key: str, i: int) -> None:
        lst = idx[key]
        if len(lst) < MAX_POST:                # soft cap → bounds build memory
            lst.append(i)

    def add_record(self, rid: str, core: str, addr: str) -> None:
        i = len(self.ids)
        self.ids.append(rid)
        self.cores.append(core)
        self.addrs.append(addr)
        for t in set(core.split()):
            self._push(self.k_tok, t, i)
            m = _mph(t)
            if m:
                self._push(self.k_mph, m, i)
        for g in c.char_ngrams(core, NGRAM):
            self._push(self.k_ng, g, i)
        for t in c.numeric_tokens(addr):
            self._push(self.k_num, t, i)
        for t in region_tokens(addr):
            self._push(self.k_aw, t, i)

    def finalize(self) -> None:
        """Drop over-frequent keys, convert postings to int32 arrays, build IDF."""
        self.ids = np.array(self.ids, dtype=object)
        self.cores = np.array(self.cores, dtype=object)
        self.addrs = np.array(self.addrs, dtype=object)
        n = len(self.ids)
        self.idf: dict[str, float] = {}
        for name in ("k_tok", "k_mph", "k_num", "k_aw", "k_ng"):
            idx = getattr(self, name)
            kept = {}
            for key, posts in idx.items():
                if len(posts) > self.df_cap:
                    continue                       # ultra-generic: noise + explosion
                kept[key] = np.asarray(posts, dtype=np.int32)
                self.idf[(name, key)] = np.log(n / len(posts))
            setattr(self, name, kept)

    def n_records(self) -> int:
        return len(self.ids)

    def n_pending(self) -> int:
        """Record count during build (before finalize turns lists into arrays)."""
        return len(self.ids)

    def ensure_norm(self) -> np.ndarray:
        """Per-record length norm sqrt(#name_tok + #addr_tok + 1), built once.

        Used to length-normalize the cheap weighted-union score (an approximate
        TF-IDF cosine) so long candidate strings no longer dominate the ranking —
        a strict recall-ceiling win at equal topk (see audit_block.py)."""
        n = getattr(self, "norm", None)
        if n is None:
            n = np.sqrt(np.fromiter(
                ((len(str(cr).split()) + len(str(ad).split()) + 1)
                 for cr, ad in zip(self.cores, self.addrs)),
                dtype=np.float32, count=len(self.ids)))
            self.norm = n
        return n


def load_country_index(files: dict, country: str, maxrec: int | None = None,
                       df_cap: int = DF_CAP) -> CountryIndex:
    ci = CountryIndex(country)
    ci.df_cap = df_cap
    cols = ["entity_id", "country", "name_core", "addr_norm"]
    for role in ("S2", "S3"):
        for chunk in c.iter_clean(files[role], usecols=cols):
            sub = chunk[chunk["country"].values == country]
            if sub.empty:
                continue
            for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
                ci.add_record(rid, core, addr)
                if maxrec and ci.n_pending() >= maxrec:
                    break
            if maxrec and ci.n_pending() >= maxrec:
                break
        if maxrec and ci.n_pending() >= maxrec:
            break
    ci.finalize()
    ci.ensure_norm()          # length norms for the cosine-approx cheap ranker
    return ci


# --- Candidate scoring --------------------------------------------------------
def score_pair(s1_tok: set, s1_grams: set, s1_num: set, s1_reg: set,
               rec_core: str, rec_addr: str) -> float:
    rtok = set(rec_core.split())
    name_jac = len(s1_tok & rtok) / (len(s1_tok | rtok) or 1)
    rgrams = c.char_ngrams(rec_core, 4)
    ng_jac = len(s1_grams & rgrams) / (len(s1_grams | rgrams) or 1)
    rnum = c.numeric_tokens(rec_addr)
    num_ov = len(s1_num & rnum)
    rreg = region_tokens(rec_addr)
    reg_jac = len(s1_reg & rreg) / (len(s1_reg | rreg) or 1) if (s1_reg or rreg) else 0.0
    return (0.42 * name_jac + 0.23 * ng_jac
            + 0.20 * min(num_ov, 3) / 3 + 0.15 * reg_jac)


def candidates_for(ci: CountryIndex, core: str, addr: str, topk: int,
                   eff_cap: int | None = None, detail_cap: int = 300,
                   fam_w: dict | None = None, return_pos: bool = False) -> list:
    """Return up to `topk` candidate record ids for one S1 entity, best first.

    Two-phase: (1) a numpy-vectorized IDF-weighted union count over the key
    families gives a cheap ranking; (2) the top `detail_cap` are re-scored with
    the address-aware `score_pair` and the best `topk` returned (detail_cap<=0
    skips step 2 and ranks purely by the weighted union — much faster).

    `eff_cap` (optional) skips any key whose posting list is longer than it, at
    query time — lets a sweep test tighter caps against one high-cap-built index.
    `fam_w` (optional) per-family {mult, bonus} overrides for tuning the ranker.
    `return_pos` (optional) returns integer record positions into ci.ids/cores/
    addrs instead of the string ids — lets a caller pull fields without a global
    id->field dict.
    """
    fw = fam_w if fam_w is not None else FAM_W
    s1_tok = set(core.split())
    s1_num = c.numeric_tokens(addr)
    s1_reg = region_tokens(addr)

    arrs: list[np.ndarray] = []
    ws: list[float] = []

    def _add(idxdict, key, fam):
        p = idxdict.get(key)
        if p is None:
            return
        if eff_cap is not None and len(p) > eff_cap:
            return
        mult, bonus = fw[fam]
        arrs.append(p)
        ws.append(ci.idf.get((fam, key), 0.0) * mult + bonus)

    for t in s1_tok:
        _add(ci.k_tok, t, "k_tok")
        m = _mph(t)
        if m:
            _add(ci.k_mph, m, "k_mph")
    for g in c.char_ngrams(core, NGRAM):
        _add(ci.k_ng, g, "k_ng")               # fuzzy: concatenation / typos / noise
    for t in s1_num:
        _add(ci.k_num, t, "k_num")              # numeric address match is strong
    for t in s1_reg:
        _add(ci.k_aw, t, "k_aw")                # rare locality word: strong rescue

    if not arrs:
        return []
    # 1) vectorized weighted union: concat postings, aggregate per record via bincount
    all_idx = np.concatenate(arrs)
    lens = np.fromiter((len(a) for a in arrs), dtype=np.int64, count=len(arrs))
    all_w = np.repeat(np.asarray(ws, dtype=np.float32), lens)
    uniq, inv = np.unique(all_idx, return_inverse=True)
    wsum = np.bincount(inv, weights=all_w)
    # length-normalize (approx TF-IDF cosine): divide by sqrt(candidate token count)
    # so long candidate strings don't dominate — a strict recall-ceiling win at
    # equal topk. Falls back to the raw union if norms are unavailable.
    norm = getattr(ci, "norm", None)
    score = wsum / norm[uniq] if norm is not None else wsum
    # cheap-only ranking (detail_cap == 0): take top-K directly by weighted union
    if detail_cap <= 0:
        if len(uniq) > topk:
            sel = np.argpartition(score, -topk)[-topk:]
            sel = sel[np.argsort(score[sel])[::-1]]
        else:
            sel = np.argsort(score)[::-1]
        pos = [int(uniq[s]) for s in sel]
        return pos if return_pos else [ci.ids[j] for j in pos]
    # 2) keep top detail_cap by cheap weight, then address-aware detail score
    if len(uniq) > detail_cap:
        sel = np.argpartition(score, -detail_cap)[-detail_cap:]
        cand = uniq[sel]
    else:
        cand = uniq
    s1_grams = c.char_ngrams(core, 4)
    scored = [(score_pair(s1_tok, s1_grams, s1_num, s1_reg, ci.cores[j], ci.addrs[j]), int(j))
              for j in cand]
    scored.sort(reverse=True)
    top = [j for _, j in scored[:topk]]
    return top if return_pos else [ci.ids[j] for j in top]


# --- Driver -------------------------------------------------------------------
def distinct_countries(files: dict) -> list[str]:
    seen: set[str] = set()
    for chunk in c.iter_clean(files["S1"], usecols=["country"]):
        seen.update(x for x in chunk["country"].unique() if x)
    return sorted(seen)


def run(split: str, topk: int, sample: int | None, maxrec: int | None = None,
        df_cap: int = DF_CAP) -> None:
    files = c.TRAIN_FILES if split == "train" else c.TEST_FILES
    countries = distinct_countries(files)
    print(f"[blocking] split={split} countries={countries} topk={topk} dfcap={df_cap}"
          + (f" sample={sample}" if sample else "")
          + (f" maxrec={maxrec}" if maxrec else ""), flush=True)

    cands: dict[str, list[str]] = {}
    n_seen = 0
    for country in countries:
        t0 = time.time()
        ci = load_country_index(files, country, maxrec=maxrec, df_cap=df_cap)
        print(f"  [{country}] records={ci.n_records():,} "
              f"keys(tok/mph/ng/num/aw)={len(ci.k_tok):,}/{len(ci.k_mph):,}/"
              f"{len(ci.k_ng):,}/{len(ci.k_num):,}/{len(ci.k_aw):,} "
              f"built in {time.time()-t0:.0f}s", flush=True)
        # query all S1 of this country
        q0 = time.time()
        nq = 0
        for chunk in c.iter_clean(files["S1"],
                                  usecols=["entity_id", "country", "name_core", "addr_norm"]):
            sub = chunk[chunk["country"].values == country]
            for rid, core, addr in zip(sub["entity_id"], sub["name_core"], sub["addr_norm"]):
                if sample and n_seen >= sample:
                    break
                cands[rid] = candidates_for(ci, core, addr, topk)
                nq += 1
                n_seen += 1
            if sample and n_seen >= sample:
                break
        print(f"  [{country}] queried {nq:,} S1 in {time.time()-q0:.0f}s", flush=True)
        del ci
        if sample and n_seen >= sample:
            break

    suffix = f".sample{sample}" if sample else ""
    out = f"{c.CAND_DIR}/candidates_{split}{suffix}.tsv"
    c.write_candidate_file(out, cands)
    tot = sum(len(v) for v in cands.values())
    print(f"[blocking] wrote {len(cands):,} S1 rows, {tot:,} candidate pairs "
          f"(avg {tot/max(len(cands),1):.1f}/S1) -> {out}", flush=True)

    if split == "train":
        audit_recall(cands)


def audit_recall(cands: dict[str, list[str]]) -> None:
    """Recall-ceiling audit vs ground truth (the honest upper bound on model recall).

    Also reports the macro-F0.5 CEILING: the best macro-F0.5 any classifier could
    reach given these candidates, i.e. assuming perfect precision (predict exactly
    candidates ∩ truth) — singletons contribute 1.0. This is what blocking caps.
    """
    gt = c.load_ground_truth()
    recalls, full = [], 0
    n_nonsingleton = 0
    covered_pairs = missed_pairs = 0
    f05_ceiling = []                      # per-S1 best-possible F0.5 (incl. singletons)
    for s1, truth in gt.items():
        if s1 not in cands:
            continue                      # not in this (possibly sampled) run
        cset = set(cands[s1])
        if not truth:
            f05_ceiling.append(1.0)       # singleton: predict empty -> 1.0
            continue
        n_nonsingleton += 1
        hit = len(cset & truth)
        r = hit / len(truth)
        recalls.append(r)
        covered_pairs += hit
        missed_pairs += len(truth) - hit
        if hit == len(truth):
            full += 1
        # perfect-precision ceiling: precision=1 if hit>0 else 0
        f05_ceiling.append(1.25 * r / (0.25 + r) if r > 0 else 0.0)
    if not recalls:
        print("[audit] no non-singleton S1 in this run", flush=True)
        return
    import numpy as _np
    print("[audit] recall-ceiling vs ground truth:", flush=True)
    print(f"  non-singleton S1 evaluated : {n_nonsingleton:,}", flush=True)
    print(f"  macro recall (mean/S1)     : {_np.mean(recalls):.4f}", flush=True)
    print(f"  micro recall (pairs)       : {covered_pairs/(covered_pairs+missed_pairs):.4f}", flush=True)
    print(f"  S1 with FULL recall        : {full:,} ({full/n_nonsingleton:.1%})", flush=True)
    print(f"  true pairs covered/missed  : {covered_pairs:,}/{missed_pairs:,}", flush=True)
    print(f"  >> macro-F0.5 CEILING      : {_np.mean(f05_ceiling):.4f} "
          f"(perfect-precision upper bound, incl. singletons)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], default="train")
    ap.add_argument("--topk", type=int, default=TOP_K)
    ap.add_argument("--sample", type=int, default=None,
                    help="limit to first N S1 entities (fast dev/recall check)")
    ap.add_argument("--maxrec", type=int, default=None,
                    help="cap records/country during build (SMOKE TEST only — recall not meaningful)")
    ap.add_argument("--dfcap", type=int, default=DF_CAP,
                    help="drop a blocking key pointing at more than this many records")
    args = ap.parse_args()
    run(args.split, args.topk, args.sample, args.maxrec, args.dfcap)


if __name__ == "__main__":
    main()
