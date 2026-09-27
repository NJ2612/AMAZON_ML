#!/usr/bin/env python3
"""Diagnose blocking recall misses: why are true matches not generated?"""
import sys
from collections import Counter
import common as c
import blocking as bl

SAMPLE = 5000

def main():
    gt = c.load_ground_truth()
    # replicate the sampled S1 set: first SAMPLE India S1 (India sorts first)
    files = c.TRAIN_FILES
    s1_info = {}   # id -> (country, core, addr)
    n = 0
    for chunk in c.iter_clean(files["S1"], usecols=["entity_id","country","name_core","addr_norm"]):
        sub = chunk[chunk["country"].values == "India"]
        for rid, ctry, core, addr in zip(sub["entity_id"], sub["country"], sub["name_core"], sub["addr_norm"]):
            s1_info[rid] = (ctry, core, addr)
            n += 1
            if n >= SAMPLE: break
        if n >= SAMPLE: break
    print(f"loaded {len(s1_info)} sampled India S1", flush=True)

    # all truth-match ids for these S1
    want = set()
    for s1 in s1_info:
        want |= gt.get(s1, set())
    print(f"truth-match record ids to locate: {len(want):,}", flush=True)

    # pull those records from S2/S3 (any country)
    rec = {}
    for role in ("S2","S3"):
        for chunk in c.iter_clean(files[role], usecols=["entity_id","country","name_core","addr_norm"]):
            m = chunk["entity_id"].isin(want).values
            if not m.any(): continue
            sub = chunk[m]
            for rid, ctry, core, addr in zip(sub["entity_id"], sub["country"], sub["name_core"], sub["addr_norm"]):
                rec[rid] = (ctry, core, addr)
    found = len(rec)
    print(f"located {found:,}/{len(want):,} truth records ({found/max(len(want),1):.1%})", flush=True)

    # analyze
    cross_country = 0
    missing_rec = 0
    no_name_token_overlap = 0
    no_metaphone_overlap = 0
    no_numeric_overlap = 0
    salvageable_by_addr = 0
    total_pairs = 0
    # expanded-key blockability
    blk_current = 0     # tok OR mph OR num
    blk_ngram = 0       # + name char-4gram (space-stripped) shares >=2
    blk_addrword = 0    # + rare address word token shares >=1
    blk_any = 0         # union of all
    still_unblockable = 0
    examples = []

    # address word document frequency over the located truth records (proxy for rarity)
    addr_df = Counter()
    for _, (_, _, raddr) in rec.items():
        for t in set(raddr.split()):
            addr_df[t] += 1

    for s1, truth in gt.items():
        if s1 not in s1_info: continue
        _, s1core, s1addr = s1_info[s1]
        s1tok = set(s1core.split()); s1mph = {bl._mph(t) for t in s1tok if bl._mph(t)}
        s1num = c.numeric_tokens(s1addr)
        s1grams = c.char_ngrams(s1core, 4)
        s1aw = {t for t in set(s1addr.split()) if len(t) >= 4 and not any(ch.isdigit() for ch in t)}
        for tid in truth:
            total_pairs += 1
            if tid not in rec:
                missing_rec += 1; continue
            rctry, rcore, raddr = rec[tid]
            if rctry != "India":
                cross_country += 1
            rtok = set(rcore.split()); rmph = {bl._mph(t) for t in rtok if bl._mph(t)}
            rnum = c.numeric_tokens(raddr)
            rgrams = c.char_ngrams(rcore, 4)
            raw = {t for t in set(raddr.split()) if len(t) >= 4 and not any(ch.isdigit() for ch in t)}
            tok_ov = bool(s1tok & rtok)
            mph_ov = bool(s1mph & rmph)
            num_ov = bool(s1num & rnum)
            ng_ov = len(s1grams & rgrams) >= 2
            aw_ov = bool(s1aw & raw)
            if not tok_ov: no_name_token_overlap += 1
            if not mph_ov: no_metaphone_overlap += 1
            if not num_ov: no_numeric_overlap += 1
            cur = tok_ov or mph_ov or num_ov
            if cur: blk_current += 1
            if cur or ng_ov: blk_ngram += 1
            if cur or aw_ov: blk_addrword += 1
            if cur or ng_ov or aw_ov: blk_any += 1
            else:
                still_unblockable += 1
                if len(examples) < 20:
                    examples.append((s1core[:45], s1addr[:45], rcore[:45], raddr[:45], rctry))
            if not cur and num_ov:
                salvageable_by_addr += 1

    print(f"\ntotal truth pairs (sampled S1): {total_pairs:,}", flush=True)
    print(f"  truth record not in S2/S3 file : {missing_rec:,}", flush=True)
    print(f"  cross-country (rec != India)   : {cross_country:,}", flush=True)
    print(f"  no name-token overlap          : {no_name_token_overlap:,}", flush=True)
    print(f"  no metaphone overlap           : {no_metaphone_overlap:,}", flush=True)
    print(f"  no numeric-addr overlap        : {no_numeric_overlap:,}", flush=True)
    print(f"\n  BLOCKABILITY (share >=1 key of the family), of {total_pairs:,} pairs:", flush=True)
    print(f"    current keys (tok|mph|num)   : {blk_current:,} ({blk_current/total_pairs:.1%})", flush=True)
    print(f"    + name char-4gram (>=2)      : {blk_ngram:,} ({blk_ngram/total_pairs:.1%})", flush=True)
    print(f"    + address word token         : {blk_addrword:,} ({blk_addrword/total_pairs:.1%})", flush=True)
    print(f"    ALL families unioned         : {blk_any:,} ({blk_any/total_pairs:.1%})", flush=True)
    print(f"    still unblockable            : {still_unblockable:,} ({still_unblockable/total_pairs:.1%})", flush=True)
    print(f"\nSTILL-UNBLOCKABLE examples (even with ngram+addrword):", flush=True)
    for s1c, s1a, rc, ra, rct in examples:
        print(f"  S1 name=[{s1c}] addr=[{s1a}]", flush=True)
        print(f"     rec[{rct}] name=[{rc}] addr=[{ra}]", flush=True)

if __name__ == "__main__":
    main()
