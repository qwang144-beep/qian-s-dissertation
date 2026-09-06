# -*- coding: utf-8 -*-
"""
oov_report.py
=============
Reports CMUdict coverage / OOV rate for the syllable and rhyme measures.
The syllable skeletons and the selector's syllable counts and rhyme features
all depend on CMUdict; words it cannot resolve fall back to heuristic vowel
counting, which is less accurate. This script quantifies how many words /
lines / terminal words hit the fallback, giving a concrete reliability figure.

Both inputs are supported:
  --ref-dir  ref_300          scan the raw lyrics directly (closest to true OOV; recommended)
  --skeletons skeletons.jsonl scan the line texts in the skeleton file (equivalent, if built)

Dependencies: lyrics_reranker.py, lyric_txt_parser.py (same directory)
Usage:
  python oov_report.py --ref-dir ref_300
  python oov_report.py --skeletons skeletons_300.jsonl
  python oov_report.py --ref-dir ref_300 --list-oov 40   # also list the most frequent OOV words
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from lyrics_reranker import tokenize_words, phones_for, last_word
from lyric_txt_parser import clean_lyric_lines, song_id_from_path


def iter_lines_from_ref(ref_dir: str):
    for fname in sorted(os.listdir(ref_dir)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(ref_dir, fname), encoding="utf-8") as f:
            lines = clean_lyric_lines(f.read())
        yield song_id_from_path(fname), lines


def iter_lines_from_skeletons(path: str):
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            yield rec["song_id"], [ln["text"] for ln in rec["lines"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--skeletons", default=None)
    ap.add_argument("--list-oov", type=int, default=0,
                    help="also list the N most frequent OOV words")
    args = ap.parse_args()

    if args.ref_dir:
        source = iter_lines_from_ref(args.ref_dir)
    elif args.skeletons:
        source = iter_lines_from_skeletons(args.skeletons)
    else:
        ap.error("give one of --ref-dir or --skeletons")

    n_songs = 0
    n_lines = 0
    n_tokens = 0
    n_oov_tokens = 0
    n_lines_with_oov = 0          # lines containing at least one OOV word
    n_tail_oov = 0                # lines whose terminal word is OOV (directly affects rhyme)
    n_tail_total = 0
    oov_counter: Counter = Counter()

    for _sid, lines in source:
        n_songs += 1
        for ln in lines:
            toks = tokenize_words(ln)
            if not toks:
                continue
            n_lines += 1
            line_has_oov = False
            for t in toks:
                n_tokens += 1
                if phones_for(t) is None:          # not in CMUdict
                    n_oov_tokens += 1
                    oov_counter[t.lower()] += 1
                    line_has_oov = True
            if line_has_oov:
                n_lines_with_oov += 1
            tw = last_word(ln)
            if tw is not None:
                n_tail_total += 1
                if phones_for(tw) is None:
                    n_tail_oov += 1

    def pct(a, b): return 100.0 * a / b if b else 0.0

    print(f"\n===== CMUdict Coverage / OOV Report =====")
    print(f"songs                : {n_songs}")
    print(f"lines                : {n_lines}")
    print(f"tokens               : {n_tokens}")
    print(f"OOV tokens           : {n_oov_tokens}  ({pct(n_oov_tokens, n_tokens):.2f}% of tokens)")
    print(f"lines with >=1 OOV   : {n_lines_with_oov}  ({pct(n_lines_with_oov, n_lines):.2f}% of lines)")
    print(f"tail-word OOV        : {n_tail_oov}/{n_tail_total}  "
          f"({pct(n_tail_oov, n_tail_total):.2f}% of line-ending words)")
    print(f"  ^ tail OOV directly affects the rhyme metric; token OOV affects syllable counts")
    print(f"distinct OOV types   : {len(oov_counter)}")

    if args.list_oov:
        print(f"\n--- top {args.list_oov} OOV words (count) ---")
        for w, c in oov_counter.most_common(args.list_oov):
            print(f"  {c:5d}  {w}")
    print()


if __name__ == "__main__":
    main()
