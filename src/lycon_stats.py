#!/usr/bin/env python3
"""
LyCon Table 1 evaluation harness
=================================
Reproduces the statistics reported in Kim & Choi (2024), "LyCon: Lyrics
Reconstruction from the Bag-of-Words Using Large Language Models", Table 1.

Point it at ANY folder of *.txt lyric files (one song per file, sections
separated by a blank line) and it will emit the same 8 statistics used in the
paper, so you can drop your Qwen2.5 reconstructions next to the released LyCon
set and compare columns directly.

    6 structural / n-gram metrics
    -----------------------------
    Faithful re-implementation of the authors' released stats.py. Verified to
    reproduce the paper's LyCon column EXACTLY on the released dataset:
        avg word count     319.42
        avg line count      42.58
        avg section count    9.97
        unique unigrams    18,921
        unique bigrams    269,139
        unique trigrams   689,852

    2 imagery metrics  (abstract / concrete word ratios)
    ----------------------------------------------------
    Per Kao & Jurafsky (2012) [ref 18]: the proportion of tokens that fall in
    the Harvard General Inquirer "Abstract" and "Object" (concrete) categories,
    x100. These are NOT in the released stats.py, so they are reconstructed
    here. You must supply the GI lexicon (see --gi). Two things to verify
    against your dissertation write-up:
      (a) which exact GI category columns you treat as abstract / concrete, and
      (b) aggregation: this script averages the per-song percentage across
          songs (consistent with the "per set" averaging of the other rows);
          the alternative is a single corpus-level pooled percentage.

Usage
-----
    python lycon_stats.py --dir ./dataset
    python lycon_stats.py --dir ./qwen_reconstructions --gi inquirerbasic.csv
    python lycon_stats.py --dir ./dataset --per-song out.csv   # dump per-song rows
"""
import argparse
import csv
import glob
import os
import re
from collections import Counter

# Same token-cleaning pattern as the authors' stats.py: keep letters, digits,
# straight/curly apostrophes and hyphens; strip everything else.
CLEAN = re.compile(r"[^a-zA-Z0-9''-]")


def iter_kept_lines(file_content):
    """Yield (section_index, kept_lines) exactly as the authors' script does.

    - sections are split on a blank line ("\\n\\n")
    - inside a section, empty lines are dropped
    - a line that is FULLY wrapped in parentheses, e.g. "(Chorus)", is dropped
      (condition: starts with '(' AND ends with ')')
    """
    sections = file_content.split("\n\n")
    for si, section in enumerate(sections):
        lines = section.split("\n")
        lines = [ln for ln in lines if len(ln)]
        lines = [ln for ln in lines if ln[0] != '(' or ln[-1] != ')']
        yield si, lines


def song_structural_stats(file_content):
    """Return (word_count, line_count, section_count, unigrams, bigrams, trigrams)
    for a single song. n-grams are built WITHIN each line (never across line
    breaks), matching stats.py."""
    section_count = len(file_content.split("\n\n"))
    line_count = 0
    word_count = 0
    unigrams, bigrams, trigrams = [], [], []

    for _si, lines in iter_kept_lines(file_content):
        line_count += len(lines)
        for line in lines:
            words = line.split(" ")
            word_count += len(words)                       # raw space-split count
            w = [CLEAN.sub('', tok.lower()) for tok in words]
            unigrams.extend(w)
            bigrams.extend((w[i], w[i + 1]) for i in range(len(w) - 1))
            trigrams.extend((w[i], w[i + 1], w[i + 2]) for i in range(len(w) - 2))

    return word_count, line_count, section_count, unigrams, bigrams, trigrams


def song_tokens(file_content):
    """Cleaned, lowercased, non-empty tokens for a song (for imagery ratios)."""
    toks = []
    for _si, lines in iter_kept_lines(file_content):
        for line in lines:
            for tok in line.split(" "):
                t = CLEAN.sub('', tok.lower())
                if t:
                    toks.append(t)
    return toks


def load_gi(path):
    """Load a General-Inquirer-style lexicon.

    Expected: a CSV whose first column is the (lower-cased) word/entry and which
    has boolean-ish columns naming the categories. We look for an 'abstract'
    column and an 'object' column (case-insensitive). Adjust ABSTRACT_COLS /
    CONCRETE_COLS below to match the exact GI category names you settle on.

    Returns (abstract_set, concrete_set).
    """
    ABSTRACT_COLS = {"abs", "abs@", "abstract"}
    CONCRETE_COLS = {"object", "obj", "concrete"}
    abstract, concrete = set(), set()
    with open(path, newline='', encoding='utf-8', errors='ignore') as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        abs_idx = [i for i, h in enumerate(header) if h in ABSTRACT_COLS]
        con_idx = [i for i, h in enumerate(header) if h in CONCRETE_COLS]
        if not abs_idx or not con_idx:
            raise ValueError(
                f"Could not find abstract/concrete columns in {path}. "
                f"Header was: {header}. Edit ABSTRACT_COLS / CONCRETE_COLS.")
        for row in reader:
            if not row:
                continue
            # GI entries are often UPPER and may carry a '#N' sense suffix
            word = row[0].split('#')[0].strip().lower()
            if any(row[i].strip() for i in abs_idx):
                abstract.add(word)
            if any(row[i].strip() for i in con_idx):
                concrete.add(word)
    return abstract, concrete


def main():
    ap = argparse.ArgumentParser(description="LyCon Table 1 statistics")
    ap.add_argument("--dir", required=True, help="folder of *.txt lyric files")
    ap.add_argument("--gi", default=None,
                    help="General Inquirer lexicon CSV (enables abstract/concrete)")
    ap.add_argument("--per-song", default=None,
                    help="optional path to dump per-song structural rows as CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "*.txt")))
    if not paths:
        raise SystemExit(f"No .txt files found in {args.dir}")

    abstract = concrete = None
    if args.gi:
        abstract, concrete = load_gi(args.gi)
        print(f"[gi] {len(abstract)} abstract / {len(concrete)} concrete entries loaded")

    word_counts, line_counts, section_counts = [], [], []
    uni, bi, tri = [], [], []
    abs_ratios, con_ratios = [], []
    per_song_rows = []

    for p in paths:
        with open(p, 'r', encoding='utf-8', errors='ignore') as fh:
            content = fh.read()
        wc, lc, sc, u, b, t = song_structural_stats(content)
        word_counts.append(wc); line_counts.append(lc); section_counts.append(sc)
        uni.extend(u); bi.extend(b); tri.extend(t)

        if abstract is not None:
            toks = song_tokens(content)
            n = len(toks) or 1
            a = sum(1 for tk in toks if tk in abstract)
            c = sum(1 for tk in toks if tk in concrete)
            abs_ratios.append(100.0 * a / n)
            con_ratios.append(100.0 * c / n)

        per_song_rows.append((os.path.basename(p), wc, lc, sc))

    n_songs = len(paths)
    def avg(xs): return sum(xs) / len(xs)

    print(f"\n=== LyCon-style statistics over {n_songs} songs in {args.dir} ===")
    print(f"Average Word Count per Set     {avg(word_counts):.2f}")
    print(f"Average Line Count per Set     {avg(line_counts):.2f}")
    print(f"Average Section Count per Set  {avg(section_counts):.2f}")
    print(f"Total Unique Unigrams          {len(Counter(uni))}")
    print(f"Total Unique Bigrams           {len(Counter(bi))}")
    print(f"Total Unique Trigrams          {len(Counter(tri))}")
    if abstract is not None:
        print(f"Abstract Words Ratio           {avg(abs_ratios):.2f}")
        print(f"Concrete Words Ratio           {avg(con_ratios):.2f}")
    else:
        print("Abstract Words Ratio           (supply --gi to compute)")
        print("Concrete Words Ratio           (supply --gi to compute)")

    if args.per_song:
        with open(args.per_song, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(["file", "word_count", "line_count", "section_count"])
            w.writerows(per_song_rows)
        print(f"\n[per-song] wrote {len(per_song_rows)} rows to {args.per_song}")


if __name__ == "__main__":
    main()
