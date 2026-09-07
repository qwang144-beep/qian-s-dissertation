# -*- coding: utf-8 -*-
"""
frame_diversity.py
==================
Tail diversity looks only at the line-final word, but templating can hide at
the start of the line:

    fire burns low / fire flickers in my spine / fire spins wild / fire drifts low
    someday I'll break the ground / someday your heart will ach / someday we break down

These lines all end differently (tail diversity looks fine) while opening
identically. This script covers the symmetric side and reports both ends
together for comparison.

  head_div     per song, distinct line-initial words / lines, mean over songs
               (higher is better)
  head2_div    per song, distinct opening bigrams / lines, mean over songs
               (higher is better)
  head_reuse@K fraction of lines whose first word matches the first word of
               one of the K preceding lines (lower is better)
  tail_div / tail_reuse@K  as above, on the line-final word (consistent with
               tail_reuse.py)

Usage:
  python frame_diversity.py --dirs ref=ref_300 noselect=metrical_300_noselect \
      strict=metrical_300_n8 rep25=metrical_300_rep25 \
      relaxed=metrical_300_relax_v2 rhyme=metrical_300_rhyme2
  # whole-song comparison of the two base models (free; answers whether
  # switching models would help):
  python frame_diversity.py --dirs ref=ref_300 qwen=out_300 llama=out_llama_300
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")
from lyrics_reranker import tokenize_words, last_word          # noqa: E402
from lyric_txt_parser import clean_lyric_lines                 # noqa: E402


def first_word(line: str) -> str | None:
    toks = tokenize_words(line)
    return toks[0] if toks else None


def first_two(line: str) -> str | None:
    toks = tokenize_words(line)
    if not toks:
        return None
    return " ".join(toks[:2])


def audit(dir_: str, k: int) -> dict:
    n_lines = 0
    head_reuse = tail_reuse = 0
    head_div, head2_div, tail_div = [], [], []
    worst: list[tuple[float, str]] = []

    for fname in sorted(os.listdir(dir_)):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(dir_, fname), encoding="utf-8") as f:
            lines = clean_lyric_lines(f.read())

        heads = [first_word(l) for l in lines]
        heads2 = [first_two(l) for l in lines]
        tails = [last_word(l) for l in lines]
        keep = [i for i, h in enumerate(heads) if h]
        if not keep:
            continue
        heads = [heads[i] for i in keep]
        heads2 = [heads2[i] for i in keep]
        tails = [tails[i] for i in keep if tails[i]]

        for i, h in enumerate(heads):
            n_lines += 1
            if h in heads[max(0, i - k):i]:
                head_reuse += 1
        for i, t in enumerate(tails):
            if t in tails[max(0, i - k):i]:
                tail_reuse += 1

        hd = len(set(heads)) / len(heads)
        head_div.append(hd)
        head2_div.append(len(set(heads2)) / len(heads2))
        if tails:
            tail_div.append(len(set(tails)) / len(tails))
        worst.append((hd, fname))

    worst.sort()
    m = lambda xs: sum(xs) / len(xs) if xs else 0.0
    return {
        "lines": n_lines,
        "head_div": m(head_div),
        "head2_div": m(head2_div),
        "tail_div": m(tail_div),
        "head_reuse": head_reuse / n_lines if n_lines else 0.0,
        "tail_reuse": tail_reuse / n_lines if n_lines else 0.0,
        "worst": worst[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name=path")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    rows = []
    for spec in args.dirs:
        name, _, path = spec.partition("=")
        if not os.path.isdir(path):
            print(f"[skip] {spec}")
            continue
        rows.append((name, audit(path, args.k)))

    kk = args.k
    print(f"{'config':<10}{'lines':>7}{'head_div':>10}{'head2_div':>11}"
          f"{'tail_div':>10}{'head_reuse@'+str(kk):>14}{'tail_reuse@'+str(kk):>14}")
    print("-" * 76)
    for name, r in rows:
        print(f"{name:<10}{r['lines']:>7}{r['head_div']:>10.4f}{r['head2_div']:>11.4f}"
              f"{r['tail_div']:>10.4f}{r['head_reuse']:>13.2%}{r['tail_reuse']:>14.2%}")
    print("\nhead_* read in the same direction as tail_*: diversity high is "
          "good, reuse low is good")
    for name, r in rows:
        print(f"\n{name}: 5 songs with the least varied openings: "
              + ", ".join(f"{f.replace('.txt','')}({d:.3f})" for d, f in r["worst"]))


if __name__ == "__main__":
    main()
