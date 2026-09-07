# -*- coding: utf-8 -*-
"""
extra_metrics.py
================
Computes the two measures that evaluate_reconstruction.py does not output
(needed for the song-level blocks of the result tables):

  vocabulary fidelity : per line, the fraction of tokens whose stem occurs
                        in the song's bag of words, averaged over all lines
  tail diversity      : per song, distinct line-final words / number of
                        lines, averaged over all songs

Both definitions match generate_metrical.vocab_fidelity and Section 3.5 of
the report.

Usage (several directories can be passed at once; prints one comparison
table):
  python extra_metrics.py --manifest prompts_300.jsonl \
      --dirs ref=ref_300 strict=metrical_300_n8 relaxed=metrical_300_relax_v2 \
             rhyme=metrical_300_rhyme2 rep25=metrical_300_rep25
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, "src")
from generate_metrical import load_manifest, parse_song_ctx, _STEM   # noqa: E402
from lyrics_reranker import tokenize_words, last_word                # noqa: E402
from lyric_txt_parser import clean_lyric_lines, song_id_from_path    # noqa: E402


def score_dir(path: str, stems_by_song: dict[str, set[str]]) -> dict:
    line_fid: list[float] = []
    song_tail: list[float] = []
    n_songs = n_lines = n_missing = 0

    for fname in sorted(os.listdir(path)):
        if not fname.endswith(".txt"):
            continue
        sid = song_id_from_path(fname)
        with open(os.path.join(path, fname), encoding="utf-8") as f:
            lines = clean_lyric_lines(f.read())
        stems = stems_by_song.get(sid)
        if stems is None:
            n_missing += 1
            stems = set()
        n_songs += 1

        tails: list[str] = []
        for ln in lines:
            toks = tokenize_words(ln)
            if not toks:
                continue
            n_lines += 1
            hits = sum(1 for t in toks if _STEM(t.lower()) in stems)
            line_fid.append(hits / len(toks))
            tw = last_word(ln)
            if tw:
                tails.append(tw)
        if tails:
            song_tail.append(len(set(tails)) / len(tails))

    return {
        "songs": n_songs,
        "lines": n_lines,
        "missing_vocab": n_missing,
        "vocab_fidelity": sum(line_fid) / len(line_fid) if line_fid else 0.0,
        "tail_diversity": sum(song_tail) / len(song_tail) if song_tail else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="name=path pairs; several may be given")
    args = ap.parse_args()

    stems_by_song = {tid: parse_song_ctx(rec).vocab_stems
                     for tid, rec in load_manifest(args.manifest).items()}

    rows = []
    for spec in args.dirs:
        name, _, path = spec.partition("=")
        if not path or not os.path.isdir(path):
            print(f"[skip] {spec} (directory not found)")
            continue
        rows.append((name, score_dir(path, stems_by_song)))

    print()
    print(f"{'config':<12}{'songs':>7}{'lines':>8}"
          f"{'vocab_fidelity':>17}{'tail_diversity':>17}")
    print("-" * 61)
    for name, r in rows:
        print(f"{name:<12}{r['songs']:>7}{r['lines']:>8}"
              f"{r['vocab_fidelity']:>17.4f}{r['tail_diversity']:>17.4f}")
    for name, r in rows:
        if r["missing_vocab"]:
            print(f"[warn] {name}: {r['missing_vocab']} songs have no "
                  f"vocabulary entry in the manifest")


if __name__ == "__main__":
    main()
