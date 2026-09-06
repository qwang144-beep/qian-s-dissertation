# -*- coding: utf-8 -*-
"""
lyric_txt_parser.py
===================
Foundation for the metrically-controlled pipeline: parses LyCon .txt lyric
files into clean line lists and extracts a **syllable skeleton** from every
reference line, for metrically-controlled line-by-line reconstruction.

Handles two formats:
  - out_300/<ID>.txt  : first line is a title (e.g. "(Mothing Wings)"), with
                        (Chorus)/(Verse 1)/(Bridge)/(Outro) section headers
                        and blank lines between sections
  - ref_300/<ID>.txt  : first line is a "**Moth's Wings**"-style title,
                        blank-line stanza breaks, no section headers

Output (extract_skeletons): per song -> per line
{text, n_syllables, per_word, template_str}, where template_str is the paper's
[SYL]/[SEP] encoding, ready for the selector / generation prompt.

Dependencies: lyrics_reranker.py (same directory) --
reuses line_to_template / expand_to_syl_seq
Usage:
  python lyric_txt_parser.py --ref_dir ref_300 --out skeletons_300.jsonl
  python lyric_txt_parser.py --ref_dir ref_300 --peek TRABZFP12903D0945A   # inspect one song
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict

from lyrics_reranker import line_to_template, expand_to_syl_seq, tokenize_words

_SECTION_WORDS = (
    r"verse|chorus|bridge|outro|intro|hook|refrain|pre[\s-]*chorus|"
    r"post[\s-]*chorus|interlude|breakdown|drop|coda|vamp"
)
_SECTION_RE = re.compile(
    rf"^\s*[\(\[]\s*(?:{_SECTION_WORDS})\b[^\)\]]*[\)\]]\s*$", re.IGNORECASE
)
_TITLE_MD_RE = re.compile(r"^\s*(\*\*.*\*\*|#.*|_.*_)\s*$")
_TITLE_PAREN_RE = re.compile(r"^\s*[\(\[].*[\)\]]\s*$")


def is_section_header(line: str) -> bool:
    return bool(_SECTION_RE.match(line))


def _looks_like_title(line: str) -> bool:
    """Title heuristic: markdown bold/heading, or the whole line wrapped in ()/[]."""
    return bool(_TITLE_MD_RE.match(line) or _TITLE_PAREN_RE.match(line))


def clean_lyric_lines(raw_text: str, drop_title: bool = True) -> list[str]:
    """
    Reduce raw .txt content to a list of lyric lines only:
      - drop the first non-empty line (if it looks like a title)
      - drop all section-header lines
      - drop blank lines
    Lyric lines keep their original text (punctuation included) for later
    G2P / rhyme processing.
    """
    lines = raw_text.splitlines()
    out: list[str] = []
    title_dropped = not drop_title

    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if is_section_header(s):
            continue
        if not title_dropped:
            title_dropped = True
            if _looks_like_title(s):
                continue
        out.append(s)
    return out


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class LineSkeleton:
    idx: int                 # line index (0-based)
    text: str                # original reference line
    per_word: list[int]      # syllables per word, e.g. [2, 1]
    n_syllables: int         # total syllables in the line
    template_str: str        # [SYL]/[SEP] encoding, e.g. "[SYL][SYL][SEP][SYL]"


def _to_template_str(per_word: list[int]) -> str:
    seq = expand_to_syl_seq(per_word)  # ['SYL','SYL','SEP','SYL']
    return "".join(f"[{tok}]" for tok in seq)


def extract_skeleton(lines: list[str]) -> list[LineSkeleton]:
    """Extract the syllable skeleton line-by-line from a cleaned line list."""
    skel: list[LineSkeleton] = []
    for i, ln in enumerate(lines):
        per_word = line_to_template(ln)          # reuse the selector's G2P + syllable counting
        skel.append(LineSkeleton(
            idx=i,
            text=ln,
            per_word=per_word,
            n_syllables=sum(per_word),
            template_str=_to_template_str(per_word),
        ))
    return skel


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def song_id_from_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def process_dir(ref_dir: str) -> dict[str, list[LineSkeleton]]:
    """Walk the reference directory; each <ID>.txt -> a list of syllable skeletons."""
    result: dict[str, list[LineSkeleton]] = {}
    for fname in sorted(os.listdir(ref_dir)):
        if not fname.endswith(".txt"):
            continue
        path = os.path.join(ref_dir, fname)
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        lines = clean_lyric_lines(raw)
        result[song_id_from_path(path)] = extract_skeleton(lines)
    return result


def write_jsonl(skeletons: dict[str, list[LineSkeleton]], out_path: str) -> None:
    """One song per output line: {song_id, n_lines, lines:[{idx,text,per_word,n_syllables,template_str}...]}"""
    with open(out_path, "w", encoding="utf-8") as f:
        for sid, skel in skeletons.items():
            rec = {
                "song_id": sid,
                "n_lines": len(skel),
                "total_syllables": sum(s.n_syllables for s in skel),
                "lines": [asdict(s) for s in skel],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", required=True, help="reference lyrics directory (ground truth)")
    ap.add_argument("--out", default="skeletons.jsonl", help="output JSONL")
    ap.add_argument("--peek", default=None, help="parse and print one song only (give its song_id)")
    args = ap.parse_args()

    if args.peek:
        path = os.path.join(args.ref_dir, f"{args.peek}.txt")
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        lines = clean_lyric_lines(raw)
        skel = extract_skeleton(lines)
        print(f"\n=== {args.peek} : {len(skel)} lines, "
              f"{sum(s.n_syllables for s in skel)} syllables ===\n")
        for s in skel:
            print(f"[{s.idx:2d}] {s.n_syllables:2d}syl  {str(s.per_word):<16} "
                  f"{s.template_str:<28} | {s.text}")
        return

    skeletons = process_dir(args.ref_dir)
    write_jsonl(skeletons, args.out)
    n_songs = len(skeletons)
    n_lines = sum(len(v) for v in skeletons.values())
    print(f"[done] {n_songs} songs, {n_lines} lines -> {args.out}")


if __name__ == "__main__":
    main()
