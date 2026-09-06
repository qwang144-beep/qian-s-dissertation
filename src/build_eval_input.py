# -*- coding: utf-8 -*-
"""
build_eval_input.py
===================
Assembles the candidate dump written by generate_metrical.py into the input
format expected by evaluate_reconstruction.py.
Key point: baseline and reranked come from **the same candidate pool**, giving
a clean best-of-N ablation --
  baseline = candidates[0] per line  (no selection, one raw sample)
  reranked = chosen per line         (the line the selector picked)
Both are scored against the reference context (fair context).

Because this assembly is offline, changing the selection weights only requires
rerunning this script, not regenerating.
Optional --reselect: ignore the recorded chosen and re-select from the
candidates under the current weights.

Dependencies: lyrics_reranker.py, lyric_txt_parser.py (same directory)
Usage:
  python build_eval_input.py --dump cand_dump.jsonl --ref-dir ref_300 \
      --skeletons skeletons_300.jsonl --out eval_input.json
  python build_eval_input.py --dump cand_dump.jsonl --ref-dir ref_300 \
      --skeletons skeletons_300.jsonl --out eval_input_w.json \
      --reselect --w-count 4 --w-rhyme 1.5
"""

from __future__ import annotations

import argparse
import json
import os

from lyric_txt_parser import clean_lyric_lines, song_id_from_path
from generate_metrical import (      # reuse the same line-selection logic
    select_best_line, load_manifest, parse_song_ctx,
)


def load_dump(path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["song_id"]] = rec["lines"]
    return out


def load_ref_lines(ref_dir: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for fname in os.listdir(ref_dir):
        if not fname.endswith(".txt"):
            continue
        with open(os.path.join(ref_dir, fname), encoding="utf-8") as f:
            out[song_id_from_path(fname)] = clean_lyric_lines(f.read())
    return out


def load_skeletons(path: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["song_id"]] = rec["lines"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="candidate dump from generate_metrical.py")
    ap.add_argument("--ref-dir", required=True, help="reference lyrics directory (ref_300)")
    ap.add_argument("--skeletons", required=True, help="skeletons_300.jsonl")
    ap.add_argument("--out", default="eval_input.json")
    ap.add_argument("--manifest", default=None,
                    help="prompts_300.jsonl. Strongly recommended with --reselect: "
                         "without it the vocabulary term is missing and re-selection "
                         "cannot reproduce the generation-time choice "
                         "(measured agreement only ~70%)")
    ap.add_argument("--k-context", type=int, default=3)
    ap.add_argument("--reselect", action="store_true",
                    help="ignore the recorded chosen; re-select from the candidates under the current weights")
    ap.add_argument("--w-count", type=float, default=3.0)
    ap.add_argument("--w-rhyme", type=float, default=1.0)
    ap.add_argument("--w-vocab", type=float, default=0.7)
    ap.add_argument("--w-rep", type=float, default=0.8)
    ap.add_argument("--select-context", choices=["ref", "self"], default="ref",
                    help="context used when RESELECT scores candidates. 'ref' (default) = "
                         "reference-lyrics context, same source as the evaluation and "
                         "therefore circular. 'self' = the context this re-selection has "
                         "itself chosen so far, reproducing generation-time behaviour, "
                         "no circularity.")
    ap.add_argument("--select-window", choices=["k", "all"], default="k",
                    help="context window for RESELECT scoring. 'k' = last k_context lines; "
                         "'all' = the whole context selected so far (generation uses 'all').")
    ap.add_argument("--self-context", choices=["ref", "baseline", "reranked"],
                    default="ref",
                    help="context used for the rhyme/repetition metrics. "
                         "'ref' (default) = the reference's preceding lines, as "
                         "before. 'baseline'/'reranked' = that system's OWN "
                         "preceding output, which is what measures "
                         "self-repetition. With a self-context you must read "
                         "only that system's column in the report; the other "
                         "column is scored against the wrong history.")
    args = ap.parse_args()

    if args.reselect and args.select_context == "ref":
        print("[warn] --reselect scores against the reference context, and the "
              "evaluation does too: the rhyme metric becomes circular (picking "
              "answers by the same standard that grades them). For a non-circular "
              "re-selection, add --select-context self --select-window all.")

    if args.self_context != "ref":
        print(f"[warn] self-context = {args.self_context}: in the resulting "
              f"report, ONLY the '{args.self_context}' column is meaningful for "
              f"rhyme / verbatim / bigram-overlap. Ignore the other column.")

    dump = load_dump(args.dump)
    ref = load_ref_lines(args.ref_dir)
    skels = load_skeletons(args.skeletons)
    if args.manifest:
        stems_by_song = {tid: parse_song_ctx(rec).vocab_stems
                         for tid, rec in load_manifest(args.manifest).items()}
    else:
        stems_by_song = {}
        if args.reselect:
            print("[warn] --reselect without --manifest: the vocabulary term is "
                  "disabled, so re-selection is not equivalent to the "
                  "generation-time choice.")
    sel_kw = dict(w_count=args.w_count, w_rhyme=args.w_rhyme,
                  w_vocab=args.w_vocab, w_rep=args.w_rep)

    songs_out = []
    n_lines = 0
    for sid, lines in dump.items():
        ref_lines = ref.get(sid, [])
        skel = skels.get(sid, [])
        song_lines = []
        hist_base: list[str] = []      # this system's own generated context (for self-context)
        hist_rr: list[str] = []
        for ln in lines:
            i = ln["idx"]
            cands = ln.get("candidates", [])
            if not cands:
                continue
            baseline = cands[0]                         # no selection: first candidate
            if args.reselect:
                if args.select_context == "self":
                    prev_sel = hist_rr if args.select_window == "all" \
                        else hist_rr[-args.k_context:]
                else:
                    prev_sel = ref_lines[:i] if args.select_window == "all" \
                        else ref_lines[max(0, i - args.k_context):i]
                choice = select_best_line(cands, ln["target_syllables"], prev_sel,
                                          stems_by_song.get(sid, set()), **sel_kw)
                reranked = choice.line if choice else baseline
            else:
                reranked = ln.get("chosen", baseline)   # already selected at generation time

            template = skel[i]["per_word"] if i < len(skel) else \
                [ln["target_syllables"]]                # fallback: treat the whole line as one word
            gt = ref_lines[i] if i < len(ref_lines) else ""

            if args.self_context == "baseline":
                prev = hist_base[-args.k_context:] if args.k_context > 0 else []
            elif args.self_context == "reranked":
                prev = hist_rr[-args.k_context:] if args.k_context > 0 else []
            else:
                prev = ref_lines[max(0, i - args.k_context):i]

            hist_base.append(baseline)
            hist_rr.append(reranked)

            song_lines.append({
                "template": template,
                "prev_lines": prev,
                "ground_truth": gt,
                "baseline": baseline,
                "reranked": reranked,
            })
            n_lines += 1
        if song_lines:
            songs_out.append({"song_id": sid, "lines": song_lines})

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(songs_out, f, ensure_ascii=False, indent=2)
    print(f"[done] {len(songs_out)} songs, {n_lines} lines -> {args.out}")
    print(f"       feed it to: python evaluate_reconstruction.py --input {args.out}")


if __name__ == "__main__":
    main()
