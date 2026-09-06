#!/usr/bin/env python3
"""
Per-song, paired quantification of differences between model reconstructions.

Give it any number of labelled output folders (all named by track_id, as
out_300/ and ref_300/ are) and it will:
  * compute per-song metrics for every track present in ALL folders,
  * write a tidy per-song CSV (one row per track, columns per model+metric),
  * print a summary with each model's mean/median and the delta vs a baseline.

Two of the metrics are LENGTH-NORMALISED on purpose, to separate "the model
just writes shorter songs" from "the model reuses words/lines more":
  * ttr           = unique tokens / total tokens   (type-token ratio; vocab
                    richness independent of length)
  * dup_line_ratio= 1 - unique_lines / total_lines (verbatim line repetition,
                    e.g. a chorus copied word-for-word)

Usage
-----
  python compare_models.py --dirs gpt4o=ref_300 qwen=out_300 \
      --baseline gpt4o --out compare.csv
  # later, add a third model:
  python compare_models.py --dirs gpt4o=ref_300 qwen=out_300 llama=out_llama_300 \
      --baseline gpt4o --out compare.csv
"""
import argparse
import csv
import glob
import os
import statistics as st

from lycon_stats import iter_kept_lines, song_structural_stats, song_tokens

METRICS = ["words", "lines", "sections", "tokens", "uniq_words",
           "ttr", "dup_line_ratio"]


def per_song(content):
    wc, lc, sc, _uni, _bi, _tri = song_structural_stats(content)
    toks = song_tokens(content)
    total = len(toks)
    uniq = len(set(toks))
    lines = [ln.strip().lower()
             for _si, ls in iter_kept_lines(content) for ln in ls]
    return {
        "words": wc,
        "lines": lc,
        "sections": sc,
        "tokens": total,
        "uniq_words": uniq,
        "ttr": (uniq / total) if total else 0.0,
        "dup_line_ratio": (1 - len(set(lines)) / len(lines)) if lines else 0.0,
    }


def load_dir(path):
    out = {}
    for p in glob.glob(os.path.join(path, "*.txt")):
        tid = os.path.splitext(os.path.basename(p))[0]
        with open(p, encoding="utf-8", errors="ignore") as f:
            out[tid] = per_song(f.read())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="label=path pairs, e.g. gpt4o=ref_300 qwen=out_300")
    ap.add_argument("--baseline", default=None, help="label to diff others against")
    ap.add_argument("--out", default="compare.csv")
    args = ap.parse_args()

    models = {}
    order = []
    for spec in args.dirs:
        label, path = spec.split("=", 1)
        models[label] = load_dir(path)
        order.append(label)
        print(f"{label:8s}: {len(models[label])} songs in {path}")

    baseline = args.baseline or order[0]
    common = set.intersection(*(set(m) for m in models.values()))
    print(f"tracks common to all models: {len(common)}  (baseline={baseline})\n")

    # per-song CSV
    cols = ["track_id"] + [f"{lab}_{met}" for lab in order for met in METRICS]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tid in sorted(common):
            row = [tid]
            for lab in order:
                for met in METRICS:
                    v = models[lab][tid][met]
                    row.append(f"{v:.4f}" if isinstance(v, float) else v)
            w.writerow(row)
    print(f"wrote per-song table -> {args.out}\n")

    # summary
    def col(lab, met):
        return [models[lab][t][met] for t in common]

    head = f"{'metric':<16}" + "".join(f"{lab:>12}" for lab in order)
    if len(order) > 1:
        head += "".join(f"{('d '+lab):>16}" for lab in order if lab != baseline)
    print(head)
    print("-" * len(head))
    for met in METRICS:
        means = {lab: st.mean(col(lab, met)) for lab in order}
        line = f"{met:<16}" + "".join(f"{means[lab]:>12.3f}" for lab in order)
        for lab in order:
            if lab == baseline:
                continue
            b = means[baseline]
            d = means[lab] - b
            pct = (100 * d / b) if b else float("nan")
            line += f"{d:>+9.2f}({pct:>+3.0f}%)"
        print(line)

    print("\n(medians)")
    for met in METRICS:
        meds = {lab: st.median(col(lab, met)) for lab in order}
        print(f"{met:<16}" + "".join(f"{meds[lab]:>14.3f}" for lab in order))


if __name__ == "__main__":
    main()
