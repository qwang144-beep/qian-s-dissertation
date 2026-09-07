# -*- coding: utf-8 -*-
"""
bootstrap_ci.py
===============
Song-clustered bootstrap over the eval_per_line.csv files produced by
evaluate_reconstruction.py, reporting 95% confidence intervals.

Why resample songs rather than lines: lines within a song share the same
metadata, the same bag of words and a common generated history, so they are
not independent, and resampling lines would give intervals that are too
narrow. The resampling unit here is the song: drawing a song brings all of
its lines, and the metric is recomputed on the pooled set of lines --- the
same estimator as the point estimates in eval_report.md.

Three families of quantities are reported:
  1. within each run: mean + CI for baseline and reranked
  2. within each run: the paired difference reranked - baseline + CI
     (same lines, paired)
  3. between runs: reranked(A) - reranked(B) + CI (same 300 songs,
     paired by song)

Usage:
  python bootstrap_ci.py \
      --csv strict=results_n8_bs/eval_per_line.csv \
            rep25=results_rep25/eval_per_line.csv \
            relaxed=results_relax_v2/eval_per_line.csv \
            rhyme=results_rhyme2/eval_per_line.csv \
      --compare rep25:strict relaxed:strict rhyme:relaxed
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict

METRICS = ["syl_exact", "syl_within1", "syl_template_acc", "rhyme",
           "verbatim", "ngram_overlap", "bert_P", "bert_R", "bert_F1"]

LABEL = {
    "syl_exact": "Syllable exact match",
    "syl_within1": "Syllable +-1 match",
    "syl_template_acc": "Syllable template acc",
    "rhyme": "Rhyme score",
    "verbatim": "Verbatim copy rate",
    "ngram_overlap": "Bigram overlap",
    "bert_P": "BERTScore P",
    "bert_R": "BERTScore R",
    "bert_F1": "BERTScore F1",
}


def load(path: str):
    """-> {system: {song: {'n': int, metric: sum}}}"""
    out: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(
        lambda: {"n": 0, **{m: 0.0 for m in METRICS}}))
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cell = out[r["system"]][r["song"]]
            cell["n"] += 1
            for m in METRICS:
                v = r.get(m, "")
                cell[m] += float(v) if v not in ("", None) else 0.0
    return {sys_: dict(songs) for sys_, songs in out.items()}


def pooled_mean(data: dict, songs, metric: str) -> float:
    s = n = 0.0
    for sid in songs:
        cell = data.get(sid)
        if cell is None:
            continue
        s += cell[metric]
        n += cell["n"]
    return s / n if n else 0.0


def ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    v = sorted(values)
    lo = v[int(alpha / 2 * len(v))]
    hi = v[min(len(v) - 1, int((1 - alpha / 2) * len(v)))]
    return lo, hi


def fmt(point: float, lo: float, hi: float, dp: int = 4) -> str:
    return f"{point:.{dp}f} [{lo:.{dp}f}, {hi:.{dp}f}]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True, help="name=path")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="A:B pairs; reports reranked(A) - reranked(B)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    runs = {}
    for spec in args.csv:
        name, _, path = spec.partition("=")
        runs[name] = load(path)

    # songs common to every run
    common = None
    for data in runs.values():
        songs = set(data["reranked"].keys())
        common = songs if common is None else (common & songs)
    songs_all = sorted(common)
    print(f"songs common to all runs: {len(songs_all)}   "
          f"bootstrap resamples: {args.boot}\n")

    rng = random.Random(args.seed)
    draws = [[songs_all[rng.randrange(len(songs_all))]
              for _ in range(len(songs_all))] for _ in range(args.boot)]

    # ---- 1 & 2: within each run ----
    for name, data in runs.items():
        print(f"===== {name} =====")
        print(f"{'metric':<24}{'baseline':>26}{'reranked':>26}"
              f"{'reranked - baseline':>30}")
        for m in METRICS:
            b_pt = pooled_mean(data["baseline"], songs_all, m)
            r_pt = pooled_mean(data["reranked"], songs_all, m)
            d_pt = r_pt - b_pt
            bs, rs, ds = [], [], []
            for d in draws:
                bb = pooled_mean(data["baseline"], d, m)
                rr = pooled_mean(data["reranked"], d, m)
                bs.append(bb); rs.append(rr); ds.append(rr - bb)
            blo, bhi = ci(bs); rlo, rhi = ci(rs); dlo, dhi = ci(ds)
            star = "" if (dlo <= 0 <= dhi) else "  *"
            print(f"{LABEL[m]:<24}{fmt(b_pt, blo, bhi):>26}"
                  f"{fmt(r_pt, rlo, rhi):>26}{fmt(d_pt, dlo, dhi):>30}{star}")
        print()

    # ---- 3: between runs ----
    for spec in args.compare:
        a, _, b = spec.partition(":")
        if a not in runs or b not in runs:
            print(f"[skip] {spec}")
            continue
        print(f"===== reranked({a}) - reranked({b}) =====")
        print(f"{'metric':<24}{a:>26}{b:>26}{'difference':>30}")
        for m in METRICS:
            a_pt = pooled_mean(runs[a]["reranked"], songs_all, m)
            b_pt = pooled_mean(runs[b]["reranked"], songs_all, m)
            d_pt = a_pt - b_pt
            as_, bs_, ds = [], [], []
            for d in draws:
                aa = pooled_mean(runs[a]["reranked"], d, m)
                bb = pooled_mean(runs[b]["reranked"], d, m)
                as_.append(aa); bs_.append(bb); ds.append(aa - bb)
            alo, ahi = ci(as_); blo, bhi = ci(bs_); dlo, dhi = ci(ds)
            star = "" if (dlo <= 0 <= dhi) else "  *"
            print(f"{LABEL[m]:<24}{fmt(a_pt, alo, ahi):>26}"
                  f"{fmt(b_pt, blo, bhi):>26}{fmt(d_pt, dlo, dhi):>30}{star}")
        print()

    print("*  = 95% CI excludes 0 (the difference is resolvable at this resolution)")
    print("no star = CI spans 0; report as 'not resolvable', not as 'equal'")


if __name__ == "__main__":
    main()
