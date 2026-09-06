# -*- coding: utf-8 -*-
"""
evaluate_reconstruction.py
==========================
Batch evaluation: compares the baseline LyCon output against "LyCon + selection"
and produces the metric comparison tables used in the dissertation's results
chapter (mirrors LYRICS paper Table I / II).

Metrics:
  1. Syllable exact match rate  -- fraction of lines whose total syllables == target
  2. Syllable +-1 match rate    -- tolerant match within one syllable (common
                                   vocal flexibility in sung lyrics)
  3. Syllable template accuracy -- position-wise accuracy of the per-word
                                   SYL/SEP sequence (1 - L_sep)
  4. Rhyme score                -- neighbour-weighted terminal-word phonetic
                                   similarity (weighted_sim, higher is better)
  5. Verbatim copy rate         -- fraction of lines copied whole from the
                                   context (lower is better; catches metric inflation)
  6. n-gram overlap             -- bigram overlap with the context (lower is better)
  7. BERTScore P / R / F1       -- semantic similarity vs ground truth (pluggable backend)
  8. (optional) ROUGE-1/2/L     -- only if rouge-score is installed

BERTScore backend priority (pluggable):
  bert_score library -> sentence-transformers cosine (F1 only) -> token-level P/R/F1 (fallback)

Input format (JSON): see DEMO_DATA at the end of this file, or pass --input your.json
  [
    {"song_id": "...",
     "lines": [
        {"template": [1,2,1,1] or "1-2-1-1" or "[SYL][SYL]...",
         "prev_lines": ["...", "..."],
         "ground_truth": "...",
         "baseline": "...",      # baseline LyCon output
         "reranked": "..."},     # LyCon + selection output
        ...
     ]}
  ]

Usage:
  python evaluate_reconstruction.py                      # run the built-in demo
  python evaluate_reconstruction.py --input runs.json --out_dir results/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as stats
from dataclasses import dataclass
from typing import Optional, Sequence

from lyrics_reranker import (
    parse_template, line_to_template, expand_to_syl_seq,
    tokenize_words, rhyme_loss, repetition_penalty,
)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
class BertScoreBackend:
    """
    Always returns (P, R, F1). Three-level degradation:
      bert_score library -> sentence-transformers cosine (P=R=F1=cos) -> token-level P/R/F1
    The token-level version uses ground-truth tokens as the recall base and
    candidate tokens as the precision base, structurally matching BERTScore,
    and runs even in an offline container.
    """

    def __init__(self, lang: str = "en"):
        self.mode = "token"
        self._bs = None
        self._sbert = None
        try:
            from bert_score import score as _bs_score  # type: ignore
            self._bs = _bs_score
            self.mode = "bert_score"
            self._lang = lang
            return
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.mode = "sbert"
            return
        except Exception:
            pass
        print("[BertScoreBackend] neither bert_score nor sentence-transformers "
              "available; using token-level P/R/F1 fallback.")

    def score_batch(self, cands: Sequence[str], refs: Sequence[str]) -> list[tuple[float, float, float]]:
        if self.mode == "bert_score":
            P, R, F = self._bs(list(cands), list(refs), lang=self._lang, verbose=False)
            return list(zip(P.tolist(), R.tolist(), F.tolist()))
        if self.mode == "sbert":
            import numpy as np
            ec = self._sbert.encode(list(cands))
            er = self._sbert.encode(list(refs))
            out = []
            for a, b in zip(ec, er):
                d = float(np.linalg.norm(a) * np.linalg.norm(b))
                cos = float(np.dot(a, b) / d) if d else 0.0
                out.append((cos, cos, cos))
            return out
        # --- token-level fallback ---
        out = []
        for c, r in zip(cands, refs):
            out.append(self._token_prf(c, r))
        return out

    @staticmethod
    def _token_prf(cand: str, ref: str) -> tuple[float, float, float]:
        ct, rt = tokenize_words(cand), tokenize_words(ref)
        if not ct or not rt:
            return (0.0, 0.0, 0.0)
        cset, rset = set(ct), set(rt)
        inter = len(cset & rset)
        P = inter / len(cset)
        R = inter / len(rset)
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        return (P, R, F)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def syllable_metrics(line: str, template) -> dict[str, float]:
    tgt = parse_template(template)
    pred = line_to_template(line)
    tgt_total, pred_total = sum(tgt), sum(pred)

    exact = 1.0 if pred_total == tgt_total else 0.0
    within1 = 1.0 if abs(pred_total - tgt_total) <= 1 else 0.0

    tgt_seq, pred_seq = expand_to_syl_seq(tgt), expand_to_syl_seq(pred)
    L = max(len(tgt_seq), len(pred_seq), 1)
    tgt_pad = tgt_seq + ["<pad>"] * (L - len(tgt_seq))
    pred_pad = pred_seq + ["<pad>"] * (L - len(pred_seq))
    match = sum(1 for a, b in zip(pred_pad, tgt_pad) if a == b) / L

    return {"syl_exact": exact, "syl_within1": within1, "syl_template_acc": match}


def rhyme_metric(line: str, prev_lines: Sequence[str], lam: float = 0.7,
                 N: Optional[int] = None, rhyme_tail_only: bool = True) -> float:
    """Higher is better: neighbour-weighted terminal-word phonetic similarity."""
    return rhyme_loss(line, prev_lines, lam=lam, N=N,
                      rhyme_tail_only=rhyme_tail_only)["weighted_sim"]


def repetition_metrics(line: str, prev_lines: Sequence[str]) -> dict[str, float]:
    r = repetition_penalty(line, prev_lines, n=2)
    return {"verbatim": r["is_verbatim"], "ngram_overlap": r["ngram_overlap"]}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class SystemResult:
    name: str
    per_line: list[dict]                      # all metrics for every line
    agg: dict[str, float]                      # aggregate means


def evaluate_system(data: list[dict], field: str, bert: BertScoreBackend,
                    lam: float = 0.7, rhyme_N: Optional[int] = None,
                    rhyme_tail_only: bool = True) -> SystemResult:
    rows: list[dict] = []
    cands, refs = [], []

    for song in data:
        for ln in song["lines"]:
            pred = ln.get(field, "") or ""
            prev = ln.get("prev_lines", [])
            gt = ln.get("ground_truth", "") or ""

            m = {}
            m.update(syllable_metrics(pred, ln["template"]))
            m["rhyme"] = rhyme_metric(pred, prev, lam, rhyme_N, rhyme_tail_only)
            m.update(repetition_metrics(pred, prev))
            m["_line"] = pred
            m["_song"] = song.get("song_id", "?")
            rows.append(m)
            cands.append(pred)
            refs.append(gt)

    prf = bert.score_batch(cands, refs)
    for row, (P, R, F) in zip(rows, prf):
        row["bert_P"], row["bert_R"], row["bert_F1"] = P, R, F

    keys = ["syl_exact", "syl_within1", "syl_template_acc", "rhyme",
            "verbatim", "ngram_overlap", "bert_P", "bert_R", "bert_F1"]
    agg = {k: stats.mean(r[k] for r in rows) for k in keys}
    return SystemResult(name=field, per_line=rows, agg=agg)


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def evaluate_ground_truth(data: list[dict], lam: float, rhyme_N, rhyme_tail_only) -> dict[str, float]:
    rhymes, verbs, overlaps = [], [], []
    for song in data:
        for ln in song["lines"]:
            gt = ln.get("ground_truth", "") or ""
            prev = ln.get("prev_lines", [])
            rhymes.append(rhyme_metric(gt, prev, lam, rhyme_N, rhyme_tail_only))
            rm = repetition_metrics(gt, prev)
            verbs.append(rm["verbatim"])
            overlaps.append(rm["ngram_overlap"])
    return {
        "rhyme": stats.mean(rhymes) if rhymes else 0.0,
        "verbatim": stats.mean(verbs) if verbs else 0.0,
        "ngram_overlap": stats.mean(overlaps) if overlaps else 0.0,
    }


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
_METRIC_ORDER = [
    ("syl_exact",        "Syllable exact match",   "↑"),
    ("syl_within1",      "Syllable ±1 match",      "↑"),
    ("syl_template_acc", "Syllable template acc",  "↑"),
    ("rhyme",            "Rhyme score",            "↑"),
    ("verbatim",         "Verbatim copy rate",     "↓"),
    ("ngram_overlap",    "Bigram overlap",         "↓"),
    ("bert_P",           "BERTScore P",            "↑"),
    ("bert_R",           "BERTScore R",            "↑"),
    ("bert_F1",          "BERTScore F1",           "↑"),
]


def _delta_str(base: float, new: float, better: str) -> str:
    if base == 0:
        return "  n/a"
    pct = (new - base) / abs(base) * 100
    return f"{pct:+.1f}%"


def make_markdown(base: SystemResult, rerank: SystemResult,
                  gt: Optional[dict], bert_mode: str) -> str:
    lines = []
    lines.append(f"# Reconstruction Evaluation: Baseline vs. Reranked\n")
    lines.append(f"BERTScore backend: `{bert_mode}`  |  "
                 f"lines evaluated: {len(base.per_line)}\n")
    lines.append("| Metric | Dir | Baseline | Reranked | Δ | GT ref |")
    lines.append("|---|:--:|--:|--:|--:|--:|")
    for key, label, direction in _METRIC_ORDER:
        b, r = base.agg[key], rerank.agg[key]
        d = _delta_str(b, r, direction)
        gtv = ""
        if gt is not None and key in gt:
            gtv = f"{gt[key]:.4f}"
        lines.append(f"| {label} | {direction} | {b:.4f} | {r:.4f} | {d} | {gtv} |")
    lines.append("\n*↑ higher is better, ↓ lower is better. "
                 "Δ = relative change of Reranked over Baseline.*")
    lines.append("\n> Note: read the Rhyme score together with Verbatim/Bigram overlap. "
                 "A baseline that inflates rhyme through repetition is exposed by "
                 "verbatim/overlap (the metric-inflation lesson of LYRICS paper Table II).")
    return "\n".join(lines)


def write_per_line_csv(path: str, base: SystemResult, rerank: SystemResult) -> None:
    fields = ["song", "system", "line", "syl_exact", "syl_within1",
              "syl_template_acc", "rhyme", "verbatim", "ngram_overlap",
              "bert_P", "bert_R", "bert_F1"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sysres in (base, rerank):
            for r in sysres.per_line:
                w.writerow({
                    "song": r["_song"], "system": sysres.name, "line": r["_line"],
                    "syl_exact": r["syl_exact"], "syl_within1": r["syl_within1"],
                    "syl_template_acc": round(r["syl_template_acc"], 4),
                    "rhyme": round(r["rhyme"], 4), "verbatim": r["verbatim"],
                    "ngram_overlap": round(r["ngram_overlap"], 4),
                    "bert_P": round(r["bert_P"], 4), "bert_R": round(r["bert_R"], 4),
                    "bert_F1": round(r["bert_F1"], 4),
                })


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
DEMO_DATA = [
    {
        "song_id": "demo_1",
        "lines": [
            {"template": [1, 1, 1, 2],
             "prev_lines": ["I walk the empty streets alone",
                            "beneath a sky of dying light"],
             "ground_truth": "I close my eyes so tight",
             "baseline": "beneath a sky of dying light",   # copied -> verbatim=1
             "reranked": "I close my eyes so tight"},
            {"template": [1, 1, 1, 1, 1],
             "prev_lines": ["beneath a sky of dying light",
                            "I close my eyes so tight"],
             "ground_truth": "and dream about the night",
             "baseline": "so tight so tight so tight",      # repeated phrase
             "reranked": "and dream about the night"},
        ],
    },
    {
        "song_id": "demo_2",
        "lines": [
            {"template": [1, 2, 1],
             "prev_lines": ["you never really cared at all",
                            "you let me watch the whole thing fall"],
             "ground_truth": "I heard you call",
             "baseline": "the whole thing fall",            # partial copy
             "reranked": "I heard you call"},
        ],
    },
]


# ----------------------------------------------------------------------------
# 7. main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="JSON data file; built-in demo if omitted")
    ap.add_argument("--out_dir", default=".", help="output directory")
    ap.add_argument("--lam", type=float, default=0.7, help="lambda for the rhyme neighbour weighting")
    ap.add_argument("--rhyme_N", type=int, default=None, help="rhyme considers only the last N lines")
    ap.add_argument("--no_tail", action="store_true",
                    help="rhyme uses whole-word features (as in the paper) instead of the rhyming tail only")
    args = ap.parse_args()

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = DEMO_DATA
        print(">>> no --input given; using the built-in demo data\n")

    os.makedirs(args.out_dir, exist_ok=True)
    tail_only = not args.no_tail

    bert = BertScoreBackend()
    print(f"BERTScore backend = {bert.mode}\n")

    base = evaluate_system(data, "baseline", bert, args.lam, args.rhyme_N, tail_only)
    rerank = evaluate_system(data, "reranked", bert, args.lam, args.rhyme_N, tail_only)
    gt = evaluate_ground_truth(data, args.lam, args.rhyme_N, tail_only)

    md = make_markdown(base, rerank, gt, bert.mode)
    print(md)

    md_path = os.path.join(args.out_dir, "eval_report.md")
    csv_path = os.path.join(args.out_dir, "eval_per_line.csv")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    write_per_line_csv(csv_path, base, rerank)
    print(f"\n\n[saved] {md_path}")
    print(f"[saved] {csv_path}")


if __name__ == "__main__":
    main()
