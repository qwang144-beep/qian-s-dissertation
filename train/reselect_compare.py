# -*- coding: utf-8 -*-
"""
reselect_compare.py -- zero-GPU pre-check for the scorer fix
============================================================
On an EXISTING cand_dump, for each line's identical candidate pool and
identical context, compares:
  OLD = the old scorer's choice (the recorded chosen; recomputed here as a
        determinism check)
  NEW = the fixed scorer's choice (exclude_same_word=True, tail_window=K)

Answers four questions:
  1. how many lines change their selection under the fix (switch rate)
  2. terminal-word reuse rate of the OLD vs NEW choices (stem-level, against
     the previous K lines of the original context)
  3. the ceiling: fraction of lines whose pool contains any non-reusing
     candidate at all (NEW can do no better than this)
  4. cost pre-check: exact syllable hit rate of the OLD vs NEW choices
     (does the fix hurt metre already at the selection level)

CAUTION: this is a diagnostic, not a result. The context is frozen to the
original run's chosen history (trajectory dependence); numbers usable in the
report must come from a full regeneration after patching. This script bounds
the first-step effect only.

Usage (from the directory above src/):
  python train/reselect_compare.py --dump cand_dump_n8.jsonl --manifest prompts_300.jsonl
Optional: --k-context 3 --lam 0.7 --w-count 3 --w-rhyme 1 --w-vocab 0.7 --w-rep 0.8 --limit 50
"""
from __future__ import annotations
import argparse, json, sys
sys.path.insert(0, "src")

from generate_metrical import (load_manifest, parse_song_ctx, select_best_line)
from lyrics_reranker import last_word, _stem_eq  # noqa: requires patch_scorer.py to have run


def load_dump(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["song_id"]] = rec["lines"]
    return out


def tail_reuses(line, prev, k):
    g = last_word(line)
    if g is None:
        return False
    recent = [last_word(p) for p in prev[-k:]]
    return any(_stem_eq(g, t) for t in recent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--k-context", type=int, default=3)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--w-count", type=float, default=3.0)
    ap.add_argument("--w-rhyme", type=float, default=1.0)
    ap.add_argument("--w-vocab", type=float, default=0.7)
    ap.add_argument("--w-rep", type=float, default=0.8)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N songs (0 = all)")
    a = ap.parse_args()

    W = dict(w_count=a.w_count, w_rhyme=a.w_rhyme, w_vocab=a.w_vocab,
             w_rep=a.w_rep, lam=a.lam)
    dump = load_dump(a.dump)
    manifest = load_manifest(a.manifest)
    K = a.k_context

    n = agree = switch = 0
    old_reuse = new_reuse = ceiling = 0
    old_exact = new_exact = 0
    from lyrics_reranker import rhyme_loss
    old_rh_fixed = new_rh_fixed = 0.0   # "true rhyme" loss of both, measured by the fixed formula

    songs = list(dump.items())
    if a.limit:
        songs = songs[:a.limit]
    for sid, lines in songs:
        rec = manifest.get(sid)
        if rec is None:
            continue
        ctx = parse_song_ctx(rec)
        history = [ln.get("chosen", "") for ln in lines]   # the original run's actual context
        for i, ln in enumerate(lines):
            cands = ln.get("candidates", [])
            if not cands:
                continue
            prev = [h for h in history[:i] if h]
            tgt = ln["target_syllables"]

            old_c = select_best_line(cands, tgt, prev, ctx.vocab_stems, **W)
            new_c = select_best_line(cands, tgt, prev, ctx.vocab_stems, **W,
                                     exclude_same_word=True, tail_window=K)
            if old_c is None or new_c is None:
                continue
            n += 1
            if old_c.line == ln.get("chosen"):
                agree += 1
            if new_c.line != old_c.line:
                switch += 1
            o_r = tail_reuses(old_c.line, prev, K)
            n_r = tail_reuses(new_c.line, prev, K)
            old_reuse += o_r
            new_reuse += n_r
            if any(not tail_reuses(c, prev, K) for c in cands):
                ceiling += 1
            old_exact += (old_c.count_diff == 0)
            new_exact += (new_c.count_diff == 0)
            old_rh_fixed += rhyme_loss(old_c.line, prev, lam=a.lam,
                                       rhyme_tail_only=True, exclude_same_word=True)["loss"]
            new_rh_fixed += rhyme_loss(new_c.line, prev, lam=a.lam,
                                       rhyme_tail_only=True, exclude_same_word=True)["loss"]

    if n == 0:
        sys.exit("no comparable lines -- check that the dump / manifest paths and song_ids correspond")
    p = lambda x: f"{x / n:7.2%}"
    print(f"lines compared            {n}")
    print(f"determinism check         {p(agree)}  (recomputed OLD vs recorded chosen; should be ~100%)")
    print(f"selection switched        {p(switch)}")
    print(f"terminal reuse  OLD       {p(old_reuse)}")
    print(f"terminal reuse  NEW       {p(new_reuse)}")
    print(f"pool ceiling              {p(n - ceiling)}  <- lines whose whole pool reuses; NEW has no way out")
    print(f"syllable exact  OLD       {p(old_exact)}")
    print(f"syllable exact  NEW       {p(new_exact)}")
    print(f"rhyme loss (fixed) OLD    {old_rh_fixed / n:.4f}")
    print(f"rhyme loss (fixed) NEW    {new_rh_fixed / n:.4f}   (lower is better; both measured with the fixed formula)")
    print("\nNOTE: first-step effect under a frozen context; report numbers require a full regeneration after patching.")


if __name__ == "__main__":
    main()
