# -*- coding: utf-8 -*-
"""
patch_scorer.py -- scorer fix (run once, before any training)
=============================================================
Two changes, both introduced as NEW OPTIONAL PARAMETERS whose defaults keep
the old behaviour: existing scripts and old results stay 100% reproducible,
and only calls that pass the new parameters enable the fix.

  1. rhyme_loss(..., exclude_same_word=False)
       When True: a candidate whose terminal word shares a STEM with a
       context line's terminal word gets sim = 0 for that pair.
       Rhyme means "similar sound, different word" -- a word does not rhyme
       with itself. This moves the global optimum of L_rhyme from
       "reuse the terminal word" back to genuine rhyme.

  2. repetition_penalty(..., tail_window=0)
       When >0: if the candidate's terminal stem appears among the terminal
       stems of the last tail_window lines, tail_reuse=1 and
       loss = max(bigram, verbatim, tail_reuse).
       Terminal-word reuse is priced explicitly for the first time
       (the term proposed in the report).

  3. Wiring into select_best_line / the generate_metrical CLI:
       adds the --fix-rhyme and --tail-window N switches.

Usage:  python patch_scorer.py            # run from the directory above src/
        python patch_scorer.py --check    # only check whether already patched

NOTE: the src/ files in this repository are already patched; this script is
kept for provenance and reports [ok] in --check mode.
"""
import argparse, sys, os

RERANKER = "src/lyrics_reranker.py"
GENERATE = "src/generate_metrical.py"

EDITS_RERANKER = [
("""def last_word(line: str) -> Optional[str]:""",
"""try:                                             # scorer-fix: for stem comparison
    from nltk.stem import PorterStemmer
    _SCORER_STEM = PorterStemmer().stem
except Exception:                                # without nltk, degrade to lower-cased word
    _SCORER_STEM = lambda w: w

def _stem_eq(a: Optional[str], b: Optional[str]) -> bool:
    \"\"\"scorer-fix: whether two words share the same Porter stem (case-insensitive).\"\"\"
    if a is None or b is None:
        return False
    return _SCORER_STEM(a.lower()) == _SCORER_STEM(b.lower())


def last_word(line: str) -> Optional[str]:"""),

("""    lam: float = 0.7,
    N: Optional[int] = None,
    rhyme_tail_only: bool = False,
) -> dict[str, float]:""",
"""    lam: float = 0.7,
    N: Optional[int] = None,
    rhyme_tail_only: bool = False,
    exclude_same_word: bool = False,
) -> dict[str, float]:"""),

("""        w = lam ** (n - i)
        s = jaccard(word_feature_set(Pi, rhyme_tail_only), G_feat)
        weights.append(w)
        sims.append(s)""",
"""        w = lam ** (n - i)
        if exclude_same_word and _stem_eq(Pi, G):
            s = 0.0                              # scorer-fix: a word does not rhyme with itself
        else:
            s = jaccard(word_feature_set(Pi, rhyme_tail_only), G_feat)
        weights.append(w)
        sims.append(s)"""),

("""def repetition_penalty(line: str, prev_lines: Sequence[str], n: int = 2) -> dict[str, float]:""",
"""def repetition_penalty(line: str, prev_lines: Sequence[str], n: int = 2,
                       tail_window: int = 0) -> dict[str, float]:"""),

("""    loss = max(overlap, verbatim)  # copying saturates the loss
    return {"ngram_overlap": overlap, "is_verbatim": verbatim, "loss": loss}""",
"""    tail_reuse = 0.0                             # scorer-fix: price terminal-word reuse explicitly
    if tail_window > 0:
        g = last_word(line)
        recent = [last_word(p) for p in list(prev_lines)[-tail_window:]]
        if g is not None and any(_stem_eq(g, t) for t in recent):
            tail_reuse = 1.0

    loss = max(overlap, verbatim, tail_reuse)  # copying / terminal reuse saturate the loss
    return {"ngram_overlap": overlap, "is_verbatim": verbatim,
            "tail_reuse": tail_reuse, "loss": loss}"""),
]

EDITS_GENERATE = [
("""                     w_count: float = 3.0, w_rhyme: float = 1.0,
                     w_vocab: float = 0.7, w_rep: float = 0.8,
                     lam: float = 0.7, rhyme_tail_only: bool = True) -> Optional[LineChoice]:""",
"""                     w_count: float = 3.0, w_rhyme: float = 1.0,
                     w_vocab: float = 0.7, w_rep: float = 0.8,
                     lam: float = 0.7, rhyme_tail_only: bool = True,
                     exclude_same_word: bool = False,
                     tail_window: int = 0) -> Optional[LineChoice]:"""),

("""        rh = rhyme_loss(c, prev_lines, lam=lam, rhyme_tail_only=rhyme_tail_only)["loss"]
        vf = vocab_fidelity(c, vocab_stems)
        rep = repetition_penalty(c, prev_lines)["loss"]""",
"""        rh = rhyme_loss(c, prev_lines, lam=lam, rhyme_tail_only=rhyme_tail_only,
                        exclude_same_word=exclude_same_word)["loss"]
        vf = vocab_fidelity(c, vocab_stems)
        rep = repetition_penalty(c, prev_lines, tail_window=tail_window)["loss"]"""),

("""    ap.add_argument("--w-rep", type=float, default=0.8)""",
"""    ap.add_argument("--w-rep", type=float, default=0.8)
    ap.add_argument("--fix-rhyme", action="store_true",
                    help="scorer-fix: same-stem terminal words score sim 0 (rhyming requires a different word)")
    ap.add_argument("--tail-window", type=int, default=0,
                    help="scorer-fix: look-back window for the terminal-reuse term in L_rep (0 = off)")"""),

("""    select_kwargs = dict(w_count=args.w_count, w_rhyme=args.w_rhyme,""",
"""    select_kwargs = dict(exclude_same_word=args.fix_rhyme,
                         tail_window=args.tail_window,
                         w_count=args.w_count, w_rhyme=args.w_rhyme,"""),
]

def apply(path, edits, check_only):
    s = open(path, encoding="utf-8").read()
    done = sum(1 for old, new in edits if new in s)
    if done == len(edits):
        print(f"[ok] {path}: already patched ({done}/{len(edits)})")
        return True
    if check_only:
        print(f"[--] {path}: not patched ({done}/{len(edits)} present)")
        return False
    for i, (old, new) in enumerate(edits, 1):
        if new in s:
            continue
        if old not in s:
            sys.exit(f"[FAIL] {path} edit {i}: target text not found -- file differs from expectation, aborting (nothing written)")
        s = s.replace(old, new, 1)
    open(path + ".bak_scorerfix", "w", encoding="utf-8").write(open(path, encoding="utf-8").read())
    open(path, "w", encoding="utf-8").write(s)
    print(f"[patched] {path} (backup at {path}.bak_scorerfix)")
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if not os.path.exists(RERANKER):
        sys.exit("run from the directory above src/ (src/lyrics_reranker.py not found)")
    ok1 = apply(RERANKER, EDITS_RERANKER, a.check)
    ok2 = apply(GENERATE, EDITS_GENERATE, a.check)
    if ok1 and ok2 and not a.check:
        print("\nVerify with: python -c \"import sys; sys.path.insert(0,'src'); "
              "from lyrics_reranker import rhyme_loss, repetition_penalty; "
              "print(rhyme_loss('we drift low',['fires burn low'],exclude_same_word=True)); "
              "print(repetition_penalty('we drift low',['fires burn low'],tail_window=3))\"")
