# -*- coding: utf-8 -*-
"""
lyrics_reranker.py
==================
Recasts the three training losses of LYRICS (Ko et al., 2025) -- syllable /
context / rhyme -- as **inference-time scoring functions**, used to rerank the
N candidate lines produced for LyCon bag-of-words reconstruction
(best-of-N selection).

Design principles:
  - Training-free: preserves LyCon's core advantage of not fine-tuning the LLM.
  - Every scorer returns a "loss" (lower is better), matching the semantics of
    the paper's L_total; best-of-N = argmin(total_loss).
  - syllable / rhyme depend only on CMUdict: lightweight, CPU-only, offline.
  - semantic scoring is pluggable: sentence-transformers when available,
    degrading to token-level Jaccard so the code never hard-crashes.
  - An additional repetition penalty guards against the metric-inflation
    failure reported in the paper's Table II (rhyme score inflated by
    verbatim copying).

Dependencies:
  pip install pronouncing            # CMUdict wrapper, required
  pip install sentence-transformers  # optional, for the semantic scorer
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import pronouncing  # CMUdict wrapper

# ----------------------------------------------------------------------------
#    (place / manner / voicing for consonants; height / backness / rounding for vowels)
# ----------------------------------------------------------------------------
ARPABET_FEATURES: dict[str, set[str]] = {
    # --- Stops / plosive ---
    "P":  {"consonant", "plosive", "bilabial", "voiceless"},
    "B":  {"consonant", "plosive", "bilabial", "voiced"},
    "T":  {"consonant", "plosive", "alveolar", "voiceless"},
    "D":  {"consonant", "plosive", "alveolar", "voiced"},
    "K":  {"consonant", "plosive", "velar", "voiceless"},
    "G":  {"consonant", "plosive", "velar", "voiced"},
    # --- Affricates ---
    "CH": {"consonant", "affricate", "postalveolar", "voiceless"},
    "JH": {"consonant", "affricate", "postalveolar", "voiced"},
    # --- Fricatives ---
    "F":  {"consonant", "fricative", "labiodental", "voiceless"},
    "V":  {"consonant", "fricative", "labiodental", "voiced"},
    "TH": {"consonant", "fricative", "dental", "voiceless"},
    "DH": {"consonant", "fricative", "dental", "voiced"},
    "S":  {"consonant", "fricative", "alveolar", "voiceless"},
    "Z":  {"consonant", "fricative", "alveolar", "voiced"},
    "SH": {"consonant", "fricative", "postalveolar", "voiceless"},
    "ZH": {"consonant", "fricative", "postalveolar", "voiced"},
    "HH": {"consonant", "fricative", "glottal", "voiceless"},
    # --- Nasals ---
    "M":  {"consonant", "nasal", "bilabial", "voiced"},
    "N":  {"consonant", "nasal", "alveolar", "voiced"},
    "NG": {"consonant", "nasal", "velar", "voiced"},
    # --- Liquids ---
    "L":  {"consonant", "liquid", "alveolar", "voiced", "lateral"},
    "R":  {"consonant", "liquid", "alveolar", "voiced", "rhotic"},
    # --- Glides / approximant ---
    "W":  {"consonant", "glide", "labiovelar", "voiced"},
    "Y":  {"consonant", "glide", "palatal", "voiced"},
    # --- Vowels ---
    "IY": {"vowel", "high", "front", "unrounded", "tense"},
    "IH": {"vowel", "high", "front", "unrounded", "lax"},
    "EY": {"vowel", "mid", "front", "unrounded", "tense", "diphthong"},
    "EH": {"vowel", "mid", "front", "unrounded", "lax"},
    "AE": {"vowel", "low", "front", "unrounded", "lax"},
    "AA": {"vowel", "low", "back", "unrounded", "tense"},
    "AO": {"vowel", "mid", "back", "rounded", "tense"},
    "AH": {"vowel", "mid", "central", "unrounded", "lax"},
    "UW": {"vowel", "high", "back", "rounded", "tense"},
    "UH": {"vowel", "high", "back", "rounded", "lax"},
    "OW": {"vowel", "mid", "back", "rounded", "tense", "diphthong"},
    "AW": {"vowel", "low", "back", "rounded", "diphthong"},
    "AY": {"vowel", "low", "front", "unrounded", "diphthong"},
    "OY": {"vowel", "mid", "back", "rounded", "diphthong"},
    "ER": {"vowel", "mid", "central", "rhotic"},
}

_WORD_RE = re.compile(r"[a-zA-Z']+")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def tokenize_words(line: str) -> list[str]:
    """Extract words from a line (strips punctuation, keeps apostrophes as in don't)."""
    return _WORD_RE.findall(line.lower())


def phones_for(word: str) -> Optional[list[str]]:
    """Look up the ARPAbet phoneme sequence in CMUdict (first pronunciation); None if OOV."""
    prons = pronouncing.phones_for_word(word.lower())
    if not prons:
        return None
    return prons[0].split()


def _strip_stress(phone: str) -> str:
    """Strip the trailing stress digit 0/1/2 from a vowel phoneme, e.g. 'AH0' -> 'AH'."""
    return re.sub(r"\d", "", phone)


def count_syllables_word(word: str) -> int:
    """
    Syllables per word = number of CMUdict phonemes carrying a stress digit
    (i.e. the vowels). For OOV words, falls back to a vowel-group heuristic.
    """
    phones = phones_for(word)
    if phones is not None:
        return sum(1 for p in phones if p[-1].isdigit())
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and n > 1:  # silent 'e'
        n -= 1
    return max(1, n)


def line_to_template(line: str) -> list[int]:
    """Project a line onto its per-word syllable counts, e.g. 'hello world' -> [2, 1]."""
    return [count_syllables_word(w) for w in tokenize_words(line)]


def parse_template(template) -> list[int]:
    """
    Accepts three input forms:
      - list[int]               : [2, 1]  (per-word syllable counts directly)
      - '[SYL][SYL][SEP][SYL]'  : the paper's token encoding
      - '2-1' / '2 1'           : compact notation
    Always returns list[int].
    """
    if isinstance(template, (list, tuple)):
        return [int(x) for x in template]
    s = str(template).strip()
    if "[SYL]" in s.upper() or "[SEP]" in s.upper():
        words = re.split(r"\[SEP\]", s, flags=re.IGNORECASE)
        return [len(re.findall(r"\[SYL\]", w, flags=re.IGNORECASE)) for w in words if w.strip()]
    nums = re.findall(r"\d+", s)
    return [int(n) for n in nums]


def expand_to_syl_seq(template_counts: Sequence[int]) -> list[str]:
    """
    Expand per-word syllable counts into the paper's SYL/SEP token sequence.
    [2, 1] -> ['SYL','SYL','SEP','SYL']
    """
    seq: list[str] = []
    for i, c in enumerate(template_counts):
        if i > 0:
            seq.append("SEP")
        seq.extend(["SYL"] * c)
    return seq


# ----------------------------------------------------------------------------
# 2. Syllable Structure Loss  (L_syllable = L_sep + L_count)
# ----------------------------------------------------------------------------
def syllable_loss(line: str, template) -> dict[str, float]:
    """
    Reimplements the paper's L_syllable:
      L_sep   : position-wise mismatch rate of the SYL/SEP token sequences (padded)
      L_count : |generated total syllables - target total syllables|

    Returns a dict {'sep': ..., 'count_raw': ..., 'count_norm': ..., 'loss': ...}.
    'loss' is normalised to roughly [0, 1+], lower is better, for weighted selection.
    """
    tgt_counts = parse_template(template)
    pred_counts = line_to_template(line)

    tgt_seq = expand_to_syl_seq(tgt_counts)
    pred_seq = expand_to_syl_seq(pred_counts)

    L = max(len(tgt_seq), len(pred_seq), 1)
    tgt_pad = tgt_seq + ["<pad>"] * (L - len(tgt_seq))
    pred_pad = pred_seq + ["<pad>"] * (L - len(pred_seq))
    mismatches = sum(1 for a, b in zip(pred_pad, tgt_pad) if a != b)
    L_sep = mismatches / L  # 0..1

    tgt_total = sum(tgt_counts)
    pred_total = sum(pred_counts)
    L_count_raw = abs(pred_total - tgt_total)
    L_count_norm = min(1.0, L_count_raw / max(1, tgt_total))

    return {
        "sep": L_sep,
        "count_raw": float(L_count_raw),
        "count_norm": L_count_norm,
        "loss": L_sep + L_count_norm,  # both terms in [0,1], sum in [0,2]
    }


# ----------------------------------------------------------------------------
# 3. Rhyme Quality Loss  (L_rhyme)
# ----------------------------------------------------------------------------
def _rhyme_tail_phones(word: str) -> list[str]:
    """Phonemes from the last stressed vowel to the end of the word (the rhyming part)."""
    phones = phones_for(word)
    if not phones:
        return []
    idx = None
    for i, p in enumerate(phones):
        if p[-1] in ("1", "2"):
            idx = i
    if idx is None:  # no primary stress found: fall back to the whole word
        idx = 0
    return phones[idx:]


def word_feature_set(word: str, rhyme_tail_only: bool = False) -> set[str]:
    """
    Phi(word): map the word's phonemes to the union of their articulatory
    feature sets.
      rhyme_tail_only=False -> as in the paper: all phonemes of the whole word
      rhyme_tail_only=True  -> variant: only the rhyming tail (last stressed
                               vowel onward), more sensitive to slant rhyme
                               (an improvement direction the paper's
                               Limitations section suggests)
    """
    if rhyme_tail_only:
        phones = _rhyme_tail_phones(word)
    else:
        phones = phones_for(word) or []
    feats: set[str] = set()
    for p in phones:
        base = _strip_stress(p)
        feats |= ARPABET_FEATURES.get(base, set())
    return feats


def jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity |A&B| / |A|B|; two empty sets count as 0."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


try:                                             # scorer-fix: for stem comparison
    from nltk.stem import PorterStemmer
    _SCORER_STEM = PorterStemmer().stem
except Exception:                                # without nltk, degrade to lower-cased word
    _SCORER_STEM = lambda w: w

def _stem_eq(a: Optional[str], b: Optional[str]) -> bool:
    """scorer-fix: whether two words share the same Porter stem (case-insensitive)."""
    if a is None or b is None:
        return False
    return _SCORER_STEM(a.lower()) == _SCORER_STEM(b.lower())


def last_word(line: str) -> Optional[str]:
    words = tokenize_words(line)
    return words[-1] if words else None


def rhyme_loss(
    line: str,
    prev_lines: Sequence[str],
    lam: float = 0.7,
    N: Optional[int] = None,
    rhyme_tail_only: bool = False,
    exclude_same_word: bool = False,
) -> dict[str, float]:
    """
    Reimplements the paper's L_rhyme = sum_i w_i (1 - sim(P_i, G)), w_i ~ lam^(N-i).
      line       : candidate line (supplies the terminal word G)
      prev_lines : preceding lines (supply the P_i); later = nearer = higher weight
      lam        : lam in (0,1); smaller = rhyme mostly with the nearest line
      N          : consider only the last N lines; None = all of prev_lines
      rhyme_tail_only : see word_feature_set

    Returns {'loss': .., 'best_sim': .., 'weighted_sim': ..}.
    'loss' is normalised by the weight sum to [0,1]; lower = better rhyme.
    """
    G = last_word(line)
    ctx = list(prev_lines)
    if N is not None:
        ctx = ctx[-N:]
    if G is None or not ctx:
        return {"loss": 1.0, "best_sim": 0.0, "weighted_sim": 0.0}

    G_feat = word_feature_set(G, rhyme_tail_only)
    n = len(ctx)
    weights, sims = [], []
    for i, pline in enumerate(ctx, start=1):
        Pi = last_word(pline)
        if Pi is None:
            continue
        w = lam ** (n - i)
        if exclude_same_word and _stem_eq(Pi, G):
            s = 0.0                              # scorer-fix: a word does not rhyme with itself
        else:
            s = jaccard(word_feature_set(Pi, rhyme_tail_only), G_feat)
        weights.append(w)
        sims.append(s)

    if not weights:
        return {"loss": 1.0, "best_sim": 0.0, "weighted_sim": 0.0}

    wsum = sum(weights)
    loss = sum(w * (1.0 - s) for w, s in zip(weights, sims)) / wsum  # normalised to [0,1]
    weighted_sim = sum(w * s for w, s in zip(weights, sims)) / wsum
    return {
        "loss": loss,
        "best_sim": max(sims),
        "weighted_sim": weighted_sim,
    }


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
class SemanticScorer:
    """
    Semantic-consistency scorer. The context may be the preceding lines or a
    string built from the bag of words -- for the LyCon setting the bag is the
    more natural context (it checks whether the candidate stays faithful to
    the given word set).
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model = None
        self._mode = "fallback"
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(model_name)
            self._mode = "sbert"
        except Exception as e:  # noqa: BLE001
            print(f"[SemanticScorer] sentence-transformers unavailable ({e}); "
                  f"falling back to token-Jaccard.")

    @property
    def mode(self) -> str:
        return self._mode

    def _cos(self, a, b) -> float:
        import numpy as np
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(a @ b / denom) if denom else 0.0

    def loss(self, line: str, context: str) -> float:
        """Return 1 - cosine(embed(line), embed(context)), clamped to [0,1]."""
        if self._mode == "sbert":
            emb = self._model.encode([line, context])
            cos = self._cos(emb[0], emb[1])
        else:
            ta = set(tokenize_words(line))
            tb = set(tokenize_words(context))
            cos = jaccard(ta, tb)
        cos = max(0.0, min(1.0, cos))
        return 1.0 - cos


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
def _ngrams(tokens: Sequence[str], n: int) -> set[tuple]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)} if len(tokens) >= n else set()


def repetition_penalty(line: str, prev_lines: Sequence[str], n: int = 2,
                       tail_window: int = 0) -> dict[str, float]:
    """
    n-gram overlap between the candidate and the preceding lines, plus
    whole-line copy detection.
    Returns {'ngram_overlap': .., 'is_verbatim': .., 'loss': ..}; loss in [0,1],
    lower is better.
    """
    cand = tokenize_words(line)
    if not cand:
        return {"ngram_overlap": 0.0, "is_verbatim": 0.0, "loss": 1.0}

    cand_ng = _ngrams(cand, n)
    prev_ng: set[tuple] = set()
    verbatim = 0.0
    for p in prev_lines:
        pt = tokenize_words(p)
        prev_ng |= _ngrams(pt, n)
        if pt == cand:  # whole-line copy
            verbatim = 1.0

    if not cand_ng:
        overlap = verbatim  # line too short for n-grams: only check whole-line copy
    else:
        overlap = len(cand_ng & prev_ng) / len(cand_ng)

    tail_reuse = 0.0                             # scorer-fix: price terminal-word reuse explicitly
    if tail_window > 0:
        g = last_word(line)
        recent = [last_word(p) for p in list(prev_lines)[-tail_window:]]
        if g is not None and any(_stem_eq(g, t) for t in recent):
            tail_reuse = 1.0

    loss = max(overlap, verbatim, tail_reuse)  # copying / terminal reuse saturate the loss
    return {"ngram_overlap": overlap, "is_verbatim": verbatim,
            "tail_reuse": tail_reuse, "loss": loss}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
@dataclass
class RerankWeights:
    """
    Per-term weights. Suggested defaults for the LyCon setting:
      - the bag of words already pins down the vocabulary -> semantic weight low
      - LyCon itself never controls syllables or rhyme -> syllable / rhyme high
      - repetition medium-high, to guard against degeneration
    """
    syllable: float = 1.0
    rhyme: float = 1.0
    semantic: float = 0.3
    repetition: float = 0.8


@dataclass
class Scored:
    line: str
    total: float
    parts: dict[str, float] = field(default_factory=dict)


def score_candidate(
    line: str,
    prev_lines: Sequence[str],
    template,
    *,
    context: Optional[str] = None,
    semantic_scorer: Optional[SemanticScorer] = None,
    weights: RerankWeights = RerankWeights(),
    lam: float = 0.7,
    rhyme_N: Optional[int] = None,
    rhyme_tail_only: bool = False,
) -> Scored:
    """Score one candidate line (total loss, lower is better) with a per-term breakdown."""
    syl = syllable_loss(line, template)
    rhy = rhyme_loss(line, prev_lines, lam=lam, N=rhyme_N, rhyme_tail_only=rhyme_tail_only)
    rep = repetition_penalty(line, prev_lines)

    if semantic_scorer is not None:
        ctx = context if context is not None else " ".join(prev_lines)
        sem_loss = semantic_scorer.loss(line, ctx)
    else:
        sem_loss = 0.0  # no semantic scorer provided: term not counted

    total = (
        weights.syllable * syl["loss"]
        + weights.rhyme * rhy["loss"]
        + weights.semantic * sem_loss
        + weights.repetition * rep["loss"]
    )
    return Scored(
        line=line,
        total=total,
        parts={
            "syllable": syl["loss"],
            "syl_sep": syl["sep"],
            "syl_count_raw": syl["count_raw"],
            "rhyme": rhy["loss"],
            "rhyme_weighted_sim": rhy["weighted_sim"],
            "semantic": sem_loss,
            "repetition": rep["loss"],
        },
    )


def rerank(
    candidates: Sequence[str],
    prev_lines: Sequence[str],
    template,
    *,
    context: Optional[str] = None,
    semantic_scorer: Optional[SemanticScorer] = None,
    weights: RerankWeights = RerankWeights(),
    lam: float = 0.7,
    rhyme_N: Optional[int] = None,
    rhyme_tail_only: bool = False,
    top_k: Optional[int] = None,
) -> list[Scored]:
    """
    Best-of-N selection over a set of candidate lines; returns Scored objects
    sorted by ascending total loss. result[0].line is the selected line.

    Typical usage:
        sample N candidates for line j ->
        rerank(candidates, prev_lines=lines already committed,
               template=target syllable template for line j) ->
        take [0] as line j -> append to prev_lines -> proceed to line j+1
    """
    scored = [
        score_candidate(
            c, prev_lines, template,
            context=context, semantic_scorer=semantic_scorer, weights=weights,
            lam=lam, rhyme_N=rhyme_N, rhyme_tail_only=rhyme_tail_only,
        )
        for c in candidates
    ]
    scored.sort(key=lambda s: s.total)
    return scored[:top_k] if top_k else scored


# ----------------------------------------------------------------------------
# 7. Demo
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    prev = [
        "I walk the empty streets alone",
        "beneath a sky of dying light",
    ]
    template = [1, 2, 1, 1]

    candidates = [
        "and dream about the night",       # rhymes light/night
        "beneath a sky of dying light",    # verbatim copy -> repetition should punish it
        "the cold wind bites my skin",     # coherent but does not rhyme
        "I close my eyes so tight",        # rhymes, syllable count close
        "no",                              # far too short -> large syllable loss
    ]

    sem = SemanticScorer()  # degrades automatically if sbert is missing
    print(f"\nSemanticScorer mode = {sem.mode}\n")

    results = rerank(
        candidates, prev_lines=prev, template=template,
        semantic_scorer=sem,
        weights=RerankWeights(syllable=1.0, rhyme=1.0, semantic=0.3, repetition=0.8),
        lam=0.7, rhyme_tail_only=True,
    )

    print(f"target per-word syllable template = {parse_template(template)} "
          f"(total {sum(parse_template(template))})\n")
    print(f"{'rank':<5}{'total':>8}  {'syl':>6}{'rhyme':>7}{'sem':>6}{'rep':>6}   line")
    print("-" * 72)
    for r, s in enumerate(results, 1):
        p = s.parts
        print(f"{r:<5}{s.total:>8.3f}  {p['syllable']:>6.2f}{p['rhyme']:>7.2f}"
              f"{p['semantic']:>6.2f}{p['repetition']:>6.2f}   {s.line}")
    print(f"\n>>> selected: {results[0].line!r}")
