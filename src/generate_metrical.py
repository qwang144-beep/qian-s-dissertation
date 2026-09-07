# -*- coding: utf-8 -*-
"""
generate_metrical.py  --  line-by-line, syllable-constrained LyCon reconstruction
=================================================================================
Replaces LyCon's whole-song generation with "line-by-line generation + per-line
syllable-skeleton constraint + best-of-N selection", still training-free
(prompt-conditioned inference + inference-time selection).

Inputs:
  --manifest  prompts.jsonl        LyCon manifest (bag of words + metadata + prompt)
  --skeletons skeletons_300.jsonl  per-line syllable skeletons extracted from the
                                   reference by lyric_txt_parser.py
Output:
  --out-dir   line-by-line reconstructions, one <track_id>.txt per song,
              lines aligned one-to-one with the reference

Per-line flow: build the line-level prompt (genre/artist/mood/title/vocabulary
         + previous K lines + target syllable count)
         -> sample N candidates in one prefill (num_return_sequences=N)
         -> select_best_line(): syllable count dominant, rhyme / vocabulary
            fidelity / repetition auxiliary -> pick one line
         -> append to the context, proceed to the next line

Compute optimisations (written for a Colab T4):
  - the N candidates are sampled in a single prefill per line
  - the model stays resident for the whole run
  - --resume skips songs whose output already exists
  - --dry-run tests the orchestration offline with stub candidates, no model load

Dependencies: lyrics_reranker.py (same directory);
      transformers/accelerate/bitsandbytes (for real runs);
      nltk (vocabulary stemming; degrades without it)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional

from lyrics_reranker import (
    line_to_template, rhyme_loss, repetition_penalty, tokenize_words,
)

try:
    from nltk.stem import PorterStemmer
    _STEM = PorterStemmer().stem
except Exception:  # noqa: BLE001
    def _STEM(w: str) -> str:  # minimal fallback: crude stemming when nltk is unavailable
        for suf in ("ing", "ed", "es", "s", "e"):
            if len(w) > len(suf) + 2 and w.endswith(suf):
                return w[: -len(suf)]
        return w


# ============================================================================
# ============================================================================
def load_manifest(path: str) -> dict[str, dict]:
    """prompts.jsonl -> {track_id: record}"""
    out: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["track_id"]] = rec
    return out


def load_skeletons(path: str) -> dict[str, list[dict]]:
    """skeletons.jsonl -> {song_id: [line dict, ...]}"""
    out: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["song_id"]] = rec["lines"]
    return out


# ============================================================================
# ============================================================================
_VOCAB_RE = re.compile(r"using the following vocabulary\s+(.*?)\.?\s*$",
                       re.IGNORECASE | re.DOTALL)
_MOOD_RE = re.compile(r"represents an?\s+([a-z\-]+)\s+mood", re.IGNORECASE)


@dataclass
class SongCtx:
    track_id: str
    title: str
    artist: str
    genre: str
    mood: str
    vocab: list[str]          # stemmed vocabulary (fed to the model as-is)
    vocab_stems: set[str]     # used for vocab-fidelity scoring


def parse_song_ctx(rec: dict) -> SongCtx:
    meta = rec.get("meta", {})
    prompt = rec.get("prompt", "")

    m = _VOCAB_RE.search(prompt)
    if m:
        vocab_raw = m.group(1).strip().rstrip(".")
        vocab = [w.strip() for w in vocab_raw.split(",") if w.strip()]
    else:
        vocab = []

    mm = _MOOD_RE.search(prompt)
    mood = mm.group(1) if mm else "neutral"

    genre = " ".join(rec["genre"]) if isinstance(rec.get("genre"), list) else str(rec.get("genre", ""))

    return SongCtx(
        track_id=rec["track_id"],
        title=meta.get("title", ""),
        artist=meta.get("artist", ""),
        genre=genre,
        mood=mood,
        vocab=vocab,
        vocab_stems={_STEM(w.lower()) for w in vocab},
    )


# ============================================================================
# ============================================================================
SYSTEM_PROMPT = (
    "You are a songwriter reconstructing a song one line at a time from a fixed "
    "vocabulary. Every line you write must (1) draw its words from the given "
    "vocabulary (you may inflect the stems), (2) fit the song's style and mood, "
    "and (3) contain EXACTLY the requested number of syllables. "
    "Output only the single line, with no quotes, no numbering, no commentary."
)

SYSTEM_PROMPT_STRICT_VARY = (
    "You are a songwriter reconstructing a song one line at a time from a fixed "
    "vocabulary. Every line you write must (1) draw its words from the given "
    "vocabulary (you may inflect the stems), (2) fit the song's style and mood, "
    "and (3) contain EXACTLY the requested number of syllables. "
    "Vary your sentence structure: do not reuse the grammatical pattern or the "
    "opening words of the previous lines. "
    "Output only the single line, with no quotes, no numbering, no commentary."
)

SYSTEM_PROMPT_RELAXED = (
    "You are a songwriter reconstructing a song one line at a time from a fixed "
    "vocabulary. Every line you write must (1) draw its words from the given "
    "vocabulary (you may inflect the stems), (2) fit the song's style and mood, "
    "and (3) be close to the requested number of syllables. "
    "Vary your sentence structure: do not reuse the grammatical pattern or the "
    "opening words of the previous lines. "
    "Output only the single line, with no quotes, no numbering, no commentary."
)

SYSTEM_PROMPT_RHYME = (
    "You are a songwriter reconstructing a song one line at a time from a fixed "
    "vocabulary. Every line you write must (1) draw its words from the given "
    "vocabulary (you may inflect the stems), (2) fit the song's style and mood, "
    "and (3) rhyme with the preceding lines where possible, but never by "
    "repeating their final words. "
    "Vary your sentence structure: do not reuse the grammatical pattern or the "
    "opening words of the previous lines. "
    "Output only the single line, with no quotes, no numbering, no commentary."
)


def build_line_user_prompt(ctx: SongCtx, prev_lines: list[str],
                           target_syllables: int, k_context: int,
                           relax: bool = False, anti_template: bool = False,
                           no_syllable: bool = False) -> str:
    prev = prev_lines[-k_context:] if k_context > 0 else []
    ctx_block = "\n".join(prev) if prev else "(this is the first line)"
    vocab_str = ", ".join(ctx.vocab)

    if no_syllable:
        syl_instr = "Write the next line. "
    elif relax:
        syl_instr = (f"Write the next line. Aim for about {target_syllables} "
                     f"syllables (a little more or less is fine). ")
    else:
        syl_instr = (f"Write the next line. It must have EXACTLY "
                     f"{target_syllables} syllables. ")

    anti = ""
    if anti_template and prev:
        openings = ", ".join(f'"{" ".join(p.split()[:2])}"' for p in prev if p.split())
        anti = (f"The previous lines begin with {openings}. "
                f"Start this line differently and use a different sentence structure. ")

    return (
        f'Song: "{ctx.title}" by {ctx.artist}. '
        f"Genre: {ctx.genre}. Mood: {ctx.mood}.\n"
        f"Vocabulary (stems, may be inflected): {vocab_str}.\n\n"
        f"Lines so far:\n{ctx_block}\n\n"
        f"{syl_instr}{anti}"
        f"Output only the line."
    )


# ============================================================================
# ============================================================================
_LINE_PREFIX_RE = re.compile(r"^\s*(line\s*\d+\s*[:.\-]?\s*|\d+[\).\-]\s*)", re.IGNORECASE)


def clean_candidate(text: str) -> str:
    """Take the first usable line of model output; strip quotes, numbering, extra whitespace."""
    if not text:
        return ""
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = _LINE_PREFIX_RE.sub("", s)          # strip "Line 3:" / "3)" prefixes
        s = s.strip().strip('"').strip("'").strip("“”").strip()
        if s:
            return s
    return ""


def vocab_fidelity(line: str, vocab_stems: set[str]) -> float:
    """Fraction of candidate stems inside the vocabulary (higher = more faithful). Empty line scores 0."""
    toks = tokenize_words(line)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if _STEM(t.lower()) in vocab_stems)
    return hits / len(toks)


@dataclass
class LineChoice:
    line: str
    n_syllables: int
    count_diff: int
    rhyme_loss: float
    vocab_fidelity: float
    repetition: float
    total: float


def first_line(candidates_clean: list[str], target_syllables: int,
               prev_lines: list[str], vocab_stems: set[str],
               lam: float = 0.7, rhyme_tail_only: bool = True) -> Optional[LineChoice]:
    """No selection: take the first non-empty candidate.

    Matches the definition baseline = candidates[0] in build_eval_input.py.
    The per-term scores are still computed, but only for logging; they play no
    part in any choice. total is recorded as 0.0 (there is no cost to speak of).
    """
    if not candidates_clean:
        return None
    c = candidates_clean[0]
    n = sum(line_to_template(c))
    rh = rhyme_loss(c, prev_lines, lam=lam, rhyme_tail_only=rhyme_tail_only)["loss"]
    vf = vocab_fidelity(c, vocab_stems)
    rep = repetition_penalty(c, prev_lines)["loss"]
    return LineChoice(c, n, abs(n - target_syllables), rh, vf, rep, 0.0)


def select_best_line(candidates: list[str], target_syllables: int,
                     prev_lines: list[str], vocab_stems: set[str],
                     w_count: float = 3.0, w_rhyme: float = 1.0,
                     w_vocab: float = 0.7, w_rep: float = 0.8,
                     lam: float = 0.7, rhyme_tail_only: bool = True,
                     exclude_same_word: bool = False,
                     tail_window: int = 0) -> Optional[LineChoice]:
    """
    Line-selection criterion: syllable-count match dominant (w_count largest,
    acting as a soft-hard filter), with rhyme / vocabulary fidelity / repetition
    auxiliary. Returns the candidate with the lowest total.
    """
    best: Optional[LineChoice] = None
    for raw in candidates:
        c = clean_candidate(raw)
        if not c:
            continue
        n = sum(line_to_template(c))
        count_diff = abs(n - target_syllables)
        rh = rhyme_loss(c, prev_lines, lam=lam, rhyme_tail_only=rhyme_tail_only,
                        exclude_same_word=exclude_same_word)["loss"]
        vf = vocab_fidelity(c, vocab_stems)
        rep = repetition_penalty(c, prev_lines, tail_window=tail_window)["loss"]

        total = (
            w_count * (count_diff / max(1, target_syllables))
            + w_rhyme * rh
            + w_vocab * (1.0 - vf)
            + w_rep * rep
        )
        choice = LineChoice(c, n, count_diff, rh, vf, rep, total)
        if best is None or choice.total < best.total:
            best = choice
    return best


# ============================================================================
# ============================================================================
class HFGenerator:
    """Qwen2.5-Instruct (optionally 4-bit). One prefill per line; N candidates sampled at once."""

    def __init__(self, model_name: str, load_4bit: bool = False,
                 max_new_tokens: int = 24, temperature: float = 0.9,
                 top_p: float = 0.95, seed: int = 13):
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"torch_dtype": "auto", "device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            import torch as _t
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=_t.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        import transformers
        transformers.set_seed(seed)

    def sample(self, system: str, user: str, n: int) -> list[str]:
        import torch
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, do_sample=True, temperature=self.temperature,
                top_p=self.top_p, max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,           # N candidates from a single prefill
                pad_token_id=self.tok.eos_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]     # keep only the newly generated part
        return [self.tok.decode(g, skip_special_tokens=True) for g in gen]


class StubGenerator:
    """For --dry-run: fabricate stub candidates from the target syllable count, to test orchestration/selection/output without loading the model."""
    _POOL = [
        "we drift away tonight", "under the deep blue sea", "i run into the fire",
        "come closer hold me now", "the shadows fall like rain", "no", "away away away",
        "your wings like diamonds shine", "we bury all our fears",
    ]

    def sample(self, system: str, user: str, n: int) -> list[str]:
        import random
        return random.sample(self._POOL, min(n, len(self._POOL)))


# ============================================================================
# ============================================================================
def reconstruct_song(ctx: SongCtx, skeleton: list[dict], gen,
                     n_candidates: int, k_context: int,
                     select_kwargs: dict,
                     relax: bool = False,
                     anti_template: bool = False,
                     no_syllable: bool = False,
                     no_select: bool = False,
                     vary_system: bool = False) -> tuple[list[str], list[LineChoice], list[dict]]:
    prev_lines: list[str] = []
    choices: list[LineChoice] = []
    dump: list[dict] = []                        # all candidates + choice per line, for offline ablation/weight tuning
    for li, ln in enumerate(skeleton):
        target = ln["n_syllables"]
        user = build_line_user_prompt(ctx, prev_lines, target, k_context,
                                      relax=relax, anti_template=anti_template,
                                      no_syllable=no_syllable)
        if no_syllable:
            sys_prompt = SYSTEM_PROMPT_RHYME
        elif relax:
            sys_prompt = SYSTEM_PROMPT_RELAXED
        elif vary_system:
            sys_prompt = SYSTEM_PROMPT_STRICT_VARY
        else:
            sys_prompt = SYSTEM_PROMPT
        raw_cands = gen.sample(sys_prompt, user, n_candidates)
        cleaned = [c for c in (clean_candidate(x) for x in raw_cands) if c]
        if no_select:
            choice = first_line(cleaned, target, prev_lines, ctx.vocab_stems)
        else:
            choice = select_best_line(raw_cands, target, prev_lines, ctx.vocab_stems,
                                      **select_kwargs)
        if choice is None:                       # degenerate case: every candidate empty
            choice = LineChoice("", 0, target, 1.0, 0.0, 1.0, 999.0)
        prev_lines.append(choice.line)
        choices.append(choice)
        dump.append({
            "idx": li,
            "target_syllables": target,
            "candidates": cleaned,               # cleaned, empties removed
            "chosen": choice.line,               # the selected line (= context at generation time)
        })
    return prev_lines, choices, dump


# ============================================================================
# 7. main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--skeletons", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip songs whose output already exists")
    ap.add_argument("--dry-run", action="store_true", help="stub candidates, no model load")
    ap.add_argument("--n-candidates", type=int, default=8)
    ap.add_argument("--k-context", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N songs")
    ap.add_argument("--songs", nargs="*", default=None, help="process only the given track_ids")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dump-candidates", default=None,
                    help="write all candidates + choice per line to this JSONL, for offline ablation/weight tuning")
    ap.add_argument("--relax-syllable", action="store_true",
                    help="prompt says about N syllables instead of EXACTLY N")
    ap.add_argument("--anti-template", action="store_true",
                    help="prompt explicitly asks to avoid the openings and sentence patterns of preceding lines")
    ap.add_argument("--no-select", action="store_true",
                    help="no weighted selection; take the first non-empty candidate per line (end-to-end baseline)")
    ap.add_argument("--no-syllable", action="store_true",
                    help="prompt never mentions syllable counts, only rhyme (use with --w-count 0)")
    
    ap.add_argument("--w-count", type=float, default=3.0)
    ap.add_argument("--w-rhyme", type=float, default=1.0)
    ap.add_argument("--w-vocab", type=float, default=0.7)
    ap.add_argument("--w-rep", type=float, default=0.8)
    ap.add_argument("--fix-rhyme", action="store_true",
                    help="scorer-fix: same-stem terminal words score sim 0 (rhyming requires a different word)")
    ap.add_argument("--tail-window", type=int, default=0,
                    help="scorer-fix: look-back window for the terminal-reuse term in L_rep (0 = off)")
    ap.add_argument("--vary-system", action="store_true",
                    help="strict prompt + system-level 'vary your sentence structure' sentence "
                         "(the vary run of the report)")
    
    args = ap.parse_args()

    manifest = load_manifest(args.manifest)
    skeletons = load_skeletons(args.skeletons)
    os.makedirs(args.out_dir, exist_ok=True)

    ids = [t for t in manifest if t in skeletons]
    if args.songs:
        ids = [t for t in ids if t in set(args.songs)]
    if args.limit:
        ids = ids[: args.limit]
    print(f"[info] {len(ids)} songs to reconstruct "
          f"(manifest {len(manifest)}, skeletons {len(skeletons)})")

    if args.dry_run:
        gen = StubGenerator()
        print("[info] DRY-RUN: using stub generator (no model loaded)")
    else:
        gen = HFGenerator(args.model, load_4bit=args.load_4bit,
                          max_new_tokens=args.max_new_tokens,
                          temperature=args.temperature, top_p=args.top_p,
                          seed=args.seed)
        print(f"[info] loaded {args.model} (4bit={args.load_4bit})")

    select_kwargs = dict(exclude_same_word=args.fix_rhyme,
                         tail_window=args.tail_window,
                         w_count=args.w_count, w_rhyme=args.w_rhyme,
                         w_vocab=args.w_vocab, w_rep=args.w_rep)

    for i, tid in enumerate(ids, 1):
        out_path = os.path.join(args.out_dir, f"{tid}.txt")
        if args.resume and os.path.exists(out_path):
            print(f"[{i}/{len(ids)}] skip {tid} (exists)")
            continue
        ctx = parse_song_ctx(manifest[tid])
        skeleton = skeletons[tid]
        lines, choices, dump = reconstruct_song(
            ctx, skeleton, gen, args.n_candidates, args.k_context, select_kwargs,
            relax=args.relax_syllable, anti_template=args.anti_template,
            no_syllable=args.no_syllable, no_select=args.no_select,
            vary_system=args.vary_system)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        if args.dump_candidates:                 # append one song's candidate dump
            with open(args.dump_candidates, "a", encoding="utf-8") as f:
                f.write(json.dumps({"song_id": tid, "lines": dump},
                                   ensure_ascii=False) + "\n")
        avg_diff = sum(c.count_diff for c in choices) / max(1, len(choices))
        avg_vf = sum(c.vocab_fidelity for c in choices) / max(1, len(choices))
        print(f"[{i}/{len(ids)}] {tid}: {len(lines)} lines, "
              f"avg |syl_diff|={avg_diff:.2f}, avg vocab_fidelity={avg_vf:.2f}")

    print("[done]")


if __name__ == "__main__":
    main()
