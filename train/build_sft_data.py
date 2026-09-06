# -*- coding: utf-8 -*-
"""
build_sft_data.py -- build SFT training pairs from the teacher dump (CPU only)
==============================================================================
One example per line: {"system","user","assistant"}, where
  system/user = the EXACT prompt seen at generation time
                (rebuilt from the chosen history in the dump)
  assistant   = the line the fixed scorer selected

Filters (so that only good behaviour is taught):
  - chosen empty                                     -> drop
  - syllable difference > --max-count-diff
    (default 0: teach exact hits only)               -> drop
  - chosen reuses a terminal word of its own
    previous 3 lines                                 -> drop
  - chosen's opening word shares a stem with any of
    the previous 3 lines' openings                   -> drop
    (do not teach templating)

Train/val split is by song (--val-frac, default 0.05), so no song spans both sets.

Usage (from the directory above src/):
  $PY train/build_sft_data.py --dump cand_dump_train_fix.jsonl \
      --manifest prompts_train.jsonl --out sft_train.jsonl --val-out sft_val.jsonl
"""
from __future__ import annotations
import argparse, hashlib, json, sys
sys.path.insert(0, "src")

from generate_metrical import (load_manifest, parse_song_ctx,
                               build_line_user_prompt, SYSTEM_PROMPT)
from lyrics_reranker import last_word, _stem_eq, line_to_template
from lyric_txt_parser import clean_lyric_lines  # noqa: importability check


def first_word(line):
    import re
    toks = re.findall(r"[a-zA-Z']+", line.lower())
    return toks[0] if toks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None)
    ap.add_argument("--k-context", type=int, default=3)
    ap.add_argument("--max-count-diff", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    a = ap.parse_args()

    manifest = load_manifest(a.manifest)
    K = a.k_context
    kept = dropped_syl = dropped_tail = dropped_head = dropped_empty = 0
    out_f = open(a.out, "w", encoding="utf-8")
    val_f = open(a.val_out, "w", encoding="utf-8") if a.val_out else None

    with open(a.dump, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            sid = rec["song_id"]
            m = manifest.get(sid)
            if m is None:
                continue
            ctx = parse_song_ctx(m)
            is_val = (val_f is not None and
                      int(hashlib.md5(sid.encode()).hexdigest(), 16) % 10000
                      < a.val_frac * 10000)
            history = []
            for ln in rec["lines"]:
                chosen = ln.get("chosen", "")
                prev = list(history)          # record the context first, then always advance the history
                history.append(chosen)
                if not chosen.strip():
                    dropped_empty += 1
                    continue
                syl = sum(line_to_template(chosen))
                if abs(syl - ln["target_syllables"]) > a.max_count_diff:
                    dropped_syl += 1
                    continue
                g = last_word(chosen)
                recent = [last_word(p) for p in prev[-K:]]
                if g is not None and any(_stem_eq(g, t) for t in recent):
                    dropped_tail += 1
                    continue
                h = first_word(chosen)
                recent_h = [first_word(p) for p in prev[-K:]]
                if h is not None and any(_stem_eq(h, t) for t in recent_h):
                    dropped_head += 1
                    continue
                user = build_line_user_prompt(ctx, prev, ln["target_syllables"],
                                              K, relax=False,
                                              anti_template=False,
                                              no_syllable=False)
                row = json.dumps({"system": SYSTEM_PROMPT, "user": user,
                                  "assistant": chosen}, ensure_ascii=False)
                (val_f if is_val else out_f).write(row + "\n")
                kept += 1
    out_f.close()
    if val_f:
        val_f.close()
    tot = kept + dropped_syl + dropped_tail + dropped_head + dropped_empty
    print(f"total lines        {tot}")
    print(f"kept               {kept}  ({kept/max(1,tot):.1%})")
    print(f"dropped: syllable  {dropped_syl}   tail-reuse {dropped_tail}   "
          f"head-reuse {dropped_head}   empty {dropped_empty}")
    if kept < 3000:
        print("WARNING: few examples (<3k): raise --max-count-diff to 1 and rerun")


if __name__ == "__main__":
    main()
