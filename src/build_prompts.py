"""
Align the three metadata sources on MSD_track_id and build one LyCon prompt per
song. Emits a JSONL manifest {track_id, prompt, meta} ready for generate.py.

Sources
-------
  Deezer Mood      -> valence, arousal, artist_name, track_name  (-> [MOOD], [ARTIST], [TITLE])
  musiXmatch BoW   -> frequency-sorted vocabulary                (-> [VOCABULARY])
  AllMusic Style   -> concatenated genres                        (-> [GENRE])

Only tracks present in ALL THREE are kept (this is what reduces the paper's
mood-annotated set down to its final 7,863 reconstructed songs).

Prompt template (verbatim from Kim & Choi 2024):
  "Compose [GENRE] lyrics, in a style reminiscent of [ARTIST] which represents a
   [MOOD] mood under the title of [TITLE] using the following vocabulary
   [VOCABULARY]."
"""
import argparse
import csv
import glob
import json
import os

from mood import mood
from bow import load_bow, load_reverse_map, vocabulary_string
from genre import load_styles, genre_string

TEMPLATE = ('Compose {genre} lyrics, in a style reminiscent of {artist} which '
            'represents a {mood} mood under the title of "{title}" using the '
            'following vocabulary {vocabulary}.')


def load_deezer(csv_dir):
    """Return {MSD_track_id: dict(valence, arousal, artist, title)}."""
    out = {}
    for path in glob.glob(os.path.join(csv_dir, "*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                out[row["MSD_track_id"]] = dict(
                    so_id=row["MSD_sng_id"],   # SO... id used to name released LyCon files
                    valence=float(row["valence"]),
                    arousal=float(row["arousal"]),
                    artist=row["artist_name"],
                    title=row["track_name"],
                )
    return out


def build_prompt(meta, genres, vocab_str):
    return TEMPLATE.format(
        genre=genre_string(genres),
        artist=meta["artist"],
        mood=mood(meta["valence"], meta["arousal"]).lower(),  # 'nervous', not 'NERVOUS'
        title=meta["title"],
        vocabulary=vocab_str,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deezer-dir", required=True, help="folder of Deezer *.csv")
    ap.add_argument("--mxm", nargs="+", required=True, help="mxm_dataset_*.txt file(s)")
    ap.add_argument("--styles", required=True, help="msd-MASD-styleAssignment.cls")
    ap.add_argument("--reverse-map", default=None, help="mxm_reverse_mapping.txt (optional un-stem)")
    ap.add_argument("--out", default="prompts.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="only build N prompts (a few hundred is plenty)")
    ap.add_argument("--seed", type=int, default=13, help="sampling seed for --limit")
    ap.add_argument("--released-dir", default=None,
                    help="if set, keep only tracks whose SO id has a released "
                         "LyCon .txt here, so you get a matched GPT-4o vs Qwen subset")
    args = ap.parse_args()

    deezer = load_deezer(args.deezer_dir)
    bow = load_bow(args.mxm)
    styles = load_styles(args.styles)
    rmap = load_reverse_map(args.reverse_map) if args.reverse_map else None

    keys = set(deezer) & set(bow) & set(styles)
    print(f"deezer={len(deezer)}  bow={len(bow)}  styles={len(styles)}  "
          f"-> intersection={len(keys)}")

    if args.released_dir:
        have = {os.path.splitext(os.path.basename(f))[0]
                for f in glob.glob(os.path.join(args.released_dir, "*.txt"))}
        keys = {t for t in keys if deezer[t]["so_id"] in have}
        print(f"restricted to tracks with a released LyCon file -> {len(keys)}")

    keys = sorted(keys)
    if args.limit and args.limit < len(keys):
        import random
        random.Random(args.seed).shuffle(keys)
        keys = sorted(keys[:args.limit])
        print(f"sampled {len(keys)} tracks (seed={args.seed})")

    n = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for tid in sorted(keys):
            vocab_str = vocabulary_string(bow[tid], rmap)
            prompt = build_prompt(deezer[tid], styles[tid], vocab_str)
            fh.write(json.dumps({"track_id": tid, "prompt": prompt,
                                 "meta": deezer[tid], "genre": styles[tid]}) + "\n")
            n += 1
    print(f"wrote {n} prompts to {args.out}")


if __name__ == "__main__":
    main()
