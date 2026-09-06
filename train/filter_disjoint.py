# -*- coding: utf-8 -*-
"""
filter_disjoint.py -- remove benchmark songs from a candidate manifest and cut a training set
Usage:
  python train/filter_disjoint.py --pool prompts_pool.jsonl \
      --exclude prompts_300.jsonl --n 600 --out prompts_train.jsonl
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True, help="large manifest produced by build_prompts")
ap.add_argument("--exclude", required=True, help="benchmark manifest (all of its track_ids are removed)")
ap.add_argument("--n", type=int, default=600)
ap.add_argument("--out", required=True)
a = ap.parse_args()

ban = set()
with open(a.exclude, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            ban.add(json.loads(line)["track_id"])

kept, skipped = [], 0
with open(a.pool, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec["track_id"] in ban:
            skipped += 1
            continue
        kept.append(line)
        if len(kept) >= a.n:
            break

with open(a.out, "w", encoding="utf-8") as f:
    f.write("\n".join(kept) + "\n")
print(f"removed {skipped} benchmark songs from the pool; wrote {len(kept)} songs -> {a.out}")
assert skipped >= 0 and len(kept) > 0
out_ids = {json.loads(l)["track_id"] for l in kept}
assert not (out_ids & ban), "BUG: output still contains benchmark songs"
print("disjointness check: OK (zero overlap with the benchmark set)")
