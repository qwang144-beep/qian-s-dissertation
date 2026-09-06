"""
Copy the authors' released LyCon (.txt, named by SO id) for exactly the tracks
in a manifest, so you can run lycon_stats.py on a MATCHED GPT-4o reference set
alongside your Qwen reconstructions (same songs, isolating the model change).

    python pull_reference.py --manifest prompts.jsonl \
        --released-dir lycon/dataset --out-dir reference_subset
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--released-dir", required=True, help="folder of released LyCon *.txt (SO-id named)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    copied = missing = 0
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            so_id = item["meta"]["so_id"]
            src = os.path.join(args.released_dir, f"{so_id}.txt")
            if os.path.exists(src):
                # name the reference file by track_id too, to line up 1:1 with Qwen output
                shutil.copy(src, os.path.join(args.out_dir, f"{item['track_id']}.txt"))
                copied += 1
            else:
                missing += 1
    print(f"copied {copied} reference files, {missing} missing, into {args.out_dir}")


if __name__ == "__main__":
    main()
