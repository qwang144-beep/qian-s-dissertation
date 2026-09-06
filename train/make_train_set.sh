#!/bin/bash
# =====================================================================
# Build the disjoint training set (600 songs): sample a pool with a
# different seed, exclude every benchmark track_id, pull references,
# and extract syllable skeletons. Adjust PY and the paths below to
# your environment.
# =====================================================================
set -euo pipefail
PY=/users/smp23qw/.conda/envs/lycon/bin/python   # path to the project's Python
cd /users/smp23qw/dissertation/lycon             # project root

DEEZER_DIR=data/deezer_mood_detection_dataset       # Deezer csv directory
MXM="data/mxm_dataset_train.txt data/mxm_dataset_test.txt"            # mxm_dataset_*.txt (several allowed, space-separated, keep the quotes)
STYLES=data/msd-MASD-styleAssignment.cls           # msd-MASD-styleAssignment.cls
RELEASED=lycon_ref/dataset         # official LyCon release .txt directory (the 7,863-song one)
N_TRAIN=600                # number of training songs; drop to 400 if a 12h wall clock is too short

$PY src/build_prompts.py --deezer-dir "$DEEZER_DIR" --mxm $MXM \
    --styles "$STYLES" --released-dir "$RELEASED" \
    --limit $((N_TRAIN + 400)) --seed 41 --out prompts_pool.jsonl

$PY train/filter_disjoint.py --pool prompts_pool.jsonl \
    --exclude prompts_300.jsonl --n $N_TRAIN --out prompts_train.jsonl

$PY src/pull_reference.py --manifest prompts_train.jsonl \
    --released-dir "$RELEASED" --out-dir ref_train

$PY src/lyric_txt_parser.py --ref_dir ref_train --out skeletons_train.jsonl

echo "=== done: prompts_train.jsonl / ref_train/ / skeletons_train.jsonl ==="
