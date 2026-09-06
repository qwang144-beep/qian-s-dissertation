#!/bin/bash
#SBATCH --job-name=traingen
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=traingen_%j.out
#SBATCH --error=traingen_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
cd /users/smp23qw/dissertation/lycon

$PY src/generate_metrical.py \
  --manifest prompts_train.jsonl \
  --skeletons skeletons_train.jsonl \
  --out-dir metrical_train_fix \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-candidates 8 --k-context 3 --temperature 0.9 \
  --fix-rhyme --tail-window 3 \
  --w-count 3.0 --w-rhyme 1.0 --w-vocab 0.7 --w-rep 0.8 \
  --resume \
  --dump-candidates cand_dump_train_fix.jsonl
