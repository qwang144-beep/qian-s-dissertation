#!/bin/bash
#SBATCH --job-name=noselect300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=noselect300_%j.out
#SBATCH --error=noselect300_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cd /users/smp23qw/dissertation/lycon

$PY src/generate_metrical.py \
  --manifest prompts_300.jsonl \
  --skeletons skeletons_300.jsonl \
  --out-dir metrical_300_noselect \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-candidates 8 \
  --k-context 3 \
  --temperature 0.9 \
  --no-select \
  --dump-candidates cand_dump_noselect.jsonl
