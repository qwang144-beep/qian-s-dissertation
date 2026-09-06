#!/bin/bash
#SBATCH --job-name=anti_test
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=anti_test_%j.out
#SBATCH --error=anti_test_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cd /users/smp23qw/dissertation/lycon

$PY src/generate_metrical.py --manifest prompts_300.jsonl --skeletons skeletons_300.jsonl --out-dir out_anti_only --songs TRZXYTD128F42928C0 --n-candidates 8 --k-context 3 --anti-template --temperature 0.95 --dump-candidates anti_only_dump.jsonl