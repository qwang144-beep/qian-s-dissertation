#!/bin/bash
#SBATCH --job-name=vary300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=vary300_%j.out
#SBATCH --error=vary300_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qwang144@sheffield.ac.uk
# ---------------------------------------------------------------------------
# A2 step 4: the 2x2 control run.
#   exact syllable target (strict user prompt + w_count 3.0)
#   + anti-template instruction in the USER prompt (as in the anti run)
#   + "Vary your sentence structure" sentence in the SYSTEM prompt
#     (as in relaxed / rhyme-only)
# Everything else identical to strict: N=8, K=3, T=0.9, weights 3/1/0.7/0.8.
# ---------------------------------------------------------------------------
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon
$PY src/generate_metrical.py \
  --manifest prompts_300.jsonl \
  --skeletons skeletons_300.jsonl \
  --out-dir metrical_300_vary \
  --model Qwen/Qwen2.5-7B-Instruct \
  --n-candidates 8 --k-context 3 --temperature 0.9 \
  --anti-template --vary-system \
  --w-count 3.0 --w-rhyme 1.0 --w-vocab 0.7 --w-rep 0.8 \
  --resume \
  --dump-candidates cand_dump_vary.jsonl
echo "generation finished: $(date)"
