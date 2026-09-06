#!/bin/bash
#SBATCH --job-name=metrical_relax
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=metrical_relax_%j.out
#SBATCH --error=metrical_relax_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

cd /users/smp23qw/dissertation/lycon

$PY src/generate_metrical.py --manifest prompts_300.jsonl --skeletons skeletons_300.jsonl --out-dir metrical_300_relax --n-candidates 8 --k-context 3 --relax-syllable --anti-template --temperature 0.95 --w-count 0.5 --w-rhyme 1.0 --w-vocab 4.0 --w-rep 1.5 --resume --dump-candidates cand_dump_relax.jsonl