#!/bin/bash
#SBATCH --job-name=lorasft
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=lorasft_%j.out
#SBATCH --error=lorasft_%j.err

PY=/users/smp23qw/.conda/envs/lycon/bin/python
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1
cd /users/smp23qw/dissertation/lycon

MERGED=/mnt/parscratch/users/smp23qw/qwen_metrical_sft   # ~15GB; keep it on scratch storage

$PY train/train_lora.py \
  --base Qwen/Qwen2.5-7B-Instruct \
  --train sft_train.jsonl --val sft_val.jsonl \
  --out-dir lora_out --merged-dir "$MERGED" \
  --epochs 3 --lr 1e-4 --bsz 8 --accum 4 --max-len 768

echo "=================================================================="
echo "Training done. Two evaluation runs follow (submit each as its own sbatch job):"
echo "  1) tuned, no selection:   generate_metrical.py --model $MERGED --no-select ..."
echo "  2) tuned with selection:  generate_metrical.py --model $MERGED --fix-rhyme --tail-window 3 ..."
echo "=================================================================="
