#!/bin/bash
#SBATCH --job-name=llama300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=llama300_%j.out
#SBATCH --error=llama300_%j.err

# Reconstruct the SAME 300 songs with Llama 3.1 8B, then do a 3-way compare
# (GPT-4o vs Qwen vs Llama). Reuses prompts_300.jsonl so the song set is identical.
# Submit from ~/dissertation/lycon:   sbatch run_llama300.sh

set -euo pipefail
cd "$HOME/dissertation/lycon"

export HF_HOME=/mnt/parscratch/users/smp23qw/hf
mkdir -p "$HF_HOME"

module load Anaconda3/2024.02-1 2>/dev/null || true
source activate lycon

# --- Llama 3.1 is GATED on HuggingFace. Two options: ---
# (A) ungated mirror (identical weights, no token needed) -- default below.
# (B) official meta-llama/Llama-3.1-8B-Instruct: accept the licence on HF, then
#     `huggingface-cli login` once (or export HF_TOKEN=hf_xxx here) and swap MODEL.
MODEL="NousResearch/Meta-Llama-3.1-8B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"   # <- if you have a token/licence

echo "node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date)"
python -c "import torch; print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"

# generate with Llama on the existing manifest (no rebuild -> same 300 songs)
python src/generate.py \
  --manifest prompts_300.jsonl \
  --out-dir out_llama_300 \
  --model "$MODEL" \
  --resume

# 3-way, same songs, GPT-4o as baseline
echo "===================================================================="
python src/compare_models.py \
  --dirs gpt4o=ref_300 qwen=out_300 llama=out_llama_300 \
  --baseline gpt4o \
  --out compare_3way.csv
echo "===================================================================="
echo "finished: $(date)"
