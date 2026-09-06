#!/bin/bash
#SBATCH --job-name=lycon300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=lycon300_%j.out
#SBATCH --error=lycon300_%j.err

# ---- Stanage: run 300-song LyCon reconstruction with Qwen2.5 + matched comparison ----
# Submit from ~/dissertation/lycon with:   sbatch run_lycon300.sh
# Watch it with:   squeue --me      and    tail -f lycon300_<jobid>.out
# If your GPU partition/QoS differ, edit the two #SBATCH lines above
# (find them with: sinfo -s | grep -i gpu).

set -euo pipefail

PROJ="$HOME/dissertation/lycon"
cd "$PROJ"

# keep the big HF weight cache off your home quota
export HF_HOME=/mnt/parscratch/users/smp23qw/hf
mkdir -p "$HF_HOME"

# activate the conda env
module load Anaconda3/2024.02-1 2>/dev/null || true
source activate lycon

echo "===================================================================="
echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}   started: $(date)"
python -c "import torch; print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "===================================================================="

# 1) build the 300-song subset (CPU, seconds)
python src/build_prompts.py \
  --deezer-dir data/deezer_mood_detection_dataset \
  --mxm data/mxm_dataset_train.txt data/mxm_dataset_test.txt \
  --styles data/msd-MASD-styleAssignment.cls \
  --released-dir lycon_ref/dataset \
  --limit 300 --seed 13 \
  --out prompts_300.jsonl

# 2) generate with Qwen2.5 (fp16, no 4-bit on Stanage). --resume => restartable
python src/generate.py \
  --manifest prompts_300.jsonl \
  --out-dir out_300 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --resume

# 3) pull the authors' GPT-4o reconstructions for the SAME 300 songs
python src/pull_reference.py \
  --manifest prompts_300.jsonl \
  --released-dir lycon_ref/dataset \
  --out-dir ref_300

# 4) matched comparison, same songs
echo "===================================================================="
echo "=== Authors GPT-4o LyCon (same 300 songs) ==="
python src/lycon_stats.py --dir ref_300
echo
echo "=== Your Qwen2.5 LyCon (300 songs) ==="
python src/lycon_stats.py --dir out_300
echo "===================================================================="
echo "finished: $(date)"
