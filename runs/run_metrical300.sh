#!/bin/bash
#SBATCH --job-name=metrical300
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=metrical300_%j.out
#SBATCH --error=metrical300_%j.err

set -euo pipefail

PROJ="$HOME/dissertation/lycon"
cd "$PROJ"

export HF_HOME=/mnt/parscratch/users/smp23qw/hf
mkdir -p "$HF_HOME"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_XET=1

PY=/users/smp23qw/.conda/envs/lycon/bin/python

# ---- tune here ----
N=8
OUT=metrical_300_n8
DUMP=cand_dump_n8.jsonl

echo "===================================================================="
echo "node: $(hostname)   job: ${SLURM_JOB_ID:-none}   started: $(date)"
$PY -c "import torch; print('cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "N=$N  OUT=$OUT  DUMP=$DUMP"
echo "===================================================================="

# 1) syllable skeletons from ref (CPU, seconds; idempotent)
$PY src/lyric_txt_parser.py --ref_dir ref_300 --out skeletons_300.jsonl

# 2) line-by-line metrical generation with best-of-N rerank (fp16)
$PY src/generate_metrical.py \
  --manifest prompts_300.jsonl \
  --skeletons skeletons_300.jsonl \
  --out-dir "$OUT" \
  --dump-candidates "$DUMP" \
  --model Qwen/Qwen2.5-7B-Instruct \
  --resume \
  --n-candidates "$N" --k-context 3

echo "===================================================================="
echo "generation finished: $(date)"
echo "next (CPU): "
echo "  \$PY src/build_eval_input.py --dump $DUMP --ref-dir ref_300 --skeletons skeletons_300.jsonl --out eval_input_n8.json"
echo "  \$PY src/evaluate_reconstruction.py --input eval_input_n8.json --out_dir results_n8"
echo "===================================================================="