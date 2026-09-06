#!/bin/bash
#SBATCH --job-name=evalfix
#SBATCH --partition=sheffield
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=evalfix_%j.out
#SBATCH --error=evalfix_%j.err
set -euo pipefail
PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon

$PY src/build_eval_input.py --dump cand_dump_strictfix.jsonl \
    --ref-dir ref_300 --skeletons skeletons_300.jsonl \
    --out eval_input_strictfix.json
$PY src/evaluate_reconstruction.py --input eval_input_strictfix.json \
    --out_dir results_strictfix

$PY bootstrap_ci.py \
  --csv strict=results_n8_bs/eval_per_line.csv \
        strictfix=results_strictfix/eval_per_line.csv \
  --compare strictfix:strict --boot 2000 --seed 13 | tee bootstrap_strictfix.txt

$PY frame_diversity.py -k 3 --dirs ref=ref_300 \
    strict=metrical_300_n8 strictfix=metrical_300_strictfix | tee frame_strictfix.txt
$PY extra_metrics.py --manifest prompts_300.jsonl --dirs ref=ref_300 \
    strictfix=metrical_300_strictfix | tee extra_strictfix.txt
echo "=== eval done ==="
