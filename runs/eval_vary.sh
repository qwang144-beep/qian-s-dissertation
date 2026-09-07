#!/bin/bash
#SBATCH --job-name=evalvary
#SBATCH --partition=sheffield
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=evalvary_%j.out
#SBATCH --error=evalvary_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=qwang144@sheffield.ac.uk
# CPU evaluation of the vary run + bootstrap against strict AND anti + song-level measures
set -euo pipefail
PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon
$PY src/build_eval_input.py --dump cand_dump_vary.jsonl --ref-dir ref_300 \
    --skeletons skeletons_300.jsonl --out eval_input_vary.json
$PY src/evaluate_reconstruction.py --input eval_input_vary.json --out_dir results_vary
$PY bootstrap_ci.py \
  --csv strict=results_n8_bs/eval_per_line.csv \
        anti=results_anti/eval_per_line.csv \
        relaxed=results_relax_v2/eval_per_line.csv \
        vary=results_vary/eval_per_line.csv \
  --compare vary:strict vary:anti relaxed:vary \
  --boot 2000 --seed 13 | tee bootstrap_vary.txt
$PY extra_metrics.py --manifest prompts_300.jsonl --dirs \
  ref=ref_300 strict=metrical_300_n8 anti=metrical_300_anti \
  vary=metrical_300_vary relaxed=metrical_300_relax_v2 | tee extra_vary.txt
$PY frame_diversity.py -k 3 --dirs ref=ref_300 strict=metrical_300_n8 \
  anti=metrical_300_anti vary=metrical_300_vary relaxed=metrical_300_relax_v2 \
  | tee frame_vary.txt
echo "=== done ==="
