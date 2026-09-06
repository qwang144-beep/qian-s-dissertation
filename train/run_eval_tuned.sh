#!/bin/bash
#SBATCH --job-name=evaltuned
#SBATCH --partition=sheffield
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=evaltuned_%j.out
#SBATCH --error=evaltuned_%j.err
set -euo pipefail
PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon

$PY frame_diversity.py -k 3 --dirs ref=ref_300 \
    noselect=metrical_300_noselect strictfix=metrical_300_strictfix \
    tunedNS=metrical_300_tunedNS tunedSEL=metrical_300_tunedSEL \
    | tee frame_tuned.txt
$PY extra_metrics.py --manifest prompts_300.jsonl --dirs ref=ref_300 \
    tunedNS=metrical_300_tunedNS tunedSEL=metrical_300_tunedSEL \
    | tee extra_tuned.txt

for tag in tunedNS tunedSEL; do
  $PY src/build_eval_input.py --dump cand_dump_${tag}.jsonl \
      --ref-dir ref_300 --skeletons skeletons_300.jsonl \
      --out eval_input_${tag}.json
  $PY src/evaluate_reconstruction.py --input eval_input_${tag}.json \
      --out_dir results_${tag}
done

$PY bootstrap_ci.py \
  --csv noselect=results_noselect/eval_per_line.csv \
        strictfix=results_strictfix/eval_per_line.csv \
        tunedNS=results_tunedNS/eval_per_line.csv \
        tunedSEL=results_tunedSEL/eval_per_line.csv \
  --compare tunedNS:noselect tunedNS:strictfix tunedSEL:strictfix \
  --boot 2000 --seed 13 | tee bootstrap_tuned.txt
echo "=== all done ==="
