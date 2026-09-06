#!/bin/bash
#SBATCH --job-name=allci
#SBATCH --partition=sheffield
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=allci_%j.out
#SBATCH --error=allci_%j.err
#
set -euo pipefail

PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon

evalone () {   # $1=dump  $2=tag
  if [ -f "results_$2/eval_per_line.csv" ]; then
    echo "[skip] results_$2/eval_per_line.csv already exists, skipping"
  else
    echo "[eval] $2"
    $PY src/build_eval_input.py \
      --dump "$1" --ref-dir ref_300 --skeletons skeletons_300.jsonl \
      --out "eval_input_$2.json"
    $PY src/evaluate_reconstruction.py \
      --input "eval_input_$2.json" --out_dir "results_$2"
  fi
}

evalone cand_dump_anti.jsonl     anti
evalone cand_dump_noselect.jsonl noselect
evalone cand_dump_rep25.jsonl    rep25
evalone cand_dump_relax_v2.jsonl relax_v2
evalone cand_dump_rhyme2.jsonl   rhyme2

echo "=== bootstrap ==="
$PY bootstrap_ci.py \
  --csv strict=results_n8_bs/eval_per_line.csv \
        noselect=results_noselect/eval_per_line.csv \
        rep25=results_rep25/eval_per_line.csv \
        anti=results_anti/eval_per_line.csv \
        relaxed=results_relax_v2/eval_per_line.csv \
        rhyme=results_rhyme2/eval_per_line.csv \
  --compare noselect:strict rep25:strict anti:strict \
            relaxed:strict rhyme:strict \
  --boot 2000 --seed 13 | tee bootstrap_all.txt

echo "=== extra_metrics ==="
$PY extra_metrics.py --manifest prompts_300.jsonl --dirs \
  ref=ref_300 \
  noselect=metrical_300_noselect \
  strict=metrical_300_n8 \
  anti=metrical_300_anti \
  rep25=metrical_300_rep25 \
  relaxed=metrical_300_relax_v2 \
  rhyme=metrical_300_rhyme2 | tee extra_metrics_all.txt

echo "=== frame_diversity ==="
$PY frame_diversity.py -k 3 --dirs \
  ref=ref_300 \
  noselect=metrical_300_noselect \
  strict=metrical_300_n8 \
  anti=metrical_300_anti \
  rep25=metrical_300_rep25 \
  relaxed=metrical_300_relax_v2 \
  rhyme=metrical_300_rhyme2 | tee frame_diversity_all.txt

echo "=== done ==="
