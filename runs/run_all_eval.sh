#!/bin/bash
set -euo pipefail
PY=/users/smp23qw/.conda/envs/lycon/bin/python
cd /users/smp23qw/dissertation/lycon

evalone () {          # $1=dump  $2=tag
  $PY src/build_eval_input.py \
    --dump "$1" --ref-dir ref_300 --skeletons skeletons_300.jsonl \
    --out "eval_input_$2.json"
  $PY src/evaluate_reconstruction.py \
    --input "eval_input_$2.json" --out_dir "results_$2"
}

evalone cand_dump_rhyme2.jsonl   rhyme2
evalone cand_dump_relax_v2.jsonl relax_v2
evalone cand_dump_rep25.jsonl    rep25

[ -f results_n8_bs/eval_report.md ] || \
  $PY src/evaluate_reconstruction.py --input eval_input_n8.json --out_dir results_n8_bs

$PY extra_metrics.py --manifest prompts_300.jsonl --dirs \
  ref=ref_300 \
  strict=metrical_300_n8 \
  relaxed=metrical_300_relax_v2 \
  rhyme=metrical_300_rhyme2 \
  rep25=metrical_300_rep25 | tee extra_metrics.txt

echo "=== done ==="
