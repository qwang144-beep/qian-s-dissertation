# Metrical and Rhyme Control for Bag-of-Words Lyric Reconstruction

Code for an MSc dissertation (University of Sheffield). The project adapts
LyCon-style bag-of-words lyric reconstruction from whole-song generation to
line-by-line generation with best-of-N selection, repurposing the training
objectives of LYRICS (Ko et al., 2025) as inference-time scorers, and closes
with a training extension that distils the fixed selection pipeline into the
model by rejection-sampling LoRA fine-tuning.

## Layout

- `src/` -- core pipeline: prompt construction, line-by-line generation
  (`generate_metrical.py`), the scoring functions (`lyrics_reranker.py`),
  skeleton extraction, evaluation and bootstrap CIs. (`lyrics_reranker.py`
  also contains an optional semantic scorer; it is not used by
  `generate_metrical.py`, which applies only the four scorers of the report.)
- `runs/` -- sbatch/run scripts for the configurations reported in Chapter 4.
  `runs/superseded/` keeps earlier iterations for provenance only (different
  temperature or weights); none of them produced a reported number.
- `train/` -- the training extension: scorer fix (`patch_scorer.py`),
  disjoint training-set construction, teacher generation, SFT data filtering
  (`build_sft_data.py`), LoRA training (`train_lora.py`), and the evaluation
  scripts for the tuned model. Note: `src/` in this repository is already
  patched; `python train/patch_scorer.py --check` reports [ok].
- `data/` -- track-identifier lists only: `test_track_ids.txt` (300 songs)
  and `train_track_ids.txt` (600 songs), disjoint by construction.
  No lyric text is distributed in this repository.
- `src/lycon_colab.ipynb` -- an early exploratory Colab path (4-bit Qwen on a
  T4). It is not the source of any reported number; all reported runs were
  made on the Stanage cluster in half precision with the scripts in `runs/`
  and `train/`.

## Environment

Python 3.11 with the packages in `requirements.txt`
(torch 2.5.1 / transformers 5.13.0 / peft 0.20.0).
Base model: Qwen/Qwen2.5-7B-Instruct. The run scripts are written for the
University of Sheffield Stanage cluster (SLURM); edit `PY`, `HF_HOME` and the
`cd` line at the top of each script for another environment.

## Data

Lyrics are not included. The pipeline consumes the musiXmatch / Million Song
Dataset derived inputs and the official LyCon release; rebuild the manifests
with `src/build_prompts.py` (evaluation set: `--limit 300 --seed 13`, as in
`runs/run_lycon300.sh`; training set: `train/make_train_set.sh`, seed 41 +
exclusion of the 300 evaluation ids).

## Reproducing the report's tables

| Report table | Configuration | Generation script | Evaluation |
| --- | --- | --- | --- |
| Table 4.1 | LyCon reproduction (Qwen, LLaMA, GPT-4o reference) | `runs/run_lycon300.sh`, `runs/run_llama300.sh` | `src/lycon_stats.py`, `src/compare_models.py` |
| Tables 4.2-4.9 | Strict (best-of-8) | `runs/run_metrical300.sh` | `runs/run_all_eval.sh` + `runs/run_all_ci.sh` |
| Table 4.2 | No-selection (end-to-end run, `--no-select`) | `runs/run_noselect.sh` | `runs/run_all_ci.sh` |
| Table 4.3 | w_rep = 2.5 | `runs/run_rep25.sh` | `runs/run_all_ci.sh` |
| Table 4.5 | Anti-template | `runs/run_anti.sh` | `runs/run_all_ci.sh` |
| Table 4.7 | Vary (anti-template + system sentence, exact target) | `runs/run_vary.sh` | `runs/eval_vary.sh` |
| Table 4.6 | Relaxed | `runs/run_relax_v2.sh` | `runs/run_all_ci.sh` |
| Table 4.8 | Rhyme-only | `runs/run_rhyme_v2.sh` | `runs/run_all_ci.sh` |
| Table 4.10 | Strict-fix (fixed scorer) | `train/run_strict_fix.sh` | `train/run_eval_strictfix.sh` |
| Table 4.11 | Tuned (`--no-select`) / Tuned+Select | `train/run_tuned_noselect.sh`, `train/run_tuned_select.sh` | `train/run_eval_tuned.sh` |

Notes on the two kinds of "single-sample" figures in Chapter 4:

- The No-selection baseline (Table 4.2) and Tuned (Table 4.11) are complete
  generation runs with `--no-select`: the first non-empty candidate is kept at
  every position and enters the context of the next line, so each run
  conditions on its own history.
- The single-sample figures quoted for other configurations (for example the
  anti and vary runs in Section 4.2.4) are the `baseline` column of that
  run's evaluation report, which `src/build_eval_input.py` reads offline as
  `candidates[0]` from the stored candidate pool.

Training data and the LoRA run (Section 3.7, Table A.3):

```
train/make_train_set.sh
train/run_traingen.sh
python train/build_sft_data.py --dump cand_dump_train_fix.jsonl \
    --manifest prompts_train.jsonl --out sft_train.jsonl \
    --val-out sft_val.jsonl --max-count-diff 1      # syllable +-1 filter
train/run_lora.sh
```

## Acknowledgements

Builds on the LyCon reconstruction task (Kim and Choi, 2024) and reimplements
the syllable and rhyme objectives of LYRICS (Ko et al., 2025) as
inference-time scorers; see the dissertation for full citations.
