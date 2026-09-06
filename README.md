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
  skeleton extraction, evaluation and bootstrap CIs.
- `runs/` -- sbatch/run scripts for the six training-free configurations
  reported in Chapter 4 (earlier iterations are kept for provenance).
- `train/` -- the training extension: scorer fix (`patch_scorer.py`),
  disjoint training-set construction, teacher generation, SFT data filtering
  (`build_sft_data.py`), LoRA training (`train_lora.py`), and the evaluation
  scripts for the tuned model. Note: `src/` in this repository is already
  patched; `patch_scorer.py --check` reports [ok].
- `data/` -- track-identifier lists only: `test_track_ids.txt` (300 songs)
  and `train_track_ids.txt` (600 songs), disjoint by construction.
  No lyric text is distributed in this repository.

## Environment

Python 3.11 with the packages in `requirements.txt`
(torch 2.5.1 / transformers 5.13.0 / peft 0.20.0).
Base model: Qwen/Qwen2.5-7B-Instruct.

## Data

Lyrics are not included. The pipeline consumes the musiXmatch / Million Song
Dataset derived inputs and the official LyCon release; rebuild the manifests
with `src/build_prompts.py` (evaluation set: seed 42 defaults; training set:
`train/make_train_set.sh`, seed 41 + exclusion of the 300 evaluation ids).

## Reproducing the report's tables

| Report table | Configuration | Generation script | Evaluation |
| --- | --- | --- | --- |
| Ch.4 main | Strict (best-of-8) | `runs/run_metrical300.sh` | `runs/run_all_eval.sh` + `runs/run_all_ci.sh` |
| Ch.4 main | No-selection | `runs/run_noselect.sh` | same |
| Ch.4 | w_rep = 2.5 | `runs/run_rep25.sh` | same |
| Ch.4 | Anti-template | `runs/run_anti.sh` | same |
| Ch.4 | Relaxed | `runs/run_relax_v2.sh` | same |
| Ch.4 | Rhyme-only | `runs/run_rhyme2.sh` | same |
| Ch.4 scorer-fix | Strict-fix | `train/run_strict_fix.sh` | `train/run_eval_strictfix.sh` |
| Ch.4 distillation | Tuned / Tuned+Select | `train/run_tuned_noselect.sh`, `train/run_tuned_select.sh` | `train/run_eval_tuned.sh` |

Training data and the LoRA run: `train/make_train_set.sh` ->
`train/run_traingen.sh` -> `train/build_sft_data.py` -> `train/run_lora.sh`.

## Acknowledgements

Builds on the LyCon reconstruction task (Kim and Choi, 2024) and reimplements
the syllable and rhyme objectives of LYRICS (Ko et al., 2025) as
inference-time scorers; see the dissertation for full citations.
