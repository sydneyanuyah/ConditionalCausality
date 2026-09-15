#!/usr/bin/env bash
set -euo pipefail

# Run from this folder or copy the scripts into a project root.
# This expects predictions in:
#   inference/spanbert/*.csv
#   inference/flan_t5/*.csv

mkdir -p evaluation/exp1a evaluation/exp1b logs

# Exp1a: synthetic gold only, stricter evaluation.
python3 eval_exp1a_synthetic.py \
  --gold syntheticgold_test.csv \
  --pred_dir inference/spanbert \
  --out_dir evaluation/exp1a/spanbert

python3 eval_exp1a_synthetic.py \
  --gold syntheticgold_test.csv \
  --pred_dir inference/flan_t5 \
  --out_dir evaluation/exp1a/flan_t5

# Exp1b: all three datasets, recall-focused relaxed evaluation.
python3 eval_exp1b_real_world.py \
  --pred_root inference \
  --models spanbert flan_t5 \
  --synthetic_gold syntheticgold_test.csv \
  --pubmed_gold PubMed-Exp1.csv \
  --reddit_gold Reddit-Exp1.csv \
  --out_dir evaluation/exp1b
