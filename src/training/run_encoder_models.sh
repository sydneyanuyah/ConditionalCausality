#!/usr/bin/env bash
set -e

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
  source "$HOME/.conda/etc/profile.d/conda.sh"
fi

conda activate phi_ft

cd "$HOME/SageMaker"

export HF_HOME=$HOME/SageMaker/hf_cache
export TRANSFORMERS_CACHE=$HOME/SageMaker/hf_cache
export HF_DATASETS_CACHE=$HOME/SageMaker/hf_datasets_cache
export HF_METRICS_CACHE=$HOME/SageMaker/hf_metrics_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE"
mkdir -p "encoder models"

nohup python train_infer_encoder_models.py \
  --task probe \
  --train_csv exp3_2_probe_splits/exp3_2_probe_train.csv \
  --validation_csv exp3_2_probe_splits/exp3_2_probe_validation.csv \
  --synthetic_test_csv exp3_2_probe_splits/exp3_2_probe_test.csv \
  --pubmed_test_csv pubmed_exp3_test_csvs/pubmed_exp3_2_probe_test.csv \
  --reddit_test_csv reddit_exp3_test_csvs/reddit_exp3_2_probe_test.csv \
  --output_dir "encoder models" \
  --models bert,scibert,pubmedbert,biobert \
  --epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_length 512 \
  --bf16 \
  > "encoder models/encoder_probe_training_inference.log" 2>&1

nohup python train_infer_encoder_models.py \
  --task multitask \
  --train_csv exp3_3_multitask_splits/exp3_3_multitask_train.csv \
  --validation_csv exp3_3_multitask_splits/exp3_3_multitask_validation.csv \
  --synthetic_test_csv exp3_3_multitask_splits/exp3_3_multitask_test.csv \
  --pubmed_test_csv pubmed_exp3_test_csvs/pubmed_exp3_3_multitask_test.csv \
  --reddit_test_csv reddit_exp3_test_csvs/reddit_exp3_3_multitask_test.csv \
  --output_dir "encoder models" \
  --models bert,scibert,pubmedbert,biobert \
  --epochs 3 \
  --train_batch_size 16 \
  --eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --max_length 512 \
  --bf16 \
  > "encoder models/encoder_multitask_training_inference.log" 2>&1

echo "Done. Encoder model outputs saved inside: $HOME/SageMaker/encoder models"
