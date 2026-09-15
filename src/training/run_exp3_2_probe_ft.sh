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

CUDA_VISIBLE_DEVICES=0 nohup python train_phi_exp3.py \
  --method probe \
  --train_csv exp3_2_probe_splits/exp3_2_probe_train.csv \
  --validation_csv exp3_2_probe_splits/exp3_2_probe_validation.csv \
  --output_dir outputs_exp3_2_phi_probe_ft \
  --model_name microsoft/Phi-4-mini-instruct \
  --epochs 3 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --use_qlora \
  --bf16 \
  > exp3_2_phi_probe_ft.log 2>&1 &

echo "Started Exp3.2 Probe-FT."
echo "Log: tail -f $HOME/SageMaker/exp3_2_phi_probe_ft.log"
echo "Output model adapter: $HOME/SageMaker/outputs_exp3_2_phi_probe_ft"
