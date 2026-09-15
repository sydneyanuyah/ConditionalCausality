#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

export HF_HOME=${HF_HOME:-$PWD/hf_cache}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PWD/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$PWD/hf_datasets_cache}
export HF_METRICS_CACHE=${HF_METRICS_CACHE:-$PWD/hf_metrics_cache}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" logs outputs

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} nohup python3 train_phi_exp3.py \
  --method probe \
  --train_csv pubmed/pubmed_exp3_2_probe_train.csv \
  --validation_csv pubmed/pubmed_exp3_2_probe_validation.csv \
  --output_dir outputs_exp3_2_qwen_probe_ft_pubmed \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --epochs 3 \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.03 \
  --use_qlora \
  --bf16 \
  --skip_prepare_kbit_training \
  > logs/exp3_2_qwen_probe_ft_pubmed.log 2>&1 &

echo "Started Qwen Exp3.2 PubMed training."
echo "Log: tail -f logs/exp3_2_qwen_probe_ft_pubmed.log"
