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

CUDA_VISIBLE_DEVICES=0 nohup python run_exp3_1_all_extraction_tests.py \
  --base_model microsoft/Phi-4-mini-instruct \
  --adapter_path outputs_exp3_1_phi_extraction_ft \
  --prompt_name two_shot \
  --synthetic_csv exp3_1_extraction_splits/exp3_1_extraction_test.csv \
  --pubmed_csv pubmed_exp3_test_csvs/pubmed_exp3_1_extraction_test.csv \
  --reddit_csv reddit_exp3_test_csvs/reddit_exp3_1_extraction_test.csv \
  --output_dir Exp3_1_Extraction_Test_Outputs \
  --input_column Paragraph \
  --batch_size 16 \
  --max_new_tokens 512 \
  --torch_dtype bfloat16 \
  --use_4bit \
  --resume \
  > exp3_1_all_extraction_tests.log 2>&1 &

echo "Started Exp3.1 extraction tests for Synthetic, PubMed, and Reddit."
echo "Log: tail -f $HOME/SageMaker/exp3_1_all_extraction_tests.log"
echo "Outputs folder: $HOME/SageMaker/Exp3_1_Extraction_Test_Outputs"
