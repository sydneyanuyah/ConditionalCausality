#!/usr/bin/env bash
set -e

conda activate phi_ft

export HF_HOME=$HOME/SageMaker/hf_cache
export TRANSFORMERS_CACHE=$HOME/SageMaker/hf_cache
export HF_DATASETS_CACHE=$HOME/SageMaker/hf_datasets_cache
export HF_METRICS_CACHE=$HOME/SageMaker/hf_metrics_cache

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE"

python -m pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -U "transformers==4.53.3" "accelerate>=1.2.0" "datasets>=3.0.0" "peft>=0.14.0" "bitsandbytes>=0.45.0" pandas scikit-learn sentencepiece protobuf

echo "Environment ready."
