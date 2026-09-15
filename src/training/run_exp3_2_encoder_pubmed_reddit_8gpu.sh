#!/usr/bin/env bash
set -e

cd "$HOME/SageMaker"

export HF_HOME=$HOME/SageMaker/hf_cache
export TRANSFORMERS_CACHE=$HOME/SageMaker/hf_cache
export HF_DATASETS_CACHE=$HOME/SageMaker/hf_datasets_cache
export HF_METRICS_CACHE=$HOME/SageMaker/hf_metrics_cache
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HF_METRICS_CACHE" logs encoder_exp3_2_domain_ft

# PubMed -> PubMed-Test
CUDA_VISIBLE_DEVICES=0 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key bert \
  --train_csv pubmed/pubmed_exp3_2_probe_train.csv \
  --validation_csv pubmed/pubmed_exp3_2_probe_validation.csv \
  --test_csv pubmed/pubmed_exp3_2_probe_test.csv \
  --train_source PubMed \
  --test_name PubMed-Test \
  --output_csv "Exp 3.2 bert PubMed to PubMed-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_bert_pubmed_to_pubmed_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key scibert \
  --train_csv pubmed/pubmed_exp3_2_probe_train.csv \
  --validation_csv pubmed/pubmed_exp3_2_probe_validation.csv \
  --test_csv pubmed/pubmed_exp3_2_probe_test.csv \
  --train_source PubMed \
  --test_name PubMed-Test \
  --output_csv "Exp 3.2 scibert PubMed to PubMed-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_scibert_pubmed_to_pubmed_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key pubmedbert \
  --train_csv pubmed/pubmed_exp3_2_probe_train.csv \
  --validation_csv pubmed/pubmed_exp3_2_probe_validation.csv \
  --test_csv pubmed/pubmed_exp3_2_probe_test.csv \
  --train_source PubMed \
  --test_name PubMed-Test \
  --output_csv "Exp 3.2 pubmedbert PubMed to PubMed-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_pubmedbert_pubmed_to_pubmed_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key biobert \
  --train_csv pubmed/pubmed_exp3_2_probe_train.csv \
  --validation_csv pubmed/pubmed_exp3_2_probe_validation.csv \
  --test_csv pubmed/pubmed_exp3_2_probe_test.csv \
  --train_source PubMed \
  --test_name PubMed-Test \
  --output_csv "Exp 3.2 biobert PubMed to PubMed-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_biobert_pubmed_to_pubmed_test.log 2>&1 &

# Reddit -> Reddit-Test
CUDA_VISIBLE_DEVICES=4 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key bert \
  --train_csv reddit/reddit_exp3_2_probe_train.csv \
  --validation_csv reddit/reddit_exp3_2_probe_validation.csv \
  --test_csv reddit/reddit_exp3_2_probe_test.csv \
  --train_source Reddit \
  --test_name Reddit-Test \
  --output_csv "Exp 3.2 bert Reddit to Reddit-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_bert_reddit_to_reddit_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key scibert \
  --train_csv reddit/reddit_exp3_2_probe_train.csv \
  --validation_csv reddit/reddit_exp3_2_probe_validation.csv \
  --test_csv reddit/reddit_exp3_2_probe_test.csv \
  --train_source Reddit \
  --test_name Reddit-Test \
  --output_csv "Exp 3.2 scibert Reddit to Reddit-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_scibert_reddit_to_reddit_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key pubmedbert \
  --train_csv reddit/reddit_exp3_2_probe_train.csv \
  --validation_csv reddit/reddit_exp3_2_probe_validation.csv \
  --test_csv reddit/reddit_exp3_2_probe_test.csv \
  --train_source Reddit \
  --test_name Reddit-Test \
  --output_csv "Exp 3.2 pubmedbert Reddit to Reddit-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_pubmedbert_reddit_to_reddit_test.log 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python3 train_infer_encoder_exp3_2_one_run.py \
  --model_key biobert \
  --train_csv reddit/reddit_exp3_2_probe_train.csv \
  --validation_csv reddit/reddit_exp3_2_probe_validation.csv \
  --test_csv reddit/reddit_exp3_2_probe_test.csv \
  --train_source Reddit \
  --test_name Reddit-Test \
  --output_csv "Exp 3.2 biobert Reddit to Reddit-Test.csv" \
  --fp16 \
  > logs/exp3_2_encoder_biobert_reddit_to_reddit_test.log 2>&1 &

echo "Started all 8 Exp3.2 encoder PubMed/Reddit fine-tuning + inference jobs."
echo "Monitor with:"
echo "tail -f logs/exp3_2_encoder_*_to_*_test.log"
