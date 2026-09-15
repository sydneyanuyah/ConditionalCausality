import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)


LABELS = ["Supported", "Not Supported", "Not Enough Evidence"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


ENCODER_MODELS = {
    "bert": "bert-base-uncased",
    "scibert": "allenai/scibert_scivocab_uncased",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "biobert": "dmis-lab/biobert-base-cased-v1.1",
}


def normalize_label(x):
    s = str(x).strip().lower()

    if s in {"supported", "support"}:
        return "Supported"

    if s in {"not supported", "unsupported", "contradiction", "contradicted"}:
        return "Not Supported"

    if s in {"not enough evidence", "nee", "not enough"}:
        return "Not Enough Evidence"

    raise ValueError(f"Unknown label: {x}")


def build_input(row, task):
    paragraph = str(row["Paragraph"])
    probe_question = str(row["Probe_Question"])

    if task == "probe":
        return (
            "Paragraph:\n"
            f"{paragraph}\n\n"
            "Probe claim:\n"
            f"{probe_question}"
        )

    tuple_text = str(row["Tuple"])

    return (
        "Paragraph:\n"
        f"{paragraph}\n\n"
        "Conditional causal tuple:\n"
        f"{tuple_text}\n\n"
        "Probe claim:\n"
        f"{probe_question}"
    )


class EncoderClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, task, max_length, has_labels=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        self.has_labels = has_labels

        self.texts = [build_input(row, task) for _, row in self.df.iterrows()]

        self.labels = None
        if has_labels:
            self.labels = [
                LABEL2ID[normalize_label(x)]
                for x in self.df["Gold_Label"].tolist()
            ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        if self.has_labels:
            enc["labels"] = self.labels[idx]

        return enc



def get_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda x: int(str(x.name).split("-")[-1]) if str(x.name).split("-")[-1].isdigit() else -1,
    )

    if not checkpoints:
        return None

    return str(checkpoints[-1])


def load_csv(path):
    df = pd.read_csv(path, low_memory=False)

    if "output" in df.columns:
        df = df.drop(columns=["output"])

    required = ["Paragraph", "Probe_Question", "Gold_Label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Found: {list(df.columns)}")

    return df


def train_one_model(model_key, model_name, args):
    print("\n" + "=" * 80)
    print(f"Training encoder: {model_key} -> {model_name}")
    print(f"Task: {args.task}")
    print("=" * 80)

    model_dir = Path(args.output_dir) / args.task / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_csv(args.train_csv)
    val_df = load_csv(args.validation_csv)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = EncoderClassificationDataset(
        train_df,
        tokenizer,
        task=args.task,
        max_length=args.max_length,
        has_labels=True,
    )

    val_ds = EncoderClassificationDataset(
        val_df,
        tokenizer,
        task=args.task,
        max_length=args.max_length,
        has_labels=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=args.logging_steps,
        report_to="none",
        seed=4000,
        data_seed=4000,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    latest_checkpoint = get_latest_checkpoint(model_dir / "checkpoints")

    if latest_checkpoint:
        print(f"Resuming {model_key} from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print(f"No checkpoint found for {model_key}. Starting from scratch.")
        trainer.train()

    final_model_dir = model_dir / "final_model"
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print(f"Saved trained model to: {final_model_dir}")

    for dataset_name, csv_path in [
        ("synthetic", args.synthetic_test_csv),
        ("pubmed", args.pubmed_test_csv),
        ("reddit", args.reddit_test_csv),
    ]:
        if not csv_path:
            continue

        path = Path(csv_path)
        if not path.exists():
            print(f"Skipping {dataset_name}: file not found -> {path}")
            continue

        print(f"Running inference on {dataset_name}: {path}")

        test_df = load_csv(path)

        test_ds = EncoderClassificationDataset(
            test_df,
            tokenizer,
            task=args.task,
            max_length=args.max_length,
            has_labels=True,
        )

        pred = trainer.predict(test_ds)
        logits = pred.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_ids = np.argmax(probs, axis=-1)

        out_df = test_df.copy()
        out_df["model_name"] = model_name
        out_df["encoder_model"] = model_key
        out_df["task"] = args.task
        out_df["predicted_label"] = [ID2LABEL[int(i)] for i in pred_ids]

        for i, label in ID2LABEL.items():
            safe_label = label.lower().replace(" ", "_")
            out_df[f"prob_{safe_label}"] = probs[:, i]

        out_path = model_dir / f"{dataset_name}_{args.task}_{model_key}_predictions.csv"
        out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to: {out_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", required=True, choices=["probe", "multitask"])

    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--validation_csv", required=True)

    parser.add_argument("--synthetic_test_csv", required=True)
    parser.add_argument("--pubmed_test_csv", required=True)
    parser.add_argument("--reddit_test_csv", required=True)

    parser.add_argument("--output_dir", default="encoder models")

    parser.add_argument("--models", default="bert,scibert,pubmedbert,biobert")
    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=50)

    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/SageMaker/hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/SageMaker/hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/SageMaker/hf_datasets_cache"))
    os.environ.setdefault("HF_METRICS_CACHE", os.path.expanduser("~/SageMaker/hf_metrics_cache"))

    for key in ["HF_HOME", "HF_DATASETS_CACHE", "HF_METRICS_CACHE"]:
        os.makedirs(os.environ[key], exist_ok=True)

    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    for model_key in selected:
        if model_key not in ENCODER_MODELS:
            raise ValueError(f"Unknown encoder model: {model_key}. Available: {list(ENCODER_MODELS)}")

    for model_key in selected:
        train_one_model(model_key, ENCODER_MODELS[model_key], args)

    print("\nAll encoder models completed.")


if __name__ == "__main__":
    main()
