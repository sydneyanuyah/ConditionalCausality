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

from train_infer_encoder_models import (
    LABELS,
    LABEL2ID,
    ID2LABEL,
    ENCODER_MODELS,
    EncoderClassificationDataset,
    load_csv,
    get_latest_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_key", required=True, choices=list(ENCODER_MODELS.keys()))
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--validation_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--train_source", required=True, choices=["PubMed", "Reddit"])
    parser.add_argument("--test_name", required=True, choices=["PubMed-Test", "Reddit-Test"])
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--output_dir", default="encoder_exp3_2_domain_ft")
    parser.add_argument("--task", default="probe", choices=["probe"])
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

    model_name = ENCODER_MODELS[args.model_key]
    run_dir = Path(args.output_dir) / args.train_source.lower() / args.model_key
    checkpoint_dir = run_dir / "checkpoints"
    final_model_dir = run_dir / "final_model"

    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Model key: {args.model_key}")
    print(f"Model name: {model_name}")
    print(f"Train source: {args.train_source}")
    print(f"Test target: {args.test_name}")
    print(f"Output CSV: {args.output_csv}")
    print("=" * 80)

    train_df = load_csv(args.train_csv)
    val_df = load_csv(args.validation_csv)
    test_df = load_csv(args.test_csv)

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

    test_ds = EncoderClassificationDataset(
        test_df,
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
        output_dir=str(checkpoint_dir),
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

    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("No checkpoint found. Starting from scratch.")
        trainer.train()

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    print(f"Saved final model to: {final_model_dir}")

    print("Running inference...")
    pred = trainer.predict(test_ds)
    logits = pred.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    pred_ids = np.argmax(probs, axis=-1)

    out_df = test_df.copy()
    out_df["model_name"] = model_name
    out_df["encoder_model"] = args.model_key
    out_df["task"] = args.task
    out_df["train_source"] = args.train_source
    out_df["test_name"] = args.test_name
    out_df["predicted_label"] = [ID2LABEL[int(i)] for i in pred_ids]

    for i, label in ID2LABEL.items():
        safe_label = label.lower().replace(" ", "_")
        out_df[f"prob_{safe_label}"] = probs[:, i]

    out_path = Path(args.output_csv)
    out_df.to_csv(out_path, index=False)

    print(f"Done. Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()
