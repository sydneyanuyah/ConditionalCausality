import os
import json
import argparse
import ast

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


os.environ.setdefault("HF_HOME", os.path.expanduser("~/SageMaker/hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/SageMaker/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/SageMaker/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.expanduser("~/SageMaker/hf_metrics_cache"))

for p in [os.environ["HF_HOME"], os.environ["HF_DATASETS_CACHE"], os.environ["HF_METRICS_CACHE"]]:
    os.makedirs(p, exist_ok=True)


def get_rank():
    return int(os.environ.get("RANK", "0"))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process():
    return get_rank() == 0


def rank_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def setup_distributed_device():
    world_size = get_world_size()
    local_rank = get_local_rank()

    if torch.cuda.is_available() and world_size > 1:
        torch.cuda.set_device(local_rank)
        rank_print(f"Distributed training detected: WORLD_SIZE={world_size}")
        rank_print(f"Each process will load the model on its own LOCAL_RANK GPU.")
        return {"": local_rank}

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        rank_print("Single-process GPU training detected.")
        rank_print("Loading 4-bit/QLoRA model on cuda:0 inside the visible CUDA devices.")
        return {"": torch.cuda.current_device()}

    rank_print("CPU training detected.")
    return None


def read_csv(path):
    return pd.read_csv(path, low_memory=False)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_jsonish(x):
    if pd.isna(x):
        return []
    if isinstance(x, (list, dict)):
        return x

    s = str(x).strip()
    if not s:
        return []

    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def to_json_string(obj):
    return json.dumps(obj, ensure_ascii=False)


def normalize_label(label):
    label = clean_text(label)
    mapping = {
        "supported": "Supported",
        "not supported": "Not Supported",
        "not enough evidence": "Not Enough Evidence",
        "nee": "Not Enough Evidence",
        "contradiction": "Not Supported",
        "contradicted": "Not Supported",
    }
    return mapping.get(label.lower(), label)


def build_chat(tokenizer, user_text, assistant_text):
    messages = [
        {"role": "user", "content": user_text.strip()},
        {"role": "assistant", "content": assistant_text.strip()},
    ]

    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    return f"User:\n{user_text.strip()}\n\nAssistant:\n{assistant_text.strip()}"


def extraction_instruction(paragraph):
    return f"""You are extracting conditional causal relations from a paragraph.

Return only valid JSON in this exact format:
{{
  "predicted_tuples": [
    {{
      "e1": "...",
      "relation": "...",
      "e2": "...",
      "condition": "..."
    }}
  ]
}}

Paragraph:
{paragraph}

Answer:
"""


def probe_instruction(paragraph, probe_question):
    return f"""You are verifying a conditional causal claim against a paragraph.

Choose exactly one decision from:
- Supported
- Not Supported
- Not Enough Evidence

Return only valid JSON in this exact format:
{{
  "decision": "...",
  "reason_type": "",
  "explanation": ""
}}

Paragraph:
{paragraph}

Probe claim:
{probe_question}

Answer:
"""


def multitask_instruction(paragraph, tuple_text, probe_question):
    return f"""You are verifying a probe claim using the paragraph and the conditional causal tuple.

The tuple contains:
- e1
- relation
- e2
- condition

Choose exactly one decision from:
- Supported
- Not Supported
- Not Enough Evidence

Return only valid JSON in this exact format:
{{
  "decision": "...",
  "reason_type": "",
  "explanation": ""
}}

Paragraph:
{paragraph}

Conditional causal tuple:
{tuple_text}

Probe claim:
{probe_question}

Answer:
"""


def format_extraction_rows(df, tokenizer):
    paragraph_col = "Paragraph" if "Paragraph" in df.columns else None

    tuple_col = None
    for c in ["Gold Annotation", "Tuple", "gold_tuples_json", "Gold_Tuples", "Gold Tuples", "Sydney Gold", "tuples"]:
        if c in df.columns:
            tuple_col = c
            break

    if paragraph_col is None:
        raise ValueError(f"Extraction file must contain Paragraph column. Found: {list(df.columns)}")

    if tuple_col is None:
        raise ValueError(f"Extraction file must contain Tuple/gold tuple column. Found: {list(df.columns)}")

    texts = []

    for _, row in df.iterrows():
        paragraph = clean_text(row[paragraph_col])
        tuples = parse_jsonish(row[tuple_col])

        if isinstance(tuples, dict):
            tuples = [tuples]

        answer = {"predicted_tuples": tuples}

        user_text = extraction_instruction(paragraph)
        assistant_text = to_json_string(answer)

        texts.append(build_chat(tokenizer, user_text, assistant_text))

    return texts


def format_probe_rows(df, tokenizer):
    required = ["Paragraph", "Probe_Question", "Gold_Label"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Probe file missing columns {missing}. Found: {list(df.columns)}")

    texts = []

    for _, row in df.iterrows():
        paragraph = clean_text(row["Paragraph"])
        probe_question = clean_text(row["Probe_Question"])
        label = normalize_label(row["Gold_Label"])

        answer = {
            "decision": label,
            "reason_type": "",
            "explanation": ""
        }

        user_text = probe_instruction(paragraph, probe_question)
        assistant_text = to_json_string(answer)

        texts.append(build_chat(tokenizer, user_text, assistant_text))

    return texts


def format_multitask_rows(df, tokenizer):
    texts = []

    if "input" in df.columns and "output" in df.columns:
        for _, row in df.iterrows():
            user_text = clean_text(row["input"])
            assistant_text = clean_text(row["output"])

            if not assistant_text.startswith("{"):
                label = normalize_label(assistant_text)
                assistant_text = to_json_string({
                    "decision": label,
                    "reason_type": "",
                    "explanation": ""
                })

            texts.append(build_chat(tokenizer, user_text, assistant_text))

        return texts

    required = ["Paragraph", "Tuple", "Probe_Question", "Gold_Label"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Multitask file missing columns {missing}. Found: {list(df.columns)}")

    for _, row in df.iterrows():
        paragraph = clean_text(row["Paragraph"])
        tuple_text = clean_text(row["Tuple"])
        probe_question = clean_text(row["Probe_Question"])
        label = normalize_label(row["Gold_Label"])

        answer = {
            "decision": label,
            "reason_type": "",
            "explanation": ""
        }

        user_text = multitask_instruction(paragraph, tuple_text, probe_question)
        assistant_text = to_json_string(answer)

        texts.append(build_chat(tokenizer, user_text, assistant_text))

    return texts


class CausalSFTDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_length):
        self.examples = []
        for text in texts:
            enc = tokenizer(text, truncation=True, max_length=max_seq_length, padding=False)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = input_ids.copy()

            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", required=True, choices=["extraction", "probe", "multitask"])
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--validation_csv", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--model_name", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--max_seq_length", type=int, default=2048)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)

    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)

    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--skip_prepare_kbit_training", action="store_true", help="Skip PEFT prepare_model_for_kbit_training to reduce Qwen 32B memory pressure.")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--no_gradient_checkpointing", action="store_true", help="Disable gradient checkpointing.")

    args = parser.parse_args()

    gradient_checkpointing = not args.no_gradient_checkpointing
    device_map = setup_distributed_device()

    rank_print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rank_print("Reading datasets...")
    train_df = read_csv(args.train_csv)
    val_df = read_csv(args.validation_csv)

    rank_print("Train columns:", list(train_df.columns))
    rank_print("Validation columns:", list(val_df.columns))

    if args.method == "extraction":
        train_texts = format_extraction_rows(train_df, tokenizer)
        val_texts = format_extraction_rows(val_df, tokenizer)
    elif args.method == "probe":
        train_texts = format_probe_rows(train_df, tokenizer)
        val_texts = format_probe_rows(val_df, tokenizer)
    else:
        train_texts = format_multitask_rows(train_df, tokenizer)
        val_texts = format_multitask_rows(val_df, tokenizer)

    rank_print(f"Train examples: {len(train_texts)}")
    rank_print(f"Validation examples: {len(val_texts)}")

    train_dataset = CausalSFTDataset(train_texts, tokenizer, args.max_seq_length)
    val_dataset = CausalSFTDataset(val_texts, tokenizer, args.max_seq_length)

    rank_print("Loading model...")

    if args.use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map=device_map,
        trust_remote_code=False,
    )

    if args.use_qlora:
        if args.skip_prepare_kbit_training:
            rank_print("Skipping prepare_model_for_kbit_training; using lightweight QLoRA preparation.")
            model.config.use_cache = False

            for param in model.parameters():
                param.requires_grad = False

            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

            if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
        else:
            model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)

    if is_main_process():
        model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        gradient_checkpointing=gradient_checkpointing,

        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to="none",
        remove_unused_columns=False,
        seed=4000,
        data_seed=4000,

        ddp_find_unused_parameters=False if get_world_size() > 1 else None,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    rank_print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    rank_print("Saving best LoRA/QLoRA adapter...")
    trainer.save_model(args.output_dir)

    if is_main_process():
        tokenizer.save_pretrained(args.output_dir)

    rank_print(f"Done. Saved adapter/tokenizer/training state to: {args.output_dir}")


if __name__ == "__main__":
    main()
