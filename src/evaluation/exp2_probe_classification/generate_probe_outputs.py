import argparse
import csv
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts_dictionary import PROMPTS


OUTPUT_COLUMNS = [
    "ProbeID",
    "Paragraph",
    "Probe Question",
    "prompt_name",
    "model_name",
    "raw_model_output",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def build_prompt(prompt_name: str, paragraph: str, probe_question: str) -> str:
    return PROMPTS[prompt_name].format(
        paragraph=str(paragraph),
        probe_question=str(probe_question),
    )


def append_rows(output_csv: str, rows):
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description="Experiment 2 raw generation only.")

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--prompt_name", required=True, choices=list(PROMPTS.keys()))

    parser.add_argument("--id_col", default="ProbeID")
    parser.add_argument("--paragraph_col", default="Paragraph")
    parser.add_argument("--probe_question_col", default="Probe Question")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_chat_template", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    set_seed(args.seed)

    data = pd.read_csv(args.input_csv)

    required_cols = [
        args.id_col,
        args.paragraph_col,
        args.probe_question_col,
    ]

    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required input columns: {missing_cols}")

    if args.limit is not None:
        data = data.head(args.limit).copy()

    if args.resume and os.path.exists(args.output_csv):
        existing = pd.read_csv(args.output_csv)
        completed = set(existing["ProbeID"].astype(str))
        data = data[~data[args.id_col].astype(str).isin(completed)].copy()
        print(f"Resume enabled. Skipping {len(completed)} completed ProbeID values.")

    if data.empty:
        print("No rows to process.")
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=get_dtype(args.torch_dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )

    model.eval()

    for start in tqdm(range(0, len(data), args.batch_size), desc="Generating"):
        batch = data.iloc[start:start + args.batch_size]

        prompts = []
        for _, row in batch.iterrows():
            prompt = build_prompt(
                args.prompt_name,
                row[args.paragraph_col],
                row[args.probe_question_col],
            )

            if not args.no_chat_template and getattr(tokenizer, "chat_template", None):
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if args.do_sample:
            generation_kwargs.update({
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
            })
        else:
            generation_kwargs["do_sample"] = False

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                **generation_kwargs,
            )

        input_length = inputs["input_ids"].shape[1]

        raw_outputs = tokenizer.batch_decode(
            generated[:, input_length:],
            skip_special_tokens=True,
        )

        rows_to_write = []
        for (_, row), raw_output in zip(batch.iterrows(), raw_outputs):
            rows_to_write.append({
                "ProbeID": row[args.id_col],
                "Paragraph": row[args.paragraph_col],
                "Probe Question": row[args.probe_question_col],
                "prompt_name": args.prompt_name,
                "model_name": args.model_name,
                "raw_model_output": raw_output.strip(),
            })

        append_rows(args.output_csv, rows_to_write)

    print(f"Saved raw model outputs to: {args.output_csv}")


if __name__ == "__main__":
    main()