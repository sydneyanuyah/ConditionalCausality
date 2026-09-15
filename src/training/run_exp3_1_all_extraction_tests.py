import argparse
import csv
import json
import os
import re
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from prompt import PROMPTS


def extract_first_json(text):
    if text is None:
        return ""

    text = str(text).strip()
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text


def validate_or_wrap(text):
    cleaned = extract_first_json(text)

    try:
        obj = json.loads(cleaned)
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return json.dumps({
            "output": {
                "predicted_tuples": []
            },
            "raw_generation": str(text).strip()
        }, ensure_ascii=False)


def make_prompt(template, text):
    text = "" if pd.isna(text) else str(text)
    return template.replace("<TEXT>", text)


def already_done(output_csv):
    path = Path(output_csv)
    if not path.exists():
        return set()

    try:
        old = pd.read_csv(path, usecols=["row_id"])
        return set(old["row_id"].astype(int).tolist())
    except Exception:
        return set()


def batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


def run_one_dataset(
    dataset_name,
    input_csv,
    output_csv,
    input_column,
    model,
    tokenizer,
    prompt_template,
    batch_size,
    max_new_tokens,
    resume
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"{dataset_name} input not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    if input_column not in df.columns:
        raise ValueError(
            f"{dataset_name}: input column '{input_column}' not found. Columns: {list(df.columns)}"
        )

    df = df.copy()
    df.insert(0, "row_id", range(len(df)))

    done = already_done(output_csv) if resume else set()

    rows = []
    for _, row in df.iterrows():
        rid = int(row["row_id"])
        if rid not in done:
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_csv.exists() or not resume
    output_cols = list(df.columns) + ["output", "error"]

    print(f"\n===== {dataset_name} =====")
    print(f"Input: {input_csv}")
    print(f"Output: {output_csv}")
    print(f"Rows total: {len(df)}")
    print(f"Rows to process: {len(rows)}")

    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_cols)

        if write_header:
            writer.writeheader()

        for batch_rows in batched(rows, batch_size):
            prompts = [
                make_prompt(prompt_template, r[input_column])
                for r in batch_rows
            ]

            try:
                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                ).to(model.device)

                with torch.no_grad():
                    generated = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                input_len = encoded["input_ids"].shape[1]
                new_tokens = generated[:, input_len:]
                texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

                for row, gen_text in zip(batch_rows, texts):
                    out = row.to_dict()
                    out["output"] = validate_or_wrap(gen_text)
                    out["error"] = ""
                    writer.writerow(out)
                    f.flush()

            except Exception as e:
                for row in batch_rows:
                    out = row.to_dict()
                    out["output"] = json.dumps({
                        "output": {
                            "predicted_tuples": []
                        }
                    }, ensure_ascii=False)
                    out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    writer.writerow(out)
                    f.flush()

    print(f"Done: {output_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--adapter_path", default="outputs_exp3_1_phi_extraction_ft")
    parser.add_argument("--prompt_name", default="two_shot", choices=sorted(PROMPTS.keys()))

    parser.add_argument("--synthetic_csv", default="exp3_1_extraction_splits/exp3_1_extraction_test.csv")
    parser.add_argument("--pubmed_csv", default="pubmed_exp3_test_csvs/pubmed_exp3_1_extraction_test.csv")
    parser.add_argument("--reddit_csv", default="reddit_exp3_test_csvs/reddit_exp3_1_extraction_test.csv")

    parser.add_argument("--output_dir", default="Exp3_1_Extraction_Test_Outputs")
    parser.add_argument("--input_column", default="Paragraph")

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/SageMaker/hf_cache"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/SageMaker/hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/SageMaker/hf_datasets_cache"))
    os.environ.setdefault("HF_METRICS_CACHE", os.path.expanduser("~/SageMaker/hf_metrics_cache"))

    for key in ["HF_HOME", "HF_DATASETS_CACHE", "HF_METRICS_CACHE"]:
        os.makedirs(os.environ[key], exist_ok=True)

    if args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    prompt_template = PROMPTS[args.prompt_name]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=False,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    quant_config = None
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=False,
    )

    print("Loading Exp3.1 adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        (
            "Synthetic",
            args.synthetic_csv,
            out_dir / "synthetic_exp3_1_extraction_test_output_phi_ft_two_shot.csv",
        ),
        (
            "PubMed",
            args.pubmed_csv,
            out_dir / "pubmed_exp3_1_extraction_test_output_phi_ft_two_shot.csv",
        ),
        (
            "Reddit",
            args.reddit_csv,
            out_dir / "reddit_exp3_1_extraction_test_output_phi_ft_two_shot.csv",
        ),
    ]

    for dataset_name, input_csv, output_csv in datasets:
        run_one_dataset(
            dataset_name=dataset_name,
            input_csv=input_csv,
            output_csv=output_csv,
            input_column=args.input_column,
            model=model,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )

    print("\nAll Exp3.1 extraction tests completed.")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
