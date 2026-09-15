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


def normalize_decision(x):
    s = str(x).strip()

    try:
        obj = json.loads(extract_first_json(s))
        if isinstance(obj, dict):
            if "decision" in obj:
                s = obj["decision"]
            elif "label" in obj:
                s = obj["label"]
            elif "output" in obj and isinstance(obj["output"], dict) and "decision" in obj["output"]:
                s = obj["output"]["decision"]
    except Exception:
        pass

    s_low = str(s).strip().lower()

    if "not enough" in s_low or s_low == "nee":
        return "Not Enough Evidence"
    if "not supported" in s_low or "unsupported" in s_low or "contradiction" in s_low or "contradicted" in s_low:
        return "Not Supported"
    if "supported" in s_low:
        return "Supported"

    return str(s).strip()


def validate_or_wrap(text):
    cleaned = extract_first_json(text)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            decision = normalize_decision(json.dumps(obj, ensure_ascii=False))
            obj["decision"] = decision
            obj.setdefault("reason_type", "")
            obj.setdefault("explanation", "")
            return json.dumps(obj, ensure_ascii=False), decision
    except Exception:
        pass

    decision = normalize_decision(text)
    wrapped = {
        "decision": decision,
        "reason_type": "",
        "explanation": "",
        "raw_generation": str(text).strip()
    }
    return json.dumps(wrapped, ensure_ascii=False), decision


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
    model,
    tokenizer,
    batch_size,
    max_new_tokens,
    resume
):
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"{dataset_name} input not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)

    # Drop old gold-output column if present. Keep Gold_Label for evaluation.
    if "output" in df.columns:
        df = df.drop(columns=["output"])

    required = ["Paragraph", "Probe_Question"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} missing columns {missing}. Found: {list(df.columns)}")

    df = df.copy()
    if "row_id" not in df.columns:
        df.insert(0, "row_id", range(len(df)))

    done = already_done(output_csv) if resume else set()

    rows = []
    for _, row in df.iterrows():
        rid = int(row["row_id"])
        if rid not in done:
            rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    write_header = not output_csv.exists() or not resume
    output_cols = list(df.columns) + ["model_output", "predicted_label", "error"]

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
                probe_instruction(r["Paragraph"], r["Probe_Question"])
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
                    model_output, predicted_label = validate_or_wrap(gen_text)
                    out = row.to_dict()
                    out["model_output"] = model_output
                    out["predicted_label"] = predicted_label
                    out["error"] = ""
                    writer.writerow(out)
                    f.flush()

            except Exception as e:
                for row in batch_rows:
                    out = row.to_dict()
                    out["model_output"] = json.dumps({
                        "decision": "",
                        "reason_type": "",
                        "explanation": ""
                    }, ensure_ascii=False)
                    out["predicted_label"] = ""
                    out["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    writer.writerow(out)
                    f.flush()

    print(f"Done: {output_csv}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", default="microsoft/Phi-4-mini-instruct")
    parser.add_argument("--adapter_path", default="outputs_exp3_2_phi_probe_ft")

    parser.add_argument("--synthetic_csv", default="exp3_2_probe_splits/exp3_2_probe_test.csv")
    parser.add_argument("--pubmed_csv", default="pubmed_exp3_test_csvs/pubmed_exp3_2_probe_test.csv")
    parser.add_argument("--reddit_csv", default="reddit_exp3_test_csvs/reddit_exp3_2_probe_test.csv")

    parser.add_argument("--output_dir", default="Exp3_2_Probe_Test_Outputs")

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
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

    print("Loading Exp3.2 adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("Synthetic", args.synthetic_csv, out_dir / "synthetic_exp3_2_probe_test_output_phi_probe_ft.csv"),
        ("PubMed", args.pubmed_csv, out_dir / "pubmed_exp3_2_probe_test_output_phi_probe_ft.csv"),
        ("Reddit", args.reddit_csv, out_dir / "reddit_exp3_2_probe_test_output_phi_probe_ft.csv"),
    ]

    for dataset_name, input_csv, output_csv in datasets:
        run_one_dataset(
            dataset_name=dataset_name,
            input_csv=input_csv,
            output_csv=output_csv,
            model=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            resume=args.resume,
        )

    print("\nAll Exp3.2 probe classification tests completed.")
    print(f"Saved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
