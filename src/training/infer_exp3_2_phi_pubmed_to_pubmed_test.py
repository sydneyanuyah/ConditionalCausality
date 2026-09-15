import os
import re
import json
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


BASE_MODEL = "microsoft/Phi-4-mini-instruct"
ADAPTER_DIR = "outputs_exp3_2_phi_probe_ft_pubmed"
TEST_CSV = "pubmed/pubmed_exp3_2_probe_test.csv"
OUTPUT_CSV = "Exp 3.2 Phi-4-mini-instruct PubMed to PubMed-Test.csv"

os.environ.setdefault("HF_HOME", os.path.expanduser("~/SageMaker/hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/SageMaker/hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.expanduser("~/SageMaker/hf_datasets_cache"))
os.environ.setdefault("HF_METRICS_CACHE", os.path.expanduser("~/SageMaker/hf_metrics_cache"))


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


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


def build_prompt(tokenizer, user_text):
    messages = [{"role": "user", "content": user_text.strip()}]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User:\n{user_text.strip()}\n\nAssistant:\n"


def extract_decision(text):
    text = str(text).strip()

    try:
        obj = json.loads(text)
        decision = obj.get("decision", "")
        if decision:
            return decision
    except Exception:
        pass

    match = re.search(r'"decision"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1).strip()

    lowered = text.lower()
    if "not enough evidence" in lowered:
        return "Not Enough Evidence"
    if "not supported" in lowered:
        return "Not Supported"
    if "supported" in lowered:
        return "Supported"

    return ""


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map={"": 0},
        trust_remote_code=False,
    )

    print("Loading PubMed adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()

    print("Reading PubMed test file...")
    df = pd.read_csv(TEST_CSV, low_memory=False)

    required = ["Paragraph", "Probe_Question"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    raw_outputs = []
    predicted_decisions = []

    for idx, row in df.iterrows():
        paragraph = clean_text(row["Paragraph"])
        probe_question = clean_text(row["Probe_Question"])

        user_text = probe_instruction(paragraph, probe_question)
        prompt = build_prompt(tokenizer, user_text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = generated[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        raw_outputs.append(decoded)
        predicted_decisions.append(extract_decision(decoded))

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(df)} rows")

    df["raw_model_output"] = raw_outputs
    df["predicted_decision"] = predicted_decisions

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done. Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
