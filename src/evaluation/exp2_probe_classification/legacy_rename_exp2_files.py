from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "exp2_probe_classification" / "base" / "raw_outputs"

domains = {
    "SyntheticResults": "Exp2a",
    "PubMedResults": "Exp2b",
    "RedditResults": "Exp2c",
}

standard_names = {
    # Kimi
    "kimi zero": "Kimi-K2 Zero Shot {exp}.csv",
    "kimi one": "Kimi-K2 One Shot {exp}.csv",
    "kimi two": "Kimi-K2 Two Shot {exp}.csv",
    "kimi cot": "Kimi-K2 Two Shot + CoT {exp}.csv",

    # Llama
    "llama zero": "Llama-3.3-70B-Instruct Zero Shot {exp}.csv",
    "llama one": "Llama-3.3-70B-Instruct One Shot {exp}.csv",
    "llama two": "Llama-3.3-70B-Instruct Two Shot {exp}.csv",
    "llama cot": "Llama-3.3-70B-Instruct Two Shot + CoT {exp}.csv",

    # Phi
    "phi zero": "Phi-4-mini-instruct Zero Shot {exp}.csv",
    "phi one": "Phi-4-mini-instruct One Shot {exp}.csv",
    "phi two": "Phi-4-mini-instruct Two Shot {exp}.csv",
    "phi cot": "Phi-4-mini-instruct Two Shot + CoT {exp}.csv",

    # Qwen
    "qwen zero": "Qwen2.5-32B-Instruct Zero Shot {exp}.csv",
    "qwen one": "Qwen2.5-32B-Instruct One Shot {exp}.csv",
    "qwen two": "Qwen2.5-32B-Instruct Two Shot {exp}.csv",
    "qwen cot": "Qwen2.5-32B-Instruct Two Shot + CoT {exp}.csv",
}

def identify_file(name: str):
    n = name.lower()

    # Ignore hidden/system files
    if name.startswith("."):
        return None

    # Model
    if "kimi" in n or name in {"one_shot_combined.csv", "two_shot_combined.csv", "cot_two_shot_combined.csv"}:
        model = "kimi"
    elif "llama" in n:
        model = "llama"
    elif "phi" in n:
        model = "phi"
    elif "qwen" in n:
        model = "qwen"
    else:
        return None

    # Prompt
    if "cot_two_shot" in n or "two shot + cot" in n:
        prompt = "cot"
    elif "zero_shot" in n or "zero shot" in n:
        prompt = "zero"
    elif "one_shot" in n or "one shot" in n:
        prompt = "one"
    elif "two_shot" in n or "two shot" in n:
        prompt = "two"
    else:
        return None

    return f"{model} {prompt}"

for folder, exp in domains.items():
    folder_path = BASE / folder

    if not folder_path.exists():
        print(f"Missing folder: {folder_path}")
        continue

    for file in sorted(folder_path.glob("*.csv")):
        key = identify_file(file.name)

        if key is None:
            print(f"SKIP unknown: {file.name}")
            continue

        new_name = standard_names[key].format(exp=exp)
        target = folder_path / new_name

        if file.name == new_name:
            print(f"OK already named: {file.name}")
            continue

        if target.exists():
            print(f"SKIP target exists: {target.name} | source: {file.name}")
            continue

        print(f"RENAMING: {file.name} -> {target.name}")
        file.rename(target)
