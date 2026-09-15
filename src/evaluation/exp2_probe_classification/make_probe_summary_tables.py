from pathlib import Path
import re
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs" / "exp2_probe_classification" / "base"
EVAL_DIR = BASE / "evaluated_outputs"
OUT_DIR = ROOT / "results" / "exp2_probe_classification" / "summary_tables_regenerated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "synthetic": "Synthetic",
    "pubmed": "PubMed",
    "reddit": "Reddit",
}

MODELS = [
    "Kimi-K2",
    "Llama-3.3-70B-Instruct",
    "Phi-4-mini-instruct",
    "Qwen2.5-32B-Instruct",
]

PROMPTS = [
    "Zero Shot",
    "One Shot",
    "Two Shot",
    "Two Shot + CoT",
]

PROMPT_ORDER = {p: i for i, p in enumerate(PROMPTS)}

LABELS = ["Supported", "Not Supported", "Not Enough Evidence"]

MODEL_KEYS = {
    "kimi_k2": "Kimi-K2",
    "llama_33_70b_instruct": "Llama-3.3-70B-Instruct",
    "phi_4_mini_instruct": "Phi-4-mini-instruct",
    "qwen25_32b_instruct": "Qwen2.5-32B-Instruct",
}

PROMPT_KEYS = {
    "zero_shot": "Zero Shot",
    "one_shot": "One Shot",
    "two_shot": "Two Shot",
    "two_shot_plus_cot": "Two Shot + CoT",
}


def clean_text(s):
    s = str(s)

    # Remove common invisible/control characters from copied filenames
    invisible_chars = [
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # BOM
        "\xa0",    # non-breaking space
    ]

    for ch in invisible_chars:
        s = s.replace(ch, " ")

    # Collapse repeated spaces into one clean space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def snake_name(s):
    s = clean_text(s)
    s = s.replace("+", " plus ")
    s = s.replace("2.5", "25").replace("3.3", "33").replace("4-", "4_")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def parse_filename(path: Path):
    dataset_code = path.parent.name.lower()
    dataset = DATASETS.get(dataset_code)
    if dataset is None:
        print(f"Skipped: could not detect dataset from folder -> {path}")
        return None

    original_name = path.name
    stem = snake_name(original_name)
    stem = stem.removesuffix("_csv").removesuffix("_evaluated")

    model = None
    model_key = None
    for key, label in MODEL_KEYS.items():
        if stem.startswith(key):
            model_key = key
            model = label
            break

    if model is None:
        print(f"Skipped: could not detect model -> {repr(original_name)} | cleaned: {repr(stem)}")
        return None

    rest = stem[len(model_key):].strip("_")

    prompt = None
    for key, label in sorted(PROMPT_KEYS.items(), key=lambda x: len(x[0]), reverse=True):
        if rest == key:
            prompt = label
            break

    if prompt is None:
        print(f"Skipped: could not detect prompt -> {repr(original_name)} | cleaned prompt: {repr(rest)}")
        return None

    return {
        "Dataset": dataset,
        "Dataset Code": dataset_code,
        "Model": model,
        "Prompt": prompt,
        "File": str(path),
    }

def safe_first(df, col):
    if col not in df.columns or df.empty:
        return np.nan
    vals = df[col].dropna()
    return vals.iloc[0] if len(vals) else np.nan


def accuracy_for_probe(df, probe_num):
    if "Probe Class" not in df.columns or "Correct" not in df.columns:
        return np.nan
    sub = df[df["Probe Class"] == probe_num]
    if len(sub) == 0:
        return np.nan
    return sub["Correct"].mean()


def class_f1(df, label):
    if "Gold Label" not in df.columns or "Class F1" not in df.columns:
        return np.nan
    sub = df[df["Gold Label"] == label]
    if len(sub) == 0:
        return np.nan
    vals = sub["Class F1"].dropna()
    return vals.iloc[0] if len(vals) else np.nan


def compute_ogr(df):
    """
    OGR = Overgeneralization Rate.
    Condition-altered probes are Probe 2, Probe 3, and Probe 5.
    It measures how often the model predicts Supported for claims where the
    condition was replaced, removed, or contradicted.
    Lower is better.
    """
    if "Probe Class" not in df.columns or "Prediction" not in df.columns:
        return np.nan

    sub = df[df["Probe Class"].isin([2, 3, 5])]
    if len(sub) == 0:
        return np.nan

    return (sub["Prediction"] == "Supported").mean()


def get_probe_group_id(probe_id):
    """
    Groups probes from the same base example.
    Example:
    P0001_T1_1 -> P0001_T1
    P0001_1    -> P0001
    """
    s = str(probe_id)
    return re.sub(r"[_-]?[1-5]$", "", s)


def compute_sps(df):
    """
    SPS = Scope Preservation Score.
    A group is correct only if all available five probes for the same base
    example are correct. Uses complete groups containing probes 1 to 5.
    Higher is better.
    """
    if "ProbeID" not in df.columns or "Probe Class" not in df.columns or "Correct" not in df.columns:
        return np.nan

    tmp = df.copy()
    tmp["Probe Group"] = tmp["ProbeID"].apply(get_probe_group_id)

    group_scores = []
    for _, g in tmp.groupby("Probe Group"):
        probes = set(g["Probe Class"].dropna().astype(int).tolist())
        if not set([1, 2, 3, 4, 5]).issubset(probes):
            continue

        needed = g[g["Probe Class"].isin([1, 2, 3, 4, 5])]
        probe_correct = needed.groupby("Probe Class")["Correct"].max()
        group_scores.append(float(probe_correct.loc[[1, 2, 3, 4, 5]].all()))

    if len(group_scores) == 0:
        return np.nan

    return float(np.mean(group_scores))


def summarize_file(path: Path):
    meta = parse_filename(path)
    if meta is None:
        return None

    df = pd.read_csv(path)

    row = {
        **meta,
        "Rows": len(df),
        "Accuracy": safe_first(df, "Overall Accuracy"),
        "Macro Precision": safe_first(df, "Overall Macro Precision"),
        "Macro Recall": safe_first(df, "Overall Macro Recall"),
        "Macro F1": safe_first(df, "Overall Macro F1"),
        "Weighted Precision": safe_first(df, "Overall Weighted Precision"),
        "Weighted Recall": safe_first(df, "Overall Weighted Recall"),
        "Weighted F1": safe_first(df, "Overall Weighted F1"),
        "Supported F1": class_f1(df, "Supported"),
        "Not Supported F1": class_f1(df, "Not Supported"),
        "Not Enough Evidence F1": class_f1(df, "Not Enough Evidence"),
        "Probe 1 Accuracy": accuracy_for_probe(df, 1),
        "Probe 2 Accuracy": accuracy_for_probe(df, 2),
        "Probe 3 Accuracy": accuracy_for_probe(df, 3),
        "Probe 4 Accuracy": accuracy_for_probe(df, 4),
        "Probe 5 Accuracy": accuracy_for_probe(df, 5),
        "OGR": compute_ogr(df),
        "SPS": compute_sps(df),
    }

    return row


def bold_best_numeric(df, columns, higher_is_better=True):
    out = df.copy()

    for col in columns:
        if col not in out.columns:
            continue

        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.dropna().empty:
            continue

        best = numeric.max() if higher_is_better else numeric.min()

        formatted = []
        for val in numeric:
            if pd.isna(val):
                formatted.append("")
            else:
                text = f"{val:.4f}"
                if np.isclose(val, best):
                    text = f"\\textbf{{{text}}}"
                formatted.append(text)

        out[col] = formatted

    return out


def save_table(df, filename, bold=True):
    path = OUT_DIR / filename
    df.to_csv(path, index=False)

    if bold:
        numeric_cols = [
            "Accuracy",
            "Macro Precision",
            "Macro Recall",
            "Macro F1",
            "Weighted F1",
            "Supported F1",
            "Not Supported F1",
            "Not Enough Evidence F1",
            "Probe 1 Accuracy",
            "Probe 2 Accuracy",
            "Probe 3 Accuracy",
            "Probe 4 Accuracy",
            "Probe 5 Accuracy",
            "SPS",
        ]

        bold_df = bold_best_numeric(df, numeric_cols, higher_is_better=True)
        if "OGR" in bold_df.columns:
            bold_df = bold_best_numeric(bold_df, ["OGR"], higher_is_better=False)

        bold_path = OUT_DIR / filename.replace(".csv", "_bolded_for_latex.csv")
        bold_df.to_csv(bold_path, index=False)

    return path


# -----------------------------
# 1. Read all evaluated files
# -----------------------------

print("Files found in EvaluatedResults:")
for path in sorted(EVAL_DIR.rglob("*_evaluated.csv")):
    print(repr(path.name))

rows = []
for path in sorted(EVAL_DIR.rglob("*_evaluated.csv")):
    item = summarize_file(path)
    if item is not None:
        rows.append(item)

summary = pd.DataFrame(rows)

if summary.empty:
    raise SystemExit(f"No evaluated CSVs found in {EVAL_DIR}")

summary["Prompt Order"] = summary["Prompt"].map(PROMPT_ORDER)
summary = summary.sort_values(["Dataset", "Model", "Prompt Order"]).reset_index(drop=True)

# This is the extra master raw summary file, useful for checking everything.
save_table(summary.drop(columns=["Prompt Order"]), "00_all_results_summary.csv", bold=False)


# -----------------------------
# 2. Per-dataset, per-model prompt comparison tables
# -----------------------------

table_count = 1

table_cols = [
    "Prompt",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Supported F1",
    "Not Supported F1",
    "Not Enough Evidence F1",
    "Probe 1 Accuracy",
    "Probe 2 Accuracy",
    "Probe 3 Accuracy",
    "Probe 4 Accuracy",
    "Probe 5 Accuracy",
    "OGR",
    "SPS",
    "Rows",
]

for dataset in ["Synthetic", "PubMed", "Reddit"]:
    for model in MODELS:
        sub = summary[(summary["Dataset"] == dataset) & (summary["Model"] == model)].copy()

        # Skip all-missing models, e.g. Llama Reddit if absent.
        if sub.empty:
            continue

        sub = sub.sort_values("Prompt Order")
        clean_model = model.replace("/", "-")
        filename = f"{table_count:02d}_{dataset}_{clean_model}_prompt_comparison.csv"
        save_table(sub[table_cols], filename, bold=True)
        table_count += 1


# -----------------------------
# 3. Overall best prompt per model within each dataset
# -----------------------------

best_rows = []

for dataset in ["Synthetic", "PubMed", "Reddit"]:
    dsub = summary[summary["Dataset"] == dataset].copy()

    for model in MODELS:
        msub = dsub[dsub["Model"] == model].copy()
        if msub.empty:
            continue

        # Primary ranking metric: Macro F1.
        # Ties broken by SPS, then lower OGR, then Accuracy.
        msub = msub.sort_values(
            by=["Macro F1", "SPS", "OGR", "Accuracy"],
            ascending=[False, False, True, False]
        )

        best_rows.append(msub.iloc[0].to_dict())

best = pd.DataFrame(best_rows)
best = best.sort_values(["Dataset", "Model"]).reset_index(drop=True)

overall_cols = [
    "Dataset",
    "Model",
    "Prompt",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Supported F1",
    "Not Supported F1",
    "Not Enough Evidence F1",
    "Probe 1 Accuracy",
    "Probe 2 Accuracy",
    "Probe 3 Accuracy",
    "Probe 4 Accuracy",
    "Probe 5 Accuracy",
    "OGR",
    "SPS",
    "Rows",
]

for dataset in ["Synthetic", "PubMed", "Reddit"]:
    sub = best[best["Dataset"] == dataset].copy()
    filename = f"{table_count:02d}_{dataset}_best_prompt_per_model.csv"
    save_table(sub[overall_cols], filename, bold=True)
    table_count += 1


# -----------------------------
# 4. Master table: best prompt per model across datasets
# -----------------------------

master = best.copy()
master = master.sort_values(["Model", "Dataset"]).reset_index(drop=True)

master_cols = [
    "Dataset",
    "Model",
    "Prompt",
    "Accuracy",
    "Macro F1",
    "Weighted F1",
    "Supported F1",
    "Not Supported F1",
    "Not Enough Evidence F1",
    "Probe 1 Accuracy",
    "Probe 2 Accuracy",
    "Probe 3 Accuracy",
    "Probe 4 Accuracy",
    "Probe 5 Accuracy",
    "OGR",
    "SPS",
    "Rows",
]

save_table(master[master_cols], f"{table_count:02d}_MASTER_best_prompt_per_model_across_datasets.csv", bold=True)
table_count += 1


# -----------------------------
# 5. Missing expected files report
# -----------------------------

expected_rows = []
for code, dataset in DATASETS.items():
    for model in MODELS:
        for prompt in PROMPTS:
            expected_rows.append({
                "Dataset": dataset,
                "Dataset Code": code,
                "Model": model,
                "Prompt": prompt,
            })

expected = pd.DataFrame(expected_rows)

present = summary[["Dataset", "Dataset Code", "Model", "Prompt"]].drop_duplicates()
missing = expected.merge(
    present,
    on=["Dataset", "Dataset Code", "Model", "Prompt"],
    how="left",
    indicator=True
)
missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])

missing.to_csv(OUT_DIR / f"{table_count:02d}_missing_expected_runs.csv", index=False)

print(f"Done. Tables saved to: {OUT_DIR}")
print(f"Runs summarized: {len(summary)}")
print(f"Dataset/model/prompt table files created, plus dataset best tables, master table, all-results summary, and missing report.")
print(f"Missing expected runs: {len(missing)}")
