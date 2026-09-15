import argparse, json
import pandas as pd

gold = {
    1: "Not Supported",
    2: "Not Enough Evidence",
    3: "Not Enough Evidence",
    4: "Supported",
    5: "Not Supported"
}

labels = ["Supported", "Not Supported", "Not Enough Evidence"]

def first_json_with_decision(x):
    text = str(x).strip().replace("```json", "").replace("```", "").strip()
    decoder = json.JSONDecoder()

    for i, ch in enumerate(text):
        if ch != "{":
            continue

        try:
            obj, _ = decoder.raw_decode(text[i:])
            if isinstance(obj, dict) and "decision" in obj:
                return obj
        except Exception:
            continue

    raise ValueError("No valid JSON object with decision found")

def parse_output(x):
    try:
        obj = first_json_with_decision(x)
        decision = str(obj.get("decision", "")).strip()

        return pd.Series({
            "Prediction": decision if decision in labels else "Invalid",
            "reason_type": obj.get("reason_type", ""),
            "explanation": obj.get("explanation", ""),
            "parse_status": "parsed"
        })

    except Exception as e:
        return pd.Series({
            "Prediction": "Invalid",
            "reason_type": "",
            "explanation": "",
            "parse_status": f"parse_error: {e}"
        })

def prf(actual, pred):
    rows = []

    for label in labels:
        tp = ((actual == label) & (pred == label)).sum()
        fp = ((actual != label) & (pred == label)).sum()
        fn = ((actual == label) & (pred != label)).sum()
        support = (actual == label).sum()

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        rows.append({
            "Label": label,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "Support": support
        })

    m = pd.DataFrame(rows)
    total = m["Support"].sum()

    macro = {
        "Precision": m["Precision"].mean(),
        "Recall": m["Recall"].mean(),
        "F1": m["F1"].mean()
    }

    weighted = {
        "Precision": (m["Precision"] * m["Support"]).sum() / total if total else 0,
        "Recall": (m["Recall"] * m["Support"]).sum() / total if total else 0,
        "F1": (m["F1"] * m["Support"]).sum() / total if total else 0
    }

    return m, macro, weighted

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()

df = pd.read_csv(args.input)

df["Probe Class"] = df["ProbeID"].astype(str).apply(lambda x: int(x.split("_")[-1]))
df["Gold Label"] = df["Probe Class"].map(gold)

parsed = df["raw_model_output"].apply(parse_output)
df = pd.concat([df, parsed], axis=1)

df["Correct"] = df["Gold Label"].eq(df["Prediction"])

df["Overall Accuracy"] = df["Correct"].mean()
df["Probe Accuracy"] = df.groupby("Probe Class")["Correct"].transform("mean")
df["Class Accuracy"] = df.groupby("Gold Label")["Correct"].transform("mean")

class_metrics, macro, weighted = prf(df["Gold Label"], df["Prediction"])

df["Overall Macro Precision"] = macro["Precision"]
df["Overall Macro Recall"] = macro["Recall"]
df["Overall Macro F1"] = macro["F1"]

df["Overall Weighted Precision"] = weighted["Precision"]
df["Overall Weighted Recall"] = weighted["Recall"]
df["Overall Weighted F1"] = weighted["F1"]

for probe_class in sorted(df["Probe Class"].dropna().unique()):
    mask = df["Probe Class"] == probe_class
    _, probe_macro, probe_weighted = prf(df.loc[mask, "Gold Label"], df.loc[mask, "Prediction"])

    df.loc[mask, "Probe Macro Precision"] = probe_macro["Precision"]
    df.loc[mask, "Probe Macro Recall"] = probe_macro["Recall"]
    df.loc[mask, "Probe Macro F1"] = probe_macro["F1"]

    df.loc[mask, "Probe Weighted Precision"] = probe_weighted["Precision"]
    df.loc[mask, "Probe Weighted Recall"] = probe_weighted["Recall"]
    df.loc[mask, "Probe Weighted F1"] = probe_weighted["F1"]

for label in labels:
    mask = df["Gold Label"] == label
    tp = ((df["Gold Label"] == label) & (df["Prediction"] == label)).sum()
    fp = ((df["Gold Label"] != label) & (df["Prediction"] == label)).sum()
    fn = ((df["Gold Label"] == label) & (df["Prediction"] != label)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    df.loc[mask, "Class Precision"] = precision
    df.loc[mask, "Class Recall"] = recall
    df.loc[mask, "Class F1"] = f1

df.to_csv(args.output, index=False)