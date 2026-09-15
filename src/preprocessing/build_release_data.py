#!/usr/bin/env python3
"""Build compact public probe/prediction files from archived experiment outputs.

This is an assembly utility. It intentionally excludes raw generations and model
weights because the saved class decisions are sufficient for metric reproduction.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile


PROBE_TYPE = {
    1: "relation_contradicted",
    2: "condition_replaced",
    3: "condition_removed",
    4: "original_condition",
    5: "condition_contradicted",
}
GOLD_BY_PROBE = {
    1: "Not Supported",
    2: "Not Enough Evidence",
    3: "Not Enough Evidence",
    4: "Supported",
    5: "Not Supported",
}


def normalize_label(value: str) -> str:
    key = " ".join(str(value).strip().split()).casefold()
    return {
        "supported": "Supported",
        "not supported": "Not Supported",
        "not enough evidence": "Not Enough Evidence",
    }.get(key, str(value).strip())

PROMPTED = {
    "kimi": ("Kimi-K2", "kimi_k2_two_shot_plus_cot_evaluated.csv", "two_shot_plus_cot"),
    "llama": ("Llama-3.3-70B-Instruct", "llama_33_70b_instruct_two_shot_plus_cot_evaluated.csv", "two_shot_plus_cot"),
    "phi": ("Phi-4-mini-instruct", "phi_4_mini_instruct_zero_shot_evaluated.csv", "zero_shot"),
    "qwen": ("Qwen2.5-32B-Instruct", "qwen25_32b_instruct_two_shot_plus_cot_evaluated.csv", "two_shot_plus_cot"),
}


def first(row: dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return ""


def probe_number(row: dict[str, str], probe_id: str) -> int:
    value = first(row, "Probe_Number", "Probe Class").strip()
    if value:
        return int(float(value))
    match = re.search(r"_(\d+)$", probe_id)
    if not match:
        raise ValueError(f"Cannot derive probe number from {probe_id!r}")
    return int(match.group(1))


def tuple_id(row: dict[str, str], probe_id: str) -> str:
    value = first(row, "Tuple_ID", "Tuple ID").strip()
    return value or re.sub(r"_\d+$", "", probe_id)


def paragraph_id(row: dict[str, str], tuple_value: str) -> str:
    value = first(row, "Paragraph_ID", "Paragraph ID", "Paragraph_id").strip()
    return value or re.sub(r"_T\d+$", "", tuple_value)


@contextmanager
def rows_from(path: Path, member: str = ""):
    archive = raw = text = None
    try:
        if member:
            archive = ZipFile(path)
            raw = archive.open(member)
            text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        else:
            text = path.open(encoding="utf-8-sig", newline="")
        yield csv.DictReader(text)
    finally:
        if text:
            text.close()
        if raw:
            raw.close()
        if archive:
            archive.close()


def write_prediction(reader, output: Path, prediction_column: str) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["Paragraph_ID", "Tuple_ID", "ProbeID", "Probe_Number", "Probe_Type", "Gold_Label", "predicted_label"]
    count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            pid = first(row, "ProbeID").strip()
            tid = tuple_id(row, pid)
            pnum = probe_number(row, pid)
            writer.writerow({
                "Paragraph_ID": paragraph_id(row, tid),
                "Tuple_ID": tid,
                "ProbeID": pid,
                "Probe_Number": pnum,
                "Probe_Type": first(row, "Probe_Type").strip() or PROBE_TYPE[pnum],
                "Gold_Label": normalize_label(first(row, "Gold_Label", "Gold Label")),
                "predicted_label": normalize_label(first(row, prediction_column)),
            })
            count += 1
    return count


def write_probe_data(cleaned: Path, output_root: Path) -> None:
    for domain in ("synthetic", "pubmed", "reddit"):
        filename = "synthetic_probes_joined_seed1_sorted.csv" if domain == "synthetic" else f"{domain}_probes.csv"
        source_path = cleaned / "inputs" / "exp2_probe_classification" / filename
        output = output_root / "data" / "probes" / f"{domain}_probes.csv"
        fields = ["Domain", "Paragraph_ID", "Source_Document_ID", "Tuple_ID", "ProbeID", "Probe_Number", "Probe_Type", "Paragraph", "Probe_Question", "Gold_Label"]
        output.parent.mkdir(parents=True, exist_ok=True)
        with rows_from(source_path) as reader, output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                pid = first(row, "ProbeID").strip()
                tid = tuple_id(row, pid)
                pnum = probe_number(row, pid)
                paragraph = paragraph_id(row, tid)
                # PubMed IDs encode article group + paragraph (e.g., P12PR3).
                # This is a stable benchmark grouping ID, not a claimed PMID.
                source = re.sub(r"PR\d+$", "", paragraph) if domain == "pubmed" else paragraph
                writer.writerow({
                    "Domain": domain.capitalize(),
                    "Paragraph_ID": paragraph,
                    "Source_Document_ID": source,
                    "Tuple_ID": tid,
                    "ProbeID": pid,
                    "Probe_Number": pnum,
                    "Probe_Type": first(row, "Probe_Type").strip() or PROBE_TYPE[pnum],
                    "Paragraph": first(row, "Paragraph"),
                    "Probe_Question": first(row, "Probe_Question"),
                    "Gold_Label": normalize_label(first(row, "Gold_Label")) or GOLD_BY_PROBE[pnum],
                })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--cleaned", type=Path, required=True)
    parser.add_argument("--finetuned-manifest", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    write_probe_data(args.cleaned, repo)
    manifest_rows: list[dict[str, str]] = []

    base = args.cleaned / "runs/exp2_probe_classification/base/evaluated_outputs"
    for domain in ("synthetic", "pubmed", "reddit"):
        for key, (model, filename, prompt) in PROMPTED.items():
            run_id = f"prompted_{key}_{domain}"
            rel = Path("predictions") / "prompted" / f"{run_id}.csv"
            with rows_from(base / domain / filename) as reader:
                nrows = write_prediction(reader, repo / rel, "Prediction")
            manifest_rows.append({
                "run_id": run_id, "model": model, "train_domain": "Prompted",
                "test_domain": domain.capitalize(), "training_fraction": "0",
                "prompt_template": prompt, "source_path": rel.as_posix(),
                "zip_member": "", "prediction_column": "predicted_label", "rows": str(nrows),
            })

    with args.finetuned_manifest.open(encoding="utf-8-sig", newline="") as handle:
        specs = list(csv.DictReader(handle))
    for spec in specs:
        rel = Path("predictions") / "finetuned" / f"{spec['run_id']}.csv"
        with rows_from(Path(spec["source_path"]), spec["zip_member"]) as reader:
            nrows = write_prediction(reader, repo / rel, spec["prediction_column"])
        manifest_rows.append({
            "run_id": spec["run_id"], "model": spec["model"],
            "train_domain": spec["train_domain"], "test_domain": spec["test_domain"],
            "training_fraction": spec["training_fraction"], "prompt_template": "fine_tuned_inference",
            "source_path": rel.as_posix(), "zip_member": "",
            "prediction_column": "predicted_label", "rows": str(nrows),
        })

    manifest = repo / "manifests" / "run_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = list(manifest_rows[0])
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Wrote {len(manifest_rows)} runs to {manifest}")


if __name__ == "__main__":
    main()
