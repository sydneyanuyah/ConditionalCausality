#!/usr/bin/env python3
"""
eval_exp1b.py
Recall-focused Exp1b evaluation across SyntheticGold, PubMed, and Reddit.

Expected folder layout:
  inference/
    flan_t5/*.csv
    spanbert/*.csv

The script automatically maps prediction CSVs to datasets by filename:
  synthetic: filenames containing synthetic or syntheticgold
  pubmed:    filenames containing pubmed
  reddit:    filenames containing reddit
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from eval_common import DATASET_SPECS, evaluate_recall_rows


DATASET_KEYS = ["synthetic", "pubmed", "reddit"]


def infer_dataset_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if "pubmed" in name:
        return "pubmed"
    if "reddit" in name:
        return "reddit"
    if "synthetic" in name or "syntheticgold" in name:
        return "synthetic"
    return None


def collect_model_prediction_files(model_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for csv_path in sorted(model_dir.glob("*.csv")):
        dataset = infer_dataset_from_name(csv_path)
        if dataset and dataset not in mapping:
            mapping[dataset] = csv_path
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1b recall-focused evaluation across synthetic, PubMed, and Reddit.")
    parser.add_argument("--pred_root", default="inference", help="Root folder with model subfolders. Default: inference")
    parser.add_argument("--models", nargs="*", default=["flan_t5", "spanbert"], help="Model subfolders under pred_root. Default: flan_t5 spanbert")
    parser.add_argument("--synthetic_gold", default="syntheticgold_test.csv")
    parser.add_argument("--pubmed_gold", default="PubMed-Exp1.csv")
    parser.add_argument("--reddit_gold", default="Reddit-Exp1.csv")
    parser.add_argument("--out_dir", default="evaluation/exp1b", help="Output directory.")
    parser.add_argument("--pred_id_col", default="id", help="Prediction ID column. Default: id")
    parser.add_argument("--pred_output_col", default="output", help="Prediction output column. Default: output")
    parser.add_argument("--iou_threshold", type=float, default=0.30, help="Recall-focused relaxed match threshold. Default: 0.30")
    parser.add_argument("--split_fields", default="e1,e2,relation,condition", help="Comma-separated tuple fields to split on and/or. Default: all fields")
    args = parser.parse_args()

    split_fields = {x.strip() for x in args.split_fields.split(",") if x.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gold_paths = {
        "synthetic": Path(args.synthetic_gold),
        "pubmed": Path(args.pubmed_gold),
        "reddit": Path(args.reddit_gold),
    }
    gold_dfs = {k: pd.read_csv(v, low_memory=False) for k, v in gold_paths.items()}

    all_summary_rows: List[pd.DataFrame] = []
    all_detail_paths: List[str] = []
    pred_root = Path(args.pred_root)

    for model in args.models:
        model_dir = pred_root / model
        if not model_dir.exists():
            print(f"WARNING: missing model folder: {model_dir}")
            continue

        pred_map = collect_model_prediction_files(model_dir)
        for dataset in DATASET_KEYS:
            pred_path = pred_map.get(dataset)
            if pred_path is None:
                print(f"WARNING: no {dataset} prediction CSV found in {model_dir}")
                continue

            spec = DATASET_SPECS[dataset]
            detail_df, summary_df = evaluate_recall_rows(
                gold_df=gold_dfs[dataset],
                pred_df=pd.read_csv(pred_path, low_memory=False),
                spec=spec,
                pred_id_col=args.pred_id_col,
                pred_output_col=args.pred_output_col,
                threshold=args.iou_threshold,
                split_fields=split_fields,
            )
            if summary_df.empty:
                print(f"WARNING: empty summary for {model} / {dataset}")
                continue

            summary_df.insert(0, "Model", model)
            summary_df.insert(1, "Prediction_File", str(pred_path))
            all_summary_rows.append(summary_df)

            details_path = out_dir / model / f"exp1b_details_{dataset}.csv"
            summary_path = out_dir / model / f"exp1b_summary_{dataset}.csv"
            details_path.parent.mkdir(parents=True, exist_ok=True)
            detail_df.to_csv(details_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            all_detail_paths.append(str(details_path))
            print(f"Saved: {summary_path}")
            print(f"Saved: {details_path}")

    if not all_summary_rows:
        raise SystemExit("No Exp1b summaries were produced. Check --pred_root and prediction filenames.")

    combined = pd.concat(all_summary_rows, ignore_index=True)
    combined_path = out_dir / "exp1b_all_model_dataset_summaries.csv"
    combined.to_csv(combined_path, index=False)

    # Model-level macro average across available datasets. Recall columns are emphasized first.
    numeric_cols = [
        "R_tuple", "F1_tuple", "R_E1", "R_R", "R_E2", "R_C", "R_RE",
        "Delta_condition", "R_RE_given_C", "R_C_given_RE", "PGC", "P_tuple",
        "total_gold_tuples", "total_pred_tuples", "rows", "parse_errors",
    ]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    macro_cols = [c for c in numeric_cols if c in combined.columns and c not in {"total_gold_tuples", "total_pred_tuples", "rows", "parse_errors"}]
    model_macro = combined.groupby("Model", as_index=False)[macro_cols].mean()
    count_cols = combined.groupby("Model", as_index=False)[["total_gold_tuples", "total_pred_tuples", "rows", "parse_errors"]].sum()
    model_macro = model_macro.merge(count_cols, on="Model", how="left")
    model_macro_path = out_dir / "exp1b_model_macro_summary.csv"
    model_macro.to_csv(model_macro_path, index=False)

    print(f"Saved combined dataset summary: {combined_path}")
    print(f"Saved model macro summary: {model_macro_path}")


if __name__ == "__main__":
    main()
