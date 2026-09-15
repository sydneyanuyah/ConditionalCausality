#!/usr/bin/env python3
"""
eval_exp1a.py
Strict Exp1a evaluation for SyntheticGold only.

This script compares one or more prediction CSVs against the synthetic gold CSV.
It writes row-level evaluation files and summary metric files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from eval_common import DATASET_SPECS, evaluate_strict_rows


def collect_prediction_files(pred_files: List[str] | None, pred_dir: str | None) -> List[Path]:
    files: List[Path] = []
    if pred_files:
        files.extend(Path(p) for p in pred_files)
    if pred_dir:
        files.extend(sorted(Path(pred_dir).glob("*.csv")))
    seen = set()
    unique = []
    for p in files:
        rp = str(p)
        if rp not in seen:
            unique.append(p)
            seen.add(rp)
    if not unique:
        raise ValueError("No prediction files were provided. Use --pred_files and/or --pred_dir.")
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp1a strict evaluation for synthetic gold only.")
    parser.add_argument("--gold", default="syntheticgold_test.csv", help="Synthetic gold CSV. Default: syntheticgold_test.csv")
    parser.add_argument("--pred_files", nargs="*", default=None, help="One or more prediction CSVs to evaluate.")
    parser.add_argument("--pred_dir", default=None, help="Optional folder containing prediction CSVs.")
    parser.add_argument("--out_dir", default="evaluation/exp1a", help="Output directory.")
    parser.add_argument("--pred_id_col", default="id", help="Prediction ID column. Default: id")
    parser.add_argument("--pred_output_col", default="output", help="Prediction output column. Default: output")
    parser.add_argument("--iou_threshold", type=float, default=0.50, help="Strict relation/condition match threshold. Default: 0.50")
    parser.add_argument("--split_fields", default="e1,e2", help="Comma-separated tuple fields to split on and/or. Default: e1,e2")
    args = parser.parse_args()

    split_fields = {x.strip() for x in args.split_fields.split(",") if x.strip()}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = DATASET_SPECS["synthetic"]
    gold_df = pd.read_csv(args.gold, low_memory=False)

    all_summaries = []
    for pred_path in collect_prediction_files(args.pred_files, args.pred_dir):
        pred_df = pd.read_csv(pred_path, low_memory=False)
        detail_df, summary_df = evaluate_strict_rows(
            gold_df=gold_df,
            pred_df=pred_df,
            spec=spec,
            pred_id_col=args.pred_id_col,
            pred_output_col=args.pred_output_col,
            threshold=args.iou_threshold,
            split_fields=split_fields,
        )
        stem = pred_path.stem
        detail_path = out_dir / f"exp1a_details_{stem}.csv"
        summary_path = out_dir / f"exp1a_summary_{stem}.csv"
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        summary_wide = {"prediction_file": str(pred_path)}
        for _, row in summary_df.iterrows():
            metric = row["metric"]
            for col in ["precision", "recall", "f1"]:
                summary_wide[f"{metric}_{col}"] = row[col]
        all_summaries.append(summary_wide)

        print(f"Saved: {detail_path}")
        print(f"Saved: {summary_path}")

    combined_path = out_dir / "exp1a_all_summaries.csv"
    pd.DataFrame(all_summaries).to_csv(combined_path, index=False)
    print(f"Saved combined summary: {combined_path}")


if __name__ == "__main__":
    main()
