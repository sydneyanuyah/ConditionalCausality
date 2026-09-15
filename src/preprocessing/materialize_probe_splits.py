#!/usr/bin/env python3
"""Materialize leakage-safe probe splits from deposited paragraph assignments."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.assignments.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    paragraph_column = next(name for name in ("Paragraph_ID", "Paragraph_id") if name in rows[0])
    split_column = next(name for name in ("split", "Split") if name in rows[0])
    assignment = {}
    for row in rows:
        value = row[split_column].strip().lower()
        assignment[row[paragraph_column].strip()] = "validation" if value == "val" else value

    with args.probes.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        probes = list(reader)
    missing = sorted({row["Paragraph_ID"] for row in probes}.difference(assignment))
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} probe paragraph IDs lack an assignment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in probes if assignment[row["Paragraph_ID"]] == split]
        path = args.output_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        print(f"{split}: {len(selected):,} rows -> {path}")


if __name__ == "__main__":
    main()
