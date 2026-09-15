#!/usr/bin/env python3
"""Check that probe tuple families and extraction grouping IDs do not cross splits."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_ids(path: Path, candidates: tuple[str, ...]) -> set[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column = next((name for name in candidates if name in (reader.fieldnames or [])), None)
        if not column:
            raise SystemExit(f"ERROR: no grouping column in {path}")
        return {row[column].strip() for row in reader if row[column].strip()}


def check_group(files: list[Path], candidates: tuple[str, ...], label: str) -> None:
    seen: dict[str, str] = {}
    for path in files:
        split = next(name for name in ("train", "validation", "test") if name in path.stem.lower())
        for item in read_ids(path, candidates):
            prior = seen.setdefault(item, split)
            if prior != split:
                raise SystemExit(f"ERROR: {label} {item} occurs in {prior} and {split}")
    print(f"PASS: {label}: {len(seen):,} grouping IDs are split-disjoint")


def main() -> None:
    probe_root = ROOT / "data/splits/probes"
    for domain in ("synthetic", "pubmed", "reddit"):
        files = sorted((probe_root / domain).glob("*probe_*.csv"))
        files = [p for p in files if any(s in p.stem for s in ("train", "validation", "test"))]
        check_group(files, ("Tuple_ID", "Tuple ID"), f"{domain} probe Tuple_ID")
        check_group(files, ("Paragraph_ID", "Paragraph ID", "ID"), f"{domain} probe Paragraph_ID")

    extraction_root = ROOT / "data/splits/extraction/domain_exp3_1_extraction_splits"
    for domain in ("synthetic", "pubmed", "reddit"):
        files = sorted((extraction_root / domain).glob("*.csv"))
        candidates = ("Document_ID", "Paragraph_ID", "ID") if domain == "reddit" else ("Paragraph_ID", "ID")
        check_group(files, candidates, f"{domain} extraction grouping")


if __name__ == "__main__":
    main()

