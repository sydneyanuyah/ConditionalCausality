#!/usr/bin/env python3
"""Validate the public CondRelBench evaluation bundle using the standard library."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS = {"Supported", "Not Supported", "Not Enough Evidence"}


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def validate_manifest() -> tuple[int, int]:
    manifest = ROOT / "manifests/run_manifest.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        specs = list(csv.DictReader(handle))
    if len(specs) != 47:
        fail(f"expected 47 reported probe runs, found {len(specs)}")
    if len({row["run_id"] for row in specs}) != len(specs):
        fail("duplicate run_id in run manifest")

    total_rows = 0
    invalid_predictions = 0
    required = {"Paragraph_ID", "Tuple_ID", "ProbeID", "Probe_Number", "Probe_Type", "Gold_Label", "predicted_label"}
    for spec in specs:
        path = ROOT / spec["source_path"]
        if not path.is_file():
            fail(f"missing prediction file: {spec['source_path']}")
        families: dict[str, set[int]] = defaultdict(set)
        nrows = 0
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                fail(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                nrows += 1
                if row["Gold_Label"] not in LABELS:
                    fail(f"{path}: invalid gold label {row['Gold_Label']!r}")
                if row["predicted_label"] not in LABELS:
                    invalid_predictions += 1
                pnum = int(row["Probe_Number"])
                if pnum not in range(1, 6):
                    fail(f"{path}: invalid Probe_Number {pnum}")
                families[row["Tuple_ID"]].add(pnum)
        if nrows != int(spec["rows"]):
            fail(f"{path}: manifest says {spec['rows']} rows, found {nrows}")
        if not families:
            fail(f"{path}: empty prediction file")
        total_rows += nrows
    return total_rows, invalid_predictions


def validate_probes() -> tuple[int, int]:
    total = 0
    incomplete_total = 0
    required = {"Domain", "Paragraph_ID", "Source_Document_ID", "Tuple_ID", "ProbeID", "Probe_Number", "Probe_Type", "Paragraph", "Probe_Question", "Gold_Label"}
    for domain in ("synthetic", "pubmed", "reddit"):
        path = ROOT / "data" / "probes" / f"{domain}_probes.csv"
        families: dict[str, Counter[int]] = defaultdict(Counter)
        probe_ids: set[str] = set()
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                fail(f"{path}: missing columns {sorted(missing)}")
            for row in reader:
                total += 1
                if not row["Source_Document_ID"]:
                    fail(f"{path}: blank source grouping ID")
                if row["ProbeID"] in probe_ids:
                    fail(f"{path}: duplicate ProbeID {row['ProbeID']}")
                probe_ids.add(row["ProbeID"])
                families[row["Tuple_ID"]][int(row["Probe_Number"])] += 1
                if row["Gold_Label"] not in LABELS:
                    fail(f"{path}: invalid gold label")
        incomplete = [tid for tid, counts in families.items() if set(counts) != set(range(1, 6))]
        expected_incomplete = {"synthetic": 37, "pubmed": 23, "reddit": 47}[domain]
        if len(incomplete) != expected_incomplete:
            fail(f"{path}: expected {expected_incomplete} incomplete source families, found {len(incomplete)}")
        incomplete_total += len(incomplete)
    return total, incomplete_total


def validate_sizes() -> int:
    largest = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            largest = max(largest, path.stat().st_size)
            if path.stat().st_size >= 100 * 1024 * 1024:
                fail(f"file exceeds GitHub's 100 MiB limit: {path.relative_to(ROOT)}")
    return largest


def validate_checksums() -> int:
    checksum_file = ROOT / "CHECKSUMS.sha256"
    if not checksum_file.exists():
        fail("CHECKSUMS.sha256 is missing")
    count = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        path = ROOT / rel
        if not path.is_file():
            fail(f"checksummed file is missing: {rel}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            fail(f"checksum mismatch: {rel}")
        count += 1
    return count


def main() -> None:
    rows, invalid = validate_manifest()
    probes, incomplete = validate_probes()
    largest = validate_sizes()
    checksums = validate_checksums()
    print(f"PASS: 47 runs, {rows:,} prediction rows, {probes:,} canonical probe rows")
    print(f"PASS: {checksums} checksums; largest file {largest / 1024 / 1024:.1f} MiB")
    print(f"INFO: {invalid:,} blank/noncanonical predictions are retained as evaluation errors")
    print(f"INFO: {incomplete} recovered source families are incomplete and listed in the completeness manifest")


if __name__ == "__main__":
    main()
