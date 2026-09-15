#!/usr/bin/env python3
"""Tuple-family percentile bootstrap for CondRelBench probe predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


LABELS = ("Supported", "Not Supported", "Not Enough Evidence")
LABEL_TO_INDEX = {label: i for i, label in enumerate(LABELS)}


def normalize_label(value: str) -> str:
    key = " ".join(str(value).strip().split()).casefold()
    mapping = {
        "supported": "Supported",
        "not supported": "Not Supported",
        "not enough evidence": "Not Enough Evidence",
    }
    return mapping.get(key, str(value).strip())


def open_prediction_csv(source_path: str, zip_member: str):
    if zip_member:
        archive = ZipFile(source_path)
        raw = archive.open(zip_member)
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        return archive, raw, text
    text = open(source_path, "r", encoding="utf-8-sig", newline="")
    return None, None, text


def load_family_arrays(spec: dict[str, str]):
    families: dict[str, list[tuple[int, int, int]]] = {}
    archive, raw, text = open_prediction_csv(spec["source_path"], spec["zip_member"])
    try:
        reader = csv.DictReader(text)
        prediction_column = spec["prediction_column"]
        required = {"Tuple_ID", "Probe_Number", "Gold_Label", prediction_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{spec['run_id']}: missing columns {sorted(missing)}")
        for row in reader:
            gold = normalize_label(row["Gold_Label"])
            predicted = normalize_label(row[prediction_column])
            if gold not in LABEL_TO_INDEX:
                raise ValueError(f"{spec['run_id']}: unknown gold label {gold!r}")
            # Invalid or blank predictions remain errors but get an extra index.
            pred_index = LABEL_TO_INDEX.get(predicted, 3)
            family_id = row["Tuple_ID"].strip()
            probe_number = int(row["Probe_Number"])
            families.setdefault(family_id, []).append(
                (probe_number, LABEL_TO_INDEX[gold], pred_index)
            )
    finally:
        text.close()
        if raw is not None:
            raw.close()
        if archive is not None:
            archive.close()

    n = len(families)
    # The fourth prediction column captures invalid/blank model outputs.
    confusion = np.zeros((n, 3, 4), dtype=np.int64)
    ogr_num = np.zeros(n, dtype=np.int64)
    ogr_den = np.zeros(n, dtype=np.int64)
    complete = np.zeros(n, dtype=np.int64)
    sps_correct = np.zeros(n, dtype=np.int64)

    for i, rows in enumerate(families.values()):
        by_probe: dict[int, list[bool]] = {}
        for probe_number, gold_index, pred_index in rows:
            confusion[i, gold_index, pred_index] += 1
            by_probe.setdefault(probe_number, []).append(gold_index == pred_index)
            if probe_number in (2, 3, 5):
                ogr_den[i] += 1
                ogr_num[i] += int(pred_index == LABEL_TO_INDEX["Supported"])
        if set(by_probe).issuperset({1, 2, 3, 4, 5}):
            complete[i] = 1
            # Matches the deposited SPS implementation: duplicate probe rows count
            # as correct when at least one prediction for that probe is correct.
            sps_correct[i] = int(all(any(by_probe[k]) for k in range(1, 6)))

    return confusion, ogr_num, ogr_den, complete, sps_correct


def metrics_from_aggregates(confusion, ogr_num, ogr_den, complete, sps_correct):
    # confusion shape is (..., 3, 4); only the first three prediction columns
    # can contribute true positives.
    total = confusion.sum(axis=(-2, -1))
    correct = np.diagonal(confusion[..., :3], axis1=-2, axis2=-1).sum(axis=-1)
    accuracy = np.divide(correct, total, out=np.zeros_like(correct, dtype=float), where=total != 0)

    f1s = []
    for k in range(3):
        tp = confusion[..., k, k]
        fp = confusion[..., :, k].sum(axis=-1) - tp
        fn = confusion[..., k, :].sum(axis=-1) - tp
        denom = 2 * tp + fp + fn
        f1s.append(np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0))
    macro_f1 = np.mean(np.stack(f1s, axis=-1), axis=-1)
    ogr = np.divide(ogr_num, ogr_den, out=np.zeros_like(ogr_num, dtype=float), where=ogr_den != 0)
    sps = np.divide(sps_correct, complete, out=np.full_like(sps_correct, np.nan, dtype=float), where=complete != 0)
    return {"accuracy": accuracy, "macro_f1": macro_f1, "ogr": ogr, "sps": sps}


def bootstrap_run(spec, repetitions: int, base_seed: int, batch_size: int):
    confusion, ogr_num, ogr_den, complete, sps_correct = load_family_arrays(spec)
    n_families = confusion.shape[0]
    point = metrics_from_aggregates(
        confusion.sum(axis=0), ogr_num.sum(), ogr_den.sum(), complete.sum(), sps_correct.sum()
    )

    seed_material = f"{base_seed}:{spec['run_id']}".encode("utf-8")
    run_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = np.random.default_rng(run_seed)
    draws = {name: np.empty(repetitions, dtype=float) for name in point}
    flat_confusion = confusion.reshape(n_families, -1)

    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        size = stop - start
        # Each row contains the multiplicity of every tuple family in one
        # bootstrap sample of n_families draws with replacement.
        weights = rng.multinomial(n_families, np.full(n_families, 1.0 / n_families), size=size)
        boot_confusion = (weights @ flat_confusion).reshape(size, 3, 4)
        boot_metrics = metrics_from_aggregates(
            boot_confusion,
            weights @ ogr_num,
            weights @ ogr_den,
            weights @ complete,
            weights @ sps_correct,
        )
        for name, values in boot_metrics.items():
            draws[name][start:stop] = values

    row = {
        "run_id": spec["run_id"],
        "model": spec["model"],
        "train_domain": spec["train_domain"],
        "test_domain": spec["test_domain"],
        "training_fraction": spec["training_fraction"],
        "n_tuple_families": n_families,
        "n_complete_families": int(complete.sum()),
        "bootstrap_repetitions": repetitions,
        "confidence_level": 0.95,
        "interval_method": "percentile",
        "base_seed": base_seed,
        "run_seed": run_seed,
        "source_path": spec["source_path"],
        "zip_member": spec["zip_member"],
    }
    for name, value in point.items():
        low, high = np.nanquantile(draws[name], [0.025, 0.975])
        row[name] = float(value)
        row[f"{name}_ci_low"] = float(low)
        row[f"{name}_ci_high"] = float(high)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        specs = list(csv.DictReader(handle))
    repository_root = manifest_path.parent.parent
    for spec in specs:
        source = Path(spec["source_path"])
        if not source.is_absolute():
            spec["source_path"] = str(repository_root / source)
    results = [bootstrap_run(spec, args.repetitions, args.seed, args.batch_size) for spec in specs]
    for row in results:
        row["source_path"] = str(Path(row["source_path"]).relative_to(repository_root))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    if args.metadata_output:
        metadata = {
            "description": "Tuple-family bootstrap over saved prompted and 100%-training predictions; no model inference or training was rerun.",
            "bootstrap_unit": "Tuple_ID (all probe rows retained together)",
            "confidence_interval": "2.5th and 97.5th percentiles of bootstrap estimates",
            "labels": LABELS,
            "metrics": ["accuracy", "macro_f1", "ogr", "sps"],
            "repetitions": args.repetitions,
            "base_seed": args.seed,
            "manifest": str(manifest_path.relative_to(repository_root)),
            "output": str(output.resolve().relative_to(repository_root)),
            "numpy_version": np.__version__,
        }
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
