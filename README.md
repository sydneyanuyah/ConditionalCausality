# CondRelBench

### A mixed-domain benchmark for condition-scoped relation understanding

[![DOI](https://img.shields.io/badge/DOI-10.6084%2Fm9.figshare.32948507-1f6feb)](https://doi.org/10.6084/m9.figshare.32948507)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-2ea44f)](LICENSE_DATA.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab)](https://www.python.org/)

CondRelBench contains benchmark data for condition-scoped relation extraction and
probe classification across Synthetic, PubMed, and Reddit domains. The release
includes gold data, handcrafted P1–P5 probes, train/validation/test splits, saved
predictions, evaluation code, training and inference programs, manifests,
documentation, and tuple-family bootstrap results.

Dataset record: [https://doi.org/10.6084/m9.figshare.32948507](https://doi.org/10.6084/m9.figshare.32948507)

## Release summary

| Artifact | Count |
|---|---:|
| Domains | 3 |
| Canonical probe rows | 52,112 |
| Tuple families | 10,449 |
| Saved prediction rows | 401,906 |
| Reported probe-classification runs | 47 |
| Bootstrap repetitions per run | 10,000 |

The counts above are produced by the included release and split validators.

## Benchmark structure

Each condition-scoped tuple has five intended probe claims:

| Probe | Transformation | Gold label |
|---|---|---|
| P1 | Contradict the relation | Not Supported |
| P2 | Replace the condition | Not Enough Evidence |
| P3 | Remove the condition | Not Enough Evidence |
| P4 | Preserve the original condition-scoped relation | Supported |
| P5 | Contradict the condition | Not Supported |

`Tuple_ID` identifies the probe family, and `ProbeID` identifies an individual
claim. The label records whether the complete claim is licensed by the supplied
paragraph.

## Evaluation

The release reports:

- **Accuracy** and **Macro-F1** over the three released labels.
- **OGR**, the proportion of P2, P3, and P5 rows predicted as `Supported`.
- **SPS**, the proportion of complete tuple families for which all five probe
  predictions are correct.
- **95% confidence intervals** from a 10,000-repetition percentile bootstrap with
  seed `20260802`.

Bootstrap sampling is performed by `Tuple_ID`, keeping all probe rows from a family
together. The reference results are in
[`results/bootstrap_all_reported_runs.csv`](results/bootstrap_all_reported_runs.csv).

## Quick start

Create a Python 3.10+ environment and run:

```bash
python -m pip install -r environments/evaluation-requirements.txt
python src/validation/validate_release.py
python src/validation/validate_splits.py
python src/evaluation/bootstrap_tuple_families.py \
  --manifest manifests/run_manifest.csv \
  --output results/bootstrap_recomputed.csv \
  --metadata-output results/bootstrap_recomputed_metadata.json \
  --repetitions 10000 \
  --seed 20260802
```

No model training or inference is required to recalculate the released evaluation
metrics and confidence intervals from the saved prediction files.

## Repository contents

| Path | Contents |
|---|---|
| [`data/extraction/`](data/extraction/) | Gold condition-scoped extraction data |
| [`data/probes/`](data/probes/) | Canonical P1–P5 probe tables and gold labels |
| [`data/splits/`](data/splits/) | Released extraction and probe splits |
| [`predictions/`](predictions/) | Saved prompted and fine-tuned predictions |
| [`results/`](results/) | Estimates and bootstrap confidence intervals |
| [`manifests/`](manifests/) | Run mappings, model settings, and family metadata |
| [`src/evaluation/`](src/evaluation/) | Extraction, classification, OGR, SPS, and bootstrap evaluation |
| [`src/training/`](src/training/) | Training and inference programs |
| [`src/validation/`](src/validation/) | Release, checksum, and split validators |
| [`docs/`](docs/) | Data, methods, metrics, and protocol documentation |
| [`provenance/`](provenance/) | Source-review and historical project materials |

## Documentation

- [`docs/data_dictionary.md`](docs/data_dictionary.md)
- [`docs/probe_construction.md`](docs/probe_construction.md)
- [`docs/annotation_and_boundary_rules.md`](docs/annotation_and_boundary_rules.md)
- [`docs/split_protocol.md`](docs/split_protocol.md)
- [`docs/evaluation_metrics.md`](docs/evaluation_metrics.md)
- [`docs/model_and_training_settings.md`](docs/model_and_training_settings.md)
- [`docs/reproducibility_scope.md`](docs/reproducibility_scope.md)
- [`docs/paper_claim_to_artifact.md`](docs/paper_claim_to_artifact.md)

## Integrity

`CHECKSUMS.sha256` contains SHA-256 checksums for the released files. After an
intentional change, regenerate and validate them with:

```bash
python src/validation/build_checksums.py
python src/validation/validate_release.py
```

## Citation and data license

Please cite **CondRelBench: A Mixed-Domain Benchmark for Condition-Scoped Relation
Understanding** and the dataset DOI. See [`CITATION.md`](CITATION.md).

The dataset is distributed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
See [`LICENSE_DATA.md`](LICENSE_DATA.md).
