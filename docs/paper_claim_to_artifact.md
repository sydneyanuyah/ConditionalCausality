# Paper claim to artifact map

| Reproducibility item | Artifact |
|---|---|
| Condition-scoped tuples | `data/extraction/` and tuple-linked probe tables |
| P1–P5 claims and gold labels | `data/probes/` |
| Tuple/family and source grouping IDs | canonical probe columns |
| Source and split metadata/logic | `data/splits/`, `docs/split_protocol.md`, and `src/preprocessing/materialize_probe_splits.py` |
| Annotation and boundary rules | `docs/annotation_and_boundary_rules.md` |
| Handcrafted probe documentation | `docs/probe_construction.md` |
| Prompt templates/demonstrations | `src/evaluation/exp2_probe_classification/prompts_dictionary.py`, `src/training/prompts_dictionary.py` |
| Training/inference code and settings | `src/training/`, `docs/model_and_training_settings.md` |
| Evaluation, OGR, SPS | `src/evaluation/` |
| Bootstrap | `src/evaluation/bootstrap_tuple_families.py` |
| Saved predictions | `predictions/` and `manifests/run_manifest.csv` |
| Machine-readable estimates and CIs | `results/` |
| Checksums | `CHECKSUMS.sha256` |
| Known historical limits | `docs/reproducibility_scope.md` |
