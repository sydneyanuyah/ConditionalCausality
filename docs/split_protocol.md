# Split protocol

Train, validation, and test partitions use 60%, 10%, and 30% of unique paragraph
groups, respectively. The deposited assignment CSVs are the authoritative allocation
and remove any dependence on library-specific random shuffling.

All rows sharing a `Paragraph_ID` stay together. Since every `Tuple_ID` belongs to a
single paragraph, this also keeps every P1–P5 tuple family together and prevents
tuple-family leakage. Reddit extraction additionally groups by `Document_ID` where
that identifier is available. `src/validation/validate_splits.py` checks these
invariants across all released partitions.

`src/preprocessing/materialize_probe_splits.py` rebuilds probe partitions directly
from a canonical probe table and the deposited assignments. Example:

```bash
python src/preprocessing/materialize_probe_splits.py \
  --probes data/probes/pubmed_probes.csv \
  --assignments data/splits/probes/pubmed/split_assignments.csv \
  --output-dir /tmp/pubmed_splits
```

Synthetic assignments are in
`data/splits/probes/synthetic/exp3_2_probe_split_assignments.csv`; PubMed and Reddit
use `split_assignments.csv` in their domain directories.

