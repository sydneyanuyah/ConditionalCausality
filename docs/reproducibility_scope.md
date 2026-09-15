# Reproducibility scope and known limitations

## Complete in this package

The package contains condition-scoped tuples, all handcrafted P1–P5 claims, gold
probe labels, tuple/family identifiers, source grouping identifiers, released split
files, prompts, recovered training/inference code, evaluation code, OGR and SPS,
tuple-family bootstrap code, 47 compact prediction files, machine-readable results,
and checksums.

## Historical information that cannot be reconstructed honestly

1. The original runs did not record immutable upstream model revision hashes or API
   provider build identifiers. The human-readable and Hugging Face model identifiers
   are preserved.
2. The original training machines did not retain a package lockfile. Recovered setup
   scripts express the dependencies used, but not exact versions for every package.
3. No separate row-level manual audit-decision log existed. The probes were
   handcrafted; final canonical probe CSVs are authoritative. Source review sheets
   are included for provenance, not represented as an exhaustive audit ledger.
4. PubMed benchmark IDs provide stable article grouping but are not explicit PMID
   mappings. This package does not invent PMID metadata.
5. Historical text encoding artifacts in Reddit data are retained rather than
   silently normalized.
6. One hundred seven tuple families in the recovered canonical source (37 Synthetic, 23
   PubMed, and 47 Reddit) contain fewer than all five probe rows. They are disclosed in
   `manifests/probe_family_completeness.csv`; SPS excludes incomplete families by
   definition, and no missing claim text or label was fabricated.

These limitations affect bit-for-bit historical training recreation, not metric
recalculation from the deposited predictions.
