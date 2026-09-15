# Release checklist

## Completed

- [x] Full condition-scoped extraction data and exact released splits
- [x] Full recovered handcrafted P1–P5 probe rows and gold labels
- [x] Tuple-family, paragraph, and source grouping identifiers
- [x] Probe/extraction split assignments and leakage validator
- [x] Annotation, boundary, handcrafted-probe, metric, and split documentation
- [x] Prompt templates and demonstrations
- [x] Recovered training and inference scripts and recorded hyperparameters
- [x] Evaluation code, including OGR and SPS
- [x] Tuple-family bootstrap code and 10,000-resample intervals for all 47 runs
- [x] Compact saved prediction files and run manifest
- [x] Checksums and release validator
- [x] Git repository initialized on branch `main`

## Owner actions

- [x] Integrate the release package into the public GitHub repository.
- [ ] Choose and add a source-code license. The dataset license is documented as
      CC BY 4.0, but the archived project did not record a code-license choice.
- [ ] Add the final author list, venue, and year to citation metadata.
- [ ] Update the Figshare record and manuscript repository link after GitHub has a
      stable public URL.
- [ ] Confirm that redistribution of PubMed- and Reddit-derived text complies with
      the source terms; CC BY on the benchmark record cannot override third-party
      rights. The repository may be published privately while this is checked.

## Disclosed historical limits (not fixable by inventing files)

- Exact original package lockfiles and immutable upstream model/API revision hashes
  were not recorded.
- No separate row-level probe-audit decision ledger existed; the final handcrafted
  probe tables are authoritative.
- 107 recovered tuple families lack one or more source probe rows and are listed in
  `manifests/probe_family_completeness.csv`; SPS excludes incomplete families.
- 2,611 blank/noncanonical saved predictions are retained as evaluation errors.
