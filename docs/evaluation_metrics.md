# Evaluation metrics and matching thresholds

Probe Accuracy and Macro-F1 use the three released labels. OGR is the proportion of
P2, P3, and P5 rows incorrectly predicted as `Supported`. SPS is the proportion of
complete tuple families for which all five P1–P5 predictions are correct.

Confidence intervals use a 10,000-repetition percentile bootstrap with seed
`20260802`. Sampling is by `Tuple_ID`, never by individual probe, so every sampled
family keeps all of its probe rows together.

Extraction matching has two explicitly different protocols, not one undocumented
threshold:

- Synthetic strict evaluation: relation and condition match by exact normalized
  text or token/span overlap at threshold **0.50**.
- PubMed/Reddit recall-focused evaluation: each tuple field uses the maximum of
  token Jaccard, token F1, character similarity, and containment, at threshold
  **0.30**.

Both defaults are exposed as `--iou_threshold` in the corresponding evaluator. The
earlier audit phrase “present but inconsistent” meant these values differed between
protocols without being explained. They are now documented here and in code.

