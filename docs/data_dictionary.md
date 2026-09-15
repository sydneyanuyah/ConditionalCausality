# Data dictionary

Canonical probe tables use these fields:

| Field | Meaning |
|---|---|
| `Domain` | Synthetic, PubMed, or Reddit |
| `Paragraph_ID` | Stable paragraph identifier |
| `Source_Document_ID` | Grouping identifier used to keep source material together |
| `Tuple_ID` | Base conditional-tuple/family identifier |
| `ProbeID` | Unique individual probe identifier |
| `Probe_Number` | Integer P1–P5 |
| `Probe_Type` | Named transformation used to form the probe |
| `Paragraph` | Evidence text |
| `Probe_Question` | Handcrafted probe claim/question |
| `Gold_Label` | Supported, Not Supported, or Not Enough Evidence |

Prediction files add `predicted_label` and omit raw generations. The run manifest
maps every result row to its prediction file and records model, train domain, test
domain, training fraction, and prompt template.

For PubMed, `Source_Document_ID` is derived from the benchmark paragraph prefix
(for example `P12PR3` groups under `P12`). It is a stable internal grouping key, not
a claimed PMID. Synthetic and Reddit canonical files use the paragraph identifier
as a conservative grouping key; the domain split CSVs preserve any more specific
source identifiers recorded during split construction.

