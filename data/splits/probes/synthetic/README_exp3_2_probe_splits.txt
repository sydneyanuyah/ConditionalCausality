Exp3.2 Probe fine-tuning splits

These files use the exact Paragraph_ID split from Exp3.1 extraction fine-tuning.
Train = 60%, Test = 30%, Validation = 10% by unique Paragraph_ID.

Columns kept in train/test/validation CSVs:
- Paragraph_ID
- Tuple_ID
- ProbeID
- Probe_Number
- Probe_Type
- Paragraph
- Probe_Question
- Gold_Label

Dropped columns include raw tuple fields, claim text, validation status, and validation notes to keep the files light for SageMaker training.
