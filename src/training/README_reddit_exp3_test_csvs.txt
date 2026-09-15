Reddit Exp3 Test CSVs

Created files:
1. reddit_exp3_1_extraction_test.csv
   Columns:
   task, Paragraph_ID, Document_ID, Paragraph, Gold Annotation, input, output

2. reddit_exp3_2_probe_test.csv
   Columns:
   task, Paragraph_ID, Tuple_ID, ProbeID, Probe_Number, Probe_Type, Paragraph, Probe_Question, Gold_Label, input, output

3. reddit_exp3_3_multitask_test.csv
   Columns:
   task, Paragraph_ID, Document_ID, Tuple_ID, ProbeID, Probe_Number, Probe_Type, Paragraph, Tuple, Probe_Question, Gold_Label, input, output

Summary:
                             file  rows  unique_paragraphs  unique_tuples                                                                                          notes
reddit_exp3_1_extraction_test.csv   695                695           1670                                 Reddit extraction test set. Uses Paragraph -> Gold Annotation.
     reddit_exp3_2_probe_test.csv  7883                681           1586             Reddit probe verification test set. Uses Paragraph + Probe_Question -> Gold_Label.
 reddit_exp3_3_multitask_test.csv  7883                681           1586 Reddit multitask test set. Joined probes to tuple annotations. Missing tuple joins dropped: 0.

Label mapping:
Probe 1 -> Not Supported
Probe 2 -> Not Enough Evidence
Probe 3 -> Not Enough Evidence
Probe 4 -> Supported
Probe 5 -> Not Supported

Note:
These are Reddit-only test files for evaluating the three fine-tuned Phi adapters:
- Exp3.1 extraction fine-tune
- Exp3.2 probe fine-tune
- Exp3.3 multitask fine-tune
