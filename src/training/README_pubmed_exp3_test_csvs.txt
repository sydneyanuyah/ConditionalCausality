PubMed Exp3 Test CSVs

Created files:
1. pubmed_exp3_1_extraction_test.csv
   Columns:
   task, Paragraph_ID, Paragraph, Gold Annotation, input, output

2. pubmed_exp3_2_probe_test.csv
   Columns:
   task, Paragraph_ID, Tuple_ID, ProbeID, Probe_Number, Probe_Type, Paragraph, Probe_Question, Gold_Label, input, output

3. pubmed_exp3_3_multitask_test.csv
   Columns:
   task, Paragraph_ID, Tuple_ID, ProbeID, Probe_Number, Probe_Type, Paragraph, Tuple, Probe_Question, Gold_Label, input, output

Summary:
                             file  rows  unique_paragraphs  unique_tuples                                                                                          notes
pubmed_exp3_1_extraction_test.csv   681                681           1401                                 PubMed extraction test set. Uses Paragraph -> Gold Annotation.
     pubmed_exp3_2_probe_test.csv  6961                678           1397             PubMed probe verification test set. Uses Paragraph + Probe_Question -> Gold_Label.
 pubmed_exp3_3_multitask_test.csv  6961                678           1397 PubMed multitask test set. Joined probes to tuple annotations. Missing tuple joins dropped: 0.

Label mapping:
Probe 1 -> Not Supported
Probe 2 -> Not Enough Evidence
Probe 3 -> Not Enough Evidence
Probe 4 -> Supported
Probe 5 -> Not Supported

Note:
These are PubMed-only test files for evaluating the three fine-tuned Phi adapters:
- Exp3.1 extraction fine-tune
- Exp3.2 probe fine-tune
- Exp3.3 multitask fine-tune
