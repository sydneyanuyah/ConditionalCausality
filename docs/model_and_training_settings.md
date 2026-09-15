# Models and recorded settings

Prompted models are identified as Kimi-K2, Llama-3.3-70B-Instruct,
`microsoft/Phi-4-mini-instruct`, and `Qwen/Qwen2.5-32B-Instruct`. Reported best
templates are two-shot plus chain-of-thought for Kimi, Llama, and Qwen, and zero-shot
for Phi. The base generation program records defaults of `max_new_tokens=128`,
`temperature=0`, `top_p=1`, and seed `4000`.

Encoder identifiers are:

- `bert-base-uncased`
- `allenai/scibert_scivocab_uncased`
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- `dmis-lab/biobert-base-cased-v1.1`

Recovered encoder settings are 3 epochs, train batch size 16, evaluation batch size
32, learning rate 2e-5, warmup ratio 0.03, weight decay 0.01, maximum length 512,
and training/data seed 4000.

Recovered Phi QLoRA settings are 3 epochs, batch size 4, gradient accumulation 4,
learning rate 2e-5, warmup ratio 0.03, bfloat16, and seed 4000. Recovered Qwen QLoRA
settings use batch size 1, gradient accumulation 16, maximum sequence length 2048,
and otherwise the same recorded epoch/learning-rate/warmup values.

See the shell and Python files in `src/training/` for command-level settings. Exact
historical library versions and upstream revision hashes were not logged and are
therefore not asserted.

