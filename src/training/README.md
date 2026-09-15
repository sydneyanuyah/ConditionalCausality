# Phi Exp3 Training Package

Files:
- train_phi_exp3.py
- setup_phi_ft_env.sh
- run_exp3_1_extraction_ft.sh
- run_exp3_2_probe_ft.sh
- run_exp3_3_multitask_ft.sh

Expected folders in $HOME/SageMaker:
- exp3_1_extraction_splits/
- exp3_2_probe_splits/
- exp3_3_multitask_splits/

Run:

bash setup_phi_ft_env.sh
bash run_exp3_1_extraction_ft.sh
bash run_exp3_2_probe_ft.sh
bash run_exp3_3_multitask_ft.sh

Model saving:
The script saves the LoRA/QLoRA adapter, tokenizer files, training arguments, and checkpoints into the output_dir for each experiment.

It does NOT save a full merged Phi model by default. This is intentional because QLoRA fine-tuning normally saves the adapter only.

Output directories:
- outputs_exp3_1_phi_extraction_ft
- outputs_exp3_2_phi_probe_ft
- outputs_exp3_3_phi_multitask_ft
