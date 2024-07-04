python scripts/get_expert_scores.py \
    --eval_datasets=intent,summary,law,translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --output_dir=results/expert_scores \
    --n_sample_tokens=8192 # this sample size is a hyperparameter

python scripts/generate_expert_config.py \
    --eval_datasets=intent,summary,law,translation \
    --expert_scores_dir=results/expert_scores \
    --output_dir=results/expert_configs \
    --score_function=token \
    --top_p=0.2 # the scoring function and top_p are hyperparameters
    # --train_shared_experts
    # --train_non_expert_modules
