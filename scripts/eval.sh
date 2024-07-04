# first, download adapter models and put them to the corresponding directories


python scripts/eval.py \
    --eval_datasets=translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=all_models/adapters/token \
    --output_dir=results/completions/token \
    --max_new_tokens=512 \
    --openai_api_key=REPLACE_WITH_YOUR_KEY \
    --eval_batch_size=2
