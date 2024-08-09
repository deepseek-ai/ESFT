# first: download adapter models and put them to the corresponding directories


python eval_multigpu.py \
    --eval_datasets=translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=all_models/adapters/token \
    --output_dir=results/completions/token \
    --max_new_tokens=512 \
    --openai_api_key=REPLACE_WITH_YOUR_KEY \
    --eval_batch_size=2 \
    --world_size=4 \
    --gpus_per_rank=2

# this script is used for single-gpu training and has been deprecated. If you have no multiple gpus, you can set above world_size=1 and gpus_per_rank=1

# python scripts/eval.py \
#     --eval_datasets=translation \
#     --base_model_path=deepseek-ai/ESFT-vanilla-lite \
#     --adapter_dir=all_models/adapters/token \
#     --output_dir=results/completions/token \
#     --max_new_tokens=512 \
#     --openai_api_key=REPLACE_WITH_YOUR_KEY \
#     --eval_batch_size=2
