
export TOKENIZERS_PARALLELISM=false

exp_name="test/eval_translation"
base_model_path="deepseek-ai/esft-vanilla-lite"
torchrun --nproc-per-node=8 train_ep.py \
    --base_model_path=${base_model_path} \
    --expert_config=results/expert_configs/translation.json \
    --train_dataset=translation \
    --train_config=configs/base.yaml \
    --output_dir=results/checkpoints/${exp_name}