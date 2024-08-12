import argparse
import json
import yaml
import os
import random
import torch
import torch.distributed as dist
from types import MethodType
from torch.utils.data import TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, logging

from benchmarks import *
from utils import get_formatted_input_and_target, get_examples_from_buffer_pad, init_parallel_groups
from esft import to_esft
from deepseek.modeling_deepseek import DeepseekV2ForCausalLM


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_AVOID_RECORD_STREAMS"] = "1"
logging.set_verbosity_error()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--expert_config", type=str, required=True)
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--wandb_api_key", type=str, required=False)
    args = parser.parse_args()

    expert_config = json.load(open(args.expert_config))
    output_dir = args.output_dir
    base_model_path = args.base_model_path
    config = yaml.safe_load(open(args.train_config))
    os.makedirs(args.output_dir, exist_ok=True)

    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

    if args.wandb_api_key is not None:
        import wandb
        wandb.login(key=args.wandb_api_key)

    ep_size = config.get("ep_size", 1)
    world_size, local_rank, ep_group, edp_group = init_parallel_groups(ep_size)
    edp_size = world_size // ep_size

    # Prepare data
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    samples = [json.loads(i) for i in open(f"datasets/train/{args.train_dataset}.jsonl").readlines()]
    buffer = []
    for instance in samples:
        input_ids, target_ids = get_formatted_input_and_target(instance['messages'], tokenizer, -100)
        buffer.append((input_ids, target_ids))
    seq_length = config['seq_length']
    random_concat_ratio = config['random_concat_ratio']
    concated_examples = get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio)

    dataset = TensorDataset(concated_examples['input_ids'], concated_examples['labels'])
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.98), len(dataset) - int(len(dataset) * 0.98)])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=config['steps'],
        per_device_train_batch_size=config['per_device_batch_size'],
        per_device_eval_batch_size=config['per_device_batch_size'],
        warmup_steps=config['warmup_steps'],
        weight_decay=config['weight_decay'],
        logging_dir=f"{output_dir}/logs",
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_strategy="steps",
        eval_steps=config['eval_steps'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=True,
        lr_scheduler_type='constant',
        save_total_limit=5,
        learning_rate=config['learning_rate'],
        optim=config['optim'],
        adam_beta1=config['adam_beta1'],
        adam_beta2=config['adam_beta2'],
        disable_tqdm=False,
        gradient_checkpointing=config['gradient_checkpointing'],
        gradient_checkpointing_kwargs={"use_reentrant": False} if config['gradient_checkpointing'] else {}, # if set to True, backward will raise bug
    )

    def data_collator(data):
        input_ids = torch.stack([item[0] for item in data])
        labels = torch.stack([item[1] for item in data])
        return {"input_ids": input_ids, "labels": labels}


    model = DeepseekV2ForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                  ep_size=ep_size, attn_implementation="flash_attention_2")
    model._ddp_params_and_buffers_to_ignore = [n for n, _ in model.named_parameters() if ".expert" in n]    # we manage grad synchronization of expert parameters
    to_esft(model, expert_config)
    model.dummy = torch.nn.Parameter(torch.zeros(1, dtype=model.dtype))    # prevent DDP from having no trainable parameters
    model._keys_to_ignore_on_save = ["dummy"]
    expert_params = [p for n, p in model.named_parameters() if p.requires_grad and ".expert" in n]
    for layer in model.model.layers:
        if type(layer.mlp).__name__ != "DeepseekV2MoE":
            continue
        layer.mlp.ep_group = ep_group
    # Force all2all backward the same number of times
    if ep_size > 1 and not expert_config["non_expert_modules"]:
        min_layer_id = min(int(k) for k, v in expert_config["experts"].items() if v)
        mlp = model.model.layers[min_layer_id].mlp
        forward = mlp.forward
        def custom_forward(self, hidden_states: torch.Tensor):
            return forward(hidden_states.requires_grad_(torch.is_grad_enabled()))
        mlp.forward = MethodType(custom_forward, mlp)

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    accelerator = trainer.accelerator
    backward = accelerator.backward
    def custom_backward(self, loss, **kwargs):
        backward(loss, **kwargs)
        if not self.sync_gradients or edp_size == 1:
            return
        for p in expert_params:
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(g, op=dist.ReduceOp.AVG, group=edp_group)
            if p.grad is not g:
                p.grad = g
    accelerator.backward = MethodType(custom_backward, accelerator)

    # Training
    ckpt_path = f"{output_dir}/last_checkpoint_ep{local_rank}"
    if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 1: # has checkpoints already
        trainer.train(resume_from_checkpoint=ckpt_path)
    else:
        trainer.train()

    # Save the model and tokenizer
    if local_rank == 0:
        trainer.save_model(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)
    elif local_rank < ep_size:
        model.save_pretrained(ckpt_path)

    print("Training complete")

if __name__ == "__main__":
    main()
