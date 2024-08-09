import json
import os
import torch
import argparse
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_formatted_input_and_target
import torch.multiprocessing as mp
from itertools import accumulate
from accelerate import dispatch_model


def infer_auto_device_map(model, pp_splits, visible_devices):
    assert len(pp_splits) == len(visible_devices)
    device_map = {
        "model.embed_tokens": 0,
        "model.norm": len(pp_splits) - 1,
        "lm_head": len(pp_splits) - 1
    }
    assert len(model.model.layers) == sum(pp_splits)
    pp_splits = [0, *list(accumulate(pp_splits))]
    for idx, (start, end) in enumerate(zip(pp_splits[:-1], pp_splits[1:])):
        for i in range(start, end):
            device_map.update({f"model.layers.{i}": idx})
    for k, v in device_map.items():
        device_map[k] = visible_devices[v]
    return device_map


def eval_expert(rank, args, model, dataset):
    try:
        print(f"Rank {rank} starting expert evaluation...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        visible_devices = list(range(rank * args.gpus_per_rank, (rank + 1) * args.gpus_per_rank))
        device_map = infer_auto_device_map(model, [14, 13], visible_devices)
        model = dispatch_model(model, device_map)
        model.config.expert_log_dir = os.path.join(args.output_dir, f"rank_{rank}")
        n_sample_tokens = args.n_sample_tokens // args.world_size
        os.makedirs(os.path.join(args.output_dir, f"rank_{rank}"), exist_ok=True)
        done_tokens = 0
        cur_dataset = dataset[rank::args.world_size]
        for instance in cur_dataset:
            input_ids, target_ids = get_formatted_input_and_target(instance['messages'], tokenizer, -100)
            model(input_ids=torch.tensor(input_ids).unsqueeze(0), labels=torch.tensor(target_ids).unsqueeze(0))
            done_tokens += len(input_ids)
            if done_tokens >= n_sample_tokens:
                break


    except Exception as e:
        print(f"Error in process {rank}: {e}", flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with adapters on a specified dataset.")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Name of the evaluation dataset")
    parser.add_argument("--base_model_path", type=str,  required=True, help="Path to the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes to use for evaluation")
    parser.add_argument("--gpus_per_rank", type=int, default=2, help="Number of GPUs per process")
    parser.add_argument("--n_sample_tokens", type=int, required=True, help="Token to sample for expert evaluation")
    args = parser.parse_args()
    random.seed(5934875)


    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16) # not using tokenizer here to aviod deadlock
    model.config.log_expert_weights = True


    print(f"Running expert evaluation on {args.eval_dataset}...")
    dataset = [json.loads(i) for i in open(f"datasets/train/{args.eval_dataset}.jsonl").readlines()]
    random.shuffle(dataset)


    print("Start Evaluating...")
    mp.spawn(eval_expert, args=(args, model, dataset), nprocs=args.world_size, join=True)
