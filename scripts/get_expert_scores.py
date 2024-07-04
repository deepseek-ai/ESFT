import json
from benchmarks import *
import os
import torch
from torch import nn
import argparse
from random import shuffle
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_formatted_input_and_target

# constants for deepseek-v2-lite
TOP_K=6
N_EXPERTS=64

parser = argparse.ArgumentParser()
parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--eval_datasets", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--n_sample_tokens", type=int, required=True)
args = parser.parse_args()

eval_datasets = args.eval_datasets.split(",")
output_dir = args.output_dir
base_model_path = args.base_model_path
n_sample_tokens = args.n_sample_tokens

model, tokenizer = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"), AutoTokenizer.from_pretrained(base_model_path)
model.config.log_expert_weights = True

for dataset_name in eval_datasets:
    dataset = [json.loads(i) for i in open(f"datasets/train/{dataset_name}.jsonl").readlines()]
    shuffle(dataset)
    model.config.expert_log_dir = os.path.join(args.output_dir, dataset_name)
    # make dir -p this
    os.makedirs(os.path.join(args.output_dir, dataset_name), exist_ok=True)
    done_tokens = 0
    for instance in dataset:
        input_ids, target_ids = get_formatted_input_and_target(instance['messages'], tokenizer, -100)
        model(input_ids=torch.tensor(input_ids).unsqueeze(0), labels=torch.tensor(target_ids).unsqueeze(0))
        done_tokens += len(input_ids)
        if done_tokens >= n_sample_tokens:
            break

    # open all files under os.path.join(args.output_dir, dataset_name). For each file, generate a summary of it
    # and write it to a file in the same directory
    files = os.listdir(os.path.join(args.output_dir, dataset_name))
    summary_file = os.path.join(args.output_dir, dataset_name, "summary.json")
    token_scores = {}
    gate_scores = {}

    for file in files:
        if not file.endswith(".txt"):
            continue
        layer_idx = file.split("_")[2].split(".")[0]
        token_scores[layer_idx] = {expert:0 for expert in range(N_EXPERTS)}
        gate_scores[layer_idx] = {expert:0 for expert in range(N_EXPERTS)}

        with open(os.path.join(args.output_dir, dataset_name, file)) as f:
            data = f.readlines()
            for line in data:
                expert_ids, expert_weights = line.split("\t\t")
                expert_ids = [int(i) for i in expert_ids.split("\t")]
                expert_weights = [float(i) for i in expert_weights.split("\t")]
                for expert_id, expert_weight in zip(expert_ids, expert_weights):
                    gate_scores[layer_idx][expert_id] += expert_weight
                    token_scores[layer_idx][expert_id] += 1. / TOP_K
            total = sum(token_scores[layer_idx].values())
            gate_scores[layer_idx] = {expert: round(gate_scores[layer_idx][expert] / total, 4) for expert in gate_scores[layer_idx]}
            token_scores[layer_idx] = {expert: round(token_scores[layer_idx][expert] / total, 4) for expert in token_scores[layer_idx]}


    with open(summary_file, "w") as f:
        f.write(json.dumps({"token_scores": token_scores, "gate_scores": gate_scores}))


