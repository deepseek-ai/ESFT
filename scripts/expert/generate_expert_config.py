import argparse
import json
import os
from multiprocessing import Pool
import numpy as np


def parse_line(line):
    expert_ids, expert_weights = line.split("\t\t")
    expert_ids = [int(i) for i in expert_ids.split("\t")]
    expert_weights = [float(i) for i in expert_weights.split("\t")]
    return expert_ids, expert_weights


def get_summary(files):
    TOP_K=6
    N_EXPERTS=64
    N_LAYERS=26 # 27 layers totally, the first layer is not MoE

    gate_scores = np.zeros((N_LAYERS, N_EXPERTS))
    token_scores = np.zeros((N_LAYERS, N_EXPERTS))

    print("loading files")
    for rank, file in files:
        layer_id = int(file.split(".")[0].split("_")[2]) - 1

        with open(os.path.join(args.expert_scores_dir, rank, file)) as f:
            data = f.readlines()
            for line in data:
                expert_ids, expert_weights = parse_line(line)
                np.add.at(gate_scores[layer_id], expert_ids, expert_weights)
                np.add.at(token_scores[layer_id], expert_ids, np.ones_like(expert_weights) / TOP_K)

    gate_scores = gate_scores / np.sum(gate_scores, axis=0)
    token_scores = token_scores / np.sum(token_scores, axis=0)

    summary = {"token_scores": token_scores, "gate_scores": gate_scores}
    summary = {k: {str(i+1): {str(j): round(v, 4) for j, v in enumerate(l)} for i, l in enumerate(v)} for k, v in summary.items()}

    return summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dataset", type=str, required=True)
    parser.add_argument("--expert_scores_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_function", type=str, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--train_shared_experts", action="store_true")
    parser.add_argument("--train_non_expert_modules", action="store_true")

    args = parser.parse_args()

    expert_cfg = { # initialize expert config
        "experts": {},
        "shared_experts": args.train_shared_experts,
        "non_expert_modules": args.train_non_expert_modules
    }

    # let's walk inside args.expert_scores_dir and get abs file names
    file_names = []
    for rank in [i for i in os.listdir(args.expert_scores_dir) if 'rank' in i]:
        for file in os.listdir(os.path.join(args.expert_scores_dir, rank)):
            file_names.append([rank, file])


    summary_file = os.path.join(args.expert_scores_dir, "summary.json")
    summary = get_summary(file_names)

    with open(summary_file, "w") as f:
        f.write(json.dumps(summary))


    scores = summary[f"{args.score_function}_scores"]
    for layer, l_score in scores.items():
        l_score = [(int(k), v) for k,v in l_score.items()]
        l_score = sorted(l_score, key=lambda x: x[1], reverse=True)
        selected_experts = []
        current_score = 0
        for expert, score in l_score:
            if current_score >= args.top_p:
                break
            selected_experts.append(expert)
            current_score += score
        expert_cfg["experts"][layer] = selected_experts

    top_p = args.top_p
    train_shared_experts = args.train_shared_experts
    train_non_expert_modules = args.train_non_expert_modules



    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(expert_cfg, f)
