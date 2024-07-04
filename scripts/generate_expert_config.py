import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--eval_datasets", type=str, required=True)
parser.add_argument("--expert_scores_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--score_function", type=str, required=True)
parser.add_argument("--top_p", type=float, required=True)
parser.add_argument("--train_shared_experts", action="store_true")
parser.add_argument("--train_non_expert_modules", action="store_true")

args = parser.parse_args()

eval_datasets = args.eval_datasets.split(",")
expert_scores_dir = args.expert_scores_dir
output_dir = args.output_dir
score_function = args.score_function
top_p = args.top_p
train_shared_experts = args.train_shared_experts
train_non_expert_modules = args.train_non_expert_modules

for dataset_name in eval_datasets:
    summary_file = f"{expert_scores_dir}/{dataset_name}/summary.json"
    expert_cfg = {"experts": {}, "shared_experts": train_shared_experts, "non_expert_modules": train_non_expert_modules}

    with open(summary_file) as f:
        data = json.load(f)
        assert score_function in ["gate", "token"], f"Unknown score function: {score_function}"
        scores = data[f"{score_function}_scores"]

        for layer, l_score in scores.items():
            l_score = [(int(k), v) for k,v in l_score.items()]
            l_score = sorted(l_score, key=lambda x: x[1], reverse=True)
            # get the top experts that make the threshold exceed top_p
            selected_experts = []
            current_score = 0
            for expert, score in l_score:
                if current_score >= top_p:
                    break
                selected_experts.append(expert)
                current_score += score
            expert_cfg["experts"][layer] = selected_experts
    
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{dataset_name}.json", "w") as f:
        json.dump(expert_cfg, f)

        
