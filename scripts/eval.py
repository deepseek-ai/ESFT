import json
import argparse
from benchmarks import *
import os
from esft import load_base_model, add_adapter

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_dir", type=str, required=True)
parser.add_argument("--base_model_path", type=str, required=True)
parser.add_argument("--eval_datasets", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--max_new_tokens", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=16)
parser.add_argument("--openai_api_key", type=str, required=True)
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

base_model_path = args.base_model_path
adapter_dir = args.adapter_dir
eval_datasets = args.eval_datasets.split(",")

config = {"max_new_tokens": args.max_new_tokens, "eval_batch_size": args.eval_batch_size, "openai_api_key": args.openai_api_key}

evaluator_map={"intent": IntentEvaluator, "summary": SummaryEvaluator, "law": LawEvaluator, "translation": TranslationEvaluator}

print("Loading base model...")
model, tokenizer = load_base_model(base_model_path)

for dataset_name in eval_datasets:
    print(f"Running evaluation on {dataset_name}...")
    dataset = [json.loads(i) for i in open(f"datasets/eval/{dataset_name}.jsonl").readlines()]
    if args.debug:
        print("Debugging. Shortening the dataset length")
        dataset = dataset[:16]

    evaluator = evaluator_map[dataset_name](dataset, config)
    print("Adding adapter...")
    model.model, original_state_dict = add_adapter(model.model, os.path.join(adapter_dir, dataset_name), return_original_states=True) # add adapter to model and convert original states to buffer.
    results, metrics = evaluator.evaluate(model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, dataset_name + ".jsonl"), "w") as f:
        for res, m in zip(results, metrics):
            obj = {
                "example": res,
                "score": m
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    model.model.load_state_dict(original_state_dict) # convert to original model



