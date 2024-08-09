import json
import argparse

from torch import device
from benchmarks import *
import os
from esft import load_base_model, add_adapter
import torch.multiprocessing as mp
from itertools import accumulate
from accelerate import dispatch_model
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def eval_model(rank, args, model, dataset):
    config = {
        "max_new_tokens": args.max_new_tokens,
        "eval_batch_size": args.eval_batch_size,
        "openai_api_key": args.openai_api_key
    }
    evaluator_map = {
        "intent": IntentEvaluator,
        "summary": SummaryEvaluator,
        "law": LawEvaluator,
        "translation": TranslationEvaluator
    }
    try:
        evaluator_cls = evaluator_map[args.eval_dataset]
        print(f"Rank {rank} starting evaluation...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
        visible_devices = list(range(rank * args.gpus_per_rank, (rank + 1) * args.gpus_per_rank))
        device_map = infer_auto_device_map(model, [14, 13], visible_devices)
        model = dispatch_model(model, device_map)
        cur_dataset = dataset[rank::args.world_size]
        evaluator = evaluator_cls(cur_dataset, config)
        with torch.no_grad():
            results, metrics = evaluator.evaluate(model, tokenizer)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path + f".rank_{rank}", "w") as f:
            for res, m in zip(results, metrics):
                obj = {
                    "example": res,
                    "score": m
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Error in process {rank}: {e}", flush=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with adapters on a specified dataset.")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Name of the evaluation dataset")
    parser.add_argument("--base_model_path", type=str,  required=True, help="Path to the base model")
    parser.add_argument("--adapter_dir", type=str, required=True, help="Directory containing the adapter")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens")
    parser.add_argument("--openai_api_key", type=str, required=True, help="API key for OpenAI")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--world_size", type=int, default=4, help="Number of processes to use for evaluation")
    parser.add_argument("--gpus_per_rank", type=int, default=2, help="Number of GPUs per process")

    args = parser.parse_args()



    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16) # not using tokenizer here to aviod deadlock

    print(f"Running evaluation on {args.eval_dataset}...")
    dataset = [json.loads(i) for i in open(f"datasets/eval/{args.eval_dataset}.jsonl").readlines()]

    print("Adding adapter...")
    model = add_adapter(model, args.adapter_dir, return_original_states=False)

    print("Start Evaluating...")
    mp.spawn(eval_model, args=(args, model, dataset), nprocs=args.world_size, join=True)
