
# Expert-Specialized Fine-Tuning


Official Repo for paper [Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models](https://arxiv.org/abs/2407.01906) by 
[Zihan Wang](https://zihanwang314.github.io), [Deli Chen](https://victorchen96.github.io/chendeli.io/), [Damai Dai](https://scholar.google.com.hk/citations?user=8b-ysf0NWVoC&hl=zh-CN), [Runxin Xu](https://runxinxu.github.io/aboutme/), 
[Zhuoshu Li](http://www.idi.zju.edu.cn/member/3053.html) and
Y. Wu. 

**ESFT** aims to efficiently customize Large Language Models (LLMs) with Mixture-of-Experts (MoE) architecture by adjusting only task-relevant parts, improving efficiency and performance while using fewer resources and storage. 
 



## 🚀 Quick Start 
### Installation and Setup
```bash
git clone https://github.com/deepseek-ai/ESFT.git
cd esft
```

### Install dependencies
```bash
pip install transformers torch safetensors
```

### Download necessary adapters
```bash
bash scripts/download_adapters.sh
```



## 🔧Key Scripts
1. **eval.py**
This script evaluates the performance of the model on various datasets. **Usage:**
```bash
python scripts/eval.py \
    --eval_datasets=translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --adapter_dir=all_models/adapters/token \
    --output_dir=results/completions/token \
    --max_new_tokens=512 \
    --openai_api_key=REPLACE_WITH_YOUR_KEY \
    --eval_batch_size=2
```

2. **get_expert_scores.py**
This script calculates the scores for each expert based on the evaluation datasets.
**Usage:**
```bash
python scripts/get_expert_scores.py \
    --eval_datasets=intent,summary,law,translation \
    --base_model_path=deepseek-ai/ESFT-vanilla-lite \
    --output_dir=results/expert_scores \
    --n_sample_tokens=8192 # the sample size hyperparameter
```

3. **generate_expert_config.py**
This script generates the configuration to convert a MoE model with only task-relevant tasks trained based on evaluation scores.
**Usage:**
```bash
python scripts/generate_expert_config.py \
    --eval_datasets=intent,summary,law,translation \
    --expert_scores_dir=results/expert_scores \
    --output_dir=results/expert_configs \
    --score_function=token \
    --top_p=0.2 # the scoring function and top_p are hyperparameters
```


## Contact and Support
For bug reports, feature requests, and general inquiries, please open an issue on our GitHub Issues page. Make sure to include as much detail as possible to help us address your issue quickly.

## 🌟Todo list
- ☑️  📝 Update models, evaluation scripts, and expert selection scripts
- 🔲 🔧 Update training scripts
- 🔲 🚀 More...


## 📚Citation
If you find our code or paper useful, please cite:
```bash
@article{wang2024letexpertsticklast,
      title={Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models}, 
      author={Zihan Wang and Deli Chen and Damai Dai and Runxin Xu and Zhuoshu Li and Y. Wu},
      year={2024},
      eprint={2407.01906},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01906}, 
}
```
