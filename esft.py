import os
import json
import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

def to_buffer(module, mark_param=True):
    """Turns all parameters of a module into buffers."""
    if module is None:
        return
    modules = module.modules()
    module = next(modules)
    delattrs = []
    for name, param in module.named_parameters(recurse=False):
        delattrs.append([module, name, param])
    if mark_param and delattrs:
        old_param_list = getattr(module, 'param_list', [])
        module.param_list = old_param_list + [name for _, name, _ in delattrs]
    for module, name, _ in delattrs:
        delattr(module, name)  # Unregister parameter
    for module, name, param in delattrs:
        module.register_buffer(name, param.data, persistent=False)
    for module in modules:
        to_buffer(module, mark_param=mark_param)


def to_param(module):
    """Turns all buffers of a module into parameters."""
    if module is None:
        return
    modules = module.modules()
    module = next(modules)
    param_list = getattr(module, 'param_list', [])
    for name in param_list:
        buffer = getattr(module, name)
        delattr(module, name)  # Delete buffer
        setattr(module, name, nn.Parameter(buffer))
    for module in modules:
        to_param(module)


def recursive_getattr(model, module_name):
    split_list = module_name.split('.')
    output = model
    for name in split_list:
        output = getattr(output, name)
    return output


def recursive_setattr(model, module_name, module):
    split_list = module_name.split('.')
    output = model
    for name in split_list[:-1]:
        output = getattr(output, name)
    output.__setattr__(split_list[-1], module)


def to_esft(model, adapter_config):
    if not adapter_config.get('non_expert_modules', False):
        to_buffer(model)
    else:
        to_param(model)
    for idx, layer in enumerate(model.model.layers):
        if type(layer.mlp).__name__ != "DeepseekV2MoE":
            continue
        if adapter_config.get('shared_experts', False):
            to_param(layer.mlp.shared_experts)
        else:
            to_buffer(layer.mlp.shared_experts)
        trainable_experts = adapter_config['experts'][str(idx)]
        for expert_id in range(len(layer.mlp.experts)):
            if expert_id in trainable_experts:
                to_param(layer.mlp.experts[expert_id])
            else:
                to_buffer(layer.mlp.experts[expert_id])
    return model


def load_state_dict(folder_path):
    # Initialize empty state_dict
    combined_state_dict = {}

    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.safetensors'):
            file_path = os.path.join(folder_path, file_name)
            state_dict = load_file(file_path)
            combined_state_dict.update(state_dict)

    # legacy for loading v1 checkpoints: add prefix "model." for parameters
    for k in list(combined_state_dict.keys()):
        if k.startswith("layers"):
            k_new = "model." + k
            combined_state_dict[k_new] = combined_state_dict[k]
            del combined_state_dict[k]

    return combined_state_dict
    

def load_esft_model(base_model_path, adapter_dir):
    adapter_config = json.load(open(adapter_dir + "/expert_cfg.json"))
    adapter_state_dict = load_state_dict(adapter_dir)

    # load pretrained model:
    model, tokenizer = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16), AutoTokenizer.from_pretrained(base_model_path)

    to_esft(model, adapter_config)
    model.load_state_dict(adapter_state_dict)

    return model, tokenizer

def load_base_model(base_model_path):
    # load pretrained model:
    model, tokenizer = AutoModelForCausalLM.from_pretrained(base_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16), AutoTokenizer.from_pretrained(base_model_path)

    return model, tokenizer

def _get_expert_id(key):
    "input: model.layers.25.mlp.experts.10.xx.xx, output: ('25', 10)"
    return str(key.split(".")[2]), int(key.split(".")[5])

def _build_expert_dict(expert_list):
    expert_dict = {}
    for layer, expert in expert_list:
        if layer not in expert_dict:
            expert_dict[layer] = []
        expert_dict[layer].append(expert)
    for layer in expert_dict:
        expert_dict[layer] = sorted(list(set(expert_dict[layer])))
    return expert_dict

def _dict_equal(dict1, dict2):
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    else:
        keys = set(dict1.keys())
        return all([sorted(tuple(dict1[k])) == sorted(tuple(dict2[k])) for k in keys])

def add_adapter(base_model, adapter_dir, return_original_states=False, expert_config=None):
    if expert_config is not None:
        adapter_config = json.load(open(expert_config))
    else:
        adapter_config = json.load(open(adapter_dir + "/expert_cfg.json"))
    adapter_state_dict = load_state_dict(adapter_dir)
    expert_in_param = list([_get_expert_id(i) for i in adapter_state_dict.keys() if "expert" in i])
    # expert_in_param: [('1', 8), ('1', 9), ('2', 0), ('2', 1), ('2', 2), ('2', 3), ('2', 4), ('2', 5), ('2', 6), ('2', 7)]
    # expert_in_param_dict: {'1': [8, 9], '2': [0, 1, 2, 3, 4, 5, 6, 7]}
    expert_in_param_dict = _build_expert_dict(expert_in_param)
    if not _dict_equal(adapter_config['experts'], expert_in_param_dict):
        print(adapter_config['experts'])
        print(expert_in_param_dict)
        raise ValueError("expert_config and expert_in_param_dict are not consistent")
    
    to_esft(base_model, adapter_config)

    if return_original_states:
        original_state_dict = {k:v.cpu() for k, v in base_model.state_dict().items()}
        base_model.load_state_dict(adapter_state_dict, strict=False)
        return base_model, original_state_dict
    else:
        base_model.load_state_dict(adapter_state_dict)
        return base_model

