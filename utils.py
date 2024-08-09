import os
import random
import torch
import torch.distributed as dist
# given a message object, convert to prompt and response

PROMPT_USER: str = 'User: {input}\n\n'
PROMPT_ASSISTANT: str = 'Assistant:'  # should not have a space at the end
ASSISTANT_RESPONSE: str = ' {input}'

def get_formatted_question(line):
    return PROMPT_USER.format(input=str(line).strip()) + PROMPT_ASSISTANT

def get_formatted_answer(line):
    return ASSISTANT_RESPONSE.format(input=str(line).strip())

def get_formatted_input_and_target(messages, tokenizer, IGNORE_TOKEN_ID=-100, mask_prompt=True):
    input_ids = []
    target_ids = []
    for idx, message in enumerate(messages):
        if idx == 0:
            input_ids.extend([tokenizer.bos_token_id])
            target_ids.extend([tokenizer.bos_token_id])

        if message['role'] == "user":
            formatted_question = get_formatted_question(message['content'])
            tokenized_line = tokenizer.encode(formatted_question, add_special_tokens=False)
            input_ids.extend(tokenized_line)
            if mask_prompt:
                target_ids.extend([IGNORE_TOKEN_ID] * len(tokenized_line))
            else:
                target_ids.extend(tokenized_line)
        elif message['role'] == "assistant":
            formatted_answer = get_formatted_answer(message['content'])
            tokenized_line = tokenizer.encode(formatted_answer, add_special_tokens=False) + [tokenizer.eos_token_id]
            input_ids.extend(tokenized_line)
            if message.get('mask', 0) == 1:
                target_ids.extend([IGNORE_TOKEN_ID] * len(tokenized_line))
            else:
                target_ids.extend(tokenized_line)
        else:
            assert False, f"Unknown role: {message['role']}"

    return [input_ids, target_ids]


def get_examples_from_buffer_pad(buffer, seq_length, tokenizer, random_concat_ratio, IGNORE_TOKEN_ID=-100):
    all_input_ids_list, all_target_ids_list = [], []
    all_input_ids, all_target_ids = [], []

    for input_ids, target_ids in buffer:
        if len(input_ids) > seq_length - len(all_input_ids):
            input_ids = input_ids[-(seq_length - len(all_input_ids)):]
            target_ids = target_ids[-(seq_length - len(all_target_ids)):]
        if len(all_input_ids) > 0 and random.random() < random_concat_ratio:
            input_ids = input_ids[1:]
            target_ids = target_ids[1:]
        all_input_ids.extend(input_ids)
        all_target_ids.extend(target_ids)
        if len(all_input_ids) >= seq_length:
            assert len(all_input_ids) == seq_length, f"{len(all_input_ids)=}, {seq_length=}, {len(buffer)=}"
            all_input_ids_list.append(all_input_ids)
            all_target_ids_list.append(all_target_ids)
            all_input_ids, all_target_ids = [], []

    all_input_ids = all_input_ids + [tokenizer.pad_token_id for i in range(seq_length - len(all_input_ids))]
    all_target_ids = all_target_ids + [IGNORE_TOKEN_ID for i in range(seq_length - len(all_target_ids))]
    all_input_ids_list.append(all_input_ids)
    all_target_ids_list.append(all_target_ids)

    if len(all_input_ids) <= 0:
        return None
    return {
        "input_ids": torch.tensor(all_input_ids_list, dtype=torch.long),
        "labels": torch.tensor(all_target_ids_list, dtype=torch.long)
    }


def init_parallel_groups(ep_size=1):
    dist.init_process_group("nccl")
    world_size = int(os.getenv("WORLD_SIZE", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    ep_group = edp_group = None
    for i in range(0, world_size, ep_size):
        ranks = list(range(i, i + ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            ep_group = group
    edp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks)
        if local_rank in ranks:
            edp_group = group
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    dist.all_reduce(torch.zeros(1, device="cuda"), group=edp_group)
    return world_size, local_rank, ep_group, edp_group
