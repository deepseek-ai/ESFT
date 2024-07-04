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
