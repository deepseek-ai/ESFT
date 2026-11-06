from unittest.mock import MagicMock

import torch

from utils import (
    get_examples_from_buffer_pad,
    get_formatted_answer,
    get_formatted_input_and_target,
    get_formatted_question,
)


def make_mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.pad_token_id = 0
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: [ord(c) for c in text]
    return tokenizer


def test_formatted_question_and_answer():
    q = get_formatted_question('What is 2+2?')
    assert q == 'User: What is 2+2?\n\nAssistant:'

    a = get_formatted_answer('4')
    assert a == ' 4'


def test_formatted_input_and_target_mask_prompt():
    tokenizer = make_mock_tokenizer()
    messages = [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello'},
    ]

    # When mask_prompt=True (default), BOS must be masked (-100)
    input_ids, target_ids = get_formatted_input_and_target(messages, tokenizer, mask_prompt=True)
    assert input_ids[0] == tokenizer.bos_token_id
    assert target_ids[0] == -100
    assert target_ids[-1] == tokenizer.eos_token_id

    # When mask_prompt=False, BOS is unmasked
    input_ids_unmasked, target_ids_unmasked = get_formatted_input_and_target(messages, tokenizer, mask_prompt=False)
    assert input_ids_unmasked[0] == tokenizer.bos_token_id
    assert target_ids_unmasked[0] == tokenizer.bos_token_id


def test_buffer_pad_empty():
    tokenizer = make_mock_tokenizer()
    result = get_examples_from_buffer_pad([], seq_length=10, tokenizer=tokenizer, random_concat_ratio=0.0)
    assert result is None


def test_buffer_pad_exact_boundary_no_phantom_sequence():
    tokenizer = make_mock_tokenizer()
    # 2 items of length 5 => total 10 tokens == seq_length
    buffer = [
        ([1, 2, 3, 4, 5], [10, 20, 30, 40, 50]),
        ([6, 7, 8, 9, 10], [60, 70, 80, 90, 100]),
    ]
    result = get_examples_from_buffer_pad(buffer, seq_length=10, tokenizer=tokenizer, random_concat_ratio=0.0)
    assert result is not None
    # Must contain exactly 1 sequence of length 10, NOT an extra all-padding sequence
    assert result['input_ids'].shape == (1, 10)
    assert result['labels'].shape == (1, 10)
    assert torch.equal(result['input_ids'][0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))


def test_buffer_pad_partial_remainder():
    tokenizer = make_mock_tokenizer()
    buffer = [
        ([1, 2, 3], [10, 20, 30]),
    ]
    result = get_examples_from_buffer_pad(buffer, seq_length=6, tokenizer=tokenizer, random_concat_ratio=0.0)
    assert result is not None
    assert result['input_ids'].shape == (1, 6)
    # Remaining 3 tokens padded with pad_token_id (0) and -100
    assert torch.equal(result['input_ids'][0], torch.tensor([1, 2, 3, 0, 0, 0]))
    assert torch.equal(result['labels'][0], torch.tensor([10, 20, 30, -100, -100, -100]))
