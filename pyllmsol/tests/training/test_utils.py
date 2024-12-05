#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-04 15:34:44
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 08:33:59
# @File path: ./pyllmsol/tests/training/test_utils.py
# @Project: PyLLMSol

""" Test training util objects. """

# Built-in packages
from unittest.mock import MagicMock

# Third party packages
import pytest
import torch

# Local packages
from pyllmsol.tests.mock import MockPreTrainedTokenizerBase, MockAutoModelForCausalLM
from pyllmsol.training.utils import (
    find_token, find_sequence, generate, get_token_size,
    set_mask, shuffle_per_batch, sort_per_tokens_size
)

__all__ = []


@pytest.fixture
def tokenizer():
    """Mock a tokenizer."""
    return MockPreTrainedTokenizerBase()


@pytest.fixture
def llm():
    """Mock a language model."""
    return MockAutoModelForCausalLM()


def test_find_token():
    input_ids = torch.tensor([10, 20, 30, 40, 50])
    token_id = 30
    idx = find_token(token_id, input_ids)
    assert idx == 2


def test_find_token_not_found():
    input_ids = torch.tensor([10, 20, 30, 40, 50])
    token_id = 60
    with pytest.raises(IndexError):
        find_token(token_id, input_ids)

    input_ids = torch.tensor([])
    with pytest.raises(IndexError):
        find_token(token_id, input_ids)


def test_find_sequence():
    input_ids = torch.tensor([10, 20, 30, 40, 50])
    seq_ids = torch.tensor([30, 40])
    idx = find_sequence(seq_ids, input_ids)
    assert idx == 4


def test_find_sequence_not_found():
    input_ids = torch.tensor([10, 20, 30, 40, 50])
    seq_ids = torch.tensor([60, 70])
    assert find_sequence(seq_ids, input_ids) is None


def test_generate(llm, tokenizer):
    result = generate(llm, tokenizer, "Hello", max_length=15, device="cpu")
    assert result == "Hello world !"


def test_get_token_size(tokenizer):
    text = "Hello world"
    size = get_token_size(text, tokenizer)
    assert size == 11 + 1


def test_set_mask():
    attention_mask = torch.ones(10)
    masked = set_mask(attention_mask, rate=0.2)
    assert (masked.sum() <= 10).item()
    with pytest.raises(ValueError):
        set_mask(attention_mask, rate=1.5)


def test_shuffle_per_batch(tokenizer):
    data = ["text1", "text2", "text3", "text4"]
    tokenizer.side_effect = lambda x: {"input_ids": [1] * len(x)}
    shuffled = shuffle_per_batch(data, tokenizer, batch_size=2)
    assert set(data) == set(shuffled)  # Ensure all data is still present


def test_sort_per_tokens_size(tokenizer):
    data = ["long text", "short"]
    tokenizer.side_effect = lambda x: {"input_ids": [1] * len(x)}
    sorted_data = sort_per_tokens_size(data, tokenizer)
    assert sorted_data == ["short", "long text"]

if __name__ == "__main__":
    pass