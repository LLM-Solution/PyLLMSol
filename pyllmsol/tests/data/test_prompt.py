#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-27 10:07:12
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 09:57:02

""" Test Prompt objects. """

# Built-in packages

# Third party packages

# Local packages

__all__ = []


# Built-in packages
from pathlib import Path
from unittest.mock import Mock

# Third party packages
import pytest

# Local packages
from pyllmsol.mock import MockTokenizer as MockLlamaTokenizer
from pyllmsol.mock import MockPreTrainedTokenizerBase
from pyllmsol.data.prompt import Prompt, PromptDataSet


__all__ = []


@pytest.fixture
def mock_llama_tokenizer():
    return MockLlamaTokenizer()


@pytest.fixture
def mock_transformers_tokenizer():
    return MockPreTrainedTokenizerBase()


# Test Prompt object
def test_prompt_initialization(mock_llama_tokenizer, mock_transformers_tokenizer):
    prompt = "This is a sample prompt for testing."
    data = Prompt(prompt, tokenizer=mock_transformers_tokenizer)
    assert data.text == prompt
    assert data.tokenizer == mock_transformers_tokenizer


def test_tokens_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    prompt = "This is a sample prompt for testing."
    data = Prompt(prompt, tokenizer=mock_transformers_tokenizer)
    assert data.tokens == [0] + [ord(i) for i in prompt]

    data = Prompt(prompt, tokenizer=mock_llama_tokenizer)
    assert data.tokens == [0] + [ord(i) for i in prompt]


def test_mask_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    prompt = "This is a sample prompt for testing."
    data = Prompt(prompt, tokenizer=mock_transformers_tokenizer)
    assert data.mask == [1] * (len(prompt) + 1)
    assert len(data.mask) == len(data.tokens)

    data = Prompt(prompt, tokenizer=mock_llama_tokenizer)
    assert data.mask == [1] * (len(prompt) + 1)
    assert len(data.mask) == len(data.tokens)


def test_from_textfile(mock_llama_tokenizer, tmp_path):
    path = tmp_path / "test_prompt.txt"
    path.write_text("File prompt content")
    data = Prompt.from_textfile(path, tokenizer=mock_llama_tokenizer)
    assert data.text == "File prompt content"


def test_add_operator(mock_transformers_tokenizer):
    data1 = Prompt("Hello World!", tokenizer=mock_transformers_tokenizer)
    data2 = Prompt("\nHi!", tokenizer=mock_transformers_tokenizer)
    result = data1 + data2
    assert result.text == "Hello World!\nHi!"


def test_get_n_tokens(mock_llama_tokenizer):
    prompt = "This is a sample prompt for testing."
    data = Prompt(prompt, tokenizer=mock_llama_tokenizer)
    assert data.get_n_tokens() == len(prompt) + 1


# Test PromptDataSet
def test_promptdataset_initialization(mock_transformers_tokenizer):
    prompt_list = ["Prompt one.", "Prompt two.", "Prompt three."]
    dataset = PromptDataSet(prompt_list, batch_size=2, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 3
    assert dataset.batch_size == 2


def test_prompt_add(mock_llama_tokenizer):
    prompt_list = ["Prompt one.", "Prompt two.", "Prompt three."]
    dataset = PromptDataSet(prompt_list, batch_size=2, tokenizer=mock_transformers_tokenizer)

    dataset.add("Prompt four", inplace=True)
    assert len(dataset.items) == 4
    assert isinstance(dataset[-1], Prompt)
    assert str(dataset.items[-1]) == "Prompt four"

    new_dataset = dataset.add(dataset, inplace=False)
    assert len(dataset.items) == 4
    assert len(new_dataset.items) == 8


def test_remaining_data(mock_llama_tokenizer):
    prompt_list = ["Prompt one.", "Prompt two.", "Prompt three."]
    dataset = PromptDataSet(prompt_list, batch_size=2, tokenizer=mock_llama_tokenizer)
    dataset._set_boundary(start=1, end=3)
    remaining_data = dataset.remaining_data()
    assert len(remaining_data) == 2


def test_from_json(mock_transformers_tokenizer, tmp_path):
    json_path = tmp_path / "test_data.json"
    json_path.write_text('["Prompt one.", "Prompt two."]')
    dataset = PromptDataSet.from_json(json_path, batch_size=1, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 2


def test_from_jsonl(mock_llama_tokenizer, tmp_path):
    jsonl_path = tmp_path / "test_data.jsonl"
    jsonl_path.write_text('"Sample prompt 1"\n"Sample prompt 2"')
    dataset = PromptDataSet.from_jsonl(jsonl_path, batch_size=1, tokenizer=mock_llama_tokenizer)
    assert len(dataset.items) == 2


if __name__ == "__main__":
    pass
