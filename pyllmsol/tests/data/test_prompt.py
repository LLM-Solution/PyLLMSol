#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-27 10:07:12
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-27 10:25:47

""" Test Prompt objects. """

# Built-in packages

# Third party packages

# Local packages

__all__ = []


# Built-in packages
from pathlib import Path
from unittest.mock import Mock

# Third party packages
from llama_cpp import LlamaTokenizer
import pytest
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol.data.prompt import Prompt, PromptDataSet


__all__ = []


class MockLlamaTokenizer(LlamaTokenizer):
    def __init__(self):
        pass

    def encode(self, text, add_bos=False, special=True):
        return [101, 102, 103]

    def __bool__(self):
        return True


class MockTransformersTokenizer(PreTrainedTokenizerBase):
    def __init__(self):
        pass

    def encode(self, text, add_special_tokens=False):
        return [101, 102, 103]

    def __bool__(self):
        return True


@pytest.fixture
def mock_llama_tokenizer():
    return MockLlamaTokenizer()


@pytest.fixture
def mock_transformers_tokenizer():
    return MockTransformersTokenizer()


# Test Prompt object
def test_prompt_initialization(mock_llama_tokenizer, mock_transformers_tokenizer):
    text = "This is a sample text for testing."
    data = Prompt(text, tokenizer=mock_transformers_tokenizer)
    assert data.text == text
    assert data.tokenizer == mock_transformers_tokenizer


def test_tokens_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    data = Prompt("Sample text", tokenizer=mock_transformers_tokenizer)
    assert data.tokens == [101, 102, 103]

    data = Prompt("Sample text", tokenizer=mock_llama_tokenizer)
    assert data.tokens == [101, 102, 103]


def test_mask_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    data = Prompt("Sample text", tokenizer=mock_transformers_tokenizer)
    assert data.mask == [1, 1, 1]
    assert len(data.mask) == len(data.tokens)

    data = Prompt("Sample text", tokenizer=mock_llama_tokenizer)
    assert data.mask == [1, 1, 1]
    assert len(data.mask) == len(data.tokens)


def test_from_textfile(mock_llama_tokenizer, tmp_path):
    path = tmp_path / "test_text.txt"
    path.write_text("File text content")
    data = Prompt.from_textfile(path, tokenizer=mock_llama_tokenizer)
    assert data.text == "File text content"


def test_add_operator(mock_transformers_tokenizer):
    data1 = Prompt("Hello, ", tokenizer=mock_transformers_tokenizer)
    data2 = Prompt("world!", tokenizer=mock_transformers_tokenizer)
    result = data1 + data2
    assert result.text == "Hello, world!"


def test_get_n_tokens(mock_llama_tokenizer):
    data = Prompt("Sample text", tokenizer=mock_llama_tokenizer)
    assert data.get_n_tokens() == 3


# Test PromptDataSet
def test_promptdataset_initialization(mock_transformers_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = PromptDataSet(text_list, batch_size=2, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 3
    assert dataset.batch_size == 2


def test_prompt_add(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = PromptDataSet(text_list, batch_size=2, tokenizer=mock_transformers_tokenizer)

    dataset.add("Text four", inplace=True)
    assert len(dataset.items) == 4
    assert isinstance(dataset[-1], Prompt)
    assert str(dataset.items[-1]) == "Text four"

    new_dataset = dataset.add(dataset, inplace=False)
    assert len(dataset.items) == 4
    assert len(new_dataset.items) == 8


def test_promptdataset_iteration(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = PromptDataSet(text_list, batch_size=2, tokenizer=mock_llama_tokenizer)
    iterator = iter(dataset)
    batch1 = next(iterator)
    assert len(batch1.items) == 2
    batch2 = next(iterator)
    assert len(batch2.items) == 1


def test_promptdataset_out_of_bounds_iteration(mock_transformers_tokenizer):
    text_list = ["Text one.", "Text two."]
    dataset = PromptDataSet(text_list, batch_size=1, tokenizer=mock_transformers_tokenizer)
    iterator = iter(dataset)
    next(iterator)
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_remaining_data(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = PromptDataSet(text_list, batch_size=2, tokenizer=mock_llama_tokenizer)
    dataset._set_boundary(start=1, end=3)
    remaining_data = dataset.remaining_data()
    assert len(remaining_data) == 2


def test_from_json(mock_transformers_tokenizer, tmp_path):
    json_path = tmp_path / "test_data.json"
    json_path.write_text('["Text one.", "Text two."]')
    dataset = PromptDataSet.from_json(json_path, batch_size=1, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 2


def test_from_jsonl(mock_llama_tokenizer, tmp_path):
    jsonl_path = tmp_path / "test_data.jsonl"
    jsonl_path.write_text('"Sample text 1"\n"Sample text 2"')
    dataset = PromptDataSet.from_jsonl(jsonl_path, batch_size=1, tokenizer=mock_llama_tokenizer)
    assert len(dataset.items) == 2


if __name__ == "__main__":
    pass
