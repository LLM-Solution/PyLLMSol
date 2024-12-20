#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 10:55:40
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 20:01:57
# @File path: ./pyllmsol/tests/data/test_base_data.py
# @Project: PyLLMSol

""" Test `data/_base_data.py` script. """

# Built-in packages
from pathlib import Path
from unittest.mock import Mock

# Third party packages
import pytest

# Local packages
from pyllmsol.tests.mock import MockTokenizer as MockLlamaTokenizer
from pyllmsol.tests.mock import MockPreTrainedTokenizerBase
from pyllmsol.data._base_data import _BaseData, _TextData, _DataSet, tokenize

__all__ = []


@pytest.fixture
def mock_llama_tokenizer():
    return MockLlamaTokenizer()


@pytest.fixture
def mock_transformers_tokenizer():
    return MockPreTrainedTokenizerBase()

# @pytest.fixture
# def data():
#     MockBase = Mock(_BaseData, name="MockBase")
#     instance = MockBase(items=["Sample text"], item_type=_TextData, fallback_type=str, tokenizer=MockLlamaTokenizer())

#     return instance


def test_tokenize(mock_llama_tokenizer, mock_transformers_tokenizer):
    tokens = tokenize("Hello", mock_llama_tokenizer)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    tokens = tokenize("Hello", mock_transformers_tokenizer)
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)

    with pytest.raises(TypeError):
        tokenize("Hello", None)


# # Test _BaseData object

# def test_process_item_valid_item(mock_transformers_tokenizer, data):
#     """Test that _process_item correctly processes an item of the expected type."""
#     text = "Sample text"
#     # item = _TextData(text, tokenizer=mock_transformers_tokenizer)
#     # data = _BaseData(items=[item], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)
#     assert isinstance(data.items[0], _TextData)
#     assert data.items[0].text == text
#     assert data.items[0].tokens == [101, 102, 103]


# def test_process_item_fallback_conversion(mock_transformers_tokenizer, data):
#     """Test that _process_item converts a string into an _TextData object using a tokenizer."""
#     text = "Sample text"
#     # data = _BaseData(items=[text], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)
#     assert isinstance(data.items[0], _TextData)
#     assert data.items[0].text == text
#     assert data.items[0].tokens == [101, 102, 103]


# def test_process_item_invalid_type(mock_transformers_tokenizer):
#     """Test that _process_item raises a TypeError when the item is not of expected types."""
#     with pytest.raises(TypeError):
#         _BaseData(items=[123], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)


# def test_append_item_valid(mock_transformers_tokenizer):
#     """Test that _append_item correctly adds a new item of valid type."""
#     text = "New item text"
#     data = _BaseData(items=[], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)
#     data._append_item(text)
#     assert len(data.items) == 1
#     assert isinstance(data.items[0], _TextData)
#     assert data.items[0].text == text


# def test_append_item_invalid(mock_transformers_tokenizer):
#     """Test that _append_item raises a TypeError when adding an invalid item."""
#     data = _BaseData(items=[], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)
#     with pytest.raises(TypeError):
#         data._append_item(123)  # Invalid type


# def test_tokenize_no_tokenizer():
#     """Test that the tokenize method raises an error when no tokenizer is set."""
#     with pytest.raises(ValueError):
#         data = _BaseData(items=["text"], item_type=_TextData, fallback_type=str, tokenizer=None)
#         data.tokenize("text")


# def test_tokenize_valid(mock_transformers_tokenizer):
#     """Test that tokenize method returns the correct tokens when tokenizer is provided."""
#     data = _BaseData(items=["Sample text"], item_type=_TextData, fallback_type=str, tokenizer=mock_transformers_tokenizer)
#     tokens = data.tokenize("Sample text")
#     assert tokens == [101, 102, 103]


# Test _TextData object

def test_text_data_initialization(mock_llama_tokenizer, mock_transformers_tokenizer):
    text = "This is a sample text for testing."
    data = _TextData(text, tokenizer=mock_transformers_tokenizer)
    assert data.text == text
    assert data.tokenizer == mock_transformers_tokenizer


def test_tokens_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    text = "Sample text"
    data = _TextData(text, tokenizer=mock_transformers_tokenizer)
    assert data.tokens == [0] + [ord(i) for i in text]

    data = _TextData(text, tokenizer=mock_llama_tokenizer)
    assert data.tokens == [0] + [ord(i) for i in text]


def test_mask_property(mock_llama_tokenizer, mock_transformers_tokenizer):
    text = "Sample text"
    data = _TextData(text, tokenizer=mock_transformers_tokenizer)
    assert data.mask == [1] * (len(text) + 1)
    assert len(data.mask) == len(data.tokens)

    data = _TextData(text, tokenizer=mock_llama_tokenizer)
    assert data.mask == [1] * (len(text) + 1)
    assert len(data.mask) == len(data.tokens)


def test_from_textfile(mock_llama_tokenizer, tmp_path):
    path = tmp_path / "test_text.txt"
    path.write_text("File text content")
    data = _TextData.from_textfile(path, tokenizer=mock_llama_tokenizer)
    assert data.text == "File text content"


def test_add_operator(mock_transformers_tokenizer):
    data1 = _TextData("Hello, ", tokenizer=mock_transformers_tokenizer)
    data2 = _TextData("world!", tokenizer=mock_transformers_tokenizer)
    result = data1 + data2
    assert result.text == "Hello, world!"

    with pytest.raises(TypeError):
        data1 + 1


def test_get_n_tokens(mock_llama_tokenizer):
    text = "Sample text"
    data = _TextData(text, tokenizer=mock_llama_tokenizer)
    assert data.get_n_tokens() == len(text) + 1


# Test _DataSet

def test_dataset_initialization(mock_transformers_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = _DataSet(text_list, batch_size=2, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 3
    assert dataset.batch_size == 2


def test_set_boundary(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two."]
    dataset = _DataSet(text_list, batch_size=2, tokenizer=mock_transformers_tokenizer)

    with pytest.raises(IndexError, match="Start index 3 is out of bounds for data of size 2"):
        dataset.set_boundary(3)

    with pytest.raises(IndexError, match="End index 4 is out of bounds for data of size 2"):
        dataset.set_boundary(start=1, end=4)

    with pytest.raises(IndexError, match="End index 0 must be greater than start index 1"):
        dataset.set_boundary(start=1, end=0)

    dataset.set_boundary(1)
    assert dataset.start == 1
    assert dataset.end == 2

    dataset.set_boundary(start=0, end=1)
    assert dataset.start == 0
    assert dataset.end == 1


def test_dataset_iteration(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = _DataSet(text_list, batch_size=2, tokenizer=mock_llama_tokenizer)
    iterator = iter(dataset)
    batch1 = next(iterator)
    assert len(batch1.items) == 2
    batch2 = next(iterator)
    assert len(batch2.items) == 1


def test_dataset_out_of_bounds_iteration(mock_transformers_tokenizer):
    text_list = ["Text one.", "Text two."]
    dataset = _DataSet(text_list, batch_size=1, tokenizer=mock_transformers_tokenizer)
    iterator = iter(dataset)
    next(iterator)
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)


def test_remaining_data(mock_llama_tokenizer):
    text_list = ["Text one.", "Text two.", "Text three."]
    dataset = _DataSet(text_list, batch_size=2, tokenizer=mock_llama_tokenizer)
    dataset.set_boundary(start=1, end=3)
    remaining_data = dataset.remaining_data()
    assert len(remaining_data) == 2


def test_from_json(mock_transformers_tokenizer, tmp_path):
    json_path = tmp_path / "test_data.json"

    with pytest.raises(FileNotFoundError):
        dataset = _DataSet.from_json(path=json_path, tokenizer=mock_transformers_tokenizer)

    json_path.write_text('["Text one.", "Text two."]')
    dataset = _DataSet.from_json(json_path, batch_size=1, tokenizer=mock_transformers_tokenizer)
    assert len(dataset.items) == 2


def test_from_jsonl(mock_llama_tokenizer, tmp_path):
    jsonl_path = tmp_path / "test_data.jsonl"

    with pytest.raises(FileNotFoundError):
        dataset = _DataSet.from_jsonl(path=jsonl_path, tokenizer=mock_llama_tokenizer)

    jsonl_path.write_text('"Sample text 1"\n"Sample text 2"')
    dataset = _DataSet.from_jsonl(jsonl_path, batch_size=1, tokenizer=mock_llama_tokenizer)
    assert len(dataset.items) == 2


if __name__ == "__main__":
    pass
