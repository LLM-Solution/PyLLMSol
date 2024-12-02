#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 10:37:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 12:38:32

""" Prompt objects for text data to inferencing or training LLMs. """

# Built-in packages

# Third party packages
from llama_cpp import LlamaTokenizer
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol.data._base_data import _BaseData, _TextData, _DataSet

__all__ = []


TokenizerType = LlamaTokenizer | PreTrainedTokenizerBase


class Prompt(_TextData):
    """ A Prompt class for handling and tokenizing text data.

    Parameters
    ----------
    text : str
        The text to be stored and formatted for inference or training.
    tokenizer : TokenizerType, optional
        A tokenizer object for encoding the text (default is None).

    Methods
    -------
    from_textfile
    get_n_tokens

    Attributes
    ----------
    text : str
        The text to be stored and formatted for inference or training.
    tokenizer : transformer.PreTrainedTokenizerBase or llama_cpp.LlamaTokenizer
        Tokenizer object.

    Properties
    ----------
    tokens : list of int
    mask

    """

    pass


class PromptDataSet(_DataSet, _BaseData):
    """ PromptDataset class to manage data loading and batch iteration.

    This class allows for iterating through a dataset in customizable batch
    sizes. It includes an optional progress bar to track the current progress in
    the iteration and allows setting descriptions for the progress bar. This
    class includes methods for loading itemss from JSON or JSONL files.

    Parameters
    ----------
    items : list of Prompt
        The items to iterate over, typically loaded from JSON or JSONL.
    batch_size : int
        The size of each data batch, default is 1.
    tokenizer : TokenizerType, optional
        Tokenizer object.

    Methods
    -------
    __getitem__
    __iter__
    __next__
    add
    from_json
    from_jsonl
    remaining_data
    set_description

    Attributes
    -----------
    items : list of Prompt
        The data to iterate over.
    batch_size : int
        The number of items to return in each batch.
    start : int
        The index to start the iteration from.
    end : int
        The index to end the iteration at.
    i : int
        The current index in the dataset for the iterator.
    tokenizer : transformer.PreTrainedTokenizerBase or llama_cpp.LlamaTokenizer
        Tokenizer object.

    """

    def __init__(
        self,
        items: list[Prompt | str],
        tokenizer: TokenizerType,
        batch_size: int = 1,
    ):
        _BaseData.__init__(self, items, Prompt, str, tokenizer)
        self.batch_size = batch_size
        self.set_boundary(0, end=None)


if __name__ == "__main__":
    pass
