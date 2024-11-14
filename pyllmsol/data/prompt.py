#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 10:37:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-14 11:41:13

""" Prompt objects for text data to inferencing or training LLMs. """

# Built-in packages

# Third party packages

# Local packages
from pyllmsol.data._base_data import _BaseData
from pyllmsol.data.utils import truncate_text

__all__ = []


class Prompt:
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


if __name__ == "__main__":
    pass
