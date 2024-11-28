#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 16:19:58
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-28 16:21:14
# @File path: ./pyllmsol/mock.py
# @Project: PyLLMSol

""" Description. """

# Built-in packages
from unittest.mock import MagicMock

# Third party packages
from llama_cpp import Llama, LlamaTokenizer

# Local packages

__all__ = []


class MockTokenizer(LlamaTokenizer):
    def __init__(self):
        self.pad_token_id = 0

    def __bool__(self):
        return True

    def encode(self, text, add_bos=False, special=True):
        tokens = [0] if add_bos else []
        tokens += [ord(char) for char in text]  # Simulate token IDs

        return tokens


class MockLlama(Llama):
    def __init__(self, *args, n_ctx=1024, **kwargs):
        self.tokenize = MockTokenizer()
        self.n_ctx = n_ctx
        self._stack = MagicMock()

    def __call__(self, *args, stream=False, **kwargs):
        if stream:
            return (c for c in "LLM response.")

        else:
            return {'choices': [{'text': "LLM response."}]}


if __name__ == "__main__":
    pass