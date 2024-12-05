#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 16:19:58
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 10:03:27
# @File path: ./pyllmsol/tests/mock.py
# @Project: PyLLMSol

""" Description. """

# Built-in packages
from random import randint
from unittest.mock import MagicMock

# Third party packages
from llama_cpp import Llama, LlamaTokenizer
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, BatchEncoding

# Local packages

__all__ = []


class MockTokenizer(LlamaTokenizer):
    def __init__(self):
        self.pad_token_id = -1

    def __bool__(self):
        return True

    def encode(self, text, add_bos=False, special=True):
        tokens = [0] if add_bos else []
        tokens += [ord(char) for char in text]  # Simulate token IDs

        return tokens


class MockPreTrainedTokenizerBase(PreTrainedTokenizerBase):
    def __init__(self):
        pass

    @property
    def pad_token_id(self):
        return -1

    @property
    def bos_token_id(self):
        return 0

    def __bool__(self):
        return True

    def encode(self, text, add_special_tokens=True):
        tokens = [self.bos_token_id] if add_special_tokens else []
        tokens += [ord(char) for char in text]  # Simulate token IDs

        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        if skip_special_tokens and self.bos_token_id in tokens:
            tokens = [token for token in tokens if token != self.bos_token_id]

        return ''.join(chr(token) for token in tokens)

    def __call__(self, texts, return_tensors='pt', padding=True, add_special_tokens=True):
        outputs = []
        attention_mask = []
        max_len = 0
        if isinstance(texts, str):
            texts = [texts]

        for text in texts:
            encoded = self.encode(text, add_special_tokens=add_special_tokens)
            outputs.append(encoded)
            max_len = max(max_len, len(encoded))

        if padding:
            outputs = [o + [self.pad_token_id] * (max_len - len(o)) for o in outputs]

        if return_tensors == "pt":
            outputs = torch.tensor(outputs)

        return BatchEncoding(data={'input_ids': outputs, 'attention_mask': torch.ones(outputs.size())})


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

    def tokenizer(self):
        return self.tokenize


class MockAutoModelForCausalLM(AutoModelForCausalLM):
    def __init__(self, *args, **kwargs):
        self.parameters = MagicMock()
        self.training = False

    def __call__(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = MagicMock()
        batch_size, seq_len = input_ids.size()
        outputs.logits = torch.randn(batch_size, seq_len, 200)
        outputs.loss = MockLoss()

        return outputs

    def train(self, mode=True):
        self.training = mode

    def generate(self, input_ids, attention_mask, max_length=64):
        n = max(0, max_length - len(input_ids))
        tokens = [ord(i) for i in " Hello world !"]
        generated = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        for i in range(len(generated)):
            while len(generated[i]) < n:
                generated[i].append(tokens[len(generated[i]) % len(tokens)])

        return generated


class MockOptimizer(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        super().__call__(*args, **kwargs)
        return self


class MockLoss(MagicMock):
    def detach(self):
        return torch.tensor(0.05)


if __name__ == "__main__":
    pass
