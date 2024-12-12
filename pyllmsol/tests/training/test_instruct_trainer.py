#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-11 17:55:16
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 18:29:03
# @File path: ./pyllmsol/tests/training/test_instruct_trainer.py
# @Project: PyLLMSol

""" Test InstructTrainer object. """

# Built-in packages
from unittest.mock import patch, MagicMock

# Third party packages
import pytest
import torch

# Local packages
from pyllmsol.tests.mock import (
    MockPreTrainedTokenizerBase,
    MockAutoModelForCausalLM,
)
from pyllmsol.data.chat import Chat, Message, ChatDataSet
from pyllmsol.training.instruct_trainer import TrainerInstruct


@pytest.fixture
def tokenizer():
    return MockPreTrainedTokenizerBase()


@pytest.fixture
def chat_dataset(tokenizer):
    messages = [
        [
            Message(role="user", content="Hello", tokenizer=tokenizer),
            Message(role="assistant", content="Hi", tokenizer=tokenizer),
        ],
        [
            Message(role="user", content="How are you?", tokenizer=tokenizer),
            Message(role="assistant", content="Good, thanks!", tokenizer=tokenizer),
        ],
    ]
    return ChatDataSet(items=messages, tokenizer=tokenizer, batch_size=2)


@pytest.fixture
def llm():
    return MockAutoModelForCausalLM()


@pytest.fixture
def instruct_trainer(llm, tokenizer, chat_dataset):
    return TrainerInstruct(
        llm=llm,
        tokenizer=tokenizer,
        dataset=chat_dataset,
        accumulation_steps=2,
    )


def test_initialization(instruct_trainer, chat_dataset, tokenizer, llm):
    """Test initialization of TrainerInstruct."""
    assert isinstance(instruct_trainer.dataset, ChatDataSet)
    assert instruct_trainer.llm == llm
    assert instruct_trainer.tokenizer == tokenizer
    assert instruct_trainer.dataset == chat_dataset

    trainer = TrainerInstruct(
        llm=llm,
        tokenizer=tokenizer,
        dataset=[
            [{"role": "system", "content": "Test"}, {"role": "user", "content": "Hello"}],
            [{"role": "system", "content": "Test-2"}, {"role": "assistant", "content": "Hello"}],
        ],
        accumulation_steps=2,
    )

    assert isinstance(trainer.dataset, ChatDataSet)
    assert len(trainer.dataset) == 2
    assert trainer.dataset.batch_size == 1


def test_next_batch(instruct_trainer):
    """Test the __next__ method for batch retrieval."""
    iter(instruct_trainer)
    input_ids, attention_mask = next(instruct_trainer)

    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.size(0) == instruct_trainer.dataset.batch_size
    assert attention_mask.size(0) == instruct_trainer.dataset.batch_size
    # Check padding
    assert input_ids[0, -1] == instruct_trainer.tokenizer.pad_token_id
    # Check bos token
    assert input_ids[0, 0] == instruct_trainer.tokenizer.bos_token_id


if __name__ == "__main__":
    pass
