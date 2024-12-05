#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 16:50:03
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 08:36:36
# @File path: ./pyllmsol/tests/training/test_trainer.py
# @Project: PyLLMSol

""" Test trainer object. """

# Built-in packages
from unittest.mock import MagicMock, patch

# Third party packages
import pytest
import torch

# Local packages
from pyllmsol.tests.mock import (MockPreTrainedTokenizerBase, MockOptimizer,
                           MockAutoModelForCausalLM,)
from pyllmsol.data.prompt import PromptDataSet
from pyllmsol.training.trainer import Trainer

__all__ = []


@pytest.fixture
def tokenizer():
    return MockPreTrainedTokenizerBase()


@pytest.fixture
def dataset(tokenizer):
    return PromptDataSet(items=["text1", "text 2", "text 3"], tokenizer=tokenizer, batch_size=2)


@pytest.fixture
def optimizer():
    return MockOptimizer()


@pytest.fixture
def llm():
    return MockAutoModelForCausalLM()


@pytest.fixture
def trainer(llm, tokenizer, dataset):
    trainer = Trainer(
        llm=llm,
        tokenizer=tokenizer,
        dataset=dataset,
        accumulation_steps=3,
    )

    return trainer


def test_initialization(trainer, llm, tokenizer, dataset):
    """Test Trainer initialization."""
    assert trainer.llm == llm
    assert trainer.tokenizer == tokenizer
    assert trainer.dataset == dataset
    assert trainer.accumulation_steps == 3


def test_set_mask(trainer):
    attention_mask = torch.ones((4, 256))
    input_ids = torch.randint(0, 100, (4, 256))

    updated_mask = trainer.set_mask(attention_mask, input_ids, rate=0.5)
    assert updated_mask.size() == attention_mask.size()
    assert updated_mask.mean() == 0.5 


def test_iteration(trainer):
    """Test the Trainer's iteration logic."""
    iter(trainer)
    input_ids, attention_mask = next(trainer)

    assert trainer.losses is not None
    assert trainer.n_accumulated_grad == 0
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert input_ids.size(0) == trainer.dataset.batch_size
    assert attention_mask.mean() == 1
    # Check padding
    assert input_ids[0, -1] == trainer.tokenizer.pad_token_id
    assert input_ids[1, -1] != trainer.tokenizer.pad_token_id
    # Check bos token
    assert input_ids[0, 0] == trainer.tokenizer.bos_token_id


def test_set_optimizer(trainer, optimizer):
    trainer.set_optimizer(optimizer, lr=0.001)
    assert trainer.llm.training == True
    trainer.llm.parameters.assert_called()
    optimizer.assert_called_with(trainer.llm.parameters(), lr=0.001)


def test_trainig_step(trainer, optimizer):
    # Set optimizer
    trainer.set_optimizer(optimizer, lr=0.001)

    # Get data
    iter(trainer)
    input_ids, attention_mask = next(trainer)

    trainer.training_step(input_ids, attention_mask)
    assert trainer.n_accumulated_grad == trainer.dataset.batch_size
    assert trainer.losses.current_loss == 0.05
    assert len(trainer.losses.loss_history) == 1

    trainer.training_step(input_ids, attention_mask)
    assert trainer.n_accumulated_grad == 0
    assert trainer.losses.current_loss == 0.05
    assert len(trainer.losses.loss_history) == 2
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()


def test_run(trainer, llm, optimizer):
    """Test the Trainer's run method."""
    trainer.set_optimizer(optimizer, lr=0.001)
    
    # Mock checkpointing
    checkpoint = MagicMock()
    
    trainer.run(device="cpu", checkpoint=checkpoint)
    
    assert trainer.n_accumulated_grad == 0
    assert trainer.losses.current_loss == 0.05
    assert len(trainer.losses.loss_history) == 2
    assert trainer.dataset.remaining_data().items == []
    # llm.assert_called()
    checkpoint.assert_called()



if __name__ == "__main__":
    pass
