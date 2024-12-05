#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-27 11:52:04
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 15:30:58
# @File path: ./pyllmsol/tests/inference/test_cli_instruct.py
# @Project: PyLLMSol

""" Test `pyllmsol.inference.cli_instruct.py` script. """

# Built-in packages
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.tests.mock import MockLlama, MockTokenizer
from pyllmsol.inference.cli_instruct import InstructCLI
from pyllmsol.data.chat import Chat, Message


__all__ = []


@pytest.fixture
def chat():
    return Chat(
        items=[
            {"role": "system", "content": "Welcome to the chat."},
            {"role": "assistant", "content": "How can I help you?"},
        ],
        tokenizer=MockTokenizer(),
    )


@pytest.fixture
def cli(chat):
    return InstructCLI(
        llm=MockLlama(n_ctx=240),
        init_prompt=chat,
        verbose=True,
    )


def test_initialization(cli, chat):
    """Test initialization of InstructCLI."""
    assert cli.stop == "<|eot_id|>"
    assert cli.init_prompt == chat
    assert isinstance(cli.init_prompt, Chat)
    assert isinstance(cli.prompt_hist, Chat)
    assert cli.init_prompt.items[0].role == "system"
    assert cli.init_prompt.items[0].content == "Welcome to the chat."
    assert len(cli.prompt_hist.items) == 2


def test_prompt_history(cli):
    """Test prompt history updates in InstructCLI."""
    assert len(cli.prompt_hist.items) == 2
    cli.ask("What is AI?")
    assert cli.prompt_hist.items[-1]["content"] == "What is AI?"
    assert cli.prompt_hist.items[-1]["role"] == "user"
    assert len(cli.prompt_hist.items) == 3
    cli.answer("Artificial Intelligence is the simulation of human intelligence.")
    assert cli.prompt_hist.items[-1]["content"] == "Artificial Intelligence is the simulation of human intelligence."
    assert cli.prompt_hist.items[-1]["role"] == "assistant"
    assert len(cli.prompt_hist.items) == 4


def test_context_limit(cli, chat):
    """Test context limit enforcement in _check_prompt_limit_context."""
    cli.prompt_hist.add({"role": "user", "content": "A short question."}, inplace=True)
    cli.prompt_hist.add({"role": "assistant", "content": "A short answer."}, inplace=True)
    cli._check_prompt_limit_context()
    assert len(cli.prompt_hist.items) == 3
    assert cli.prompt_hist.items[0]['role'] == "system"
    assert cli.prompt_hist.items[0]['content'] == "Welcome to the chat."
    assert cli.prompt_hist.items[1]['role'] == "user"
    assert cli.prompt_hist.items[1]['content'] == "A short question."
    assert cli.prompt_hist.items[2]["role"] == "assistant"
    assert cli.prompt_hist.items[2]["content"] == "A short answer."


@patch('pyllmsol.inference._base_cli.print')
def test_answer(mock_print, cli):
    """Test the answer method."""
    response = "Machine learning is a subset of AI."
    cli.answer(response)
    assert cli.prompt_hist.items[-1]['role'] == "assistant"
    assert cli.prompt_hist.items[-1]['content'] == response

    cli.answer("This is a test response.")
    mock_print.assert_called_with("This is a test response.", end='', flush=True)


if __name__ == "__main__":
    pass
