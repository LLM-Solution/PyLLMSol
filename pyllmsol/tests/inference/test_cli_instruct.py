#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-27 11:52:04
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-27 16:18:34

""" Test `pyllmsol.inference.cli_instruct.py` script. """

# Built-in packages
from unittest.mock import patch, MagicMock

# Third party packages
from llama_cpp import LlamaTokenizer
import pytest

# Local packages
from pyllmsol.inference.cli_instruct import InstructCLI
from pyllmsol.data.chat import Chat, Message

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
@patch('pyllmsol.inference._base_cli.Llama')
def cli(mock_llama, chat):
    mock_llama_instance = MagicMock()
    mock_llama.return_value = mock_llama_instance

    # Attach the MockTokenizer to the mock_llama_instance
    mock_llama_instance.tokenize = MockTokenizer()

    return InstructCLI(
        model_path="dummy/path/to/model",
        init_prompt=chat,
        verbose=True,
    )


def test_initialization(cli, chat):
    """Test initialization of InstructCLI."""
    assert cli.stop == "<|eot_id|>"
    assert cli.init_prompt == chat
    assert isinstance(cli.init_prompt, Chat)
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
    print("Prompt 0", str(cli.prompt_hist))
    print("Prompt 0", len(str(cli.prompt_hist)))
    cli.n_ctx = 240
    cli.prompt_hist.add({"role": "user", "content": "A short question."}, inplace=True)
    cli.prompt_hist.add({"role": "assistant", "content": "A short answer."}, inplace=True)
    print("Prompt 1", str(cli.prompt_hist))
    print("Prompt 1", len(str(cli.prompt_hist)))
    cli._check_prompt_limit_context()
    print("Prompt 2", str(cli.prompt_hist))
    print("Prompt 2", len(str(cli.prompt_hist)))
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
