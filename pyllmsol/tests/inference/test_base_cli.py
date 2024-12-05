#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-26 17:39:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-05 08:35:21
# @File path: ./pyllmsol/tests/inference/test_base_cli.py
# @Project: PyLLMSol

""" Test base CLI object. """

# Built-in packages
from time import strftime
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.tests.mock import MockLlama
from pyllmsol.data._base_data import _TextData
from pyllmsol.inference._base_cli import _BaseCommandLineInterface


__all__ = []


@pytest.fixture
def cli():
    return _BaseCommandLineInterface(
        llm=MockLlama(),
        init_prompt="Hello!",
        verbose=True
    )


def test_initialization(cli):
    """Test that the CLI initializes correctly."""
    assert cli.ai_name == "Assistant"
    assert cli.user_name == "User"
    assert cli.llm is not None
    assert cli.stop == [f"\n{cli.user_name}:", f"\n{cli.ai_name}:"]
    assert isinstance(cli.init_prompt, _TextData)
    assert isinstance(cli.prompt_hist, _TextData)


@patch('pyllmsol.inference._base_cli.print')
def test_answer(mock_print, cli):
    """Test the answer method."""
    output = "Hello, how can I help you?"
    cli.answer(output)
    mock_print.assert_called_with(output, end='', flush=True)


@patch('pyllmsol.inference._base_cli.input', return_value='exit')
@patch('pyllmsol.inference._base_cli.print')
def test_run_exit(mock_print, mock_input, cli):
    """Test that the run method exits gracefully."""
    cli.run(stream=False)
    mock_print.assert_called_with(
        (f"{strftime("%H:%M:%S")} | {cli.ai_name}: Goodbye {cli.user_name} ! I hope to see you soon !\n"),
        end='', flush=True
    )


def test_ask(cli):
    """Test that the ask method correctly formats the input."""
    question = "What is AI?"
    expected_prompt = f"Hello!\n{cli.user_name}: {question}\n{cli.ai_name}: "
    cli.ask(question)
    assert str(cli.prompt_hist) == expected_prompt


@patch('pyllmsol.inference._base_cli.sleep', return_value=None)
@patch('pyllmsol.inference._base_cli.print')
def test_stream(mock_print, mock_sleep, cli):
    """Test the _stream method."""
    text = "Hello"
    cli._stream(text)
    # Check the text was printed character by character
    mock_print.assert_any_call("H", end="", flush=True)
    mock_print.assert_any_call("e", end="", flush=True)
    mock_print.assert_any_call("l", end="", flush=True)
    mock_print.assert_any_call("o", end="", flush=True)


def test_set_init_prompt(cli):
    """Test that reset_prompt initializes with a valid tokenizer."""
    init_prompt = "Test prompt"
    cli.set_init_prompt(init_prompt)
    expected_tokens = [0] + [ord(char) for char in init_prompt]
    assert cli.init_prompt.tokens == expected_tokens


def test_reset_prompt(cli):
    """Test that reset_prompt initializes with a valid tokenizer."""
    init_prompt = str(cli.init_prompt)
    question = "What is AI?"
    expected_prompt = f"Hello!\n{cli.user_name}: {question}\n{cli.ai_name}: "
    cli.ask(question)
    assert str(cli.prompt_hist) == expected_prompt
    cli.reset_prompt()
    assert str(cli.prompt_hist) == init_prompt


if __name__ == "__main__":
    pass
