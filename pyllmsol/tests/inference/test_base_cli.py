#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-26 17:39:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 16:53:03
# @File path: ./pyllmsol/tests/inference/test_base_cli.py
# @Project: PyLLMSol

""" Test base CLI object. """

# Built-in packages
from io import StringIO
from pathlib import Path
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


@pytest.fixture
def mock_llama_class():
    """Fixture to mock the Llama class."""
    with patch('pyllmsol.inference._base_cli.Llama', MockLlama):
        yield MockLlama


def test_initialization(cli):
    """Test that the CLI initializes correctly."""
    assert cli.ai_name == "Assistant"
    assert cli.user_name == "User"
    assert cli.llm is not None
    assert cli.stop == [f"\n{cli.user_name}:", f"\n{cli.ai_name}:"]
    assert isinstance(cli.init_prompt, _TextData)
    assert isinstance(cli.prompt_hist, _TextData)


def test_from_path(mock_llama_class):
    """Test the from_path method of _BaseCommandLineInterface."""
    # Mock parameters
    model_path = Path("/mock/model/path")
    init_prompt = "Hello, I am your assistant."
    verbose = True

    # Call from_path
    cli_instance = _BaseCommandLineInterface.from_path(
        model_path=model_path,
        init_prompt=init_prompt,
        verbose=verbose,
        n_ctx=512,  # Additional kwargs
    )

    # Assertions
    assert isinstance(cli_instance, _BaseCommandLineInterface)
    assert cli_instance.llm._n_ctx == 512  # Check additional arguments passed correctly
    assert cli_instance.init_prompt.text == init_prompt  # Ensure prompt is set
    assert cli_instance.verbose == verbose


def test_run(cli, monkeypatch):
    """Test the run method of _BaseCommandLineInterface."""
    # Mock user inputs
    inputs = iter(["User question?", "exit"])
    monkeypatch.setattr('builtins.input', lambda _: next(inputs))

    # Capture printed output
    with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
        # Run the method
        cli.run(stream=True)

        # Retrieve printed output
        output = mock_stdout.getvalue()

    # Assertions
    assert "Welcome, I am Assistant your custom AI" in output
    assert "User question?" in cli.prompt_hist.text
    assert "LLM response." in output
    assert "Goodbye User !" in output


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
