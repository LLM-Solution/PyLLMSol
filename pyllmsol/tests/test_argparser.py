#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-11 17:16:58
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 17:25:10
# @File path: ./pyllmsol/tests/test_argparser.py
# @Project: PyLLMSol

""" Test ArgParser objects. """

# Built-in packages
from os import cpu_count

# Third party packages
import pytest

# Local packages
from pyllmsol.argparser import _BasisArgParser, CLIArgParser

__all__ = []


def test_basis_argparser_str():
    """Test the string representation of _BasisArgParser."""
    parser = _BasisArgParser(description="Test parser", file="test_script.py")
    parser.add_argument("--arg1", type=str, default="value1")
    parser.add_argument("--arg2", type=int, default=42)

    # Mock command-line arguments
    test_args = ["--arg1", "test_value", "--arg2", "100"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", ["test_script.py"] + test_args)
        output = str(parser)

    expected_output = (
        "\nRun test_script.py\n"
        "arg1                 = test_value\n"
        "arg2                 = 100"
    )

    assert output.strip() == expected_output.strip()


def test_cliargparser_defaults():
    """Test CLIArgParser with default values."""
    parser = CLIArgParser(file="cli_script.py")
    args = parser()  # No arguments passed, so defaults should be used

    n_threads_default = max(1, cpu_count() - 1)
    assert args.n_ctx == 32768
    assert args.n_threads == n_threads_default
    assert args.model_path is None
    assert args.init_prompt is None
    # assert args.verbose is False
    assert args.lora_path is None


def test_cliargparser_custom_values():
    """Test CLIArgParser with custom arguments."""
    parser = CLIArgParser(file="cli_script.py")

    test_args = [
        "--model_path", "/path/to/model",
        "--init_prompt", "Hello, world!",
        "--verbose",
        "--lora_path", "/path/to/lora",
        "--n_ctx", "4096",
        "--n_threads", "4",
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", ["cli_script.py"] + test_args)
        args = parser()

    assert args.model_path == "/path/to/model"
    assert args.init_prompt == "Hello, world!"
    assert args.verbose is True
    assert args.lora_path == "/path/to/lora"
    assert args.n_ctx == 4096
    assert args.n_threads == 4


def test_cliargparser_invalid_args():
    """Test CLIArgParser with invalid arguments."""
    parser = CLIArgParser(file="cli_script.py")

    test_args = [
        "--n_ctx", "invalid",  # Non-integer value for an int argument
    ]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("sys.argv", ["cli_script.py"] + test_args)

        with pytest.raises(SystemExit):  # argparse raises SystemExit on error
            parser()


if __name__ == "__main__":
    pass
