#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 09:47:15
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 17:01:01
# @File path: ./pyllmsol/tests/inference/test_cli.py
# @Project: PyLLMSol

""" Test CLI object. """

# Built-in packages
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.tests.mock import MockLlama
from pyllmsol.data.prompt import Prompt
from pyllmsol.inference.cli import CommandLineInterface


__all__ = []


@pytest.fixture
def cli():
    return CommandLineInterface(
        llm=MockLlama(),
        init_prompt="Prompt: You are friendly.\nUser: Hello!",
        verbose=True
    )


def test_initialization(cli):
    """Test that the CLI initializes correctly."""
    assert isinstance(cli.init_prompt, Prompt)
    assert isinstance(cli.prompt_hist, Prompt)


def test_check_prompt_limit_context(cli):
    with pytest.raises(NotImplementedError):
        cli._check_prompt_limit_context()



if __name__ == "__main__":
    pass