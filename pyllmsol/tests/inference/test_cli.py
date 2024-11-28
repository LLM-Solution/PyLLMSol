#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 09:47:15
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-28 16:30:33
# @File path: ./pyllmsol/tests/inference/test_cli.py
# @Project: PyLLMSol

""" Test CLI object. """

# Built-in packages
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.mock import MockLlama
from pyllmsol.data.prompt import Prompt
from pyllmsol.inference.cli import CommandLineInterface


__all__ = []


@pytest.fixture
def cli():
    return CommandLineInterface(
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
    assert isinstance(cli.init_prompt, Prompt)
    assert isinstance(cli.prompt_hist, Prompt)



if __name__ == "__main__":
    pass