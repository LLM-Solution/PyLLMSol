#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 11:01:07
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-14 11:11:45

""" Test data/utils.py script. """

# Built-in packages

# Third party packages

# Local packages
from pyllmsol.data.utils import truncate_text

__all__ = []


def test_truncate_text_function():
    assert truncate_text("Short text", max_length=20) == "Short text"
    long_text = "This is a much longer text that needs truncation"
    truncated_text = truncate_text(long_text, max_length=20, front=10, back=5)
    assert truncated_text == "This is a ...ation"

if __name__ == "__main__":
    pass
