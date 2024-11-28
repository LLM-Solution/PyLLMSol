#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 10:59:04
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-28 09:12:13
# @File path: ./pyllmsol/data/cli_instruct.py
# @Project: PyLLMSol

""" Util function for data objects. """

# Built-in packages

# Third party packages

# Local packages

__all__ = []


def truncate_text(text, max_length=50, front=20, back=20):
    """ Truncate text if it exceeds max_length.

    Parameters
    ----------
    text : str
        The input text to truncate.
    max_length : int, optional
        Maximum allowed length of the text; truncation occurs if length exceeds
        this value (default is 50).
    front : int, optional
        Number of characters to retain from the start of the text (default is
        20).
    back : int, optional
        Number of characters to retain from the end of the text (default is 20).

    Returns
    -------
    str
        The truncated text if `max_length` is exceeded; otherwise, the original
        text.

    """
    if len(text) > max_length:

        return f"{text[:front]}...{text[-back:]}"

    return text


if __name__ == "__main__":
    pass
