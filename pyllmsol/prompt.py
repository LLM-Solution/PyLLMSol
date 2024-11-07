#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 10:37:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-07 16:56:37

""" Prompt objects. """

# Built-in packages

# Third party packages
from transformers import AutoTokenizer

# Local packages

__all__ = []


class Prompt:
    """A class to handle and format prompt text.

    Parameters
    ----------
    text : str
        The prompt text to be stored and formatted for display.
    tokenizer : transformer.AutoTokenizer, optional
        Tokenizer object.

    Methods
    -------
    set_tokenizer
    tokenize

    Attributes
    ----------
    text : str
        The prompt text to be stored and formatted for display.
    tokenizer : transformer.AutoTokenizer
        Tokenizer object.

    """

    def __init__(self, text, tokenizer: AutoTokenizer = None):
        self.text = text
        self.tokenizer = tokenizer

    def tokenize(self):
        """ Get tokenized prompts. """
        return self.tokenizer(self.text)

    def set_tokenizer(self, tokenizer: AutoTokenizer):
        """ Set tokenizer object. """
        self.tokenizer = tokenizer

    def _truncate_text(self, text, max_length=50, front=20, back=20):
        """Truncate text if it exceeds max_length.

        Parameters
        ----------
        text : str
            The text to be truncated.
        max_length : int
            The maximum allowed length of the text before truncation.
        front : int
            The number of characters to keep from the start of the text.
        back : int
            The number of characters to keep from the end of the text.

        Returns
        -------
        str
            The truncated text if it exceeds `max_length`, otherwise the
            original text.

        """
        if len(text) > max_length:

            return f"{text[:front]}...{text[-back:]}"

        return text

    def __str__(self):
        return self._truncate_text(self.text)

    def __repr__(self):
        truncated_text = self._truncate_text(self.text)

        return (f"{self.__class__.__name__}(text='{truncated_text}', length="
                f"{len(self.text)})")

    def __format__(self, format_spec):
        return format(self.text, format_spec)

    def __len__(self):
        return len(self.text)

    def __add__(self, other: str):
        text = self.text + other

        return Prompt(text=text, tokenizer=self.tokenizer)

    def __iadd__(self, other: str):
        text = self.text + other

        return Prompt(text=text, tokenizer=self.tokenizer)


if __name__ == "__main__":
    pass
