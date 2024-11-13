#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 10:37:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-13 08:10:04

""" Prompt objects. """

# Built-in packages
from pathlib import Path

# Third party packages
from transformers import PreTrainedTokenizerBase

# Local packages

__all__ = []


def truncate_text(text, max_length=50, front=20, back=20):
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


class Prompt:
    """A class to handle and format prompt text.

    Parameters
    ----------
    text : str
        The prompt text to be stored and formatted for display.
    tokenizer : transformer.PreTrainedTokenizerBase, optional
        Tokenizer object.

    Methods
    -------
    from_text
    get_n_tokens
    set_tokenizer
    tokenize

    Attributes
    ----------
    text : str
        The prompt text to be stored and formatted for display.
    tokenizer : transformer.PreTrainedTokenizerBase
        Tokenizer object.

    """

    tokenizer = None
    tokens = None

    def __init__(self, text, tokenizer: PreTrainedTokenizerBase = None):
        self.text = text

        if tokenizer:
            self.set_tokenizer(tokenizer=tokenizer)

    @classmethod
    def from_text(
        cls,
        path: Path,
        tokenizer: PreTrainedTokenizerBase = None,
    ) -> 'Prompt':
        """ Create a Prompt instance from a text file.

        This method reads text from a specified file and initializes a `Prompt`
        object with it. Optionally, a tokenizer can be provided to process the 
        text.

        Parameters
        ----------
        path : Path
            The file path of the text file to read.
        tokenizer : transformers.PreTrainedTokenizerBase, optional
            A tokenizer object to be associated with the prompt for text
            processing (default is None).

        Returns
        -------
        Prompt
            A new instance of `Prompt` containing the text from the specified
            file.

        """
        with path.open('r') as f:
            text = f.read()

        return cls(text, tokenizer=tokenizer)

    def _tokenize(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[str]:
        """ Get tokenized prompts.

        Parameters
        ----------
        text : str
            Text to tokenize.
        add_special_tokens : bool, optional
            If `True` add special tokens (e.g BOS and EOS), default is `False`.

        Returns
        -------
        list of str
            Sequence of tokens.

        """
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            # Tokenizer from `transformers` library

            return self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )

        else:
            # Tokenizer from `llama_cpp` library

            return self.tokenizer(
                text.encode('utf-8'),
                add_special_tokens=add_special_tokens,
            )

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        """ Set tokenizer object.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizerBase, optional
            A tokenizer object to be associated with the prompt for text
            processing.

        """
        self.tokenizer = tokenizer
        self.tokens = self._tokenize(self.text, add_special_tokens=True)

    def get_n_tokens(self) -> int:
        """ Compute the number of tokens in the prompt.

        Returns
        -------
        int
            Number of tokens in the prompt text.

        """
        if self.tokenizer:

            return len(self.tokens)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        truncated_text = truncate_text(self.text)

        return (f"{self.__class__.__name__}(text='{truncated_text}', length="
                f"{len(self.text)}, n_tokens={self.get_n_tokens()})")

    def __format__(self, format_spec) -> str:
        return format(truncate_text(self.text), format_spec)

    def __len__(self) -> int:
        return len(self.text)

    def __add__(self, other: str) -> 'Prompt':
        text = self.text + other

        return Prompt(text=text, tokenizer=self.tokenizer)

    def __iadd__(self, other: str) -> 'Prompt':
        text = self.text + other

        return Prompt(text=text, tokenizer=self.tokenizer)


if __name__ == "__main__":
    pass
