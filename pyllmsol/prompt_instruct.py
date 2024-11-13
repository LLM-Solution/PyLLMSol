#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-09 15:41:38
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-13 20:17:38

""" Prompt objects for Llama 3.1 instruct format.

References
----------
https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

"""

# Built-in packages
from dataclasses import dataclass
from json import loads
from pathlib import Path

# Third party packages
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol.prompt import Prompt

__all__ = []

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"

INSTRUCT_ROLES = {
    "system": "<|start_header_id|>system<|end_header_id|>\n\n",
    "user": "<|start_header_id|>user<|end_header_id|>\n\n",
    "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

INSTRUCT_SEP = "<|eot_id|>"

REPR_ROLES = {
    "system": "",
    "user": "\nUser: ",
    "assistant": "\nAssistant: ",
}

REPR_SEP = ""


@dataclass(eq=False)
class MessageInstruct:
    role: str  # Must be one of {'user', 'assistant', 'system'}
    content: str

    def __init__(
        self,
        role: str,
        content: str,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.role = role
        self.content = content
        self.header = INSTRUCT_ROLES[self.role]
        self.footer = INSTRUCT_SEP

        self.set_tokens(tokenizer)
        self.set_mask()

    def set_tokens(self, tokenizer):
        kwargs = dict(add_special_tokens=False)
        self.content_tokens = tokenizer.encode(self.content, **kwargs)
        self.header_tokens = tokenizer.encode(self.header, **kwargs)
        self.footer_tokens = tokenizer.encode(self.footer, **kwargs)

    @property
    def text(self):
        if not self.content:
            return self.header

        return self.header + self.content + self.footer

    @property
    def tokens(self):
        return self.header_tokens + self.content_tokens + self.footer_tokens

    @property
    def mask(self):
        return self.header_mask + self.content_mask + self.footer_mask

    def set_mask(self, p=0.5):
        if self.role == "assistant":
            self.content_mask = [1 if random() > p else 0 for _ in self.content_tokens]

        else:
            self.content_mask = [1 for _ in self.content_tokens]

        self.header_mask = [1 for _ in self.header_tokens]
        self.footer_mask = [1 for _ in self.footer_tokens]

    def __str__(self):
        return self.header + self.content + self.footer

    def __repr__(self):
        return f"Messsage(from={self.role}, content={self.content})"

    def __format__(self):
        return f"{self.role}: {self.content}"

    def __getitem__(self, key):
        if key == "role":
            return self.role

        elif key == "content":
            return self.content

    def __contains__(self, key):
        return key in ['role', 'content']


def formater(
    messages: list[dict[str, str]],
    roles: dict[str, str],
    sep: str="",
    add_bos: bool = False,
    add_eos: bool = False,
):
    """ Format a chat conversation by combining roles and content.

    Parameters
    ----------
    messages : list of dict
        List of chat messages, where each message is a dictionary with the keys
        "role" and "content". Available roles are {"system", "user",
        "assistant"}.
    roles : dict
        Dictionary mapping each role ({"system", "user", "assistant"}) to its
        formatted string.
    sep : str, optional
        String separator to use between each formatted chat message, default is
        an empty string.
    add_bos, add_eos : bool, optional
        If `True` then add respectively begining and end of sentence tokens.
        Default is `False`. 

    Returns
    -------
    str
        The formatted chat conversation as a single string.

    Raises
    ------
    TypeError
        Raised if `messages` is not a list of dictionaries or if `sep` is not a
        string.
    KeyError
        Raised if a message's role is not in the provided `roles` dictionary.
    ValueError
        Raised if a message lacks the required "role" or "content" keys.

    """
    if not isinstance(messages, list):

        raise TypeError("`messages` should be a list of dictionaries with keys "
                        "'role' and 'content'.")

    if not isinstance(sep, str):

        raise TypeError("`sep` should be a string.")

    text = f"{BOS}" if add_bos else ""
    for message in messages:
        if not isinstance(message, dict) and not isinstance(message, MessageInstruct):

            raise TypeError("Each message should be a dictionary with 'role' "
                            "and 'content'.")

        if 'role' not in message or 'content' not in message:

            raise ValueError("Each message must contain 'role' and 'content' "
                             "keys.")

        role = message['role']

        if role not in roles:
            raise KeyError(f"Role '{role}' is not recognized. Available roles: "
                           f"{list(roles.keys())}.")

        content = message["content"]

        text += f"{roles[role]}{content}{sep}"

    text += f"{EOS}" if add_eos else ""

    return text


class PromptInstruct(Prompt):
    """ A class to handle and format chat messages for instruct-style prompts.

    This class is an extension of `Prompt` designed to manage chat-style prompts
    in a format compatible with models like LLaMA 3.1. It supports multiple
    message roles and allows formatting of conversations with role-specific
    identifiers.

    Parameters
    ----------
    messages : list of dict
        List of chat messages, where each message is a dictionary with "role"
        and "content" keys. Available roles are {"system", "user", "assistant"}.
    tokenizer : transformers.PreTrainedTokenizerBase, optional
        Tokenizer instance from Hugging Face's transformers library, used to
        manage tokenization of messages.

    Attributes
    ----------
    messages : list of dict
        Chat messages stored in a list, where each message is a dictionary with
        "role" and "content" keys.
    tokenizer : transformers.PreTrainedTokenizerBase
        Tokenizer instance for managing tokenization tasks.

    Properties
    ----------
    text : str
        Returns the formatted conversation as a single string, based on the 
        roles and separator defined.

    Methods
    -------
    from_json
        Creates a `PromptInstruct` instance from a JSON file.
    from_jsonl
        Creates a `PromptInstruct` instance from a JSON Lines file.
    get_n_tokens
        Retrieves the number of tokens in the formatted chat.
    set_tokenizer
        Sets a tokenizer for processing chat messages.
    tokenize
        Tokenizes the formatted chat message.
    append
        Adds a new message to the conversation.
    repr
        Returns a string representation of the conversation.
    __setitem__
        Add a message to the chat using dictionary-like syntax.
    __getitem__
        Get the chat text with the specified role appended for completion.

    Raises
    ------
    TypeError
        Raised if `messages` is not a list of dictionaries or if a message is
        not properly structured.
    ValueError
        Raised if a message is missing required keys ("role" and "content").
    KeyError
        Raised if an invalid role is provided in the messages or for completion.

    Examples
    --------
    >>> prompt = PromptInstruct.from_json(Path('conversation.json'))
    >>> print(prompt.text)
    "system: Initialize. user: How are you? assistant: I am fine."

    """

    def __init__(
        self,
        messages: list[dict[str, str]] | MessageInstruct,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        if all(isinstance(m, MessageInstruct) for m in messages):
            pass

        elif not isinstance(messages, list) or not all(
            isinstance(m, dict) and 'role' in m and 'content' in m
            for m in messages
        ):

            raise TypeError("`messages` must be a list of dictionaries with "
                            "'role' and 'content' keys.")

        self.messages = messages
        self.tokenizer = tokenizer

    @classmethod
    def from_json(cls, path: Path, tokenizer: PreTrainedTokenizerBase = None):
        """ Create a Prompt instance from a JSON file.

        This method reads JSON from a specified file and initializes a `Prompt`
        object with it. Optionally, a tokenizer can be provided to process the 
        text.

        Parameters
        ----------
        path : Path
            The file path of the JSON file to read.
        tokenizer : transformers.PreTrainedTokenizerBase, optional
            A tokenizer object to be associated with the prompt for text
            processing (default is None).

        Returns
        -------
        Prompt
            A new instance of `Prompt` containing the messages from the
            specified file.

        """
        with path.open("r", encoding='utf-8') as f:
            messages = loads(f.read())

        return cls(messages, tokenizer=tokenizer)

    @classmethod
    def from_jsonl(cls, path: Path, tokenizer: PreTrainedTokenizerBase = None):
        """ Create a Prompt instance from a JSONL file.

        This method reads a JSON Lines (.jsonl) file, where each line is a JSON
        object representing a chat message, and initializes a `PromptInstruct`
        object with it. Optionally, a tokenizer can be provided.

        Parameters
        ----------
        path : Path
            The file path of the JSONL file to read.
        tokenizer : transformers.PreTrainedTokenizerBase, optional
            A tokenizer object to be associated with the prompt for text
            processing (default is None).

        Returns
        -------
        PromptInstruct
            A new instance of `PromptInstruct` containing the messages from
            the specified JSONL file.

        Raises
        ------
        ValueError
            Raised if a line in the JSONL file does not contain "role" and
            "content" keys.
        JSONDecodeError
            Raised if a line is not a valid JSON object.

        Examples
        --------
        >>> prompt = PromptInstruct.from_jsonl(Path('conversation.jsonl'))
        >>> print(prompt.text)

        """
        messages = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                message = loads(line.strip())
                # Validate that each message contains "role" and "content" keys
                if not isinstance(message, dict):

                    raise ValueError("Each line must be a dict JSON object")

                elif 'role' not in message or 'content' not in message:

                    raise ValueError("Each line must be a JSON object with "
                                     "'role' and 'content' keys.")

                messages.append(message)

        return cls(messages, tokenizer=tokenizer)

    @property
    def text(self):
        # return formater(self.messages, roles=INSTRUCT_ROLES, sep=INSTRUCT_SEP)
        return "".join(m.text for m in self.messages)

    @property
    def tokens(self):
        if not self.tokenizer:

            return None

        return self._tokenize(self.text, add_special_tokens=True)

    def set_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        """ Set tokenizer object.

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizerBase, optional
            A tokenizer object to be associated with the prompt for text
            processing.

        """
        self.tokenizer = tokenizer

        for message in self.messages:
            content = message['content']
            tokens = self._tokenize(content, add_special_tokens=False)
            message.update(tokens=tokens)

    def append(self, message: dict[str, str]):
        """ Append a new message to the conversation.

        Parameters
        ----------
        message : dict
            A dictionary representing a chat message with required keys "role"
            and "content". Available roles are {"system", "user", "assistant"}.

        Raises
        ------
        ValueError
            Raised if `message` is not a dictionary with both "role" and
            "content" keys.

        """
        if not isinstance(message, dict):

            raise ValueError("`message` must be a dictionary with 'role' and "
                             "'content' keys.")

        elif 'role' not in message or 'content' not in message:

            raise ValueError("`message` must be a dictionary with 'role' and "
                             "'content' keys.")

        # Check if tokenizer is set
        if self.tokenizer:
            # Then compute tokenize new messages
            content = message['content']
            tokens = self._tokenize(content, add_special_tokens=False)
            message.update(tokens=tokens)

        self.messages.append(message)

    def __add__(self, other: dict[str: str] | list[dict[str, str]]) -> 'PromptInstruct':
        messages = self.messages.copy()

        if isinstance(other, list):
            for arg in other:
                if isinstance(arg, MessageInstruct):
                    messages.append(arg)

                else:
                    messages.append(MessageInstruct(**arg, tokenizer=self.tokenizer))

        elif isinstance(other, MessageInstruct):
            messages.append(other)

        else:
            messages.append(MessageInstruct(**other, tokenizer=self.tokenizer))

        return self.__class__(messages, tokenizer=self.tokenizer)

    def __iadd__(self, other: dict[str, str]) -> 'PromptInstruct':
        self.append(other)

        return self

    def repr(self):
        """ Returns a string representation of the conversation.

        Returns
        -------
        str
            Formated conversation.

        """
        return formater(self.messages, roles=REPR_ROLES, sep=REPR_SEP)

    def __setitem__(self, role: str, content: str):
        """ Add a message to the chat using dictionary-like syntax.

        Parameters
        ----------
        role : str
            Role of the message, available roles are {"system", "user",
            "assistant"}.
        content : str
            Content of the message.
        
        Raises
        ------
        KeyError
            If the role is not in the predefined roles.

        """
        if role not in INSTRUCT_ROLES:

            raise KeyError(f"Role '{role}' is not recognized. Available roles: "
                           f"{list(INSTRUCT_ROLES.keys())}.")

        self.append({"role": role, "content": content})

    def __getitem__(self, role: str) -> str:
        """ Get the chat text with the specified role appended for completion.

        Parameters
        ----------
        role : str
            Role to be added at the end, typically "assistant" for LLM
            completion.

        Returns
        -------
        str
            Formatted text with all messages and a placeholder for the specified
            role.

        Raises
        ------
        KeyError
            If the role is not in the predefined roles.

        """
        if role not in INSTRUCT_ROLES:
            raise KeyError(f"Role '{role}' is not recognized. Available roles: "
                           f"{list(INSTRUCT_ROLES.keys())}.")
        
        # Formatted text with existing messages
        formatted_text = self.text

        # Add the final role header as a placeholder for LLM completion
        formatted_text += f"{INSTRUCT_ROLES[role]}"

        return Prompt(formatted_text)


if __name__ == "__main__":
    pass
