#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 08:57:28
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 12:37:27
# @File path: ./pyllmsol/data/chat.py
# @Project: PyLLMSol

""" Chat data objects for dialogue data to inferencing or training LLMs.

This module use Llama 3.1 instruct format for chat dialogue.

References
----------
https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

"""

# Built-in packages
from __future__ import annotations
from pathlib import Path
from json import loads

# Third party packages
from llama_cpp import LlamaTokenizer
import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol.data._base_data import _BaseData, _TextData, _DataSet

__all__ = []


TokenizerType = LlamaTokenizer | PreTrainedTokenizerBase

BOS = "<|begin_of_text|>"
EOS = "<|end_of_text|>"

ROLES = {
    "system": "<|start_header_id|>system<|end_header_id|>\n\n",
    "user": "<|start_header_id|>user<|end_header_id|>\n\n",
    "assistant": "<|start_header_id|>assistant<|end_header_id|>\n\n",
}

SEP = "<|eot_id|>"

REPR_ROLES = {
    "system": "",
    "user": "\nUser: ",
    "assistant": "\nAssistant: ",
}

REPR_SEP = ""


class Message(_BaseData):
    """ Handles individual chat messages with specific roles and tokenization.

    This class formats and tokenizes chat messages for LLM inference or
    training, using the Llama 3.1 instruct format. It provides properties for
    generating the full message text, token IDs, and attention masks, and
    includes custom methods for accessing or displaying message attributes.

    Parameters
    ----------
    role : str
        Role of the message, expected to be one of {'system', 'user',
        'assistant'}.
    content : str
        Content of the message.
    tokenizer : TokenizerType
        Tokenizer object for processing text content.

    Methods
    -------
    __str__
    __repr__
    __format__

    Attributes
    ----------
    role : str
        Role of the message.
    content : str
        The message content.
    header : str
        Header formatted according to the role.
    footer : str
        Footer separator.

    Properties
    ----------
    text : str
        Full text of the message, including header and footer.
    tokens : list[int]
        Tokenized representation of `text`.
    mask : list
        Mask for `tokens`, masking content tokens if role is `assistant`.

    Raises
    ------
    ValueError
        If an invalid role is provided.
    KeyError
        If `__getitem__` is called with an invalid key.

    Examples
    --------
    >>> tokenizer = MockTokenizer()
    >>> msg = Message(role='user', content='Hello!', tokenizer=tokenizer)
    >>> str(msg)
    '<|start_header_id|>user<|end_header_id|>\n\nHello!<|eot_id|>'
    >>> repr(msg)
    Messsage(from=user, content=Hello!)
    >>> msg['role']
    'user'
    >>> msg = Message(role='assistant', content='Hello World !', tokenizer=tokenizer)
    >>> msg.tokens
    [128006, 78191, 128007, 271, 9906, 4435, 758, 128009]
    >>> msg.mask
    [1, 1, 1, 1, 0, 0, 0, 0]

    """

    def __init__(
        self,
        role: str,
        content: str,
        tokenizer: TokenizerType,
        **kwargs,
    ):
        if role not in ROLES:
            raise ValueError(f"Invalid role '{role}'. Expected one of "
                             f"{list(ROLES.keys())}.")

        self.tokenizer = tokenizer
        self.role = role
        self.content = content
        self.header = ROLES[self.role]
        self.footer = SEP
        self.metadata = kwargs

    @property
    def text(self):
        """ Construct the full text of the message, including header and footer.

        If `content` is empty, returns only the header.

        Returns
        -------
        str
            Formatted text of the message.

        """
        if not self.content:

            return self.header

        return self.header + self.content + self.footer

    @property
    def tokens(self) -> list[int]:
        """ Get token IDs of the full message text.

        Uses the tokenizer to convert the `text` property into a sequence of
        token IDs.

        Returns
        -------
        list of int
            Tokenized representation of `text`.

        """
        return self.tokenize(self.text, add_special_tokens=False)

    @property
    def mask(self) -> list[int]:
        """ Generate an attention mask for tokens.

        The mask is a list of 1s with the same length as `tokens`. If the role
        is 'assistant', content tokens are masked by setting their values to 0.

        Returns
        -------
        list of int
            Attention mask for list `tokens`.

        """
        mask = [1 for _ in self.tokens]

        if self.role == "assistant":
            header_tokens = self.tokenize(self.header, add_special_tokens=False)
            mask[len(header_tokens): ] = [0] * (len(mask) - len(header_tokens))

        return mask

    def __str__(self) -> str:
        """ Return the full formatted text of the message.

        Returns
        -------
        str
            Formated text of Message object.

        """
        return self.text

    def __repr__(self) -> str:
        """ Provide a concise string representation of the Message object.

        Useful for debugging and logging purposes.

        Returns
        -------
        str
            A string representation of the Message object, including role and
            content.

        """
        return f"Message(from={self.role}, content={self.content})"

    def __format__(self, format_spec) -> str:
        return format(f"{self.role}: {self.content}", format_spec)

    def __getitem__(self, key: str) -> str:
        if key == "role":
            return self.role

        elif key == "content":
            return self.content

        else:
            raise KeyError(f"'{key}' is not a valid key. Use 'role' or "
                           f"'content'.")

    def __contains__(self, key: str) -> bool:
        return key in ['role', 'content']

    def add(self, content: str, inplace: bool = False) -> 'Message':
        """ Add text to content.

        Parameters
        ----------
        content : str
            Text to add to the current content.
        inplace : bool, optional
            Add text to the current object if True, otherwise return a new
            object with the current content plus the new one (default).

        Returns
        -------
        Message
            Message with incremented content.

        """
        if not isinstance(content, str):
            raise TypeError("Message addition support only string.")

        if inplace:
            self.content += content

            return self

        else:
            return Message(
                role=self.role,
                content=self.content + content,
                tokenizer=self.tokenizer,
            )

    def to_json(self):
        return {'role': self.role, 'content': self.content, **self.metadata}


class Chat(_TextData, _BaseData):
    """ Chat object that manages a sequence of messages within a conversation.

    The Chat class facilitates structured conversation data handling, providing
    methods for text and token concatenation, padding, masking, and message
    addition. It supports the Llama 3.1 instruct format, making it compatible
    with tokenized data for LLM models.

    Parameters
    ----------
    items : Message or list of dict
        A list of message items, each being either a dictionary with 'role'
        and 'content' keys or a `Message` instance.
    tokenizer : TokenizerType
        Tokenizer object for processing text content.

    Methods
    -------
    from_json
    from_jsonl
    pad
    append
    add

    Attributes
    ----------
    items : list of Message
        List of `Message` objects representing each message in the chat.
    tokenizer : TokenizerType
        Tokenizer used to tokenize the text of each message.

    Properties
    ----------
    text : str
        Property to get the full concatenated text of all messages.
    tokens : list[int]
        Property to get tokenized representation of the full chat text.
    mask : list[int]
        Property to get the full attention mask for the chat conversation.

    Raises
    ------
    FileNotFoundError
        If the specified JSON or JSONL file is not found in from_json or from_jsonl.
    ValueError
        If `total_tokens` is less than the current length of `tokens`.
    KeyError
        If `__setitem__` or `__getitem__` is called with an unrecognized role.

    """

    def __init__(
        self,
        items: list[dict[str, str] | Message],
        tokenizer: TokenizerType,
    ):
        _BaseData.__init__(self, items, Message, dict, tokenizer)

    def _process_item(self, item: object) -> object:
        """ Process an item into the expected type if necessary.

        Parameters
        ----------
        item : object
            The item to process.

        Returns
        -------
        object
            An instance of `_item_type`.

        Raises
        ------
        TypeError
            If `item` is not of type `_item_type` or `_fallback_type`.

        """
        if isinstance(item, self._item_type):
            return item

        elif isinstance(item, self._fallback_type):
            return self._item_type(**item, tokenizer=self.tokenizer)

        else:
            raise TypeError(f"Item must be either {self._item_type} or "
                            f"{self._fallback_type}, not {type(item)} ")

    @classmethod
    def from_json(cls, path: Path, tokenizer: PreTrainedTokenizerBase):
        """ Create a Chat instance from a JSON file.

        This method reads JSON from a specified file and initializes a `Chat`
        object with it. Optionally, a tokenizer can be provided to process the 
        text.

        Parameters
        ----------
        path : Path
            The file path of the JSON file to read.
        tokenizer : TokenizerType
            A tokenizer object to be associated with the prompt for text
            processing.

        Returns
        -------
        Chat
            A new instance of `Chat` containing the messages from the
            specified file.

        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding='utf-8') as f:
            items = loads(f.read())

        return cls(items, tokenizer=tokenizer)

    @classmethod
    def from_jsonl(cls, path: Path, tokenizer: PreTrainedTokenizerBase):
        """ Create a Chat instance from a JSONL file.

        This method reads a JSON Lines (.jsonl) file, where each line is a JSON
        object representing a chat message, and initializes a `Chat` object with
        it.

        Parameters
        ----------
        path : Path
            The file path of the JSONL file to read.
        tokenizer : TokenizerType
            A tokenizer object to be associated with the prompt for text
            processing.

        Returns
        -------
        Chat
            A new instance of `Chat` containing the messages from the specified
            JSONL file.

        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open('r', encoding='utf-8') as f:
            items = [loads(line.strip()) for line in f]

        return cls(items, tokenizer=tokenizer)

    @property
    def text(self):
        """ Concatenate the full text of all messages in the chat.

        Returns
        -------
        str
            A string with the concatenated text of all messages.

        """
        return "".join(message.text for message in self.items)

    @property
    def tokens(self):
        """ Tokenize the concatenated chat text.

        Returns
        -------
        list of int
            Token IDs representing the full chat text.

        """
        return self.tokenize(self.text, add_special_tokens=True)

    @property
    def mask(self):
        """ Generate a full attention mask for the chat conversation.

        Combines the masks of individual messages to create a mask for the
        entire conversation.

        Returns
        -------
        list of int
            Concatenated attention mask for the conversation.

        """
        mask = [1]  # for BOS token added
        mask += [token for message in self.items for token in message.mask]

        return mask

    def pad(self, total_tokens: int):
        """ Pad the `tokens` and `mask` attributes to a specified length.

        Pads the `tokens` and `mask` to `total_tokens` by adding the pad token
        ID and mask values. Raises an error if `total_tokens` is less than the
        current length of `tokens`.

        Parameters
        ----------
        total_tokens : int
            Total number of tokens after padding.

        Returns
        -------
        list of int
            Extended with PAD tokens list of tokens.
        list of int
            Extended with 0 list of mask.

        Raises
        ------
        ValueError
            If `total_tokens` is less than the current token length.

        """
        n = total_tokens - self.get_n_tokens()

        if n < 0:
            raise ValueError(f"`total_tokens` ({total_tokens}) must be greater "
                             f"than or equal to the current size of "
                             f"`self.tokens` ({len(self.tokens)}).")

        tokens = self.tokens + [self.tokenizer.pad_token_id] * n
        mask = self.mask + [0] * n

        return tokens, mask

    def append(self, message: Message | dict) -> "Chat":
        """Append a message to the chat.

        Parameters
        ----------
        message : Message or dict
            A `Message` instance or dictionary with 'role' and 'content' keys to 
            add to the conversation.

        Returns
        -------
        Chat
            Self object.

        """
        return self.add(message, inplace=True)

    def add(self, item: '_TextData' | 'Message' | dict, inplace: bool = False):
        """ Add a new message or chat data to the conversation.

        This method adds an item to the chat. The item can be another `Chat`
        instance, a `Message` object, or a dictionary with 'role' and 'content'
        keys.

        Parameters
        ----------
        item : Chat or Message or dict
            The new chat message or data to add to the conversation.
        inplace : bool, optional
            If `True`, modifies the current instance. If `False`, returns a 
            new `Chat` instance with the added item. Default is `False`.

        Returns
        -------
        Chat
            The modified chat instance (self if `inplace=True`, otherwise a new
            instance).

        Examples
        --------
        >>> items = [{"role": "user", "content": "New message"}]
        >>> chat = Chat(items, tokenizer=tokenizer)
        >>> chat.add(
        ...     Message(role="assistant", content="Rep", tokenizer=tokenizer),
        ...     inplace=True,
        ... )

        """
        if isinstance(item, self.__class__):
            # If item is a Chat object then add thus messages to self messages
            new_messages = self.items + item.items

        else:
            # If item is not a Chat try to process it (list of message or list
            # of sting)
            new_messages = self.items + [self._process_item(item)]

        if inplace:
            # Update self Chat object
            self.items = new_messages

            return self

        else:
            # Create a new Chat object
            return self.__class__(
                new_messages,
                tokenizer=self.tokenizer,
            )

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
        if role not in ROLES:

            raise KeyError(f"Role '{role}' is not recognized. Available roles: "
                           f"{list(ROLES.keys())}.")

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
        if role not in ROLES:
            raise KeyError(f"Role '{role}' is not recognized. Available roles: "
                           f"{list(ROLES.keys())}.")

        new_message = Message(role=role, content='', tokenizer=self.tokenizer)

        # return self.append(new_message)
        return self.add(new_message, inplace=False)


class ChatDataSet(_DataSet, _BaseData):
    """ A DataSet class for managing and processing a collection of chat data.

    The DataSet class provides utilities for loading, batching, padding, and
    tokenizing chat data. It is designed for use in training and inference
    tasks with language models. This class extends `_DataSet`, enabling
    efficient data handling and iteration.

    Parameters
    ----------
    items : list of Chat or list of list of Message
        A collection of chats or nested lists. Each element can be:
        - A `Chat` object.
        - A list of `Message` objects.
        - A list of dictionaries, where each dictionary represents a message 
          with keys 'role' and 'content'.
    tokenizer : TokenizerType
        Tokenizer object for processing text content in the dataset.
    batch_size : int, optional
        The number of items to include in each batch during iteration (default
        is 1).
    start : int, optional
        The index to start iteration from (default is 0).
    end : int, optional
        The index to stop iteration (default is None, meaning until the end).

    Methods
    -------
    __iter__
    __next__
    get_padded
    from_json
    from_jsonl
    set_description
    remaining_data

    Attributes
    ----------
    items : list of Chat
        The collection of chats in the dataset, converted to `Chat` objects 
        if not already.
    batch_size : int
        The number of items in each batch during iteration.
    start : int
        The starting index for iteration.
    end : int
        The ending index for iteration.
    tokenizer : TokenizerType
        Tokenizer object for processing text data.

    Properties
    ----------
    text : str
        Full concatenated text of all items in the dataset.
    tokens : list of int
        Tokenized representation of the dataset's text.
    mask : list of int
        Attention mask for the dataset.

    Raises
    ------
    FileNotFoundError
        If the specified JSON or JSONL file does not exist.
    ValueError
        If batch size, start, or end values are invalid.
    TypeError
        If items in the dataset are of incompatible types.

    Examples
    --------
    >>> tokenizer = MockTokenizer()
    >>> path = Path('root/data.json')
    >>> dataset = DataSet.from_json(
    ...     path=path,
    ...     batch_size=2,
    ...     tokenizer=tokenizer,
    ... )
    >>> padded_tokens, padded_mask = dataset.get_padded()
    >>> for batch in dataset:
    ...     print(batch)

    """

    def __init__(
        self,
        items: list[Chat | list[Message | dict[str, str]]],
        tokenizer: TokenizerType,
        batch_size: int = 1,
    ):
        _BaseData.__init__(self, items, Chat, list, tokenizer)
        self.batch_size = batch_size
        self.set_boundary(0, end=None)

    def get_padded(
        self,
        return_tensor: bool = False,
    ) -> list[list[int]] | Tensor:
        """ Pad all chats in the dataset to the same length.

        Pads each chat in the dataset to the length of the longest chat,
        creating uniform-sized token and mask arrays for each chat.

        Parameters
        ----------
        return_tensor : bool, optional
            If True, returns data as `torch.Tensor` objects; otherwise, returns
            lists.

        Returns
        -------
        list of list of int or torch.tensor
            Extended with PAD tokens list of tokens.
        list of list of int or torch.tensor
            Extended with 0 list of mask.

        """
        max_tokens = max(chat.get_n_tokens() for chat in self.items)

        tokens, mask = [], []
        for chat in self.items:
            _tokens, _mask = chat.pad(max_tokens)
            tokens.append(_tokens)
            mask.append(_mask)

        if return_tensor:
            tokens = torch.tensor(tokens)
            mask = torch.tensor(mask)

        return tokens, mask


if __name__ == "__main__":
    pass
