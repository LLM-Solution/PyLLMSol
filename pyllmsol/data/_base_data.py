#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-14 08:57:05
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 10:16:52
# @File path: ./pyllmsol/data/_base_data.py
# @Project: PyLLMSol

""" Base objects for text data to inferencing or training LLMs. """

# Built-in packages
from __future__ import annotations
from abc import ABC, abstractmethod
from json import loads
from pathlib import Path

# Third party packages
from llama_cpp import LlamaTokenizer
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol.data.utils import truncate_text

__all__ = []


TokenizerType = LlamaTokenizer | PreTrainedTokenizerBase


class _BaseData(ABC):
    """ Base class to handle initialization and processing of multiple item types.

    Parameters
    ----------
    items : list
        List of items to initialize.
    item_type : Type
        Primary type of items.
    fallback_type : Type
        Alternative type that can be converted to `item_type`.
    tokenizer : TokenizerType
        Tokenizer instance to apply to items if needed.

    Attributes
    ----------
    items : list
        List of processed items, all of type `item_type`.
    item_type : Type
        Primary type of items.
    fallback_type : Type
        Alternative type that can be converted to `item_type`.
    tokenizer : transformers.PreTrainedTokenizerBase or llama_cpp.LlamaTokenizer
        Tokenizer used for item conversion.

    Raises
    ------
    TypeError
        If an item is neither `item_type` nor `fallback_type`.

        """

    def __init__(
        self,
        items: list,
        item_type: type,
        fallback_type: type,
        tokenizer: TokenizerType,
    ):
        self.tokenizer = tokenizer
        self._item_type = item_type
        self._fallback_type = fallback_type
        self.items = [self._process_item(item) for item in items]

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
            return self._item_type(item, tokenizer=self.tokenizer)

        else:
            raise TypeError(f"Item must be either {self._item_type} or "
                            f"{self._fallback_type}, not {type(item)} ")

    def _append_item(self, item: object):
        """ Append a new item to the list, processing it if necessary.

        Parameters
        ----------
        item : object
            The item to append.

        Raises
        ------
        TypeError
            If `item` is not of type `_item_type` or `_fallback_type`.

        """
        self.items.append(self._process_item(item))

    def tokenize(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """ Get tokenized text.

        Parameters
        ----------
        text : str
            Text to tokenize.
        add_special_tokens : bool, optional
            If `True` add special tokens (e.g BOS), default is `False`.

        Returns
        -------
        list of str
            Sequence of token IDs.

        """
        if not self.tokenizer:
            raise ValueError("Tokenizer is not set. Please provide a valid "
                             "tokenizer.")

        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            return self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
            )

        elif isinstance(self.tokenizer, LlamaTokenizer):
            return self.tokenizer.encode(
                text,  # .encode('utf-8'),
                add_bos=add_special_tokens,
                special=True,
            )

        else:
            raise TypeError(f"Unsupported tokenizer type: {type(self.tokenizer)}. "
                            f"Expected {TokenizerType}.")

    @abstractmethod
    def add(self, item: object, inplace: bool = False):
        """ Add an object to _BaseData objects. """
        pass

    def __add__(self, item: object):
        """Combine this instance with another item to create a new instance.

        Uses the `add` method to perform non-in-place addition, creating and
        returning a new instance that combines this instance with `item`.

        Parameters
        ----------
        item : object
            The item to add to this instance. Must be compatible with the
            instance type or a fallback type that can be processed into the
            instance type.

        Returns
        -------
        _BaseData or subclass instance
            A new instance containing the combined items of `self` and `item`.

        """
        return self.add(item, inplace=False)

    def __iadd__(self, item: object):
        """Add an item to this instance in place.

        Uses the `add` method to perform in-place addition, modifying this
        instance by combining it with `item`.

        Parameters
        ----------
        item : object
            The item to add to this instance. Must be compatible with the
            instance type or a fallback type that can be processed into the
            instance type.

        Returns
        -------
        _BaseData or subclass instance
            The modified instance with the added item, allowing for chainable
            operations with `+=`.

        """
        return self.add(item,  inplace=True)

    def __len__(self):
        return len(self.items)

    def to_json(self) -> list:
        """ Convert object to JSON format. """
        return [item.to_json() for item in self.items]


class _TextData(_BaseData):
    """ A base class for handling and tokenizing text data.

    Parameters
    ----------
    text : str
        The text to be stored and formatted for inference or training.
    tokenizer : TokenizerType, optional
        A tokenizer object for encoding the text (default is None).

    Methods
    -------
    from_textfile
    get_n_tokens
    tokenize

    Attributes
    ----------
    text : str
        The text to be stored and formatted for inference or training.
    tokenizer : transformer.PreTrainedTokenizerBase or llama_cpp.LlamaTokenizer
        Tokenizer object.

    Properties
    ----------
    tokens : list of int
    mask

    """

    def __init__(self, text, tokenizer: TokenizerType = None):
        self.text = text
        self.tokenizer = tokenizer

    @property
    def tokens(self) -> list[int]:
        """ Get token IDs of text data.

        Returns
        -------
        list of int
            Tokenized representation of `text`.

        """
        return self.tokenize(self.text, add_special_tokens=True)

    @property
    def mask(self) -> list[int]:
        """ Get attention mask for `tokens`.

        Returns
        -------
        list of int
            Mask attention for list `tokens`.

        """
        return [1 for _ in self.tokens]

    @classmethod
    def from_textfile(
        cls,
        path: Path,
        tokenizer: TokenizerType = None,
    ) -> '_TextData':
        """ Create an instance object from a text file.

        This method reads text from a specified file and initializes a
        `_TextData` object with it. Optionally, a tokenizer can be provided to
        process the text.

        Parameters
        ----------
        path : Path
            The file path of the text file to read.
        tokenizer : TokenizerType, optional
            A tokenizer object to be associated with the data for text
            processing (default is None).

        Returns
        -------
        _TextData
            A new instance of `_TextData` containing the text from the specified
            file.

        """
        with path.open('r') as f:
            text = f.read()

        return cls(text, tokenizer=tokenizer)

    def get_n_tokens(self) -> int:
        """ Compute the number of tokens in the text data.

        Returns
        -------
        int
            Number of tokens in the text data.

        """
        return len(self.tokens)

    def __str__(self) -> str:
        """ Get full text data. """
        return self.text

    def __repr__(self) -> str:
        """ Class representation. """
        truncated_text = truncate_text(self.text)

        return (f"{self.__class__.__name__}(text='{truncated_text}', length="
                f"{len(self.text)}, n_tokens={self.get_n_tokens()})")

    def __format__(self, format_spec) -> str:
        """ Format text data (truncated). """
        return format(truncate_text(self.text), format_spec)

    def __len__(self) -> int:
        """ Get number of caracters of text data. """
        return len(self.text)

    def add(self, item: str | '_TextData', inplace=False) -> '_TextData':
        """ Add an text to the current text collection.

        This method adds an item to the existing list of items, either by
        creating a new instance with the combined items (if `inplace=False`) or
        by modifying the current instance (if `inplace=True`). The item can be
        of the same type as the instance or a convertible fallback type, which
        will be processed into the expected type.

        Parameters
        ----------
        item : object
            The item to add. Must be of a compatible type, either the same a
            the instance type or a fallback type convertible to the instance
            type.
        inplace : bool, optional
            If `True`, the addition modifies the current instance (in-place).
            If `False` (default), the addition creates and returns a new
            instance with the combined items.

        Returns
        -------
        _BaseData or subclass instance
            A new instance with combined items if `inplace=False`. If
            `inplace=True`, returns `self` after modification.

        """
        if isinstance(item, str):
            text = item

        elif isinstance(item, self.__class__):
            text = item.text

        else:
            raise TypeError(f"Unsupported type for addition: {type(item)}. "
                            f"Expected 'str' or '_TextData'.")

        if inplace:
            self.text += text

            return self

        else:
            return self.__class__(self.text + text, tokenizer=self.tokenizer)

    def to_json(self):
        return {"text": self.text}


class _DataSet(_BaseData):
    """ Base Dataset class to manage data loading and batch iteration.

    This class allows for iterating through a dataset in customizable batch
    sizes. It includes an optional progress bar to track the current progress in
    the iteration and allows setting descriptions for the progress bar. This
    class includes methods for loading itemss from JSON or JSONL files.

    Parameters
    ----------
    items : list of _TextData
        The items to iterate over, typically loaded from JSON or JSONL.
    batch_size : int
        The size of each data batch, default is 1.
    start : int, optional
        The index to start iterating from, default is 0.
    end : int, optional
        The index to stop iterating, default is None, which iterates to the end.
    tokenizer : TokenizerType, optional
        Tokenizer object.

    Methods
    -------
    __iter__()
        Initializes the iterator and progress bar.
    __next__()
        Retrieves the next batch of data and updates the progress bar.
    set_description(text)
        Sets a description for the progress bar.
    remaining_data()
        Returns the remaining data that has not yet been iterated.
    from_json
        Load dataset from a JSON file.
    from_jsonl
        Load dataset from a JSONL file.

    Attributes
    -----------
    items : list of _TextData
        The data to iterate over.
    batch_size : int
        The number of items to return in each batch.
    start : int
        The index to start the iteration from.
    end : int
        The index to end the iteration at.
    i : int
        The current index in the dataset for the iterator.
    tokenizer : transformer.PreTrainedTokenizerBase or llama_cpp.LlamaTokenizer
        Tokenizer object.

    """

    def __init__(
        self,
        items: list[_TextData | str],
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
        tokenizer: TokenizerType = None,
    ):
        super(_DataSet, self).__init__(items, _TextData, str, tokenizer)
        self.batch_size = batch_size
        self.i = None
        self.pbar = None
        self._set_boundary(start, end=end)

    def _set_boundary(self, start: int, end: int = None):
        max_end = len(self.items)
        if start < 0 or start > max_end:
            raise IndexError(f"Start index {start} is out of bounds for "
                             f"data of size {max_end}")

        else:
            self.i = self.start = start

        if end is None:
            self.end = max_end

        elif end > max_end:
            raise IndexError(f"End index {end} is out of bounds for data "
                             f"of size {max_end}")

        elif start >= end:
            raise IndexError(f"End index {end} must be greater than start "
                             f"index {start}")

        else:
            self.end = end

    def __iter__(self):
        """ Initialize the iterator and progress bar.

        This method sets the starting index for the iteration and initializes
        a tqdm progress bar to track progress. Returns the instance itself to
        allow for batch iteration using the iterator protocol.

        Returns
        -------
        DataBrowser
            The instance itself, initialized for iteration.

        """
        self.i = self.start
        self.pbar = tqdm(total=self.end - self.start)

        return self

    def __next__(self) -> '_DataSet':
        """ Retrieve the next batch of data and update progress.

        This method retrieves a batch of data from the items, advancing the
        index by the batch size. It also updates the progress bar and raises
        `StopIteration` when the end of the items is reached.

        Returns
        -------
        _DataSet
            The next batch of data from the items.

        Raises
        ------
        StopIteration
            When the iteration reaches the end of the specified range.

        """
        if self.i >= self.end:
            self.pbar.close()

            raise StopIteration

        i = self.i
        j = min(i + self.batch_size, self.end)

        self.i = j

        self.pbar.update(j - i)

        return self.__class__(
            self.items[i: j],
            batch_size=self.batch_size,
            start=0,
            end=None,
            tokenizer=self.tokenizer,
        )

    def __getitem__(self, key) -> _TextData:
        return self.items[key]

    def set_description(self, text: str):
        """ Set a description for the progress bar.

        Parameters
        ----------
        text : str
            The description text to display alongside the progress bar.

        """
        self.pbar.set_description(text)

    def remaining_data(self) -> list[_TextData]:
        """ Return the data that has not yet been iterated.

        This method returns a list of data items from the current index `i` to
        the end index, representing the data that has not yet been retrieved in
        the iteration.

        Returns
        -------
        list of str
            The portion of the items that has not yet been iterated.

        """
        return self.__class__(
            self.items[self.i:],
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
        )

    @classmethod
    def from_json(
        cls,
        path: Path,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
        tokenizer: TokenizerType = None,
    ):
        """ Create a DataSet instance from a JSON file.

        This method reads data from a JSON file and initializes a `DataSet`
        instance with it. Each item in the JSON should represent a single
        element in the items (typically a list of dictionaries).

        Parameters
        ----------
        path : Path
            The file path of the JSON file to load.
        batch_size : int
            The size of each data batch for iteration, default is 1.
        start : int, optional
            The index to start iterating from, default is 0.
        end : int, optional
            The index to stop iterating, default is None, which iterates to the
            end of the items.
        tokenizer : TokenizerType, optional
            Tokenizer object.

        Returns
        -------
        DataSet
            A new instance of `DataSet` containing the data loaded from the
            specified JSON file.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        JSONDecodeError
            Raised if the JSON file is not in a valid format.

        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        with path.open("r", encoding='utf-8') as f:
            items = loads(f.read())

        return cls(items, batch_size=batch_size, start=start, end=end,
                   tokenizer=tokenizer)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
        tokenizer: TokenizerType = None,
    ):
        """ Create a DataSet instance from a JSONL file.

        This method reads data from a JSON Lines (JSONL) file, where each line
        is a JSON object representing a single element in the items. It then
        initializes a `DataSet` instance with this data.

        Parameters
        ----------
        path : Path
            The file path of the JSONL file to load.
        batch_size : int
            The size of each data batch for iteration, default is 1.
        start : int, optional
            The index to start iterating from, default is 0.
        end : int, optional
            The index to stop iterating, default is None, which iterates to the
            end of the items.
        tokenizer : TokenizerType, optional
            Tokenizer object.

        Returns
        -------
        DataSet
            A new instance of `DataSet` containing the data loaded from the
            specified JSONL file.

        Raises
        ------
        FileNotFoundError
            If the specified file path does not exist.
        JSONDecodeError
            Raised if any line in the JSONL file is not a valid JSON object.

        """
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        items = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                data = loads(line.strip())
                items.append(data)

        return cls(items, batch_size=batch_size, start=start, end=end,
                   tokenizer=tokenizer)

    def __repr__(self):
        remaining = len(self.remaining_data())

        return (f"DataSet(length={len(self.items)}, start={self.start}, end="
                f"{self.end}, batch_size={self.batch_size}, remaining="
                f"{remaining})")

    def add(self, item: object, inplace: bool = False):
        """Add an item to the collection of items.

        This method adds an item to the existing list of items, either by
        creating a new instance with the combined items (if `inplace=False`) or
        by modifying the current instance (if `inplace=True`). The item can be
        of the same type as the instance or a convertible fallback type, which
        will be processed into the expected type.

        Parameters
        ----------
        item : object
            The item to add. Must be of a compatible type, either the same as
            the instance type or a fallback type convertible to the instance
            type.
        inplace : bool, optional
            If `True`, the addition modifies the current instance (in-place).
            If `False` (default), the addition creates and returns a new
            instance with the combined items.

        Returns
        -------
        _BaseData or subclass instance
            A new instance with combined items if `inplace=False`. If
            `inplace=True`, returns `self` after modification.

        """
        if isinstance(item, self.__class__):
            # If item is a DataSet object then add thus data to self data
            new_items = self.items + item.items

        else:
            # If item is not a DataSet try to process it (list of data or list
            # of list of str)
            new_items = self.items + [self._process_item(item)]

        if inplace:
            # Update self DataSet object
            self.items = new_items

            return self

        else:
            # Create a new DataSet object
            return self.__class__(
                new_items,
                batch_size=self.batch_size,
                tokenizer=self.tokenizer,
            )


if __name__ == "__main__":
    pass
