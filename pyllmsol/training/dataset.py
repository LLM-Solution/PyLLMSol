#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-12 16:31:57
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-13 15:58:21

""" Dataset objects. """

# Built-in packages
from json import loads
from pathlib import Path

# Third party packages
from tqdm import tqdm

# Local packages

__all__ = []


class DataBrowser:
    """ Class to facilitate batch iteration over dataset with progress tracking.

    This class allows for iterating through a dataset in customizable batch
    sizes. It includes an optional progress bar to track the current progress in
    the iteration and allows setting descriptions for the progress bar.

    Parameters
    ----------
    dataset : list
        The data to iterate over, typically a list of strings or dictionaries.
    batch_size : int, optional
        The size of each batch of data to be returned by the iterator, default
        is 1.
    start : int, optional
        The starting index for the iteration, default is 0.
    end : int, optional
        The ending index for the iteration, default is None, which means to
        iterate until the end of the dataset.

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

    Attrtibutes
    -----------
    dataset : list
        The data to iterate over.
    batch_size : int
        The number of items to return in each batch.
    start : int
        The index to start the iteration from.
    end : int
        The index to end the iteration at.
    i : int
        The current index in the dataset for the iterator.

    """

    def __init__(
        self,
        dataset: list,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._set_boundary(start, end=end)
        self.i = self.start

    def _set_boundary(self, start: int, end: int = None):
        T = len(self.dataset)
        if start < 0 or start > T:
            raise IndexError(f"Start index {start} is out of bounds for "
                             f"data of size {T}")

        else:
            self.start = start

        if end is None:
            self.end = T

        elif end > T:
            raise IndexError(f"End index {end} is out of bounds for data "
                             f"of size {T}")

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

    def __next__(self) -> list:
        """ Retrieve the next batch of data and update progress.

        This method retrieves a batch of data from the dataset, advancing the
        index by the batch size. It also updates the progress bar and raises
        `StopIteration` when the end of the dataset is reached.

        Returns
        -------
        list of str
            The next batch of data from the dataset.

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

        return self.dataset[i: j]

    def __getitem__(self, key):
        return self.dataset[key]

    def set_description(self, text: str):
        """ Set a description for the progress bar.

        Parameters
        ----------
        text : str
            The description text to display alongside the progress bar.

        """
        self.pbar.set_description(text)

    def remaining_data(self) -> list[str]:
        """ Return the data that has not yet been iterated.

        This method returns a list of data items from the current index `i` to
        the end index, representing the data that has not yet been retrieved in
        the iteration.

        Returns
        -------
        list of str
            The portion of the dataset that has not yet been iterated.

        """
        return self.dataset[self.i:]


class DataSet(DataBrowser):
    """ Dataset class to manage data loading and batch iteration.

    Inherits from `DataBrowser` to enable batch iteration through a dataset.
    This class includes methods for loading datasets from JSON or JSONL files.

    Parameters
    ----------
    dataset : list
        The dataset to iterate over, typically loaded from JSON or JSONL.
    batch_size : int
        The size of each data batch, default is 1.
    start : int, optional
        The index to start iterating from, default is 0.
    end : int, optional
        The index to stop iterating, default is None, which iterates to the end.

    Methods
    -------
    from_json
        Load dataset from a JSON file.
    from_jsonl
        Load dataset from a JSONL file.

    """

    def __init__(
        self,
        dataset: list,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
    ):
        super(DataSet, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            start=start,
            end=end,
        )

    @classmethod
    def from_json(
        cls,
        path: Path,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
    ):
        """Create a DataSet instance from a JSON file.

        This method reads data from a JSON file and initializes a `DataSet`
        instance with it. Each item in the JSON should represent a single
        element in the dataset (typically a list of dictionaries).

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
            end of the dataset.

        Returns
        -------
        DataSet
            A new instance of `DataSet` containing the data loaded from the
            specified JSON file.

        Raises
        ------
        JSONDecodeError
            Raised if the JSON file is not in a valid format.

        """
        with path.open("r", encoding='utf-8') as f:
            dataset = loads(f.read())

        return cls(dataset, batch_size=batch_size, start=start, end=end)

    @classmethod
    def from_jsonl(
        cls,
        path: Path,
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
    ):
        """ Create a DataSet instance from a JSONL file.

        This method reads data from a JSON Lines (JSONL) file, where each line
        is a JSON object representing a single element in the dataset. It then
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
            end of the dataset.

        Returns
        -------
        DataSet
            A new instance of `DataSet` containing the data loaded from the
            specified JSONL file.

        Raises
        ------
        JSONDecodeError
            Raised if any line in the JSONL file is not a valid JSON object.

        """
        dataset = []
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                data = loads(line.strip())
                dataset.append(data)

        return cls(dataset, batch_size=batch_size, start=start, end=end)

    def __repr__(self):
        remaining = len(self.remaining_data())

        return (f"DataSet(length={len(self.dataset)}, start={self.start}, end="
                f"{self.end}, batch_size={self.batch_size}, remaining="
                f"{remaining})")




if __name__ == "__main__":
    pass
