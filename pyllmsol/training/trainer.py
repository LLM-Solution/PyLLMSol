#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-29 15:33:52
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 15:13:35

""" Trainer objects. """

# Built-in packages

# Third party packages
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

# Local packages
from pyllmsol.training.checkpoint import Checkpoint
from pyllmsol.training.utils import set_mask

__all__ = []


class Losses:
    """ Class for tracking and managing the loss history during training.

    Parameters
    ----------
    loss_history : list of float, optional
        List containing the history of loss values.

    Methods
    -------
    __str__
    __iadd__
    append

    Attributes
    ----------
    current_loss : float
        The most recent loss value added.
    loss_history : list of float
        List containing the history of loss values.

    """

    def __init__(
        self,
        loss_history: list[float] = None
    ):
        self.loss_history = [] if loss_history is None else loss_history

        if self.loss_history:
            self.current_loss = loss_history[-1]

        else:
            self.current_loss = None

    def __str__(self) -> str:
        """ Returns a string representation of the current loss. """
        if self.current_loss is None:

            return "Current loss = None"

        return f"Current loss = {self.current_loss:.2e}"

    def append(self, loss: float):
        """ Append a new loss to the history and the new loss is current one.

        Parameters
        ----------
        loss : float
            Loss to add to the history and become the current loss.

        Returns
        -------
        Losses
            Self object.

        """
        self.loss_history.append(loss)
        self.current_loss = loss

        return self

    def __iadd__(self, loss: float):
        return self.append(loss)


class DataBrowser:
    """ Class to facilitate browsing through a dataset in batches.

    Parameters
    ----------
    dataset : list of str
        The data to iterate over.
    batch_size : int
        The size of each data batch.
    start : int, optional
        Index to start iterating from, by default 0.
    end : int, optional
        Index to stop iterating, by default None, which means the end.

    Methods
    -------
    __iter__
    __next__
    set_description
    remaining_data

    Attrtibutes
    -----------
    dataset : list of str
        The data to iterate over.
    batch_size : int
        The size of each data batch.
    start : int
        The starting index for iteration.
    end : int
        The ending index for iteration.
    i : int
        Current index in the data iteration.

    """

    def __init__(
        self,
        dataset: list[str],
        batch_size: int,
        start: int = 0,
        end: int = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self._set_boundary(start, end=end)

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
        """ Initializes the iterator. """
        self.i = self.start
        self.pbar = tqdm(total=self.end - self.start)

        return self

    def __next__(self) -> list[str]:
        """ Retrieves the next batch of data. """
        if self.i >= self.end:
            raise StopIteration

        i = self.i
        j = min(i + self.batch_size, self.end)

        self.i = j

        self.pbar.update(j - i)

        return self.dataset[i: j]

    def set_description(self, text: str, logger=None):
        """ Sets a description for the progress bar and logs it.

        Parameters
        ----------
        text : str
            The text to display in the progress bar and log.
        logger : logging.Logger, optional
            Logger instance for logging the progress. If None, logging is
            skipped.

        """
        self.pbar.set_description(text)

        if logger is not None:
            displayer(
                f"{self.i}/{len(self.dataset)} data - {text}"
            )

    def remaining_data(self) -> list[str]:
        """ Returns the remaining data that has not been iterated.

        Returns
        -------
        list of str
            Data not yet used.

        """
        return self.dataset[self.i:]


class Trainer:
    """ Class to train a language model on a given dataset.

    Parameters
    ----------
    llm : transformers.ModelForCausalLM
        The language model to train.
    tokenizer : transformers.Tokenizer
        The tokenizer used for encoding the data.
    dataset : list of str
        List of text samples for training.
    batch_size : int
        Number of samples per batch.
    accumulation_steps : int, optional
        Number of steps for gradient accumulation, by default 1.

    Methods
    -------
    __iter__
    __next__
    run
    set_mask
    set_optimizer
    training_step

    Attributes
    ----------
    accumulation_steps : int
        Number of steps for gradient accumulation, by default 1.
    batch_size : int
        Number of samples per batch.
    dataset : list of str
        List of text samples for training.
    losses : Losses
        History loss object.
    llm : transformers.ModelForCausalLM
        The language model to train.
    n_accumulated_grad : int
        Number of step accumulated gradient.
    optimizer : torch.optim.Optimizer
        Optimizer object.
    tokenizer : transformers.Tokenizer
        The tokenizer used for encoding the data.

    """

    losses = None
    n_accumulated_grad = None
    optimizer = None

    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: list[str],
        batch_size: int,
        accumulation_steps: int = 1,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.accumulation_steps = accumulation_steps

    def __iter__(self):
        """ Initializes the training iterator. """
        self.losses = Losses()
        self.n_accumulated_grad = 0
        self._data_browser = DataBrowser(self.dataset, self.batch_size)
        self._data_browser.__iter__()

        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        """ Retrieves the next batch of input IDs and attention masks. """
        data = self._data_browser.__next__()

        # Set input data batch
        encoded_data = self.tokenizer(data, return_tensors='pt', padding=True)
        input_ids = encoded_data['input_ids']
        attention_mask = self.set_mask(encoded_data.attention_mask, input_ids)

        # Display current loss and token size of data
        self._data_browser.set_description(
            f"{self.losses} - Token size = {encoded_data['input_ids'].size(1)}"
        )

        return input_ids, attention_mask

    def run(self, device: str, checkpoint: bool | Checkpoint):
        """ Runs the training loop on a specified device with checkpointing.

        Parameters
        ----------
        device : str
            The device to make computation.
        checkpoint : bool or Checkpoint
            If True or Checkpoint object then make regular checkpoints.

        """
        for input_ids, attention_mask in self:
            with torch.enable_grad():
                self.training_step(
                    input_ids.to(device),
                    attention_mask.to(device)
                )

            if checkpoint:
                data = self._data_browser.remaining_data()
                checkpoint(self.llm, data, tokenizer=self.tokenizer)

    def set_mask(self, attention_mask: Tensor, input_ids: Tensor) -> Tensor:
        """ Randomly sets the attention mask.

        Parameters
        ----------
        attention_mask : torch.Tensor
            The attention masks to update.
        input_ids : torch.Tensor
            The input IDs.

        Returns
        -------
        torch.Tensor
            Updated attention masks.

        """
        for i in range(attention_mask.size()[0]):
            attention_mask[i] = set_mask(attention_mask[i])

        return attention_mask

    def set_optimizer(self, optimizer, parameters=None, **kwargs):
        """ Initializes the optimizer for training.

        Parameters
        ----------
        optimizer : torch.Optimizer
            Torch optimizer object.
        parameters :
            Parameters to trains, default is None then train all parameters
            of the model.
        **kwargs
            Keyword arguments of optimizer object.

        """
        self.llm.train()

        if parameters is None:
            parameters = self.llm.parameters()

        self.optimizer = optimizer(parameters, **kwargs)

    def training_step(self, input_ids: Tensor, attention_mask: Tensor):
        """ Executes a single training step.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input data to train model.
        attention_mask : torch.Tensor
            Mask attention.

        """
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )

        # Compute gradient and update weights
        loss = outputs.loss
        loss.backward()
        self.n_accumulated_grad += self.batch_size

        if self.n_accumulated_grad >= self.accumulation_steps:
            self.optimizer.step()

            # Update learning rate
            # lr_scheduler.step()

            # Reset gradient to zero
            self.optimizer.zero_grad()

            self.n_accumulated_grad = 0

        self.losses += loss.detach()


if __name__ == "__main__":
    pass
