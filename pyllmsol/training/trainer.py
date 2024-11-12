#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-29 15:33:52
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-12 18:34:27

""" Trainer objects. """

# Built-in packages
from logging import getLogger

# Third party packages
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

# Local packages
from pyllmsol._base import _Base
from pyllmsol.training.checkpoint import Checkpoint
from pyllmsol.training.dataset import DataSet
from pyllmsol.training.loss import Losses
from pyllmsol.training.utils import set_mask

__all__ = []


class Trainer(_Base):
    """ Class to train a language model on a specified dataset.

    This class manages the training loop for a language model using PyTorch.
    It supports gradient accumulation, custom batch sizes, and tracking of
    training losses with optional checkpointing.

    Parameters
    ----------
    llm : transformers.AutoModelForCausalLM
        The language model to be trained.
    tokenizer : transformers.PreTrainedTokenizerBase
        The tokenizer for encoding the data.
    dataset : list of str or DataSet
        A list of text samples or a `DataSet` object for training.
    batch_size : int
        Number of samples per batch.
    accumulation_steps : int, optional
        Number of steps to accumulate gradients before updating, default is 1.

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
        The gradient accumulation steps.
    batch_size : int
        Number of samples per batch.
    dataset : DataSet
        Training DataSet instance.
    losses : Losses
        The object tracking loss history during training.
    llm : transformers.AutoModelForCausalLM
        The language model to be trained.
    n_accumulated_grad : int
        Counter for accumulated gradient steps.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model weights.
    tokenizer : transformers.PreTrainedTokenizerBase
        The tokenizer used for data encoding.

    """

    losses = None
    n_accumulated_grad = None
    optimizer = None

    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: list | DataSet,
        batch_size: int,
        accumulation_steps: int = 1,
    ):
        if isinstance(dataset, list):
            dataset = DataSet(dataset, batch_size=batch_size)

        else:
            dataset.batch_size = batch_size

        super(Trainer, self).__init__(
            llm,
            tokenizer,
            dataset,
            batch_size=batch_size,
            accumulation_steps=accumulation_steps,
        )

        self.llm = llm
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.batch_size = batch_size

        self.accumulation_steps = accumulation_steps
        self.logger.debug("Trainer is instancied")

    def __iter__(self):
        """ Initializes the training iterator. """
        self.losses = Losses()
        self.n_accumulated_grad = 0
        # self._data_browser = DataBrowser(self.dataset, self.batch_size)
        # self._data_browser.__iter__()
        self.dataset.__iter__()
        self.logger.debug("Start iterate")

        return self

    def __next__(self) -> tuple[Tensor, Tensor]:
        """ Retrieves the next batch of input IDs and attention masks. """
        # data = self._data_browser.__next__()
        data = self.dataset.__next__()

        # Set input data batch
        encoded_data = self.tokenizer(data, return_tensors='pt', padding=True)
        input_ids = encoded_data['input_ids']
        attention_mask = self.set_mask(encoded_data.attention_mask, input_ids)

        # Display current loss and token size of data
        token_size = encoded_data['input_ids'].size(1)
        descr = f"{self.losses} - Token size = {token_size}"
        # self._data_browser.set_description(descr)
        self.dataset.set_description(descr)
        self.logger.info(descr)

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
                # data = self._data_browser.remaining_data()
                self.dataset.set_description(descr)
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

        Sets the optimizer for training, with optional keyword arguments to
        configure it. This method prepares the model for training mode.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer class to use for training.
        parameters : iterable, optional
            Parameters to be optimized. If None, uses all model parameters.
        **kwargs
            Additional arguments to configure the optimizer.

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
