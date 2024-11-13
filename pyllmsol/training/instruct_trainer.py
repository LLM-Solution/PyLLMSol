#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-12 16:27:52
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-13 20:10:00

""" Description. """

# Built-in packages

# Third party packages
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
from torch import Tensor

# Local packages
from pyllmsol.prompt_instruct import PromptInstruct, MessageInstruct, INSTRUCT_ROLES
from pyllmsol.training.dataset import DataSet as _DataSet
from pyllmsol.training.trainer import Trainer

__all__ = []


class Message(MessageInstruct):
    pass


class Chat(PromptInstruct):
    def __init__(
        self,
        messages: list[dict[str, str] | Message],
        tokenizer: PreTrainedTokenizerBase,
    ):
        if not all(isinstance(m, MessageInstruct) for m in messages):
            messages = [Message(**m, tokenizer=tokenizer) for m in messages]

        super(Chat, self).__init__(
            messages=messages,
            tokenizer=tokenizer,
        )
        self.set_tokens()

    def set_tokens(self):
        """ Initialize the tokens and mask attributes.

        This method combines the tokens and mask values from each message in
        `self.messages` to create the full sequence of tokens and mask for
        the chat object.

        """
        self._tokens = [self.tokenizer.bos_token_id]
        self.mask = [1]

        for message in self.messages:
            self._tokens += message.tokens
            self.mask += message.mask

        # self._tokens += [self.tokenizer.eos_token_id]
        # self.mask += [0]

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
            raise ValueError(f"`total_tokens` must be greater or equal than "
                             f"size of `self.tokens`")

        tokens = self.tokens + [self.tokenizer.pad_token_id for _ in range(n)],
        mask = self.mask + [0 for _ in range(n)]

        return tokens, mask

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

        new_message = Message(role=role, content='', tokenizer=self.tokenizer)

        return self.__class__(
            self.messages + [new_message],
            tokenizer=self.tokenizer,
        )


class DataSet(_DataSet):
    def __init__(
        self,
        dataset: list[Chat],
        batch_size: int = 1,
        start: int = 0,
        end: int = None,
    ):
        super(DataSet, self).__init__(
            dataset,
            batch_size=batch_size,
            start=start,
            end=end,
        )

    def __next__(self):
        """ Retrieve the next batch of chats.

        Returns
        -------
        DataSet or Chat
            The next batch of chats as a DataSet object if batch_size > 1,
            otherwise return the chat object.

        """
        data = super(DataSet, self).__next__()

        if self.batch_size == 1:

            return data

        return DataSet(
            data,
            batch_size=self.batch_size,
            start=0,
            end=self.batch_size,
        )

    def get_padded(self, torch_tensor: bool = False):
        """ Pad all chats in the dataset to the same length.

        Pads each chat in the dataset to the length of the longest chat,
        creating uniform-sized token and mask arrays for each chat.

        Parameters
        ----------
        torch_tensor : bool, optional
            If True, returns data as `torch.Tensor` objects; otherwise, returns
            lists.

        Returns
        -------
        list of list of int
            Extended with PAD tokens list of tokens.
        list of list of int
            Extended with 0 list of mask.

        """
        max_tokens = max(data.get_n_tokens() for data in self.dataset)

        tokens, mask = [], []
        for data in self.dataset:
            _tokens, _mask = data.pad(max_tokens)
            tokens.append(_tokens)
            mask.append(_mask)

        if torch_tensor:
            tokens = torch.tensor(tokens)
            mask = torch.tensor(mask)

        return tokens, mask

    def remaining_data(self) -> 'DataSet':
        return DataSet(
            super(DataSet, self).remaining_data(),
            batch_size=batch_size,
        )


class TrainerInstruct(Trainer):
    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: list | DataSet,
        batch_size: int,
        accumulation_steps: int = 1,
    ):
        if isinstance(dataset, list):
            if not isinstance(dataset[0], Chat):
                dataset = [Chat(data, tokenizer=tokenizer) for data in dataset]

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

    def __next__(self) -> tuple[Tensor, Tensor]:
        """ Retrieve the next batch of input IDs and attention masks.

        Prepares the next batch of tokenized chat data, including padding, for
        input into the language model.

        Returns
        -------
        list of list of int or torch.Tensor
            Extended with PAD tokens list of tokens.
        list of list of int or torch.Tensor
            Extended with 0 list of mask.

        """
        data = self.dataset.__next__()

        # Set input data batch
        input_ids, attention_mask = data.get_padded(torch_tensor=True)

        # Display current loss and token size of data
        token_size = input_ids.size(1)
        descr = f"{self.losses} - Token size = {token_size}"
        self.dataset.set_description(descr)
        self.logger.info(descr)

        return input_ids, attention_mask


if __name__ == "__main__":
    pass