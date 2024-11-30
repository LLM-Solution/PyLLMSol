#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-12 16:27:52
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 10:35:30

""" Trainer objects for model with chat data template.

This module use Llama 3.1 instruct format for chat dialogue.

References
----------
https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

"""

# Built-in packages

# Third party packages
from transformers import PreTrainedTokenizerBase, AutoModelForCausalLM
from torch import Tensor

# Local packages
from pyllmsol.data.chat import Chat, Message, ChatDataSet
from pyllmsol.training.trainer import Trainer

__all__ = []


class TrainerInstruct(Trainer):
    """ Trainer instruct class. """

    def __init__(
        self,
        llm: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizerBase,
        dataset: ChatDataSet | list[Chat | list[Message | dict[str, str]]],
        batch_size: int,
        accumulation_steps: int = 1,
    ):
        if not isinstance(dataset, ChatDataSet):
            dataset = ChatDataSet(dataset, tokenizer, batch_size=batch_size)

        else:
            dataset.batch_size = batch_size

        Trainer.__init__(
            self,
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
        torch.Tensor
            Extended with PAD tokens list of tokens.
        torch.Tensor
            Extended with 0 list of mask.

        """
        data = self.dataset.__next__()

        # Set input data batch
        input_ids, attention_mask = data.get_padded(return_tensor=True)

        # Display current loss and token size of data
        token_size = input_ids.size(1)
        descr = f"{self.losses} - Token size = {token_size}"
        self.dataset.set_description(descr)

        return input_ids, attention_mask


if __name__ == "__main__":
    pass
