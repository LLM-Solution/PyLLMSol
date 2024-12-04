#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-10-29 14:35:28
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-04 17:15:13
# @File path: ./pyllmsol/training/utils.py
# @Project: PyLLMSol

""" Util functions for training. """

# Built-in packages
from itertools import chain
from random import shuffle, sample

# Third party packages
import torch
from torch import Tensor

# Local packages

__all__ = []


def find_token(token_id: int, input_ids: Tensor, start: int = 0) -> int:
    """ Look for an ID token in the input IDs.

    Parameters
    ----------
    token_id : int
        Token ID to looking for.
    inputs_ids : torch.Tensor
        Input IDs.
    start : int, optional
        Index to start the research in the input IDs, default start at the
        begining (i.e 0).

    Returns
    -------
    int
        Index of the finded token ID.

    """

    idx = (torch.where(input_ids[start:] == token_id, 1., 0.)
                .nonzero(as_tuple=True)[0][-1])

    return start + idx


def find_sequence(seq_ids: Tensor, input_ids: Tensor, start: int = 0) -> int:
    """ Look for a sequence of token IDs in the input IDs.

    Parameters
    ----------
    seq_ids : torch.Tensor
        Sequence of int to looking for.
    inputs_ids : torch.Tensor
        Input IDs.
    start : int, optional
        Index to start the research in the input IDs, default start at the
        begining (i.e 0).

    Returns
    -------
    int
        Index of the end of the finded sequence.

    """
    n = len(seq_ids)
    for i in range(start, len(input_ids) - n + 1):
        if torch.equal(seq_ids, input_ids[i: i + n]):

            return i + n

    return None


def generate(
    llm,
    tokenizer,
    sentence: str,
    max_length: int = 16,
    device: str = 'cpu',
) -> str:
    """ Generate a sentence.

    Parameters
    ----------
    llm : transformer.ModelForCausalLM
        Large language model.
    tokenizer : transformers.Tokenizer
        Object to tokenize text.
    sentence : str
        Begining of the sentence to complete by the LLM.
    max_length : int, optional
        Maximum token size to the generate sentence.
    device : str, optional
        Allow 'cuda' or 'cpu', default is 'cpu'.

    Returns
    -------
    str
        Generated sentence.

    """
    encoded = tokenizer(sentence, return_tensors='pt').to(device=device)
    # encoded = encoded.input_ids
    generated_encoded = llm.generate(**encoded, max_length=max_length)[0]

    return tokenizer.decode(generated_encoded)


def get_token_size(text: str, tokenizer) -> int:
    """ Get the number of token for a piece of text.

    Parameters
    ----------
    text : str
        Text to count the number of token.
    tokenizer : transformers.Tokenizer
        Object to tokenize.

    Returns
    -------
    int
        Number of tokens for the piece of text.

    """
    # return len(tokenizer(text).input_ids)
    return tokenizer(text).input_ids.size(1)


def set_mask(
    attention_mask: Tensor,
    rate: float = 0.05,
    beginning_idx: int = 0,
    end_idx: int = None,
) -> Tensor:
    """ Set mask attention randomly.

    Parameters
    ----------
    attention_mask : torch.Tensor
        Attention mask to update (tensor of one dimension).
    rate : float, optional
        Rate of input to mask randomly, default is 5%.
    beginning_idx : int, optional
        Beginning of the mask, default is `0`.
    end_idx : int, optional
        End of the mask, default is end the sentence.

    Returns
    -------
    torch.Tensor
        Updated attention mask.

    """
    # Fallback if end index is not provided
    if end_idx is None:
        end_idx = int(attention_mask.sum())

    k = int((end_idx - beginning_idx) * rate)
    population = range(beginning_idx, end_idx)
    index = sample(population, k)

    attention_mask[index] = 0


    return attention_mask


def shuffle_per_batch(
    data: list[str],
    tokenizer,
    batch_size: int
) -> list[str]:
    """ Shuffle data per batch without modify sort in each batch.

    Data is shuffled but each batch is approximatively of the same token size.

    Parameters
    ----------
    data : list of str
        Data to shuffle.
    tokenizer : transformers.Tokenizer
        Tokenizer object to know the number of token for each data.
    batch_size : int
        Number of data in each batch

    Returns
    -------
    list of str
        Shuffled data.

    """
    sorted_data = sort_per_tokens_size(data, tokenizer)

    batchs = [*zip(*[sorted_data[i::batch_size] for i in range(batch_size)])]

    shuffle(batchs)

    return list(chain.from_iterable(batchs))


def sort_per_tokens_size(data: list[str], tokenizer) -> list[str]:
    """ Sort data per token size.

    Parameters
    ----------
    data : list of str
        Data to sort.
    tokenizer : transformers.Tokenizer
        Object to tokenize each piece of text.

    Returns
    -------
    list of str
        Sorted data per token size.

    """
    # get list of pair, number of token and text
    size_and_text = [(get_token_size(a, tokenizer), a) for a in data]

    return [a for _, a in sorted(size_and_text)]


if __name__ == "__main__":
    pass
