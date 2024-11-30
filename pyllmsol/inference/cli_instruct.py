#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-09 16:49:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 11:05:38
# @File path: ./pyllmsol/inference/cli_instruct.py
# @Project: PyLLMSol

""" Command Line Interface for instruct-based models.

This module defines a subclass of `_BaseCommandLineInterface` designed to
interact with LLMs using instruction-based chat formats. It implements the
`LLaMa 3.2` chat template [1]_, supporting structured message formats like roles
(`user`, `assistant`, `system`) and token-based context management.

References
----------
.. [1] https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

"""

# Built-in packages
from typing import Generator

# Third party packages
from llama_cpp import Llama

# Local packages
from pyllmsol.inference._base_cli import _BaseCommandLineInterface
from pyllmsol.data.chat import Chat

__all__ = []


class InstructCLI(_BaseCommandLineInterface):
    """ Command Line Interface for instruction-based LLM interactions.

    This class provides a CLI for interacting with instruction-based LLMs,
    following the `LLaMa 3.2` chat template [1]_. It manages role-specific
    messages, ensures context window constraints, and supports real-time
    response streaming.

    Parameters
    ----------
    llm : Llama
        LLM object from `llama_cpp` library.
    init_prompt : str, optional
        Initial prompt to feed the LLM.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.

    Methods
    -------
    __call__
    answer
    ask
    exit
    reset_prompt
    run
    set_init_prompt

    Attributes
    ----------
    ai_name, user_name : str
        Respectively the name of AI and of the user.
    llm : object
        Large language model.
    init_prompt : Chat
        Initial prompt to start the conversation.
    prompt_hist : Chat
        History of the conversation, managed as role-based messages.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    today : str
        Current date in "Month Day, Year" format.
    verbose : bool
        Indicates whether verbose mode is enabled.

    References
    ----------
    .. [1] https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/

    """

    PromptFactory = Chat

    def __init__(
        self,
        llm: Llama,
        init_prompt: str | Chat = None,
        verbose: bool = False,
    ):
        super().__init__(llm, init_prompt=init_prompt, verbose=verbose)
        self.stop = "<|eot_id|>"

    def answer(self, output: str | Generator[str, None, None]):
        """ Display the answer of the LLM to the user.

        Parameters
        ----------
        output : str or generator
            The generated output of the LLM.

        """
        answer = self._answer(output)
        self.prompt_hist['assistant'] = answer
        self.logger.debug(f"ANSWER - {self.ai_name}: {answer}")

    def ask(
        self,
        question: str,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """ Submit a question to the LLM and retrieve its response.

        Parameters
        ----------
        question : str
            Question asked by the user to the LLM.
        stream : bool, optional
            If True, streams the response as it is generated. Default is False.

        Returns
        -------
        str or Generator of str
            The output of the LLM, if `stream` is `True` then return generator
            otherwise return a string.

        """
        self.logger.debug(f"ASK - {self.user_name}: {question}")

        # Update prompt
        self.prompt_hist['user'] = question

        # Remove older prompt if exceed context limit
        self._check_prompt_limit_context()

        # Feed the LLM with all the prompt historic available
        return self(self.prompt_hist['assistant'], stream=stream)

    def _check_prompt_limit_context(self):
        """ Ensure the conversation history remains within the token limit.

        This method removes the oldest non-system messages from the history if
        the total number of tokens exceeds the context window (`llm.n_ctx`).

        """
        while self.prompt_hist.get_n_tokens() > self.llm.n_ctx:
            # Remove the second first item of messages list
            for i, message in enumerate(self.prompt_hist.items):
                if message["role"] != "system":
                    break

            role = self.prompt_hist.items[i]['role']

            if role != "system":
                poped_prompt = self.prompt_hist.items.pop(i)
                self.logger.debug(f"Pop the following part: {poped_prompt}")

            else:
                self.logger.error("Prompt exceed limit but only messages from "
                                  "'system' in history")
                break


if __name__ == "__main__":
    pass
