#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-30 17:24:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 09:41:21
# @File path: ./pyllmsol/inference/_base_cli.py
# @Project: PyLLMSol

""" Basis Command Line Interface object for interacting with an LLM.

This module defines a base Command Line Interface (CLI) class to interact with 
a Large Language Model (LLM). It supports features such as token streaming, 
conversation history management, and customizable initialization options, 
enabling efficient and user-friendly interactions through the terminal.

"""

# Built-in packages
from copy import deepcopy
from pathlib import Path
from random import random
from time import sleep, strftime
from typing import Generator

# Third party packages
from llama_cpp import Llama

# Local packages
from pyllmsol._base import _Base
from pyllmsol.data._base_data import _TextData

__all__ = []


class _BaseCommandLineInterface(_Base):
    """ Command Line Interface for interacting with a Large Language Model.

    This class provides an interface for engaging in conversations with an LLM
    via a command line. It supports features like token streaming, history
    management, and dynamic initialization with custom prompts or LoRA (Low-Rank
    Adaptation) weights.

    Parameters
    ----------
    llm : Llama
        LLM object.
    init_prompt : str, optional
        Initial prompt to feed the LLM.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.

    Methods
    -------
    from_path
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
    llm : llama_cpp.Llama
        Large language model.
    init_prompt : _TextData
        Initial prompt to start the conversation.
    prompt_hist : _TextData
        Prompt to feed the model. The prompt will be increment with all the
        conversation, except if you call the `reset_prompt` method.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    today : str
        Current date in "Month Day, Year" format.
    verbose : bool
        Indicates whether verbose mode is enabled.

    """

    PromptFactory = _TextData

    def __init__(
        self,
        llm: Llama,
        init_prompt: str | _TextData = None,
        verbose: bool = False,
    ):
        # Set LLM model
        self.llm = llm

        self.set_init_prompt(init_prompt)

        super(_BaseCommandLineInterface, self).__init__(
            logger=True,
            llm=self.llm,
            init_prompt=self.init_prompt,
            verbose=verbose,
        )
        self.verbose = verbose

        self.today = strftime("%B %d, %Y")
        self.user_name = "User"
        self.ai_name = "Assistant"
        self.stop = [f"\n{self.user_name}:", f"\n{self.ai_name}:"]
        self.logger.debug(f"user_name={self.user_name}&ai_name={self.ai_name}")

        self.reset_prompt()

    @classmethod
    def from_path(
        cls,
        model_path: Path | str,
        lora_path: Path | str = None,
        init_prompt: str | _TextData = None,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads: int = 6,
        **kwargs,
    ):
        """ Instanciate CLI object with Large Language Model loaded from path.

        Parameters
        ----------
        model_path : Path or str
            Path to load weight of the model.
        lora_path : Path or str, optional
            Path to load LoRA weights.
        n_ctx : int, optional
            Maximum number of input tokens for LLM, default is 32 768.
        n_threads : int, optional
            Number of threads to compute the inference.
        **kwargs
            Additional arguments passed to the `llama_cpp.Llama` object [1]_.

        References
        ----------
        .. [1] https://llama-cpp-python.readthedocs.io/en/latest/api-reference/

        """
        # Set LLM model
        llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            verbose=False,
            n_threads=n_threads,
            lora_path=lora_path,
            **kwargs,
        )

        return cls(llm, init_prompt=init_prompt, verbose=verbose)


    def set_init_prompt(self, prompt: str | _TextData):
        """ Initialize or update the starting prompt for the LLM.

        Parameters
        ----------
        prompt : str or _TextData
            The initial prompt as a string or a `_TextData` object.

        """
        if isinstance(prompt, str):
            self.init_prompt = self.PromptFactory(
                prompt,
                tokenizer=self.llm.tokenize,
            )

        else:
            self.init_prompt = prompt

    def run(self, stream: bool = True):
        """ Start the command line interface for interacting with the chatbot.

        Parameters
        ----------
        stream : bool, optional
            If True (default), streams the LLM's response as it is generated. 
            If False, waits for the complete response before displaying it.

        """
        self._output(f"\n\nWelcome, I am {self.ai_name} your custom AI, "
                      f"you can ask me anything about LLM Solution.\nPress "
                      f"`ctrl + C` or write 'exit' to exit\n\n")

        self.logger.debug("<Run>")
        question = ""

        try:
            str_time = strftime("%H:%M:%S")
            question = self._input(f"{str_time} | {self.user_name}: ")

            while question.lower() != "exit":
                output = self.ask(question, stream=stream)
                self.answer(output)

                str_time = strftime("%H:%M:%S")
                question = self._input(f"{str_time} | {self.user_name}: ")

        except Exception as e:
            self.exit(f"\n\nAN ERROR OCCURS.\n\n{type(e)}: {e}")

        self.exit(f"Goodbye {self.user_name} ! I hope to see you "
                  f"soon !\n")

    def _answer(self, output: str | Generator[str, None, None]) -> str:
        str_time = strftime("%H:%M:%S")
        self._output(f"{str_time} | {self.ai_name}:", end="", flush=True)

        if isinstance(output, str):
            answer = output
            self._output(answer)

        else:
            answer = ""
            for text in output:
                answer += text
                self._output(text, end='', flush=True)

            self._output("\n")

        return answer

    def answer(self, output: str | Generator[str, None, None]):
        """ Display the answer of the LLM to the user.

        Parameters
        ----------
        output : str or generator
            The generated output of the LLM.

        """
        answer = self._answer(output)
        self.prompt_hist += f"{answer}"
        self.logger.debug(f"ANSWER - {self.ai_name}: {_TextData(answer)}")


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
        # self.logger.debug(f"type {type(question)}")

        # Get prompt at the suitable format
        prompt = f"\n{self.user_name}: {question}\n{self.ai_name}: "

        self.prompt_hist += prompt

        # Remove older prompt if exceed context limit
        # self._check_prompt_limit_context()

        # Feed the LLM with all the prompt historic available
        return self(self.prompt_hist, stream=stream)

    def __call__(
        self,
        prompt: str | _TextData,
        stream: bool = False,
        max_tokens: int = None,
    ) -> str | Generator[str, None, None]:
        """ Generate a response from the LLM for a given prompt.

        Parameters
        ----------
        prompt : str or _TextData
            The input prompt for the LLM.
        stream : bool, optional
            If True, streams the response as it is generated. Default is False.
        max_tokens : int, optional
            The maximum number of tokens to generate in the response.

        Returns
        -------
        str or Generator of str
            The generated response from the LLM. If `stream` is True, returns a
            generator.

        """
        self.logger.debug(f"CALL - {prompt}")

        r = self.llm(
            str(prompt),
            stop=self.stop,
            stream=stream,
            max_tokens=max_tokens,
        )

        if stream:
            return self._stream_call(r)

        else:

            return r['choices'][0]['text']

    def _stream_call(self, response: Generator) -> Generator[str, None, None]:
        text = ""
        for g in response:
            text += g['choices'][0]['text']
            yield g['choices'][0]['text']

    def _check_prompt_limit_context(self):
        # FIXME : How deal with too large prompt such that all the
        #         conversation is removed ?
        #         Currently not working
        raise NotImplementedError
        while self._get_n_token(str(self.prompt_hist)) > self.llm.n_ctx:
            chunked_prompt = str(self.prompt_hist).split("\n")
            poped_prompt = chunked_prompt.pop(1)
            self.logger.debug(f"Pop the following part: {poped_prompt}")
            self.prompt_hist = "\n".join(chunked_prompt)

    def _get_n_token(self, sentence: str) -> int:
        return len(self.llm.tokenize(sentence.encode('utf-8')))

    def exit(self, txt: str = None, stream: bool = False):
        """ Terminate the CLI session and display an exit message.

        Parameters
        ----------
        txt : str, optional
            Message to display before exiting. Default is None.
        stream : bool, optional
            If `True`, streams the exit message character by character. Default
            is `False`.

        """
        if txt is not None:

            txt = f"{self.ai_name}: {txt}"

            if stream:
                self._stream(txt)

            else:
                self._output(txt)

        if self.verbose:
            self.logger.info(f"The full prompt is:\n\n{str(self.prompt_hist)}\n"
                             f"{repr(self.prompt_hist)}")

        self.logger.debug("<Exit>")

    def reset_prompt(self):
        """ Reset the current prompt history with `self.init_prompt`. """
        self.logger.debug("Reset prompt:\n" + repr(self.init_prompt))

        if self.init_prompt:
            self.prompt_hist = deepcopy(self.init_prompt)

        else:
            self.prompt_hist = self.PromptFactory(
                "",
                tokenizer=self.llm.tokenize,
            )

        self.logger.debug("Loading initial prompt")
        r = self(self.prompt_hist, stream=False, max_tokens=1)

    def _stream(self, txt: str):
        for chars in txt:
            self._output(chars)
            sleep(random() / 10)

    def _output(self, txt: str, end: str = "", flush: bool = True):
        print(txt, end=end, flush=flush)

    def _input(self, prompt: str) -> str:
        return input(prompt)


if __name__ == "__main__":
    pass
