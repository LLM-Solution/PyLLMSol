#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-09 16:49:20
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-27 16:15:32

""" CLI object for instruct models. """

# Built-in packages
from pathlib import Path
from time import strftime
from typing import Generator

# Third party packages

# Local packages
from pyllmsol.inference._base_cli import _BaseCommandLineInterface
from pyllmsol.data.chat import Chat

__all__ = []


class InstructCLI(_BaseCommandLineInterface):
    """ Command line interface object to chat with the LLM.

    Parameters
    ----------
    model_path : Path or str
        Path to load weight of the model.
    lora_path : Path or str, optional
        Path to load LoRA weights.
    init_prompt : str, optional
        Initial prompt to feed the LLM.
    verbose : bool, optional
        If True then LLM is run with verbosity. Default is False.
    n_ctx : int, optional
        Maximum number of input tokens for LLM, default is 32 768.
    n_threads : int, optional
        Number of threads to compute the inference.
    **kwargs
        Keyword arguments for llama_cpp.Llama object, cf documentation.

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
    prompt : str
        Prompt to feed the model. The prompt will be increment with all the
        conversation, except if you call the `reset_prompt` method.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    today : str
        Date of today.
    verbose : bool
        Verbosity.

    """

    PromptFactory = Chat

    def __init__(
        self,
        model_path: Path | str,
        lora_path: Path | str = None,
        init_prompt: str | Chat = None,
        verbose: bool = False,
        n_ctx: int = 32768,
        n_threads=4,
        **kwargs,
    ):
        super(InstructCLI, self).__init__(
            model_path=model_path,
            lora_path=lora_path,
            init_prompt=init_prompt,
            verbose=verbose,
            n_ctx=n_ctx,
            n_threads=n_threads,
            **kwargs,
        )
        self.stop = "<|eot_id|>"

    def answer(self, output: str | Generator[str, None, None]):
        """ Display the answer of the LLM.

        Parameters
        ----------
        output : str or generator
            Output of the LLM.

        """
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

        self.prompt_hist['assistant'] = answer
        self.logger.debug(f"ANSWER - {self.ai_name}: {answer}")

    def ask(
        self,
        question: str,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """ Ask a question to the LLM.

        Parameters
        ----------
        question : str
            Question asked by the user to the LLM.
        stream : bool, optional
            If false (default) the full answer is waited before to be printed,
            otherwise the answer is streamed.

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
        while self.prompt_hist.get_n_tokens() > self.n_ctx:
            # Remove the second first item of messages list
            for i, message in enumerate(self.prompt_hist.items):
                if message["role"] != "system":
                    break

            if self.prompt_hist.items[i]['role'] != "system":
                poped_prompt = self.prompt_hist.items.pop(i)
                self.logger.debug(f"Pop the following part: {poped_prompt}")

            else:
                self.logger.error("Only system messages are in chat")
                break


if __name__ == "__main__":
    pass
