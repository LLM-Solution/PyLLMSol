#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-28 09:42:55
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 13:09:42
# @File path: ./pyllmsol/inference/cli.py
# @Project: PyLLMSol

""" Command Line Interface object for interacting with an LLM.

This module defines a Command Line Interface (CLI) class to interact with a
Large Language Model (LLM). It supports features such as token streaming,
conversation history management, and customizable initialization options,
enabling efficient and user-friendly interactions through the terminal.

"""

# Built-in packages

# Third party packages

# Local packages
from pyllmsol.inference._base_cli import _BaseCommandLineInterface
from pyllmsol.data.prompt import Prompt

__all__ = []


class CommandLineInterface(_BaseCommandLineInterface):
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
    init_prompt : Prompt
        Initial prompt to start the conversation.
    prompt_hist : Prompt
        Prompt to feed the model. The prompt will be increment with all the
        conversation, except if you call the `reset_prompt` method.
    stop : list of str
        List of paterns to stop the text generation of the LLM.
    today : str
        Current date in "Month Day, Year" format.
    verbose : bool
        Indicates whether verbose mode is enabled.

    """

    PromptFactory = Prompt

    def _check_prompt_limit_context(self):
        # TODO : How deal with too large prompt such that all the
        #        conversation is removed ?
        raise NotImplementedError


if __name__ == "__main__":
    from pyllmsol.argparser import CLIArgParser
    # import logging.config

    parser = CLIArgParser(file=__file__)
    args = parser()
    print(parser)

    if args.verbose:
        # Load logging configuration
        # logging.config.fileConfig('./logging.ini')
        pass

    cli = CommandLineInterface.from_path(
        model_path=args.model_path,
        init_prompt=args.init_prompt,
        verbose=args.verbose,
        lora_path=args.lora_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
    )
    cli.run()
