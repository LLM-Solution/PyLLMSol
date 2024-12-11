#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-29 15:24:56
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 17:22:02

""" Argument Parser objects. """

# Built-in packages
from argparse import ArgumentParser
from os import cpu_count

# Third party packages

# Local packages

__all__ = []


class _BasisArgParser(ArgumentParser):
    """ Basis object for argument parser. """
    def __init__(self, description: str, file: str = None):
        super().__init__(description=description)
        self.file = file

    def __str__(self) -> str:
        args = self.parse_args()
        kw = vars(args)
        str_args = "\n".join([f"{k:20} = {v}" for k, v in kw.items()])

        return f"\nRun {self.file}\n" + str_args


class CLIArgParser(_BasisArgParser):
    """ CLI argument parser object. """

    def __init__(self, file: str = None):
        super().__init__("CLI arguments parser", file=file)

    def __call__(self, n_ctx: int = 32768):
        """ Parse arguments. """
        self.add_argument(
            "--model_path", "--model-path",
            type=str,
            help="Path to load LLM weights at GGUF format.",
        )
        self.add_argument(
            "--init_prompt", "--init-prompt",
            type=str,
            help="Initial prompt for the LLM.",
        )
        self.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Flag to set verbosity.",
            default=False,
        )
        self.add_argument(
            "--lora_path", "--lora-path",
            type=str,
            help="Path to load LoRA weights (optional)",
        )
        self.add_argument(
            "--n_ctx", "--n-ctx",
            default=n_ctx,
            type=int,
            help=(f"Maximum number of tokens allowed by the model, default is"
                  f"{n_ctx}"),
        )

        n_threads = max(cpu_count() - 1, 1)
        self.add_argument(
            "--n_threads", "--n-threads",
            default=n_threads,
            type=int,
            help=(f"Number of threads allowed to compute the generation "
                  f"default is {n_threads}."),
        )

        return self.parse_args()


if __name__ == "__main__":
    pass
