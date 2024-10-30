#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-29 15:24:56
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-29 15:33:17

""" Argument Parser objects. """

# Built-in packages
from argparse import ArgumentParser

# Third party packages

# Local packages

__all__ = []


class _BasisArgParser(ArgumentParser):
    """ Basis object for argument parser. """
    def __init__(self, description: str):
        super(_BasisParser, self).__init__(description=description)

    def __str__(self) -> str:
        args = self.parse_args()
        kw = vars(args)
        str_args = "\n".join([f"{k:20} = {v}" for k, v in kw.items()])

        return f"\nRun {self.file}\n" + str_args


if __name__ == "__main__":
    pass
