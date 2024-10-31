#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 09:41:32
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-31 10:29:17

""" Base objects. """

# Built-in packages
from logging import getLogger
from pprint import pformat

# Third party packages

# Local packages

__all__ = []


class _Base:
    """ Base object with optional logger.

    Parameters
    ----------
    logger : str, optional
        Name of the logger to use. If `None` (default), a `NullHandler` is
        assigned to ignore all logging calls.
    *args : tuple
        Positional arguments to pass to the subclass.
    **kwargs : dict
        Keyword arguments to pass to the subclass.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for the class, which will either use the specified
        logger name or default to a `NullHandler` to discard logs.
    
    Examples
    --------
    >>> obj = _Base(logger='example')
    >>> obj.logger.info("This will log to 'example' logger")

    """

    def __init__(self, *args, logger: str = None, **kwargs):
        if logger:
            self.logger = getLogger(logger)

        else:
            self.logger = logging.getLogger(__name__)
            self.logger.addHandler(logging.NullHandler())

        args_debug = ", ".join(f"{type(arg).__name__}({arg})" for arg in args)
        kwargs_debug = ", ".join(
            f"{k}={type(v).__name__}({v})" for k, v in kwargs.items()
        )

        if args or kwargs:
            self.logger.debug(f"Init {self.__class__.__name__}(args=[{args_debug}], "
                              f"kwargs=[{kwargs_debug}])")

    def __str__(self):
        params = ', '.join(
            f'{k}={repr(v)}' for k, v in self.__dict__.items() if not k.startswith('_')
        )

        return f"{self.__class__.__name__}({params})"

    def __repr__(self):
        params = {
            k: v for k, v in self.__dict__.items() if not k.startswith('_')
        }

        return f"{self.__class__.__name__}({pformat(params)})"


if __name__ == "__main__":
    pass
