#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 09:41:32
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 17:14:03

""" Base objects. """

# Built-in packages
from logging import getLogger, NullHandler
from pprint import pformat

# Third party packages

# Local packages

__all__ = []


class _Base:
    """ Base object with optional logger and structured argument handling.

    Parameters
    ----------
    logger : bool, optional
        If `True` (default), initializes a logger with the class name as the
        logger name.
        If `False`, a `NullHandler` is added to discard all logging calls.
    *args : tuple
        Positional arguments to pass to the subclass.
    **kwargs : dict
        Keyword arguments to pass to the subclass.

    Methods
    -------
    __str__
    __repr__
    to_dict

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for the class, using the class name by default. Logs
        are discarded if `logger` is set to `False` at initialization.
    
    Examples
    --------
    >>> obj = _Base(logger=True)
    >>> obj.logger.info("Logging enabled")

    >>> obj_no_log = _Base(logger=False)
    >>> obj_no_log.logger.info("This will not appear in the logs")

    Notes
    -----
    Positional (`args`) and keyword arguments (`kwargs`) passed to the
    constructor are logged with their types for easier debugging if logging
    is enabled.

    """

    def __init__(self, *args, logger: bool = True, **kwargs):
        self._set_logger(self.__class__.__name__)

        if not logger:
            self.logger.addHandler(NullHandler())

        args_debug = ", ".join(f"{type(arg).__name__}({arg})" for arg in args)
        kwargs_debug = ", ".join(f"{k}={repr(v)}" for k, v in kwargs.items())

        if args or kwargs:
            self.logger.debug(f"Init {self.__class__.__name__}(args=["
                              f"{args_debug}], kwargs=[{kwargs_debug}])")

    def _set_logger(self, name):
        self.logger = getLogger(name)

    def __str__(self):
        """ Return a readable string representation of the instance.

        This method provides a summary of the instance's attributes, excluding
        any private or protected attributes (those starting with '_'). If the 
        instance has complex or nested attributes, they are displayed in a 
        readable, truncated format for clarity.

        Returns
        -------
        str
            A formatted string displaying the class name and selected
            attributes.

        """
        params = ', '.join(
            f'{k}={repr(v)}'
            for k, v in self.to_dict(exclude=["logger"]).items()
        )

        return f"{self.__class__.__name__}({params})"

    def __repr__(self):
        """ Return a readable, string representation of the instance.

        This method provides a full representation of the instance's attributes
        for debugging purposes, formatted in a structured way for readability.
        Only public attributes are displayed.

        Returns
        -------
        str
            A detailed string showing the class name and all non-private
            attributes formatted with `pprint` for easy inspection.

        """
        params = self.to_dict(exclude=["logger"])

        return f"{self.__class__.__name__}({pformat(params)})"

    def to_dict(self, exclude=None):
        """ Convert instance attributes to a dictionary.

        Returns
        -------
        dict
            Dictionary of all non-private attributes and their values.

        """
        if exclude is None:
            exclude = []

        return {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith('_') and k not in exclude
        }


if __name__ == "__main__":
    pass
