#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-11 17:02:24
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 17:16:07
# @File path: ./pyllmsol/tests/test_base.py
# @Project: PyLLMSol

""" Test _Base object. """

# Built-in packages
from unittest.mock import patch, MagicMock

# Third party packages
from logging import NullHandler, getLogger
import pytest

# Local packages
from pyllmsol._base import _Base

__all__ = []


@pytest.fixture
def base_instance():
    """Fixture to create a _Base instance with logging enabled."""
    return _Base(logger=True)


def test_init_logging_enabled(base_instance):
    """Test that logging is enabled and the logger is set correctly."""
    assert base_instance.logger is not None
    assert base_instance.logger.name == "_Base"


def test_init_logging_disabled():
    """Test that logging is disabled when the logger argument is False."""
    base_instance = _Base(logger=False)
    assert isinstance(base_instance.logger.handlers[0], NullHandler)


def test_to_dict_method(base_instance):
    """Test the to_dict method for correct attribute conversion."""
    # Add attributes to the instance
    base_instance.some_attr = "test"
    base_instance.another_attr = 42

    # Expected dictionary
    expected_dict = {
        "some_attr": "test",
        "logger": base_instance.logger,
        "another_attr": 42,
    }

    assert base_instance.to_dict() == expected_dict


def test_str_method(base_instance):
    """Test the __str__ method for correct formatting."""
    base_instance.some_attr = "test"
    base_instance.another_attr = 42

    expected_str = "_Base(some_attr='test', another_attr=42)"
    assert str(base_instance) == expected_str


def test_repr_method(base_instance):
    """Test the __repr__ method for detailed representation."""
    base_instance.another_attr = 42

    expected_repr = "_Base({'another_attr': 42})"
    assert repr(base_instance) == expected_repr


def test_logging_calls():
    """Test that logging captures arguments and kwargs."""
    with patch("pyllmsol._base.getLogger") as mock_get_logger:
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger

        # Create an instance with args and kwargs
        _Base(1, "string", test_kwarg="value", logger=True)

        # Verify logging calls
        mock_logger.debug.assert_any_call(
            "Init _Base(args=[int(1), str(string)], kwargs=[test_kwarg='value'])"
        )


def test_private_attributes_exclusion(base_instance):
    """Test that private attributes are excluded from to_dict and __str__."""
    base_instance._private_attr = "hidden"
    base_instance.public_attr = "visible"

    expected_dict = {"public_attr": "visible"}
    assert base_instance.to_dict(exclude=['logger']) == expected_dict
    assert "hidden" not in str(base_instance)


if __name__ == "__main__":
    pass