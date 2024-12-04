#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-04 15:25:46
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-04 15:34:03
# @File path: ./pyllmsol/tests/training/test_loss.py
# @Project: PyLLMSol

""" Test loss objects. """

# Built-in packages
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.training.loss import Losses

__all__ = []


@pytest.fixture
def empty_losses():
    """Fixture for an empty Losses object."""
    return Losses()


@pytest.fixture
def prefilled_losses():
    """Fixture for a Losses object with initial history."""
    return Losses(loss_history=[0.5, 0.4, 0.35])


def test_append_loss(empty_losses):
    """Test appending losses to the history."""
    empty_losses.append(0.5)
    assert empty_losses.loss_history == [0.5]
    assert empty_losses.current_loss == 0.5


def test_iadd_loss(prefilled_losses):
    """Test adding a loss using +=."""
    prefilled_losses += 0.3
    assert prefilled_losses.loss_history == [0.5, 0.4, 0.35, 0.3]
    assert prefilled_losses.current_loss == 0.3


def test_append_loss(empty_losses):
    """Test appending losses to the history."""
    empty_losses.append(0.5)
    assert empty_losses.loss_history == [0.5]
    assert empty_losses.current_loss == 0.5


def test_iadd_loss(prefilled_losses):
    """Test adding a loss using +=."""
    prefilled_losses += 0.3
    assert prefilled_losses.loss_history == [0.5, 0.4, 0.35, 0.3]
    assert prefilled_losses.current_loss == 0.3


def test_str_empty(empty_losses):
    """Test string representation for an empty Losses object."""
    assert str(empty_losses) == "Current loss = None"


def test_str_prefilled(prefilled_losses):
    """Test string representation for a prefilled Losses object."""
    expected_str = "Current loss = 3.50e-01, Average loss = 4.17e-01"
    assert str(prefilled_losses) == expected_str


def test_to_dataframe(prefilled_losses):
    """Test conversion to pandas DataFrame."""
    df = prefilled_losses.to_dataframe()
    assert list(df['loss']) == [0.5, 0.4, 0.35]
    assert list(df.index) == [1, 2, 3]


def test_moving_average(prefilled_losses):
    """Test moving average calculation."""
    ma = prefilled_losses.moving_average(window_size=2)
    assert ma == [0.45, 0.375]


def test_moving_average_invalid_window(empty_losses):
    """Test moving average with invalid window size."""
    with pytest.raises(ValueError):
        empty_losses.moving_average(window_size=0)


def test_plot_prefilled(prefilled_losses):
    """Test plotting functionality."""
    with patch("matplotlib.pyplot.show"):  # Prevent the plot from displaying
        prefilled_losses.plot(window_size=2)
        # Ensure no exceptions are raised during plotting


def test_append_negative_loss(empty_losses):
    """Test appending a negative loss."""
    empty_losses.append(-0.1)
    assert empty_losses.loss_history == [-0.1]
    assert empty_losses.current_loss == -0.1


def test_moving_average_empty(empty_losses):
    """Test moving average with an empty loss history."""
    ma = empty_losses.moving_average(window_size=1)
    assert ma == []


if __name__ == "__main__":
    pass
