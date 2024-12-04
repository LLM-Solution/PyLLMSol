#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-11-12 18:29:10
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-04 15:31:43
# @File path: ./pyllmsol/training/loss.py
# @Project: PyLLMSol

""" Loss objects. """

# Built-in packages

# Third party packages
import matplotlib.pyplot as plt
import pandas as pd

# Local packages

__all__ = []


class Losses:
    """ Class for tracking and managing the loss history during training.

    Parameters
    ----------
    loss_history : list of float, optional
        List containing the history of loss values.

    Methods
    -------
    __str__
    __iadd__
    append

    Attributes
    ----------
    current_loss : float
        The most recent loss value added.
    loss_history : list of float
        List containing the history of loss values.

    """

    def __init__(
        self,
        loss_history: list[float] = None
    ):
        self.loss_history = loss_history or []
        self.current_loss = loss_history[-1] if loss_history else None

    def __str__(self) -> str:
        """ Returns a string representation of the current loss. """
        if self.current_loss is None:

            return "Current loss = None"

        avg_loss = sum(self.loss_history) / len(self.loss_history)

        return (f"Current loss = {self.current_loss:.2e}, Average loss = "
                f"{avg_loss:.2e}")

    def __add__(self, loss: float):
        return self.append(loss)

    def __iadd__(self, loss: float):
        return self.append(loss)

    def append(self, loss: float):
        """ Append a new loss to the history and update the current loss.

        Parameters
        ----------
        loss : float
            The new loss to be added to the history.

        Returns
        -------
        Losses
            Self object for method chaining.

        """
        self.loss_history.append(loss)
        self.current_loss = loss

        return self

    def to_dataframe(self) -> pd.DataFrame:
        """ Converts the loss history to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with a single column 'loss' containing the loss history.

        """
        return pd.DataFrame({
            'loss': self.loss_history,
            'steps': range(1, len(self.loss_history) + 1),
        }).set_index('steps')

    def plot(self, window_size: int = 1):
        """ Plots the loss history with an optional moving average.

        Parameters
        ----------
        window_size : int, optional
            The window size for computing the moving average, default is 1 (no
            smoothing).

        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Loss')

        if window_size >= 1:
            ma_loss = self.moving_average(window_size)
            plt.plot(ma_loss, label=f'Moving Average (window={window_size})')

        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Loss over Training Steps')
        plt.legend()
        plt.show()

    def moving_average(self, window_size: int) -> list[float]:
        """ Calculates the moving average of the loss history.

        Parameters
        ----------
        window_size : int
            The size of the window for calculating the moving average.

        Returns
        -------
        list of float
            A list containing the moving average of the loss history.

        """
        if window_size < 1:
            raise ValueError("Window size must be at least 1.")

        return [
            sum(self.loss_history[i:i + window_size]) / window_size
            for i in range(len(self.loss_history) - window_size + 1)
        ]


if __name__ == "__main__":
    pass
