#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-10-09 17:57:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 17:35:30
# @File path: ./pyllmsol/training/checkpoint.py
# @Project: PyLLMSol

""" Objects to load, save and/or make a checkpoint of models and data. """

# Built-in packages
from json import loads, dumps
from pathlib import Path
from time import time

# Third party packages
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import PreTrainedTokenizerBase

# Local packages
from pyllmsol._base import _Base

__all__ = []


class Checkpoint(_Base):
    """ Object to create checkpoint at regular timestep.

    Parameters
    ----------
    path : str or Path
        Path of the folder to save checkpoint of model and data, default is
        `checkpoints` folder at the root of the project.
    timestep : int
        Timestep in seconds to save the checkpoint, default is 300 (5 minutes).

    Methods
    -------
    __bool__
    __call__
    delete
    load
    save
    save_trained_model

    Attributes
    ----------
    path : Path
        Path of the folder to save checkpoint of model and data.
    timestep : int
        Timestep in seconds to save the checkpoint.
    ts : int
        Timestamp of the last checkpoint.

    """

    def __init__(
        self,
        path: str | Path = "./checkpoint/",
        timestep: int = 300
    ):
        super().__init__(logger=True, path=path, timestep=timestep)
        # Set variables
        self.path = Path(path) if isinstance(path, str) else path
        self.timestep = timestep
        self.ts = time()

        # Create path if not already exist
        self.path.mkdir(parents=True, exist_ok=True)
        self.logger.debug("<Checkpoint object is initiated>")

    def __repr__(self):
        """ Representative method. """
        return f"Checkpoint(path={self.path}, timestep={self.timestep})"

    def __bool__(self) -> bool:
        """ Check if the last checkpoint is older than the timestep.

        Returns
        -------
        bool
            `True` if the last checkpoint is older than the timestep, otherwise
            `False`.

        """
        return time() - self.ts > self.timestep

    def __call__(
        self,
        llm: AutoModelForCausalLM,
        data: list,
        tokenizer: PreTrainedTokenizerBase = None
    ):
        """ Save checkpoint if the last checkpoint is older than the timestep.

        Parameters
        ----------
        llm : AutoModelForCausalLM
            Model to make the checkpoint.
        data :
            Data to make the checkpoint.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        if self:
            self.save(llm, data, tokenizer=tokenizer)

        else:
            pass

    def save(
        self,
        llm: AutoModelForCausalLM,
        data: list = None,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        """ Save the checkpoint of the LLM model and data.

        Parameters
        ----------
        llm : transformers.AutoModelForCausalLM
            Model to make the checkpoint.
        data : list of dict
            Data to make the checkpoint.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        self.logger.debug("Checkpoint is saving model and data")

        # Save model
        model_path = self.path / "model"
        llm.save_pretrained(model_path)
        self.logger.debug(f"Model saved at {model_path}")

        # save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(model_path)
            self.logger.debug(f"Tokenizer saved at {model_path}")

        # Save data
        if data is not None:
            data_path = self.path / "data.json"
            with (data_path).open("w", encoding="utf8") as f:
                f.write(dumps(data, ensure_ascii=False))
                self.logger.debug(f"Data saved at {data_path}, size "
                                  f"{len(data):,}")

        self.logger.info(f"<Checkpoint saved at: '{self.path}'>")

        # update timestamp
        self.ts = time()

    def load(self, **kwargs) -> tuple[AutoModelForCausalLM, list[dict]]:
        """ Load the checkpoint of LLM model and data.

        Parameters
        ----------
        **kwargs
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`
            method, cf transformer documentation.

        Returns
        -------
        AutoModelForCausalLM
            Model from the checkpoint.
        Data
            Data from the checkpoint.

        """
        # Load model
        model_path = self.path / "model"
        llm = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        self.logger.debug(f"Model loaded from {model_path}")

        # Load LoRA weights
        # try:
        #     llm = PeftModel.from_pretrained(llm, self.path / "model")
        #     llm = llm.merge_and_unload()
        #     self.logger.debug("Previous trained LoRA weights are loaded and merged")

        # except Exception as e:
        #     print(e)
        #     self.logger.debug("There is no previous trained LoRA weights")

        # Load data
        data_path = self.path / "data.json"
        with (data_path).open("r") as f:
            data = loads(f.read())
            self.logger.debug(f"Data loaded from {data_path}, size "
                              f"{len(data):,}")

        self.logger.info("<Model and dataset loaded from checkpoint>")

        return llm, data

    def delete(self):
        """ Delete irreversibly the checkpoint. """
        # TODO : python3.12 => use walk method
        if self.path.exists():
            model_path = self.path / "model"
            if model_path.exists():
                # Delete model files
                for f in model_path.rglob("*"):
                    f.unlink()

                model_path.rmdir()
                self.logger.debug(f"Model deleted from {model_path}")

            # Delete data file
            data_path = self.path / "data.json"
            if data_path.exists():
                data_path.unlink()
                self.logger.debug(f"Data deleted from {data_path}")

            # Delete folder
            self.path.rmdir()
            self.logger.info(f"Checkpoint deleted from {self.path}")

    def save_trained_model(
        self,
        llm: AutoModelForCausalLM,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase = None,
    ):
        """ Save the trained model and delete checkpoint.

        Must be called when training has finished.

        Parameters
        ----------
        llm : AutoModelForCausalLM
            Trained model to save.
        path : Path
            Path to save the trained model.
        tokenizer : transformers.Tokenizer
            Object to tokenize text data.

        """
        # Save model and tokenizer
        path.mkdir(parents=True, exist_ok=True)

        llm.save_pretrained(path)
        self.logger.debug(f"Model saved at {path}")

        if tokenizer is not None:
            tokenizer.save_pretrained(path)
            self.logger.debug(f"Tokenizer saved at {path}")

        # Delete checkpoint
        self.delete()

        self.logger.info(f"Trained model saved at {path} and checkpoint "
                         f"deleted from {self.path}")


class LoaderLLM(_Base):
    """ Base object to load tokenizer, LLM and data.

    Parameters
    ----------
    model_path : Path
        Path of the model to load.
    data_path : Path
        Path of the dataset to load.
    checkpoint : bool or Checkpoint
        If True or Checkpoint object then make checkpoint of trained model and
        data at regular timestep.
    **kwargs
        Keyword arguments for the class method
        `transformers.AutoModelForCausalLM.from_pretrained`, cf transformers
        documentation.

    """

    def __init__(
        self,
        model_path: Path,
        data_path: Path,
        checkpoint: bool | Checkpoint,
        **kwargs,
    ):
        super().__init__(
            logger=True,
            model_path=model_path,
            data_path=data_path,
            checkpoint=checkpoint,
            **kwargs
        )
        self.checkpoint = checkpoint
        self.load_tokenizer(model_path)

        # Load model and data (or last available checkpoint)
        try:
            self.llm, self.data = checkpoint.load(**kwargs)

        except (AttributeError, ValueError):
            self.load_model(model_path, **kwargs)
            self.load_data(data_path)

            if checkpoint:
                self.checkpoint = Checkpoint()
                self.logger.debug(f"Initialize {self.checkpoint}")

    def load_model(self, path: str | Path, **kwargs):
        """ Load LLM.

        Parameters
        ----------
        path : str or Path
            Path to load LLM.
        **kwargs
            Keyword arguments for `AutoModelForCausalLM.from_pretrained`
            method, cf transformer documentation.

        """
        self.llm = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        self.logger.debug(f"Model loaded from {path}")

    def load_data(self, path: str | Path):
        """ Load data.

        Parameters
        ----------
        path : str or Path
            Path to load data at JSON format.

        """
        with open(path, "r", encoding='utf-8') as f:
            self.data = loads(f.read())
            self.logger.debug(f"Data of size {len(self.data):,} loaded from "
                              f"{path}")

    def load_tokenizer(self, path, use_fast: bool = False):
        """ Load tokenizer.

        Parameters
        ----------
        path : str or Path
            Path to load tokenizer.
        use_fast : bool, optional
            Default is False.

        """
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=use_fast)

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            self.logger.info(f"Set pad with eos token "
                             f"{self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token


if __name__ == "__main__":
    pass
