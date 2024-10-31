#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-10-09 17:57:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-31 16:44:43

""" Objects to load, save and/or make a checkpoint of models and data. """

# Built-in packages
from json import loads, dumps
from logging import getLogger
from pathlib import Path
from time import time

# Third party packages
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        super(Checkpoint, self).__init__(
            logger=True,
            path=path,
            timestep=timestep,
        )
        # self.logger =getLogger(__name__)
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
        tokenizer: AutoTokenizer = None
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
        tokenizer: AutoTokenizer = None,
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
        self.logger.debug(f"Checkpoint is saving model and data")

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
                self.logger.debug(f"Data of size {len(data):,} saved at {data_path}")

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
            self.logger.debug(f"Data of size {len(data):,} loaded from {data_path}")

        self.logger.info(f"<Model and dataset loaded from checkpoint>")

        return llm, data

    def delete(self):
        """ Delete irreversibly the checkpoint. """
        # TODO : python3.12 => use walk method
        if self.path.exists():
            model_path = (self.path / "model")
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
        tokenizer: AutoTokenizer = None,
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

        self.logger.info(f"<Trained model saved at {path} and checkpoint deleted from "
                 f"{self.path}>")


class LoaderLLM(_Base):
    """ Load tokenizer and model.

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
        # self.logger = getLogger(__name__)
        super(LoaderLLM, self).__init__(
            logger=True,
            model_path=model_path,
            data_path=data_path,
            checkpoint=checkpoint,
            **kwargs
        )
        self.checkpoint = checkpoint
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
        )

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            self.logger.info(f"Set pad with eos token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and data (or last available checkpoint)
        try:
            self.llm, self.data = checkpoint.load(**kwargs)

        except Exception:
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                **kwargs,
            )
            self.logger.debug(f"<Model loaded from {model_path}>")

            self.data = loads(Path(data_path).open("r").read())
            self.logger.debug(f"<Dataset of size ({len(self.data):,}) loaded "
                              f"from {data_path}>")

            if checkpoint:
                self.checkpoint = Checkpoint()
                self.logger.debug(f"Initialize {self.checkpoint}")


if __name__ == "__main__":
    pass
