#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2023-10-09 17:57:37
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-30 15:15:29

""" Objects to load, save and/or make a checkpoint of models and data. """

# Built-in packages
from json import loads, dumps
from logging import getLogger
from pathlib import Path
from time import time

# Third party packages
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local packages

__all__ = []


LOG = getLogger('checkpoint')


class Checkpoint:
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
        # Set variables
        self.path = Path(path) if isinstance(path, str) else path
        self.timestep = timestep
        self.ts = time()

        # Create path if not already exist
        self.path.mkdir(parents=True, exist_ok=True)
        LOG.debug("<Checkpoint object is initiated>")

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
        LOG.debug(f"Checkpoint is saving model and data")

        # Save model
        model_path = self.path / "model"
        llm.save_pretrained(model_path)
        LOG.debug(f"Model saved at {model_path}")

        # save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(model_path)
            LOG.debug(f"Tokenizer saved at {model_path}")

        # Save data
        if data is not None:
            data_path = self.path / "data.json"
            with (data_path).open("w", encoding="utf8") as f:
                f.write(dumps(data, ensure_ascii=False))
                LOG.debug(f"Data of size {len(data):,} saved at {data_path}")

        LOG.info(f"<Checkpoint saved at: '{self.path}'>")

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
        LOG.debug(f"Model loaded from {model_path}")

        # Load LoRA weights
        # try:
        #     llm = PeftModel.from_pretrained(llm, self.path / "model")
        #     llm = llm.merge_and_unload()
        #     LOG.debug("Previous trained LoRA weights are loaded and merged")

        # except Exception as e:
        #     print(e)
        #     LOG.debug("There is no previous trained LoRA weights")

        # Load data
        data_path = self.path / "data.json"
        with (data_path).open("r") as f:
            data = loads(f.read())
            LOG.debug(f"Data of size {len(data):,} loaded from {data_path}")

        LOG.info(f"<Model and dataset loaded from checkpoint>")

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
                LOG.debug(f"Model deleted from {model_path}")

            # Delete data file
            data_path = self.path / "data.json"
            if data_path.exists():
                data_path.unlink()
                LOG.debug(f"Data deleted from {data_path}")

            # Delete folder
            self.path.rmdir()
            LOG.info(f"Checkpoint deleted from {self.path}")

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
        LOG.debug(f"Model saved at {path}")

        if tokenizer is not None:
            tokenizer.save_pretrained(path)
            LOG.debug(f"Tokenizer saved at {path}")

        # Delete checkpoint
        self.delete()

        LOG.info(f"<Trained model saved at {path} and checkpoint deleted from "
                 f"{self.path}>")


def loader(
    model_path: str | Path,
    data_path: str | Path,
    checkpoint: bool | Checkpoint = False,
    **kwargs,
):
    try:
        llm, data = checkpoint.load(**kwargs)

    except Exception:
        llm = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        LOG.debug(f"<Model loaded from {model_path}>")

        data = loads(Path(data_path).open("r").read())
        LOG.debug(f"<Dataset of size ({len(data):,}) loaded from {data_path}>")

    return llm, data


class LoaderLLM:
    """ Load tokenizer and model. """

    def __init__(
        self,
        model_name: Path,
        data_path: Path,
        checkpoint: bool | Checkpoint,
        **kw_load_model,
    ):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
        )

        # /!\ LLaMa model have not pad token
        if self.tokenizer.pad_token is None:
            LOG.info(f"Set pad with eos token {self.tokenizer.eos_token}")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model and data (or last available checkpoint)
        self.llm, self.data = loader(model_name, data_path,
                                     checkpoint=checkpoint, **kw_load_model)


if __name__ == "__main__":
    pass
