#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-02 16:49:44
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 18:03:24
# @File path: ./pyllmsol/tests/training/test_checkpoint.py
# @Project: PyLLMSol

""" Test checkpoint objects. """

# Built-in packages
from pathlib import Path
import tempfile
import time
from unittest.mock import MagicMock, patch

# Third party packages
import pytest
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Local packages
from pyllmsol.training.checkpoint import Checkpoint

__all__ = []


@pytest.fixture
def temp_dir():
    """Fixture for a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_dir_2():
    """Fixture for a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_llm():
    """Mock a language model."""
    llm = MagicMock(spec=PreTrainedModel)
    return llm


@pytest.fixture
def mock_tokenizer():
    """Mock a tokenizer."""
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    return tokenizer


@pytest.fixture
def checkpoint(temp_dir):
    """Initialize a Checkpoint instance with a temporary directory."""
    return Checkpoint(path=temp_dir, timestep=300)


def test_checkpoint_initialization(checkpoint, temp_dir):
    """Test that the checkpoint initializes correctly."""
    assert checkpoint.path == temp_dir
    assert checkpoint.timestep == 300
    assert checkpoint.path.exists()
    assert checkpoint.path.is_dir()


def test_checkpoint_bool(checkpoint):
    """Test the boolean logic for checkpoint timing."""
    initial_ts = checkpoint.ts
    assert not checkpoint  # Should be False since not enough time has passed
    time.sleep(1)
    checkpoint.timestep = 1  # Adjust timestep to test timing
    assert checkpoint  # Should be True now


def test_save_checkpoint(checkpoint, mock_llm, mock_tokenizer):
    """Test saving a checkpoint."""
    data = [{"key": "value"}]
    checkpoint.save(mock_llm, data, tokenizer=mock_tokenizer)

    # Check model and tokenizer were saved
    model_path = checkpoint.path / "model"
    # assert model_path.exists()
    mock_llm.save_pretrained.assert_called_with(model_path)
    mock_tokenizer.save_pretrained.assert_called_with(model_path)

    # Check data was saved
    data_path = checkpoint.path / "data.json"
    assert data_path.exists()
    with data_path.open("r", encoding="utf-8") as f:
        saved_data = f.read()
    assert '{"key": "value"}' in saved_data


def test_load_checkpoint(checkpoint, mock_llm, mock_tokenizer):
    """Test loading a checkpoint."""
    with patch("pyllmsol.training.checkpoint.AutoModelForCausalLM.from_pretrained") as mock_model_loader:
        mock_model_loader.return_value = mock_llm
        
        # Create mock data
        data = [{"key": "value"}]
        data_path = checkpoint.path / "data.json"
        with data_path.open("w", encoding="utf-8") as f:
            f.write('["mock_data"]')

        llm, loaded_data = checkpoint.load()
        mock_model_loader.assert_called_with(checkpoint.path / "model")
        assert llm == mock_llm
        assert loaded_data == ["mock_data"]


def test_delete_checkpoint(checkpoint):
    """Test deletion of a checkpoint."""
    # Create mock files
    model_path = checkpoint.path / "model"
    data_path = checkpoint.path / "data.json"
    model_path.mkdir(parents=True)
    data_path.touch()

    checkpoint.delete()
    
    # Ensure files and directory are deleted
    assert not model_path.exists()
    assert not data_path.exists()
    assert not checkpoint.path.exists()


def test_save_trained_model(checkpoint, mock_llm, mock_tokenizer, temp_dir, temp_dir_2):
    """Test saving the trained model."""
    save_path = temp_dir_2 / "final_model"
    checkpoint.save_trained_model(mock_llm, save_path, tokenizer=mock_tokenizer)

    # Check final model and tokenizer were saved
    assert save_path.exists()
    mock_llm.save_pretrained.assert_called_with(save_path)
    mock_tokenizer.save_pretrained.assert_called_with(save_path)

    # Check checkpoint was deleted
    assert not checkpoint.path.exists()


if __name__ == "__main__":
    pass
