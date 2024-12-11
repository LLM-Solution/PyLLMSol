#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-11 15:30:53
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 15:37:17
# @File path: ./pyllmsol/test/inference/test_session_manager.py
# @Project: PyLLMSol

""" Test session manager object. """

# Built-in packages
from datetime import timedelta

# Third party packages
import pytest

# Local packages
from pyllmsol.inference.session_manager import SessionManager
from pyllmsol.tests.mock import MockLlama, MockTokenizer
from pyllmsol.inference._base_cli import _BaseCommandLineInterface

__all__ = []


@pytest.fixture
def session_manager():
    """Fixture to create a SessionManager instance with mocked LLM and CLI."""
    mock_llm = MockLlama()
    return SessionManager(llm=mock_llm, session_timeout=1, cli_class=_BaseCommandLineInterface)


def test_create_session(session_manager):
    """Test creating a new session."""
    session_id = session_manager.create_session()
    assert session_id in session_manager.sessions
    assert isinstance(session_manager.sessions[session_id]["cli"], _BaseCommandLineInterface)
    assert "last_active" in session_manager.sessions[session_id]


def test_get_session(session_manager):
    """Test retrieving an active session."""
    session_id = session_manager.create_session()
    cli = session_manager.get_session(session_id)
    assert isinstance(cli, _BaseCommandLineInterface)


def test_get_session_expired(session_manager):
    """Test retrieving an expired session."""
    session_id = session_manager.create_session()

    # Simulate session expiration
    session_manager.sessions[session_id]["last_active"] -= timedelta(minutes=2)

    with pytest.raises(TimeoutError):
        session_manager.get_session(session_id)


def test_set_session(session_manager):
    """Test manually setting a session."""
    session_id = "test_session"
    mock_cli = _BaseCommandLineInterface(MockLlama())
    session_manager.set_session(session_id, mock_cli)

    assert session_id in session_manager.sessions
    assert session_manager.sessions[session_id]["cli"] == mock_cli


def test_delete_session(session_manager):
    """Test deleting a session."""
    session_id = session_manager.create_session()
    session_manager.del_session(session_id)

    assert session_id not in session_manager.sessions


def test_dict_like_access(session_manager):
    """Test dictionary-like access to set, get, and delete sessions."""
    session_id = "dict_session"
    mock_cli = _BaseCommandLineInterface(MockLlama())

    # Set session
    session_manager[session_id] = mock_cli
    assert session_id in session_manager.sessions

    # Get session
    retrieved_cli = session_manager[session_id]
    assert retrieved_cli == mock_cli

    # Delete session
    del session_manager[session_id]
    assert session_id not in session_manager.sessions


if __name__ == "__main__":
    pass