#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-06 10:49:22
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-11 14:52:42
# @File path: ./pyllmsol/inference/session_manager.py
# @Project: PyLLMSol

""" Session manager object. """

# Built-in packages
from datetime import datetime, timedelta
import uuid

# Third party packages
from llama_cpp import Llama

# Local packages
from pyllmsol._base import _Base
from pyllmsol.data._base_data import _TextData
from pyllmsol.inference._base_cli import _BaseCommandLineInterface

__all__ = []


class SessionManager(_Base):
    """ Manages user sessions and their interactions with a LLM.

    This class creates, retrieves, and manages user sessions, each associated
    with a command-line interface (CLI) instance. Sessions are identified by
    unique IDs and have a configurable timeout to ensure efficient memory usage.

    Parameters
    ----------
    llm : Llama
        The LLM instance to be used for all sessions.
    init_prompt : str or _TextData, optional
        The initial prompt to initialize each session's CLI.
    session_timeout : int, optional
        The duration in minutes after which a session is considered inactive.
        Default is 30 minutes.
    cli_class : type, optional
        The CLI class to be instantiated for each session. Default is
        `_BaseCommandLineInterface`.

    Methods
    -------

    Attributes
    ----------
    llm : Llama
        The LLM instance used by all sessions.
    init_prompt : str or _TextData
        The initial prompt for each session.
    session_timeout : timedelta
        The timeout duration for sessions.
    cli_class : type
        The class used to create CLI instances for sessions.
    sessions : dict
        A dictionary mapping session IDs to their associated data, including CLI
        instances and last activity timestamps.

    """

    def __init__(
        self,
        llm: Llama,
        init_prompt: str | _TextData = None,
        session_timeout: int = 30,
        cli_class: type = _BaseCommandLineInterface,
    ):
        super().__init__(
            llm,
            init_prompt=init_prompt,
            session_timeout=session_timeout,
        )
        self.llm = llm
        self.init_prompt = init_prompt
        self.session_timeout = timedelta(minutes=session_timeout)
        self.cli_class = cli_class
        self.sessions = {}

    def create_session(self, session_id: str = None):
        """ Create a new session with a unique ID.

        Parameters
        ----------
        session_id : str, optional
            A custom session ID. If None, a UUID will be generated. Default is
            None.

        Returns
        -------
        str
            The ID of the created session.

        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        cli = self.cli_class(llm=self.llm, init_prompt=self.init_prompt)
        self.set_session(session_id, cli)

        return session_id

    def set_session(self, session_id: str, cli: _BaseCommandLineInterface):
        """ Store a session in the manager.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        cli : _BaseCommandLineInterface
            The CLI instance associated with the session.

        """
        self.sessions[session_id] = {
            "cli": cli,
            "last_active": datetime.now(),
        }

    def get_session(self, session_id: str):
        """ Retrieve an active session by its ID.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.

        Returns
        -------
        _BaseCommandLineInterface
            The CLI instance associated with the session.

        Raises
        ------
        KeyError
            If the session ID is not found.
        TimeoutError
            If the session has expired.

        """
        session_data = self.sessions.get(session_id)

        if not session_data:
            raise KeyError("Session not found")

        if datetime.now() - session_data['last_active'] > self.session_timeout:
            raise TimeoutError("Session expired")

        session_data["last_active"] = datetime.now()

        return session_data['cli']

    def del_session(self, session_id: str):
        """ Delete a session from the manager.

        Parameters
        ----------
        session_id : str
            The unique identifier of the session to delete.

        """
        del self.sessions[session_id]

    def __setitem__(self, session_id: str, cli: _BaseCommandLineInterface):
        """ Set a session using dictionary-like syntax.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        cli : _BaseCommandLineInterface
            The CLI instance associated with the session.

        """
        return self.set_session(session_id, cli)

    def __getitem__(self, session_id: str):
        """ Retrieve a session using dictionary-like syntax.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.

        Returns
        -------
        _BaseCommandLineInterface
            The CLI instance associated with the session.

        """
        return self.get_session(session_id)

    def __delitem__(self, session_id: str):
        """ Delete a session using dictionary-like syntax.

        Parameters
        ----------
        session_id : str
            The unique identifier of the session to delete.

        """
        return self.del_session(session_id)


if __name__ == "__main__":
    pass
