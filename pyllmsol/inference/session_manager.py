#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-06 10:49:22
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-10 17:15:02
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


class UserManager(_Base):
    def __init__(self):
        super().__init__()
        self.users = {}


class SessionManager(_Base):
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
        if session_id is None:
            session_id = str(uuid.uuid4())

        cli = self.cli_class(llm=self.llm, init_prompt=self.init_prompt)
        self.set_session(session_id, cli)

        return session_id

    def set_session(self, session_id, cli):
        self.sessions[session_id] = {
            "cli": cli,
            "last_active": datetime.now(),
        }

    def get_session(self, session_id):
        session_data = self.sessions.get(session_id)

        if not session_data:
            raise KeyError("Session not found")

        elif datetime.now() - session_data['last_active'] > self.session_timeout:
            raise TimeoutError("Session expired")

        session_data["last_active"] = datetime.now()

        return session_data['cli']

    def del_session(self, session_id):
        del self.sessions[session_id]

    def __setitem__(self, session_id, cli):
        return self.set_session(session_id, cli)

    def __getitem__(self, session_id):
        return self.get_session(session_id)

    def __delitem__(self, session_id):
        return self.del_session(session_id)


if __name__ == "__main__":
    pass
