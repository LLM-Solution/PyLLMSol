#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-02 11:39:56
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-02 11:48:04
# @File path: ./pyllmsol/tests/inference/test_api.py
# @Project: PyLLMSol

""" Test API object. """

# Built-in packages
from unittest.mock import patch

# Third party packages
import pytest

# Local packages
from pyllmsol.mock import MockLlama
from pyllmsol.inference.api import API
from pyllmsol.inference.cli import CommandLineInterface


__all__ = []


@pytest.fixture
def api():
    llm = MockLlama()
    cli = CommandLineInterface(llm, init_prompt="Init prompt.")

    # Initialize the API normally
    api = API(cli, debug=True)

    yield api


@pytest.fixture
def client(api):
    return api.app.test_client()


def test_ping(client):
    response = client.get('/ping')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == 'pong'


def test_health(client):
    response = client.get('/health')
    assert response.status_code == 200


def test_ask(client, api):
    response = client.post('/ask', json={"question": "What is AI?", "stream": False})
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "LLM response."


def test_get_prompt(client, api):
    # api.cli.prompt_hist.to_json.return_value = '{"prompt": "current prompt"}'
    response = client.get('/get_prompt')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == '{"text":"Init prompt."}\n'
    response = client.post('/ask', json={"question": "What is AI?", "stream": False})
    response = client.get('/get_prompt')
    assert response.status_code == 200
    assert response.json == {'text': "Init prompt.\nUser: What is AI?\nAssistant: "}


def test_reset_prompt(client, api):
    response = client.post('/ask', json={"question": "What is AI?", "stream": False})
    response = client.get('/reset_prompt')
    response = client.get('/get_prompt')
    assert response.status_code == 200
    assert response.data.decode('utf-8') == '{"text":"Init prompt."}\n'
    assert response.get_json() == {"text":"Init prompt."}


def test_set_init_prompt(client, api):
    response = client.post('/set_init_prompt', json={"init_prompt": "New prompt"})
    assert response.status_code == 200
    api.cli.init_prompt = "New prompt"


def test_call(client, api):
    response = client.post('/call', json={"prompt": "Tell me a story.", "stream": False})
    assert response.status_code == 200
    assert response.data.decode('utf-8') == "LLM response."


@patch('pyllmsol.inference.api.Thread')
def test_run_with_timer(mock_thread, api):
    api.run(timer=5)
    mock_thread.assert_called()


def test_debug_mode(api):
    assert api.debug is True


if __name__ == "__main__":
    pass
