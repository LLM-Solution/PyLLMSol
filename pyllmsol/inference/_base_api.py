#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-31 08:59:41
# @Last modified by: ArthurBernard
# @Last modified time: 2024-10-31 09:23:10

""" Description. """

# Built-in packages
from functools import wraps
from logging import getLogger
from pathlib import Path

# Third party packages
from flask import Flask, request, make_response, jsonify, stream_with_context, Response

# Local packages

__all__ = []


class API:
    """ Flask API object to run a LLM chatbot.

    Parameters
    ----------
    model_path : str
        Path of the model to load (must be GGUF format).
    init_prompt : str
        Initial prompt to feed the LLM in the CLI.
    lora_path : str, optional
        Path of LoRA weights to load.
    n_ctx : int, optional
        Max number of tokens in the prompt, default is 4096.
    debug : bool, optional
        If True then don't block app if lock_file already exist. It means
        several servers can run simultanously.

    Methods
    -------
    add_get_cli_route
    add_post_cli_route
    add_route
    run

    Attributes
    ----------
    app : Flask
       Flask application.
    cli : CommandLineInterface
       Chatbot object.
    debug : bool, optional
        If True then don't block app if lock_file already exist. It means
        several servers can run simultanously.
    init_prompt : str
        Initial prompt to feed the LLM in the CLI.

    """

    lock_file = Path("/tmp/api.py.lock")

    def __init__(
        self,
        model_path: str | Path,
        init_prompt: str,
        lora_path: str | Path = None,
        n_ctx: int = 32768,
        debug: bool = False,
    ):
        self.init_prompt = init_prompt
        self.debug = debug

        LOG.debug("Start init Flask API object")
        self.app = Flask(__name__)
        self.add_route()

        # Set CLI object
        lora_path = str(lora_path) if lora_path else None
        self.cli = _BaseCommandLineInterface(
            model_path,
            lora_path=lora_path,
            init_prompt=self.init_prompt,
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            n_threads=None,
        )

        # Add GET and POST to CLI routes
        self.add_get_cli_route()
        self.add_post_cli_route()

        LOG.debug("Flask API object is initiated")

    def add_route(self):
        """ Add classical routes. """
        @self.app.route("/shutdown", methods=["POST"])
        def shutdown():
            """ Shutdown flask API server.

            Returns
            -------
            str
                Server sent "Server shutting down...".

            """
            LOG.debug("Shutdown call")
            func = request.environ.get("werkzeug.server.shutdown")

            if func is None:

                raise RuntimeError("Not running with the Werkzeug Server")

            func()

            return "Server shutting down..."

        @self.app.route("/health", methods=['GET'])
        def health_check():
            """ Check status.

            Returns
            -------
            flask.Response
                Status code 200.

            """
            LOG.debug("GET health")

            return Response(status=200)

        @self.app.route("/ping", methods=['GET'])
        def ping():
            """ Ping the server.

            Returns
            -------
            str
                Server sent "pong".

            """
            LOG.debug("pong")

            return 'pong'

    def add_get_cli_route(self):
        """ Add GET routes to communicate with the CLI. """
        @self.app.route("/reset_prompt", methods=['GET'])
        def reset_prompt():
            """ Reset the prompt of the LLM.

            Returns
            -------
            flask.Response
                Status code 200.

            Examples
            --------
            >>> requests.get("http://0.0.0.0:5000/reset_prompt")
            <Response [200]>

            """
            self.cli.reset_prompt()
            LOG.debug(f"GET reset prompt")

            return Response(status=200)

        @self.app.route("/get_prompt", methods=['GET'])
        def get_prompt():
            """ Return the prompt of the LLM.

            Returns
            -------
            str
                The prompt of LLM.

            Examples
            --------
            >>> requests.get("http://0.0.0.0:5000/get_prompt")
            [Q]Who is the president of USA ?[/Q][A]Joe Biden.[/A]

            """
            prompt = self.cli.prompt_hist.to_json()
            LOG.debug(f"GET prompt : {prompt}")

            return prompt

    def add_post_cli_route(self):
        """ Add POST routes to communicate with the CLI. """
        @self.app.route("/set_init_prompt", methods=['POST'])
        def set_init_prompt():
            """ Set the prompt to the LLM.

            Returns
            -------
            flask.Response
                Status code 200.

            Examples
            --------
            >>> output = resquests.post(
            ...     "http://0.0.0.0:5000/set_init_prompt",
            ...     json={
            ...         "init_prompt": ("Conversation between an helpfull AI" 
            ...                         "assistant and a human.")
            ...     },
            ... )
            <Response streamed [200 OK]>

            """
            init_prompt = request.json.get("init_prompt")
            LOG.debug(f"POST set prompt : {init_prompt}")
            self.cli.init_prompt = init_prompt

            return Response(status=200)

        @self.app.route("/ask", methods=['POST', 'OPTIONS'])
        @cors_required
        def ask():
            """ Ask a question to the LLM.

            Returns
            -------
            str or generator of str
                If stream is True then the answer of the LLM is streamed,
                otherwise the answer is a string.

            Examples
            --------
            >>> output = requests.post(
            ...     "http://0.0.0.0:5000/ask",
            ...     json={
            ...         "question": "Who is the president of USA ?",
            ...         "stream": True,
            ...     },
            ... )
            >>> for txt in output.iter_content():
            ...     print(txt.decode('utf8'), end='', flush=True)
            Bot: Joe Biden is the president of USA.

            """
            question = request.json.get("question")
            stream = request.json.get("stream", True)
            session_id = request.json.get("session_id")
            LOG.debug(f"ask: {question}")

            # FIXME : should be escaped ? to avoid code injection
            # return self.cli.ask(escape(question), stream=stream)

            answer = self.cli.ask(question, stream=stream)  # session_id

            if stream:
                return Response(answer, content_type='text/event-stream')

            else:
                return answer

        @self.app.route("/call", methods=['POST', 'OPTIONS'])
        @cors_required
        def call():
            """ Call the LLM with a given raw prompt.

            Returns
            -------
            str or generator of str
                If stream is True then the output of the LLM is streamed,
                otherwise the output is a string.

            Examples
            --------
            >>> output = requests.post(
            ...     "http://0.0.0.0:5000/call",
            ...     json={
            ...         "prompt": ("Conversation between an helpfull Chatbot" 
            ...                    "assistant and a human.\nUser: Who is the "
            ...                    "president of USA ?"),
            ...         "stream": False,
            ...     },
            ... )
            >>> output.text
            Bot: Joe Biden is the president of USA.

            """
            prompt = request.json.get("prompt")
            stream = request.json.get("stream", True)
            LOG.debug(f"call: {prompt}")

            answer = self.cli(prompt, stream=stream)

            if stream:
                return Response(answer, content_type='text/event-stream')

            else:
                return answer

    def run(self, timer=0, **kwargs):
        """ Start to run the flask API.

        Parameters
        ----------
        timer : int, optional
            A timer in secondes to shutdown the flask app, default is 0 (never
            shutdown).
        **kwargs
            Keywords arguments for `flask.Flask.run` method, cf flask
            documentation.

        """
        if timer > 0:
            LOG.debug(f"Flask API is running for {timer} seconds")
            flask_thread = Thread(
                target=self.app.run,
                kwargs=kwargs,
                daemon=True,
            )
            flask_thread.start()

            timer_thread = Thread(
                target=self._timer_to_shutdown,
                args=(timer,),
            )
            timer_thread.start()
            timer_thread.join()

        else:
            LOG.debug("Flask API is running")
            self.app.run(**kwargs)

    def _timer_to_shutdown(self, duration):
        LOG.debug(f"API will shutdown in {duration} seconds")
        sleep(duration)
        LOG.debug("End of timer, API server shutting down")
        exit(0)

    def __enter__(self):
        LOG.debug("Enter in control manager")
        # Lock file to block an other app to run
        self.lock_file.touch(exist_ok=self.debug)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Unlock file
        self.lock_file.unlink(missing_ok=self.debug)
        LOG.debug("Exit of control manager")

        return False


if __name__ == "__main__":
    pass
