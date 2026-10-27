# SPDX-License-Identifier: Apache-2.0
"""Chat translation for the Strata external engine.

The child is a fake HTTP server or an httpx mock. These tests do not start
the Metal engine and do not post a completion at a live model.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from omlx.engine.strata import StrataBinary, StrataEngine, StrataTarget
from omlx.exceptions import InvalidRequestError


def _mock_engine(tmp_path, handler) -> StrataEngine:
    engine = _engine(tmp_path, "http://strata")
    engine._client = httpx.AsyncClient(
        base_url="http://strata",
        headers={"Authorization": f"Bearer {engine._api_key}"},
        transport=httpx.MockTransport(handler),
        timeout=30,
    )
    return engine


def _engine(tmp_path, base_url: str) -> StrataEngine:
    profile = tmp_path / "strata-flash.json"
    profile.write_text(
        json.dumps(
            {
                "exe": "component:strata-engine",
                "tokenizer": "tokenizer",
                "args": ["--max-context", "524288"],
                "model_name": "qwen3.8-flash-next-iq3_s",
            }
        )
    )
    engine = StrataEngine(
        model_name=str(tmp_path / "model"),
        model_id="Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S",
        target=StrataTarget(
            StrataBinary(tmp_path / "strata", "0.1.51", True, "metal"),
            profile,
        ),
        log_dir=tmp_path / "logs",
    )
    engine._client = httpx.AsyncClient(
        base_url=base_url,
        headers={"Authorization": f"Bearer {engine._api_key}"},
        timeout=30,
    )
    return engine


class _Server:
    """One loopback HTTP server on a free port, in a daemon thread."""

    def __init__(self, handler: type[BaseHTTPRequestHandler]) -> None:
        self.httpd = HTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=2)
        self.httpd.server_close()


class TestCount:
    async def test_count_returns_the_child_input_tokens(self, tmp_path):
        """The chat route counts on the child before it will forward a chat."""
        seen: dict = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                seen["path"] = self.path
                seen["auth"] = self.headers.get("Authorization")
                seen["body"] = json.loads(self.rfile.read(length).decode())
                payload = json.dumps({"input_tokens": 42}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        messages = [{"role": "user", "content": "hello"}]
        with _Server(Handler) as server:
            engine = _engine(tmp_path, server.url)
            count = engine.count_chat_tokens(messages)

        assert count == 42
        assert seen["path"].split("?", 1)[0].rstrip("/") == "/v1/messages/count_tokens"
        assert seen["auth"] == f"Bearer {engine._api_key}"
        assert seen["body"]["messages"] == messages

    async def test_a_failed_count_raises_the_child_message(self, tmp_path):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                payload = json.dumps(
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "message": "messages must be a list of objects",
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        with _Server(Handler) as server:
            engine = _engine(tmp_path, server.url)
            with pytest.raises(
                InvalidRequestError, match="messages must be a list of objects"
            ):
                engine.count_chat_tokens([{"role": "user", "content": "hello"}])


class TestChat:
    async def test_non_streaming_chat_folds_reasoning_and_usage(self, tmp_path):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "qwen3.8-flash-next-iq3_s",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "four",
                                "reasoning_content": "2+2",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 6,
                        "total_tokens": 17,
                        "prompt_tokens_details": {"cached_tokens": 3},
                    },
                },
            )

        engine = _mock_engine(tmp_path, handler)
        output = await engine.chat(
            [{"role": "user", "content": "2+2"}],
            max_tokens=32,
            temperature=0.2,
            top_p=0.8,
            top_k=100,
            min_p=0.05,
            repetition_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.1,
            seed=7,
            stop=["\n\n", "END", "A", "B", "C"],
            thinking_budget=128,
            repetition_context_size=64,
            tools=[{"type": "function", "function": {"name": "add"}}],
            xtc_probability=0.5,
            xtc_threshold=0.1,
            specprefill=True,
            response_format={"type": "json_object"},
            compiled_grammar="nope",
        )

        body = seen["body"]
        assert seen["path"] == "/v1/chat/completions"
        assert body["stream"] is False
        assert body["model"] == "qwen3.8-flash-next-iq3_s"
        assert body["model"] != engine._model_id
        assert body["messages"] == [{"role": "user", "content": "2+2"}]
        assert body["max_tokens"] == 32
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.8
        assert body["top_k"] == 64
        assert body["min_p"] == 0.05
        assert body["repetition_penalty"] == 1.1
        assert body["presence_penalty"] == 0.2
        assert body["frequency_penalty"] == 0.1
        assert body["seed"] == 7
        assert body["stop"] == ["\n\n", "END", "A", "B"]
        assert body["thinking_budget_tokens"] == 128
        assert body["penalty_last_n"] == 64
        for key in (
            "tools",
            "xtc_probability",
            "xtc_threshold",
            "specprefill",
            "response_format",
            "compiled_grammar",
            "thinking_token_budget",
        ):
            assert key not in body
        assert output.finished
        assert output.text == "<think>2+2</think>four"
        assert output.new_text == output.text
        assert output.finish_reason == "stop"
        assert (
            output.prompt_tokens,
            output.completion_tokens,
            output.cached_tokens,
        ) == (11, 6, 3)

    async def test_reasoning_without_content_is_closed(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": None, "reasoning_content": "hmm"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 1},
                },
            )

        output = await _mock_engine(tmp_path, handler).chat(
            [{"role": "user", "content": "hi"}]
        )
        assert output.text == "<think>hmm</think>"
        assert output.finish_reason == "length"
        assert output.cached_tokens == 0
