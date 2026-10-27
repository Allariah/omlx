# SPDX-License-Identifier: Apache-2.0
"""Chat translation for the Strata external engine.

The child is a fake HTTP server or an httpx mock. These tests do not start
the Metal engine and do not post a completion at a live model.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import HTTPException

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
        assert "tools" not in seen["body"]

    async def test_count_forwards_tools(self, tmp_path):
        seen: dict = {}
        tools = [{"type": "function", "function": {"name": "add"}}]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                seen["body"] = json.loads(self.rfile.read(length).decode())
                payload = json.dumps({"input_tokens": 9}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *_args):
                return

        with _Server(Handler) as server:
            engine = _engine(tmp_path, server.url)
            count = engine.count_chat_tokens(
                [{"role": "user", "content": "add"}], tools=tools
            )

        assert count == 9
        assert seen["body"]["tools"] == tools

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
            logprobs=True,
            top_logprobs=5,
            splash_kv_format="int8",
            splash_ssd_cache_gb=32,
            splash_ane_prefill=True,
            kv_format="bf16",
            ane_prefill=False,
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
        assert body["tools"] == [{"type": "function", "function": {"name": "add"}}]
        for key in (
            "logprobs",
            "top_logprobs",
            "splash_kv_format",
            "splash_ssd_cache_gb",
            "splash_ane_prefill",
            "kv_format",
            "ane_prefill",
            "xtc_probability",
            "xtc_threshold",
            "specprefill",
            "response_format",
            "compiled_grammar",
            "thinking_token_budget",
        ):
            assert key not in body
        assert engine.supports_logprobs is False
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

    async def test_a_tool_call_is_forwarded_and_mapped(self, tmp_path):
        seen: dict = {}
        tools = [{"type": "function", "function": {"name": "add"}}]

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "add",
                                            "arguments": {"a": 1},
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                },
            )

        engine = _mock_engine(tmp_path, handler)
        output = await engine.chat(
            [{"role": "user", "content": "1+1"}],
            tools=tools,
        )

        assert seen["body"]["tools"] == tools
        assert engine.supports_tool_calling is True
        assert output.finish_reason == "tool_calls"
        assert output.tool_calls == [
            {"id": "call_1", "name": "add", "arguments": '{"a": 1}'}
        ]


def _sse_line(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _post_chat_route(
    tmp_path,
    monkeypatch,
    payload: dict,
    handler: type[BaseHTTPRequestHandler] | None = None,
    seen: list[str] | None = None,
):
    """POST /v1/chat/completions at a Strata engine backed by ``handler``."""
    from fastapi.testclient import TestClient

    import omlx.server as srv

    recorded = [] if seen is None else seen
    if handler is None:

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                recorded.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = json.dumps({"input_tokens": 1}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                return

        handler = Handler

    with _Server(handler) as server:
        engine = _engine(tmp_path, server.url)
        pool = MagicMock()
        pool.preload_pinned_models = AsyncMock()
        pool.check_ttl_expirations = AsyncMock()
        pool.shutdown = AsyncMock()
        pool.get_entry.return_value = None
        monkeypatch.setattr(srv._server_state, "engine_pool", pool)
        monkeypatch.setattr(
            srv, "get_engine_for_model", AsyncMock(return_value=engine)
        )
        monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
        monkeypatch.setattr(
            srv, "get_model_settings_for_request", lambda name: None
        )
        monkeypatch.setitem(
            srv.app.dependency_overrides,
            srv.verify_inference_api_key,
            lambda: True,
        )
        try:
            with TestClient(srv.app, raise_server_exceptions=False) as client:
                return client.post("/v1/chat/completions", json=payload)
        finally:
            srv.app.dependency_overrides.pop(srv.verify_inference_api_key, None)


class TestImages:
    def _image_message(self) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,aaaa"},
                    },
                ],
            }
        ]

    async def test_vision_off_rejects_an_image(self, tmp_path):
        engine = _engine(tmp_path, "http://strata")
        assert engine.accepts_image_input is False
        with pytest.raises(InvalidRequestError, match="vision is off"):
            await engine.chat(self._image_message())

    async def test_vision_on_forwards_the_image(self, tmp_path):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {"content": "a dot"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            )

        engine = _mock_engine(tmp_path, handler)
        profile = json.loads(engine._target.profile_path.read_text())
        profile["ui_vision_enabled"] = True
        engine._target.profile_path.write_text(json.dumps(profile))
        messages = self._image_message()

        output = await engine.chat(messages)

        assert engine.accepts_image_input is True
        assert seen["body"]["messages"] == messages
        assert output.text == "a dot"

    def test_the_route_rejects_an_image_before_the_child(self, tmp_path, monkeypatch):
        seen: list[str] = []
        response = _post_chat_route(
            tmp_path,
            monkeypatch,
            {
                "model": "flash",
                "stream": False,
                "messages": self._image_message(),
            },
            seen=seen,
        )

        assert response.status_code == 400, response.text
        assert "vision is off" in response.json()["error"]["message"]
        assert seen == []

    async def test_anthropic_messages_refuse_an_image(self, tmp_path, monkeypatch):
        """The messages route uses the same vision refusal as chat completions."""
        import omlx.server as srv
        from omlx.api.anthropic_models import MessagesRequest

        engine = _engine(tmp_path, "http://strata")
        monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
        request = MessagesRequest(
            model="flash",
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "aaaa",
                            },
                        }
                    ],
                }
            ],
        )
        with pytest.raises(InvalidRequestError, match="vision is off"):
            await srv.create_anthropic_message(request, MagicMock(), True)


class TestStream:
    async def test_stream_yields_deltas_then_usage(self, tmp_path):
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["body"] = json.loads(request.content)
            raw = (
                ": keep-alive\n\n"
                + _sse_line({"choices": [{"delta": {"reasoning_content": "hmm"}}]})
                + _sse_line({"choices": [{"delta": {"content": "Hi"}}]})
                + ": keep-alive\n\n"
                + _sse_line(
                    {
                        "choices": [{"delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 7,
                            "completion_tokens": 3,
                            "total_tokens": 10,
                            "prompt_tokens_details": {"cached_tokens": 4},
                        },
                    }
                )
                + "data: [DONE]\n\n"
                + _sse_line({"choices": [{"delta": {"content": "after"}}]})
            )
            return httpx.Response(200, content=raw.encode())

        engine = _mock_engine(tmp_path, handler)
        outputs = [
            item
            async for item in engine.stream_chat(
                [{"role": "user", "content": "hi"}],
                max_tokens=16,
                return_progress=True,
            )
        ]

        body = seen["body"]
        assert body["stream"] is True
        assert body["model"] == "qwen3.8-flash-next-iq3_s"
        assert body["stream_options"] == {"include_usage": True}
        assert body["return_progress"] is True
        assert [item.new_text for item in outputs if not item.finished] == [
            "<think>hmm",
            "</think>Hi",
        ]
        assert all(item.text.endswith(item.new_text) for item in outputs if not item.finished)
        final = outputs[-1]
        assert final.finished
        assert final.text == "<think>hmm</think>Hi"
        assert "after" not in final.text
        assert final.finish_reason == "stop"
        assert (
            final.prompt_tokens,
            final.completion_tokens,
            final.cached_tokens,
        ) == (7, 3, 4)

    async def test_streamed_tool_calls_are_joined(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            raw = (
                _sse_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_9",
                                            "function": {"name": "add", "arguments": ""},
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                + _sse_line(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '{"a": 2}'},
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ]
                    }
                )
                + "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=raw.encode())

        outputs = [
            item
            async for item in _mock_engine(tmp_path, handler).stream_chat(
                [{"role": "user", "content": "add"}]
            )
        ]
        final = outputs[-1]
        assert final.finished
        assert final.finish_reason == "tool_calls"
        assert final.tool_calls == [
            {"id": "call_9", "name": "add", "arguments": '{"a": 2}'}
        ]


def _error_response(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"error": {"message": message}})


class TestErrors:
    async def test_a_child_400_is_an_invalid_request(self, tmp_path):
        engine = _mock_engine(
            tmp_path,
            lambda request: _error_response(400, "bad top_p"),
        )
        with pytest.raises(InvalidRequestError, match="^bad top_p$"):
            await engine.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.parametrize("status", [401, 409, 429, 502, 503])
    async def test_a_child_status_stays_that_status(self, tmp_path, status):
        message = f"strata {status}"
        engine = _mock_engine(
            tmp_path,
            lambda request: _error_response(status, message),
        )
        with pytest.raises(HTTPException) as caught:
            await engine.chat([{"role": "user", "content": "hi"}])
        assert caught.value.status_code == status
        assert caught.value.detail == message

    async def test_an_open_stream_carries_the_child_message(self, tmp_path):
        engine = _mock_engine(
            tmp_path,
            lambda request: _error_response(409, "model is busy"),
        )
        with pytest.raises(RuntimeError, match="^model is busy$"):
            async for _item in engine.stream_chat([{"role": "user", "content": "hi"}]):
                pass

    async def test_a_mid_stream_error_event_carries_the_message(self, tmp_path):
        def handler(request: httpx.Request) -> httpx.Response:
            raw = (
                _sse_line({"choices": [{"delta": {"content": "Hi"}}]})
                + _sse_line({"error": {"message": "the engine stopped"}})
                + "data: [DONE]\n\n"
            )
            return httpx.Response(200, content=raw.encode())

        engine = _mock_engine(tmp_path, handler)
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="^the engine stopped$"):
            async for item in engine.stream_chat([{"role": "user", "content": "hi"}]):
                seen.append(item.new_text)
        assert seen == ["Hi"]

    def test_the_route_keeps_a_child_409(self, tmp_path, monkeypatch):
        """A Strata 409 stays 409. A RuntimeError would be Internal server error."""

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                path = self.path.split("?", 1)[0].rstrip("/")
                if path.endswith("/count_tokens"):
                    status, payload = 200, {"input_tokens": 4}
                else:
                    status, payload = 409, {
                        "error": {
                            "type": "model_busy",
                            "message": "a request is running",
                        }
                    }
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return

        response = _post_chat_route(
            tmp_path,
            monkeypatch,
            {
                "model": "flash",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            },
            handler=Handler,
        )

        assert response.status_code == 409, response.text
        assert response.json()["error"]["message"] == "a request is running"
        assert "Internal server error" not in response.text

    def test_logprobs_are_rejected_before_the_child(self, tmp_path, monkeypatch):
        seen: list[str] = []
        response = _post_chat_route(
            tmp_path,
            monkeypatch,
            {
                "model": "flash",
                "stream": False,
                "logprobs": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            seen=seen,
        )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["message"] == (
            "logprobs is not supported for this model."
        )
        assert seen == []


def _json_response(handler: BaseHTTPRequestHandler, payload: dict) -> None:
    body = json.dumps(payload).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class TestContext:
    async def test_health_max_context_is_the_window(self, tmp_path):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    payload = {
                        "status": "ok",
                        "max_context": 524288,
                        "model": "qwen3.8-flash-next-iq3_s",
                        "loaded": True,
                    }
                else:
                    payload = {
                        "data": [
                            {
                                "id": "qwen3.8-flash-next-iq3_s",
                                "meta": {"n_ctx": 111},
                            }
                        ]
                    }
                _json_response(self, payload)

            def log_message(self, *_args):
                return

        with _Server(Handler) as server:
            engine = _engine(tmp_path, server.url)
            await engine._remember_context()
        assert engine.max_context_tokens == 524288

    async def test_models_meta_supplies_the_window_without_health(self, tmp_path):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/health":
                    payload = {"status": "ok", "loaded": True}
                else:
                    payload = {
                        "data": [
                            {
                                "id": "qwen3.8-flash-next-iq3_s",
                                "meta": {"n_ctx": 524288},
                            }
                        ]
                    }
                _json_response(self, payload)

            def log_message(self, *_args):
                return

        with _Server(Handler) as server:
            engine = _engine(tmp_path, server.url)
            await engine._remember_context()
        assert engine.max_context_tokens == 524288

    def test_the_context_check_uses_the_child_window(self, tmp_path):
        """The GGUF config says 262144. This child was started at 524288."""
        from omlx.server import get_max_context_window, validate_context_window

        engine = _engine(tmp_path, "http://strata")
        engine._max_context = 524288
        entry = MagicMock()
        entry.model_context_length = 262144
        entry.engine = engine
        pool = MagicMock()
        pool.get_entry.return_value = entry
        state = MagicMock()
        state.engine_pool = pool
        state.settings_manager = None
        state.sampling.max_context_window = 32768
        state.sampling.max_context_window_policy = None

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("omlx.server._server_state", state)
            patch.setattr("omlx.server.get_model_settings_for_request", lambda _name: None)
            patch.setattr("omlx.server.resolve_model_id", lambda name: name)
            assert get_max_context_window("flash") == 524288
            validate_context_window(400_000, "flash")
            with pytest.raises(HTTPException):
                validate_context_window(524_289, "flash")

    def _unloaded(self, tmp_path, *, engine_id, policy=None, override=None):
        from omlx.server import get_max_context_window

        model = tmp_path / "model"
        model.mkdir()
        (model / "strata-flash.json").write_text(
            json.dumps(
                {
                    "exe": "component:strata-engine",
                    "tokenizer": "tokenizer",
                    "args": ["--max-context", "524288"],
                }
            )
        )
        entry = MagicMock()
        entry.model_context_length = 262144
        entry.model_format = "gguf"
        entry.model_path = str(model)
        entry.engine = None
        pool = MagicMock()
        pool.get_entry.return_value = entry
        state = MagicMock()
        state.engine_pool = pool
        state.settings_manager = None
        state.sampling.max_context_window = 32768
        state.sampling.max_context_window_policy = policy
        settings = MagicMock()
        settings.external_engine = engine_id
        settings.max_context_window = override
        return get_max_context_window, state, settings, model

    def test_an_unloaded_pack_advertises_the_profile_window(self, tmp_path):
        """The GGUF card says 262144. The profile starts the child at 524288."""
        from omlx.admin.routes import _listed_context_length

        get_max_context_window, state, settings, model = self._unloaded(
            tmp_path, engine_id=None
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("omlx.server._server_state", state)
            patch.setattr(
                "omlx.server.get_model_settings_for_request", lambda _name: settings
            )
            patch.setattr("omlx.server.resolve_model_id", lambda name: name)
            assert get_max_context_window("flash") == 524288

        info = {
            "model_path": str(model),
            "model_format": "gguf",
            "model_context_length": 262144,
        }
        assert _listed_context_length(info, settings) == 524288

    def test_splash_keeps_the_gguf_length_while_unloaded(self, tmp_path):
        from omlx.admin.routes import _listed_context_length

        get_max_context_window, state, settings, model = self._unloaded(
            tmp_path, engine_id="splash"
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("omlx.server._server_state", state)
            patch.setattr(
                "omlx.server.get_model_settings_for_request", lambda _name: settings
            )
            patch.setattr("omlx.server.resolve_model_id", lambda name: name)
            assert get_max_context_window("flash") == 262144

        assert (
            _listed_context_length(
                {
                    "model_path": str(model),
                    "model_format": "gguf",
                    "model_context_length": 262144,
                },
                settings,
            )
            == 262144
        )

    def test_install_root_profile_advertises_its_window(self, tmp_path, monkeypatch):
        """No strata-*.json in the folder. The install root still names 524288."""
        from types import SimpleNamespace

        from omlx.admin.routes import _listed_context_length
        from omlx.server import get_max_context_window

        model = tmp_path / "model"
        model.mkdir()
        shard = model / "model.gguf"
        shard.write_bytes(b"not-a-gguf-header")
        install = tmp_path / "install"
        install.mkdir()
        launcher = install / "strata"
        launcher.write_bytes(b"")
        (install / "strata-iq3_s.json").write_text(
            json.dumps(
                {
                    "exe": "component:strata-engine",
                    "tokenizer": "tokenizer",
                    "args": [
                        "--max-context",
                        "524288",
                        "--ple-gguf",
                        str(shard),
                    ],
                }
            )
        )
        seen: list[str | None] = []

        def roots(configured: str | None) -> list:
            seen.append(configured)
            return [install]

        def no_gguf(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("GGUF header was read")

        monkeypatch.setattr("omlx.engine.strata._install_roots", roots)
        monkeypatch.setattr("omlx.engine.strata._gguf_tensor_names", no_gguf)
        entry = MagicMock()
        entry.model_context_length = 262144
        entry.model_format = "gguf"
        entry.model_path = str(model)
        entry.engine = None
        pool = MagicMock()
        pool.get_entry.return_value = entry
        state = MagicMock()
        state.engine_pool = pool
        state.settings_manager = None
        state.sampling.max_context_window = 32768
        state.sampling.max_context_window_policy = None
        state.global_settings = SimpleNamespace(
            integrations=SimpleNamespace(strata_path=str(launcher))
        )
        settings = MagicMock()
        settings.external_engine = None
        settings.max_context_window = None
        info = {
            "model_path": str(model),
            "model_format": "gguf",
            "model_context_length": 262144,
        }
        monkeypatch.setattr("omlx.server._server_state", state)
        monkeypatch.setattr(
            "omlx.server.get_model_settings_for_request", lambda _name: settings
        )
        monkeypatch.setattr("omlx.server.resolve_model_id", lambda name: name)
        monkeypatch.setattr(
            "omlx.admin.routes._get_global_settings",
            lambda: state.global_settings,
        )

        assert get_max_context_window("flash") == 524288
        assert _listed_context_length(info, settings) == 524288
        assert seen == [str(launcher), str(launcher)]

        settings.external_engine = "splash"
        assert get_max_context_window("flash") == 262144
        assert _listed_context_length(info, settings) == 262144
        assert seen == [str(launcher), str(launcher)]

    def test_an_in_folder_config_wins_over_the_install_root(
        self, tmp_path, monkeypatch
    ):
        from omlx.server import get_max_context_window

        model = tmp_path / "model"
        model.mkdir()
        (model / "strata-flash.json").write_text(
            json.dumps(
                {
                    "exe": "component:strata-engine",
                    "tokenizer": "tokenizer",
                    "args": ["--max-context", "111111"],
                }
            )
        )
        install = tmp_path / "install"
        install.mkdir()
        (install / "strata-iq3_s.json").write_text(
            json.dumps(
                {
                    "exe": "component:strata-engine",
                    "tokenizer": "tokenizer",
                    "args": ["--max-context", "524288", "--model", str(model)],
                }
            )
        )
        seen: list[str | None] = []

        def roots(configured: str | None) -> list:
            seen.append(configured)
            return [install]

        monkeypatch.setattr("omlx.engine.strata._install_roots", roots)
        entry = MagicMock()
        entry.model_context_length = 262144
        entry.model_format = "gguf"
        entry.model_path = str(model)
        entry.engine = None
        pool = MagicMock()
        pool.get_entry.return_value = entry
        state = MagicMock()
        state.engine_pool = pool
        state.settings_manager = None
        state.sampling.max_context_window = 32768
        state.sampling.max_context_window_policy = None
        state.global_settings = None
        settings = MagicMock()
        settings.external_engine = None
        settings.max_context_window = None
        monkeypatch.setattr("omlx.server._server_state", state)
        monkeypatch.setattr(
            "omlx.server.get_model_settings_for_request", lambda _name: settings
        )
        monkeypatch.setattr("omlx.server.resolve_model_id", lambda name: name)

        assert get_max_context_window("flash") == 111111
        assert seen == []

    def test_a_per_model_override_still_wins(self, tmp_path):
        get_max_context_window, state, settings, _model = self._unloaded(
            tmp_path, engine_id="strata", override=8192
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("omlx.server._server_state", state)
            patch.setattr(
                "omlx.server.get_model_settings_for_request", lambda _name: settings
            )
            patch.setattr("omlx.server.resolve_model_id", lambda name: name)
            assert get_max_context_window("flash") == 8192

    def test_policy_clamps_the_profile_window(self, tmp_path):
        get_max_context_window, state, settings, _model = self._unloaded(
            tmp_path, engine_id="strata", policy=128_000
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("omlx.server._server_state", state)
            patch.setattr(
                "omlx.server.get_model_settings_for_request", lambda _name: settings
            )
            patch.setattr("omlx.server.resolve_model_id", lambda name: name)
            assert get_max_context_window("flash") == 128_000
