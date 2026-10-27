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
