# SPDX-License-Identifier: Apache-2.0
"""Strata engine: serve a model through a Strata server process.

Strata (https://github.com/Niko1221/Strata) is a separate program. This
module does not vendor it and does not translate its Metal kernels into MLX.
``start()`` launches ``strata serve`` on a loopback port for a prepared
profile. Request translation and the memory contract are not part of this
scaffold: chat and completions refuse, and the pool does not invent a
resident-byte budget.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..exceptions import InvalidRequestError
from ..utils.proc_memory import get_phys_footprint, list_process_group_pids
from .base import BaseEngine, GenerationOutput

logger = logging.getLogger(__name__)

# Homebrew prefixes: the menu bar app starts with a minimal PATH.
_BREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
_USER_BIN_DIR = Path.home() / ".local" / "bin"

STARTUP_TIMEOUT_SECONDS = 600
STARTUP_POLL_SECONDS = 2
STOP_TIMEOUT_SECONDS = 60
STOP_POLL_SECONDS = 0.2
WATCHDOG_POLL_SECONDS = 2
REQUEST_TIMEOUT_SECONDS = 3600
VERSION_TIMEOUT_SECONDS = 30
VERSION_RETRY_SECONDS = 60
LOG_TAIL_LINES = 5

# Engine packages (BUILD.json) and the terminal (`strata --version`) use
# different version lines. 0.1.50 is the Qwen engine package that ships with
# the 0.2.6 terminal; a terminal build reports 0.2.x and passes this floor.
MIN_STRATA_VERSION = (0, 1, 50)
_NON_METAL_BACKENDS = frozenset({"cuda", "hip", "sycl"})
STRATA_INSTALL_HINT = (
    "Strata is not installed: install the Strata terminal "
    "(https://github.com/Niko1221/Strata) and set strata_path if it is not on PATH"
)
_REQUEST_UNAVAILABLE = (
    "Strata cannot serve this request yet: chat translation is not implemented"
)


@dataclass(frozen=True)
class StrataBinary:
    """A Strata launcher oMLX is willing to start."""

    path: Path
    version: str
    auto_detected: bool
    backend: str = ""

    @property
    def version_tuple(self) -> tuple[int, int, int]:
        return _version_tuple(self.version) or (0, 0, 0)

    @property
    def label(self) -> str:
        return "Strata"


@dataclass(frozen=True)
class StrataTarget:
    """A launcher plus the profile ``strata serve`` should start."""

    binary: StrataBinary
    profile_path: Path


class _PlaceholderTokenizer:
    """Satisfies the pool's tokenizer check without loading a pack."""

    def encode(self, text: str = "", add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [] if not text else [1]


def _version_tuple(version: str) -> tuple[int, int, int] | None:
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_launcher(name: str) -> Path | None:
    found = shutil.which(name)
    if found:
        return Path(found)
    for directory in (*_BREW_BIN_DIRS, str(_USER_BIN_DIR)):
        candidate = Path(directory) / name
        if _executable(candidate):
            return candidate
    return None


def _read_build(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _build_backend(build: dict | None) -> str:
    """cuda, hip, sycl, or metal when BUILD.json says so; else empty."""
    if not build:
        return ""
    for name in ("cuda", "hip", "sycl"):
        if build.get(name):
            return name
    if build.get("metal"):
        return "metal"
    backend = build.get("backend")
    return backend.lower() if isinstance(backend, str) else ""


# (path, binary mtime, BUILD.json mtime) -> (version or None, monotonic time)
_versions: dict[tuple[str, int, int], tuple[str | None, float]] = {}


def _read_cli_version(path: str) -> str | None:
    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            env=_strata_env(None),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Strata: %s --version failed: %s", path, exc)
        return None
    match = re.search(r"\d+\.\d+\.\d+", result.stdout + result.stderr)
    return match.group(0) if match else None


def _describe_binary(path: Path) -> tuple[str | None, dict | None]:
    """Version from BUILD.json beside the binary, else ``strata --version``."""
    build_path = path.parent / "BUILD.json"
    build = _read_build(build_path)
    version = None
    if build is not None and isinstance(build.get("version"), str):
        version = build["version"]
    if version is not None and _version_tuple(version) is not None:
        return version, build
    try:
        key = (
            str(path),
            path.stat().st_mtime_ns,
            build_path.stat().st_mtime_ns if build_path.is_file() else 0,
        )
    except OSError:
        return None, build
    cached = _versions.get(key)
    now = time.monotonic()
    if cached is not None and (
        cached[0] is not None or now - cached[1] < VERSION_RETRY_SECONDS
    ):
        return cached[0], build
    cli_version = _read_cli_version(str(path))
    _versions[key] = (cli_version, now)
    return cli_version, build


def inspect_strata_binary(
    path: Path, *, auto_detected: bool
) -> tuple[StrataBinary | None, str]:
    """Check that ``path`` is a Strata launcher oMLX can start."""
    if not _executable(path):
        return None, f"{path} is not an executable file"
    version, build = _describe_binary(path)
    if version is None or _version_tuple(version) is None:
        return None, f"{path} did not report a Strata version"
    parsed = _version_tuple(version)
    assert parsed is not None
    if parsed < MIN_STRATA_VERSION:
        minimum = ".".join(str(part) for part in MIN_STRATA_VERSION)
        return None, f"Strata {version} is too old; oMLX needs {minimum} or later"
    backend = _build_backend(build)
    if sys.platform == "darwin" and backend in _NON_METAL_BACKENDS:
        return (
            None,
            f"Strata at {path} is a {backend.upper()} build; "
            "oMLX on a Mac needs the Metal engine",
        )
    return StrataBinary(path, version, auto_detected, backend), ""


def find_strata_binary(configured: str | None) -> tuple[StrataBinary | None, str]:
    """The Strata launcher to run: ``configured``, or ``strata`` on PATH."""
    if configured:
        return inspect_strata_binary(Path(configured).expanduser(), auto_detected=False)
    path = _find_launcher("strata")
    if path is None:
        return None, STRATA_INSTALL_HINT
    return inspect_strata_binary(path, auto_detected=True)


def _profile_path(folder: Path) -> Path | None:
    named = folder / "profile.json"
    if named.is_file():
        return named
    matches = sorted(path for path in folder.glob("strata-*.json") if path.is_file())
    return matches[0] if matches else None


def _gguf_path(folder: Path) -> Path | None:
    matches = [
        path
        for path in sorted(folder.glob("*.gguf"))
        if path.is_file() and not path.name.lower().startswith("mmproj")
    ]
    return matches[0] if matches else None


def _ngram_path(folder: Path) -> Path | None:
    matches = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and ("ngram" in path.name.lower() or "n-gram" in path.name.lower())
    ]
    return matches[0] if matches else None


def resolve_strata_target(
    model_path: str | Path, configured_binary: str | None
) -> tuple[StrataTarget | None, str]:
    """Whether Strata can serve ``model_path``.

    A pack is a profile (``profile.json`` or ``strata-*.json``), a GGUF body,
    and an n-gram table in the model folder. This does not read the weights.
    """
    binary, reason = find_strata_binary(configured_binary)
    if binary is None:
        return None, reason
    folder = Path(model_path)
    if not folder.is_dir():
        return None, "failed to read the model folder"
    profile = _profile_path(folder)
    gguf = _gguf_path(folder)
    ngram = _ngram_path(folder)
    missing = [
        name
        for name, found in (
            ("profile", profile),
            ("GGUF", gguf),
            ("n-gram table", ngram),
        )
        if found is None
    ]
    if missing:
        return None, (
            "Strata needs a pack in the model folder "
            "(a profile, a GGUF, and an n-gram table); "
            f"missing {', '.join(missing)}"
        )
    assert profile is not None
    return StrataTarget(binary, profile), ""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _signal_group(group: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, sig)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _wait_group_exit(group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while _group_alive(group):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(STOP_POLL_SECONDS)
    return True


def _strata_env(api_key: str | None) -> dict[str, str]:
    """Environment for the Strata launcher.

    oMLX's bundled interpreter sets PYTHONHOME/PYTHONPATH, which would point
    Strata's own Python at the wrong standard library. The API key travels
    in the environment so it never shows up in the process list.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")
    }
    path = env.get("PATH", "/usr/bin:/bin").split(":")
    env["PATH"] = ":".join([directory for directory in _BREW_BIN_DIRS if directory not in path] + path)
    if api_key is not None:
        env["STRATA_API_KEY"] = api_key
    return env


def _watchdog_command(command: list[str]) -> list[str]:
    """Wrap ``command`` so the child dies when oMLX (its parent) exits."""
    script = (
        'parent=$1; shift; "$@" & child=$!; '
        'while kill -0 "$parent" 2>/dev/null && kill -0 "$child" 2>/dev/null; '
        f"do sleep {WATCHDOG_POLL_SECONDS}; done; "
        'if kill -0 "$child" 2>/dev/null; then trap "" TERM; kill -TERM 0; fi; '
        'wait "$child"'
    )
    return ["/bin/sh", "-c", script, "strata-watchdog", str(os.getpid()), *command]


def _failure_summary(path: Path, offset: int) -> str:
    try:
        with open(path, "rb") as log:
            log.seek(offset)
            lines = log.read().decode(errors="replace").splitlines()
    except OSError:
        return ""
    lines = [line.strip() for line in lines if line.strip()]
    errors = [line for line in lines if line.lower().startswith("error")]
    return "\n".join((errors or lines)[-LOG_TAIL_LINES:])


class StrataEngine(BaseEngine):
    """Serves one model through a ``strata serve`` child process.

    The child is supervised the way Splash is: free loopback port, random
    API key in the environment, new session, watchdog, idempotent start,
    SIGTERM then SIGKILL. It does not load weights itself.
    """

    external_engine_id = "strata"
    # Strata holds its own memory. There is no oMLX scheduler to resize.
    _prefill_memory_guard_managed_externally = True

    def __init__(
        self,
        model_name: str,
        model_id: str,
        target: StrataTarget,
        log_dir: Path,
    ) -> None:
        self._path = Path(model_name)
        self._model_id = model_id
        self._target = target
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._base_url = ""
        self._api_key = secrets.token_urlsafe(32)
        slug = model_id.replace("/", "--")
        self._log_path = log_dir / f"{slug}.log"
        self._log_offset = 0
        self._tokenizer: Any = _PlaceholderTokenizer()

    @property
    def model_name(self) -> str:
        return str(self._path)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        return None

    @property
    def log_path(self) -> Path:
        return self._log_path

    def has_active_requests(self) -> bool:
        return False

    @property
    def runtime_failed_reason(self) -> str | None:
        process = self._process
        if process is None or process.poll() is None:
            return None
        return (
            f"Strata exited with code {process.returncode}: "
            f"{_failure_summary(self._log_path, self._log_offset)} "
            f"(log: {self._log_path})"
        )

    def reserved_bytes(self) -> int:
        """No planned budget until the memory contract is measured."""
        return 0

    def _group_pids(self) -> list[int]:
        if self._process is None or self._process.poll() is not None:
            return []
        return list_process_group_pids(self._process.pid)

    def process_memory_bytes(self) -> int:
        """Footprint of the running Strata process group, or 0."""
        pids = self._group_pids()
        if not pids:
            return 0
        return sum(get_phys_footprint(pid) for pid in pids)

    def _serve_command(self, port: int) -> list[str]:
        return [
            str(self._target.binary.path),
            "serve",
            "--profile",
            str(self._target.profile_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--quiet",
        ]

    async def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is None:
                return
            raise RuntimeError(self.runtime_failed_reason)
        port = _free_port()
        self._base_url = f"http://127.0.0.1:{port}"
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        command = self._serve_command(port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Strata: starting %s (log: %s)", " ".join(command), self._log_path)
        with open(self._log_path, "ab") as log:
            self._log_offset = log.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_strata_env(self._api_key),
                start_new_session=True,
            )
        try:
            await self._wait_until_ready()
        except BaseException:
            await self.stop()
            raise

    async def _serves(self) -> bool:
        if self._client is None:
            return False
        try:
            response = await self._client.get("/v1/models", timeout=2)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    f"Strata exited with code {self._process.returncode} "
                    f"while starting {self._model_id}: "
                    f"{_failure_summary(self._log_path, self._log_offset)} "
                    f"(log: {self._log_path})"
                )
            if await self._serves():
                logger.info("Strata: serving %s at %s", self._model_id, self._base_url)
                return
            await asyncio.sleep(STARTUP_POLL_SECONDS)
        raise RuntimeError(
            f"Strata did not serve {self._model_id} within "
            f"{STARTUP_TIMEOUT_SECONDS}s; see {self._log_path}"
        )

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        process, self._process = self._process, None
        if process is None:
            return
        group = process.pid
        _signal_group(group, signal.SIGTERM)
        if not await _wait_group_exit(group, STOP_TIMEOUT_SECONDS):
            logger.warning("Strata: server ignored SIGTERM; killing it")
            _signal_group(group, signal.SIGKILL)
            await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
        await asyncio.to_thread(process.wait)

    def _refuse_request(self) -> None:
        raise InvalidRequestError(_REQUEST_UNAVAILABLE)

    async def generate(self, prompt: str | list[int], **kwargs: Any) -> GenerationOutput:
        del prompt, kwargs
        self._refuse_request()
        raise AssertionError("unreachable")

    async def stream_generate(
        self, prompt: str | list[int], **kwargs: Any
    ) -> AsyncIterator[GenerationOutput]:
        del prompt, kwargs
        self._refuse_request()
        yield GenerationOutput(text="", tokens=[], finished=True)

    async def chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> GenerationOutput:
        del messages, kwargs
        self._refuse_request()
        raise AssertionError("unreachable")

    async def stream_chat(
        self, messages: list[dict[str, Any]], **kwargs: Any
    ) -> AsyncIterator[GenerationOutput]:
        del messages, kwargs
        self._refuse_request()
        yield GenerationOutput(text="", tokens=[], finished=True)

    def get_stats(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        return {
            "running": running,
            "process_memory_bytes": self.process_memory_bytes(),
            "log_path": str(self._log_path),
        }

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None
