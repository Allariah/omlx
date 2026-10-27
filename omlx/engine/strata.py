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
import struct
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
LOG_TAIL_LINES = 5

# The terminal reports 0.2.x (`strata --version`). The engine package the
# server actually starts is a separate component whose floor is 0.1.50.
MIN_STRATA_VERSION = (0, 1, 50)
_NGRAM_TENSOR = "per_layer_token_embd"
_MODEL_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]|\.(?!\.)){0,127}\Z")
_NON_METAL_BACKENDS = frozenset({"cuda", "hip", "sycl"})
_MAX_GGUF_ITEMS = 1 << 20
_MAX_GGUF_STRING = 1 << 20
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


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _components_dir() -> Path:
    """Where the terminal keeps engine packages.

    Same folder ``strata`` uses: ``STRATA_COMPONENTS_DIR``, else
    ``$XDG_DATA_HOME/strata/components``.
    """
    explicit = os.environ.get("STRATA_COMPONENTS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home) if data_home else Path.home() / ".local" / "share"
    return base / "strata" / "components"


def _component_manifests(binary: Path) -> list[Path]:
    """The engine manifest for ``binary``.

    A manifest beside the binary is the engine entrypoint's own package.
    The terminal has none there; its packages live under the components dir.
    """
    beside = binary.parent / "engine-component.json"
    if beside.is_file():
        return [beside]
    root = _components_dir() / "engine"
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob("*/engine-component.json") if path.is_file())


def _component_record(path: Path) -> tuple[str, str] | None:
    data = _read_json(path)
    if data is None:
        return None
    version = data.get("version")
    backend = data.get("backend")
    if not isinstance(version, str) or not isinstance(backend, str):
        return None
    if _version_tuple(version) is None:
        return None
    return version, backend.lower()


def _choose_component(records: list[tuple[str, str]]) -> tuple[str, str] | None:
    """The engine this machine would start.

    On a Mac a Metal package wins over a newer CUDA/HIP/SYCL package. The
    caller still refuses whatever this returns when it cannot run here.
    """
    parsed: list[tuple[tuple[int, int, int], str, str]] = []
    for version, backend in records:
        key = _version_tuple(version)
        if key is not None:
            parsed.append((key, version, backend))
    if not parsed:
        return None
    if sys.platform == "darwin":
        metal = [item for item in parsed if item[2] == "metal"]
        if metal:
            parsed = metal
    parsed.sort(key=lambda item: item[0])
    _key, version, backend = parsed[-1]
    return version, backend


def _describe_binary(path: Path) -> tuple[str | None, str]:
    """Engine-component version and backend. The terminal line is not used."""
    records = [
        record
        for manifest in _component_manifests(path)
        if (record := _component_record(manifest)) is not None
    ]
    chosen = _choose_component(records)
    if chosen is None:
        return None, ""
    return chosen


def inspect_strata_binary(
    path: Path, *, auto_detected: bool
) -> tuple[StrataBinary | None, str]:
    """Check that ``path`` is a Strata launcher oMLX can start."""
    if not _executable(path):
        return None, f"{path} is not an executable file"
    version, backend = _describe_binary(path)
    if version is None or _version_tuple(version) is None:
        return None, (
            f"{path} has no Strata engine component (engine-component.json)"
        )
    parsed = _version_tuple(version)
    assert parsed is not None
    if parsed < MIN_STRATA_VERSION:
        minimum = ".".join(str(part) for part in MIN_STRATA_VERSION)
        return None, f"Strata {version} is too old; oMLX needs {minimum} or later"
    if sys.platform == "darwin" and backend != "metal":
        label = backend.upper() if backend in _NON_METAL_BACKENDS else backend or "non-Metal"
        return (
            None,
            f"Strata at {path} is a {label} build; "
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


def _valid_run_config(path: Path) -> bool:
    """True when ``strata serve --profile`` would accept ``path``.

    ``read_run_config`` requires ``exe``, ``tokenizer``, and an ``args`` array
    of strings. A saved profile (``profile.json``) is a different file.
    """
    data = _read_json(path)
    if data is None or not isinstance(data.get("args"), list):
        return False
    if not all(isinstance(arg, str) for arg in data["args"]):
        return False
    return all(isinstance(data.get(key), str) and data[key] for key in ("exe", "tokenizer"))


def _run_config(folder: Path) -> Path | None:
    matches = []
    try:
        paths = list(folder.glob("strata-*.json"))
    except OSError:
        return None
    for path in paths:
        name = path.name
        if not path.is_file() or name.endswith(".shared-settings.json"):
            continue
        model_id = name[len("strata-") : -len(".json")]
        if _MODEL_ID.fullmatch(model_id) is None or not _valid_run_config(path):
            continue
        matches.append(path)
    return sorted(matches)[0] if matches else None


def _gguf_tensor_names(path: Path) -> set[str]:
    """Tensor names in a GGUF header. The weight bytes are not read."""

    def take(size: int) -> bytes:
        data = stream.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF")
        return data

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    def u64() -> int:
        value = struct.unpack("<Q", take(8))[0]
        if value > _MAX_GGUF_ITEMS:
            raise ValueError("GGUF length out of range")
        return value

    def string() -> str:
        length = u64()
        if length > _MAX_GGUF_STRING:
            raise ValueError("GGUF string out of range")
        return take(length).decode("utf-8", "replace")

    def skip_value(kind: int) -> None:
        if kind <= 7 or kind in (10, 11, 12):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            take(sizes[kind])
            return
        if kind == 8:
            string()
            return
        if kind != 9:
            raise ValueError(f"unknown GGUF type {kind}")
        item = u32()
        count = u64()
        if item == 8:
            for _ in range(count):
                string()
        elif item <= 7 or item in (10, 11, 12):
            sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            take(sizes[item] * count)
        else:
            raise ValueError(f"unsupported GGUF array of type {item}")

    try:
        with open(path, "rb") as stream:
            if take(4) != b"GGUF" or u32() < 2:
                return set()
            tensor_count = u64()
            for _ in range(u64()):
                string()
                skip_value(u32())
            names = set()
            for _ in range(tensor_count):
                names.add(string())
                n_dims = u32()
                if n_dims > 8:
                    raise ValueError("GGUF tensor rank out of range")
                take(8 * n_dims)
                u32()
                take(8)
            return names
    except (OSError, ValueError, struct.error):
        return set()


def _ngram_gguf(folder: Path) -> Path | None:
    """The shard that holds the n-gram table, not a file named like one."""
    try:
        paths = sorted(folder.rglob("*.gguf"))
    except OSError:
        return None
    for path in paths:
        if not path.is_file() or path.name.lower().startswith("mmproj"):
            continue
        if _NGRAM_TENSOR in _gguf_tensor_names(path):
            return path
    return None


def strata_pack(model_path: str | Path) -> tuple[Path | None, str]:
    """The run config ``strata serve`` accepts, or why the folder is not a pack.

    A pack is ``strata-<model>.json`` with ``exe``, ``tokenizer``, and
    ``args``, plus a GGUF shard that contains ``per_layer_token_embd``.
    ``profile.json`` and ``strata-*.shared-settings.json`` are not packs.
    This does not read weight bytes.
    """
    folder = Path(model_path)
    if not folder.is_dir():
        return None, "failed to read the model folder"
    config = _run_config(folder)
    ngram = _ngram_gguf(folder)
    missing = [
        name
        for name, found in (("run config", config), ("n-gram table", ngram))
        if found is None
    ]
    if missing:
        return None, (
            "Strata needs a pack in the model folder "
            "(a strata-<model>.json run config and a GGUF shard "
            "with the per_layer_token_embd tensor); "
            f"missing {', '.join(missing)}"
        )
    assert config is not None
    return config, ""


def resolve_strata_target(
    model_path: str | Path, configured_binary: str | None
) -> tuple[StrataTarget | None, str]:
    """Whether Strata can serve ``model_path``."""
    binary, reason = find_strata_binary(configured_binary)
    if binary is None:
        return None, reason
    config, reason = strata_pack(model_path)
    if config is None:
        return None, reason
    return StrataTarget(binary, config), ""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _signal_group(group: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(group, sig)


def _signal_pid(pid: int, sig: int) -> None:
    """Signal ``pid``, or its process group when it leads one.

    ``serve/lifeline.py`` calls ``setpgid(0, 0)`` on the Metal engine, so a
    signal to the server's group does not reach it.
    """
    try:
        group = os.getpgid(pid)
    except ProcessLookupError:
        return
    if group == pid:
        _signal_group(pid, sig)
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, sig)


def _process_table() -> dict[int, tuple[int, int]]:
    """pid -> (ppid, pgid) for the processes ``ps`` can see."""
    try:
        result = subprocess.run(
            ["ps", "-ax", "-o", "pid=,ppid=,pgid="],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, pgid = (int(part) for part in parts)
        except ValueError:
            continue
        rows[pid] = (ppid, pgid)
    return rows


def _outside_group(root: int, group: int) -> list[int]:
    """Descendants of ``root`` that lifeline moved out of ``group``."""
    rows = _process_table()
    children: dict[int, list[int]] = {}
    for pid, (ppid, _pgid) in rows.items():
        children.setdefault(ppid, []).append(pid)
    found: list[int] = []
    stack = list(children.get(root, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        stack.extend(children.get(pid, []))
        if rows.get(pid, (0, group))[1] != group:
            found.append(pid)
    return found


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True
    state = result.stdout.strip()
    return bool(state) and not state.startswith("Z")


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


async def _wait_pids(pids: list[int], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while any(_pid_alive(pid) for pid in pids):
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
        # Snapshot before the signal. Once the server dies, an engine that
        # lifeline moved to its own process group is reparented and the
        # watchdog group no longer names it.
        outsiders = _outside_group(group, group)
        _signal_group(group, signal.SIGTERM)
        for pid in outsiders:
            _signal_pid(pid, signal.SIGTERM)
        group_done = await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
        outsiders_done = await _wait_pids(outsiders, STOP_TIMEOUT_SECONDS)
        if not group_done or not outsiders_done:
            logger.warning("Strata: server ignored SIGTERM; killing it")
            if not group_done:
                _signal_group(group, signal.SIGKILL)
            for pid in outsiders:
                if _pid_alive(pid):
                    _signal_pid(pid, signal.SIGKILL)
            await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
            await _wait_pids(outsiders, STOP_TIMEOUT_SECONDS)
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
