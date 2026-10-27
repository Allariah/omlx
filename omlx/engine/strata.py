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
_NGRAM_TENSOR = "per_layer_token_embd.weight"
_COMPONENT_PREFIX = "component:"
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


def _beside_engine(binary: Path) -> tuple[str, str] | None:
    """Version and backend recorded next to ``binary``.

    ``engine-component.json`` wins when it names both. A from-source Mac
    build leaves ``BUILD.json`` beside ``build-metal/strata`` with the same
    two strings. ``{"metal": true}`` is not a backend, and the terminal's
    ``strata --version`` line is not consulted.
    """
    component = binary.parent / "engine-component.json"
    if component.is_file():
        record = _component_record(component)
        if record is not None:
            return record
    build = binary.parent / "BUILD.json"
    if build.is_file():
        return _component_record(build)
    return None


def _installed_engine_records() -> list[tuple[str, str]]:
    root = _components_dir() / "engine"
    if not root.is_dir():
        return []
    try:
        manifests = sorted(
            path for path in root.glob("*/engine-component.json") if path.is_file()
        )
    except OSError:
        return []
    records = []
    for manifest in manifests:
        record = _component_record(manifest)
        if record is not None:
            records.append(record)
    return records


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


def _describe_binary(path: Path, *, components: bool) -> tuple[str | None, str]:
    """Engine version and backend for ``path``. The terminal line is not used.

    A record beside ``path`` is that engine, including when it is too old or
    the wrong backend. Only a binary with nothing beside it may fall through
    to the newest installed package, and only when ``components`` is set. The
    exe named by a run config passes ``components=False`` so a newer package
    cannot stand in for it.
    """
    beside = _beside_engine(path)
    if beside is not None:
        return beside
    if not components:
        return None, ""
    chosen = _choose_component(_installed_engine_records())
    if chosen is None:
        return None, ""
    return chosen


def inspect_strata_binary(
    path: Path, *, auto_detected: bool, components: bool = True
) -> tuple[StrataBinary | None, str]:
    """Check that ``path`` is a Strata engine oMLX can start."""
    if not _executable(path):
        return None, f"{path} is not an executable file"
    version, backend = _describe_binary(path, components=components)
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


def _config_base(data: dict, config_path: Path) -> Path:
    cwd = data.get("cwd")
    if isinstance(cwd, str) and cwd:
        return Path(cwd).expanduser()
    return config_path.parent


def _resolve_config_path(text: str, base: Path) -> Path:
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate


def _string_values(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            found.extend(_string_values(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_string_values(item))
    return found


def _inside(path: Path, folder: Path) -> bool:
    try:
        resolved = path.resolve()
        root = folder.resolve()
    except OSError:
        return False
    return resolved == root or root in resolved.parents


def _run_configs(folder: Path) -> list[Path]:
    matches = []
    try:
        paths = list(folder.glob("strata-*.json"))
    except OSError:
        return []
    for path in paths:
        name = path.name
        if not path.is_file() or name.endswith(".shared-settings.json"):
            continue
        model_id = name[len("strata-") : -len(".json")]
        if _MODEL_ID.fullmatch(model_id) is None or not _valid_run_config(path):
            continue
        matches.append(path)
    return matches


def _run_config(folder: Path) -> Path | None:
    matches = _run_configs(folder)
    return sorted(matches)[0] if matches else None


def _install_roots(configured: str | None) -> list[Path]:
    """Directories where ``setup_mac.py`` writes ``strata-<tag>.json``.

    A from-source binary lives in ``<root>/build-metal``. The run config is
    written at ``<root>``, not beside the shards.
    """
    raw = configured
    if not raw:
        found = _find_launcher("strata")
        raw = str(found) if found else None
    ordered: list[Path] = []
    if raw:
        parent = Path(raw).expanduser().parent
        if parent.name == "build-metal":
            ordered.append(parent.parent)
        ordered.append(parent)
    ordered.append(Path("/usr/local/bin"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in ordered:
        try:
            key = path.resolve()
        except OSError:
            key = path
        if key in seen or not path.is_dir():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _config_mentions_folder(config_path: Path, folder: Path) -> bool:
    """True when some path in the config is the model folder or inside it."""
    data = _read_json(config_path)
    if data is None:
        return False
    base = _config_base(data, config_path)
    return any(
        _inside(_resolve_config_path(text, base), folder)
        for text in _string_values(data)
    )


def _matching_install_config(folder: Path, configured: str | None) -> Path | None:
    try:
        folder_key = folder.resolve()
    except OSError:
        folder_key = folder
    matches = []
    for root in _install_roots(configured):
        try:
            if root.resolve() == folder_key:
                continue
        except OSError:
            continue
        for path in _run_configs(root):
            if _config_mentions_folder(path, folder):
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


def _candidate_ggufs(folder: Path, config: Path) -> list[Path]:
    """Shards worth opening: paths named in the config, plus top-level GGUFs.

    Nested shards are not walked. A folder with no run config never gets here.
    """
    data = _read_json(config) or {}
    base = _config_base(data, config)
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        try:
            key = path.resolve()
        except OSError:
            return
        if key in seen or not path.is_file() or path.name.lower().startswith("mmproj"):
            return
        seen.add(key)
        found.append(path)

    for text in _string_values(data):
        if text.lower().endswith(".gguf"):
            add(_resolve_config_path(text, base))
    try:
        for path in sorted(folder.glob("*.gguf")):
            add(path)
    except OSError:
        return found
    return found


def _ngram_gguf(folder: Path, config: Path) -> Path | None:
    """The shard that holds the n-gram table, not a file named like one."""
    for path in _candidate_ggufs(folder, config):
        if _NGRAM_TENSOR in _gguf_tensor_names(path):
            return path
    return None


_PACK_MISSING = (
    "Strata needs a pack in the model folder "
    "(a strata-<model>.json run config and a GGUF shard "
    "with the per_layer_token_embd.weight tensor); "
)


def strata_pack(
    model_path: str | Path, configured_binary: str | None = None
) -> tuple[Path | None, str]:
    """The run config ``strata serve`` accepts, or why the folder is not a pack.

    A pack is ``strata-<model>.json`` with ``exe``, ``tokenizer``, and
    ``args``, plus a GGUF shard that contains ``per_layer_token_embd.weight``.
    On a Mac the config is often written at the Strata install root while the
    shards stay in the model folder. ``profile.json`` and
    ``strata-*.shared-settings.json`` are not packs. Weight bytes are not read.
    With no run config, no shard header is read either.
    """
    folder = Path(model_path)
    if not folder.is_dir():
        return None, "failed to read the model folder"
    config = _run_config(folder) or _matching_install_config(folder, configured_binary)
    if config is None:
        return None, _PACK_MISSING + "missing run config, n-gram table"
    if _ngram_gguf(folder, config) is None:
        return None, _PACK_MISSING + "missing n-gram table"
    return config, ""


def _expand_component_exe(reference: str, cfg: dict) -> Path | None:
    """The entrypoint ``component:strata-engine`` names, if one is installed."""
    if not reference.startswith(_COMPONENT_PREFIX):
        return None
    name = reference[len(_COMPONENT_PREFIX) :]
    if name not in ("", "strata-engine"):
        return None
    root = _components_dir() / "engine"
    if not root.is_dir():
        return None
    hint = cfg.get("backend")
    wanted = hint.lower() if isinstance(hint, str) and hint else None
    found: list[tuple[tuple[int, int, int], str, Path]] = []
    try:
        manifests = sorted(
            path for path in root.glob("*/engine-component.json") if path.is_file()
        )
    except OSError:
        return None
    for manifest in manifests:
        record = _component_record(manifest)
        if record is None:
            continue
        version, backend = record
        if wanted is not None and backend != wanted:
            continue
        data = _read_json(manifest) or {}
        entry = data.get("entrypoint")
        if not isinstance(entry, str) or not entry or entry != Path(entry).name:
            entry = "strata"
        key = _version_tuple(version)
        if key is None:
            continue
        found.append((key, backend, manifest.parent / entry))
    if not found:
        return None
    if wanted is None and sys.platform == "darwin":
        metal = [item for item in found if item[1] == "metal"]
        if metal:
            found = metal
    found.sort(key=lambda item: item[0])
    return found[-1][2]


def _engine_from_run_config(config_path: Path) -> tuple[StrataBinary | None, str]:
    """The engine the run config's ``exe`` names, not a newer package nearby."""
    data = _read_json(config_path)
    if data is None:
        return None, f"cannot read {config_path}"
    exe = data.get("exe")
    if not isinstance(exe, str) or not exe:
        return None, f"{config_path} needs an engine exe"
    if exe.startswith(_COMPONENT_PREFIX):
        resolved = _expand_component_exe(exe, data)
        if resolved is None:
            return None, (
                f"{config_path} names {exe}, and no matching "
                "Strata engine package is installed"
            )
        return inspect_strata_binary(resolved, auto_detected=False, components=False)
    return inspect_strata_binary(
        _resolve_config_path(exe, _config_base(data, config_path)),
        auto_detected=False,
        components=False,
    )


def _serve_launcher(configured: str | None, engine: Path) -> tuple[Path, bool]:
    """The program that accepts ``strata serve``.

    That is the configured path when it is executable, otherwise ``strata``
    on PATH, otherwise the engine itself. The version still comes from the
    engine the run config names.
    """
    if configured:
        path = Path(configured).expanduser()
        if _executable(path):
            return path, False
    found = _find_launcher("strata")
    if found is not None:
        return found, True
    return engine, False


def resolve_strata_target(
    model_path: str | Path, configured_binary: str | None
) -> tuple[StrataTarget | None, str]:
    """Whether Strata can serve ``model_path``."""
    config, reason = strata_pack(model_path, configured_binary)
    if config is None:
        return None, reason
    engine, reason = _engine_from_run_config(config)
    if engine is None:
        return None, reason
    launcher, auto_detected = _serve_launcher(configured_binary, engine.path)
    binary = StrataBinary(launcher, engine.version, auto_detected, engine.backend)
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


def _living(pids: list[int]) -> list[int]:
    return [pid for pid in pids if _pid_alive(pid)]


def _any_pid_alive(pids: list[int]) -> bool:
    return any(_pid_alive(pid) for pid in pids)


async def _wait_pids(pids: list[int], timeout: float) -> bool:
    """Wait until every pid has exited. ``ps`` runs off the event loop."""
    if not pids:
        return True
    deadline = time.monotonic() + timeout
    while await asyncio.to_thread(_any_pid_alive, pids):
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


def _strata_error_message(response: httpx.Response) -> str:
    """``error.message`` from a Strata JSON body, or the response text."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message:
                return message
    text = response.text.strip()
    return text or f"Strata returned HTTP {response.status_code}"


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
        # Snapshot before any signal. Once the server dies, an engine that
        # lifeline moved to its own process group is reparented and the
        # watchdog group no longer names it. Signal the group first so the
        # server can STOP/QUIT the engine itself, then signal whatever is
        # still alive. ``ps`` stays off the event loop.
        outsiders = await asyncio.to_thread(_outside_group, group, group)
        _signal_group(group, signal.SIGTERM)
        group_done = await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
        if not group_done:
            logger.warning("Strata: server ignored SIGTERM; killing it")
            _signal_group(group, signal.SIGKILL)
            await _wait_group_exit(group, STOP_TIMEOUT_SECONDS)
        living = await asyncio.to_thread(_living, outsiders)
        for pid in living:
            _signal_pid(pid, signal.SIGTERM)
        if not await _wait_pids(living, STOP_TIMEOUT_SECONDS):
            logger.warning("Strata: engine ignored SIGTERM; killing it")
            for pid in living:
                if await asyncio.to_thread(_pid_alive, pid):
                    _signal_pid(pid, signal.SIGKILL)
            await _wait_pids(living, STOP_TIMEOUT_SECONDS)
        await asyncio.to_thread(process.wait)

    def _refuse_request(self) -> None:
        raise InvalidRequestError(_REQUEST_UNAVAILABLE)

    def _child_model_name(self) -> str:
        """The model id the child serves, from the run config's ``model_name``."""
        data = _read_json(self._target.profile_path) or {}
        name = data.get("model_name")
        return name if isinstance(name, str) and name else ""

    def _post_sync(self, path: str, body: dict[str, Any]) -> httpx.Response:
        """POST ``path`` with the bearer key from the readiness client.

        The chat route calls ``count_chat_tokens`` synchronously on the
        server loop, so this cannot await that async client. The URL and
        the key are the ones ``GET /v1/models`` already uses.
        """
        client = self._client
        if client is None:
            raise InvalidRequestError("Strata engine is not started")
        headers: dict[str, str] = {}
        auth = client.headers.get("authorization")
        if auth:
            headers["Authorization"] = auth
        try:
            with httpx.Client(
                base_url=client.base_url,
                headers=headers,
                timeout=REQUEST_TIMEOUT_SECONDS,
            ) as sync:
                return sync.post(path, json=body)
        except httpx.HTTPError as exc:
            raise InvalidRequestError(str(exc)) from exc

    def count_chat_tokens(
        self, messages, tools=None, chat_template_kwargs=None, is_partial=None
    ) -> int:
        # The chat route counts before chat() or stream_chat(). Strata
        # tokenizes the messages; the placeholder encode() is not a BPE.
        del tools, chat_template_kwargs, is_partial
        body: dict[str, Any] = {"messages": list(messages or [])}
        model = self._child_model_name()
        if model:
            body["model"] = model
        response = self._post_sync("/v1/messages/count_tokens", body)
        if response.status_code != 200:
            raise InvalidRequestError(_strata_error_message(response))
        try:
            payload = response.json()
        except ValueError as exc:
            raise InvalidRequestError("Strata token count was not JSON") from exc
        count = payload.get("input_tokens") if isinstance(payload, dict) else None
        if isinstance(count, bool) or not isinstance(count, int):
            raise InvalidRequestError("Strata token count did not include input_tokens")
        return count

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
