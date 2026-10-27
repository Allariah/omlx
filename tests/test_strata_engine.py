# SPDX-License-Identifier: Apache-2.0
"""Scaffold tests for the Strata external engine.

No Strata binary and no GGUF pack are required. A fake launcher stands in
for ``strata serve``, the way Splash's lifecycle tests stand in for Splash.
"""

import asyncio
import json
import os
import signal
import stat
import struct
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import omlx.admin.routes as admin_routes
from omlx.engine import strata as strata_mod
from omlx.engine.splash import SplashEngine
from omlx.engine.strata import (
    StrataBinary,
    StrataEngine,
    StrataTarget,
    find_strata_binary,
)
from omlx.engine_pool import EngineEntry, EnginePool
from omlx.exceptions import InvalidRequestError, ModelLoadingError
from omlx.model_settings import ModelSettings
from omlx.settings import GlobalSettings

GIB = 1024**3


def _fake_launcher(path: Path, version: str = "0.2.6", body: str = "exit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/bin/sh\n"
        f'if [ "$1" = --version ]; then echo "strata {version}"; exit 0; fi\n' + body
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    _write_engine_component(path.parent)
    return path


def _write_pack(folder: Path, exe: Path | None = None) -> Path:
    """A run config and n-gram shard that ``strata serve`` can be pointed at."""
    folder.mkdir(parents=True, exist_ok=True)
    _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd.weight")
    return _write_run_config(folder, exe=exe)


def _gguf_with_tensor(path: Path, tensor: str) -> None:
    """A GGUF v3 header whose tensor list is exactly ``tensor``."""
    name = tensor.encode()
    path.write_bytes(
        b"".join(
            (
                b"GGUF",
                struct.pack("<I", 3),
                struct.pack("<Q", 1),
                struct.pack("<Q", 0),
                struct.pack("<Q", len(name)),
                name,
                struct.pack("<I", 1),
                struct.pack("<Q", 1),
                struct.pack("<I", 0),
                struct.pack("<Q", 0),
            )
        )
    )


def _write_run_config(
    folder: Path, model: str = "flash", exe: Path | None = None
) -> Path:
    """A strata-<model>.json that ``strata serve --profile`` accepts."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"strata-{model}.json"
    path.write_text(
        json.dumps(
            {
                "exe": str(exe or (folder / "engine")),
                "tokenizer": str(folder / "tokenizer.model"),
                "args": ["--model", str(folder / "model.gguf")],
            }
        )
    )
    return path


def _write_build_json(
    directory: Path, *, version: str = "0.1.50", backend: str = "metal"
) -> Path:
    """The BUILD.json a from-source ``build-metal/strata`` leaves behind."""
    path = directory / "BUILD.json"
    path.write_text(
        json.dumps({"version": version, "backend": backend, "source": "test"})
    )
    return path


def _write_engine_component(
    directory: Path, *, version: str = "0.1.50", backend: str = "metal"
) -> Path:
    path = directory / "engine-component.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "id": "strata-engine",
                "version": version,
                "backend": backend,
                "platform": "darwin-arm64",
                "entrypoint": "strata",
            }
        )
    )
    return path


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    stat_line = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return bool(stat_line) and not stat_line.startswith("Z")


@pytest.fixture
def global_settings(tmp_path, monkeypatch):
    settings = GlobalSettings(base_path=tmp_path / "omlx")
    monkeypatch.setattr("omlx.engine_pool.get_global_settings", lambda: settings)
    monkeypatch.setattr("omlx.engine.strata._find_launcher", lambda name: None)
    return settings


def _pool(tmp_path) -> tuple[EnginePool, EngineEntry]:
    pool = EnginePool()
    model = tmp_path / "models" / "flash"
    model.mkdir(parents=True)
    entry = EngineEntry(
        model_id="flash",
        model_path=str(model),
        model_type="llm",
        engine_type="batched",
        estimated_size=16 * GIB,
    )
    pool._entries[entry.model_id] = entry
    return pool, entry


class TestDiscovery:
    def test_version_comes_from_the_engine_component(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        binary = _fake_launcher(tmp_path / "strata", body="echo should-not-run; exit 1\n")
        (binary.parent / "BUILD.json").write_text(json.dumps({"version": "9.9.9", "metal": True}))

        found, reason = find_strata_binary(str(binary))

        assert reason == ""
        assert found is not None
        assert (found.version, found.backend) == ("0.1.50", "metal")

        binary2 = _fake_launcher(tmp_path / "bin" / "strata", version="0.2.6")
        found, reason = find_strata_binary(str(binary2))
        assert reason == ""
        assert found is not None
        assert found.version == "0.1.50"
        assert found.backend == "metal"

    def test_configured_path_wins_and_search_uses_path(self, tmp_path, monkeypatch):
        installed = _fake_launcher(tmp_path / "installed" / "strata")
        custom = _fake_launcher(tmp_path / "custom" / "strata", version="0.2.7")
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: installed)

        found, _ = find_strata_binary(None)
        assert found is not None and found.path == installed and found.auto_detected
        found, _ = find_strata_binary(str(custom))
        assert found is not None and found.path == custom and not found.auto_detected

    def test_missing_old_and_cuda_builds_explain_why(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        assert find_strata_binary(None) == (None, strata_mod.STRATA_INSTALL_HINT)

        missing, reason = find_strata_binary(str(tmp_path / "missing"))
        assert missing is None and "not an executable" in reason

        old = _fake_launcher(tmp_path / "old" / "strata")
        _write_engine_component(old.parent, version="0.1.49", backend="metal")
        _, reason = find_strata_binary(str(old))
        assert "too old" in reason

        cuda = _fake_launcher(tmp_path / "cuda" / "strata")
        _write_engine_component(cuda.parent, version="0.1.50", backend="cuda")
        monkeypatch.setattr(strata_mod.sys, "platform", "darwin")
        _, reason = find_strata_binary(str(cuda))
        assert "CUDA" in reason and "Metal" in reason
        monkeypatch.setattr(strata_mod.sys, "platform", "linux")
        found, reason = find_strata_binary(str(cuda))
        assert reason == "" and found is not None and found.backend == "cuda"

    def test_a_missing_pack_is_a_model_settings_reason(self, tmp_path, global_settings):
        binary = _fake_launcher(tmp_path / "strata")
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        settings = ModelSettings(external_engine="strata")

        with pytest.raises(ModelLoadingError, match="Strata cannot serve 'flash':"):
            pool._external_engine_for(entry, settings, "batched")
        _target, reason = pool.strata_target(entry.model_id, settings)
        assert "missing run config, n-gram table" in reason

        _write_pack(Path(entry.model_path), exe=binary)
        engine = pool._external_engine_for(entry, settings, "batched")
        assert isinstance(engine, StrataEngine)
        assert engine._target.profile_path.name == "strata-flash.json"

    def test_rows_report_each_engine(self, tmp_path, global_settings):
        pool, entry = _pool(tmp_path)
        info = {"id": entry.model_id, "engine_type": "batched", "is_helper": False}
        rows = {
            row["id"]: row
            for row in admin_routes._external_engines_for_model(pool, info, None)
        }
        assert rows["splash"]["compatible"] is False
        assert rows["strata"]["compatible"] is False
        assert "Strata" in rows["strata"]["reason"]

    def test_engine_component_not_the_terminal(self, tmp_path, monkeypatch):
        """The floor is the engine package, not the 0.2.6 terminal line."""
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        monkeypatch.setattr(strata_mod.sys, "platform", "darwin")
        terminal = _fake_launcher(tmp_path / "bin" / "strata", version="0.2.6")

        metal = _write_engine_component(terminal.parent, version="0.1.50", backend="metal")
        found, reason = find_strata_binary(str(terminal))
        assert reason == ""
        assert found is not None
        assert (found.version, found.backend) == ("0.1.50", "metal")

        metal.unlink()
        _write_engine_component(terminal.parent, version="0.1.49", backend="metal")
        _, reason = find_strata_binary(str(terminal))
        assert "too old" in reason

        _write_engine_component(terminal.parent, version="0.1.50", backend="cuda")
        _, reason = find_strata_binary(str(terminal))
        assert "CUDA" in reason and "Metal" in reason

        (terminal.parent / "engine-component.json").unlink()
        (terminal.parent / "BUILD.json").write_text(
            json.dumps({"version": "0.1.50", "metal": True})
        )
        found, reason = find_strata_binary(str(terminal))
        assert found is None
        assert "engine-component.json" in reason

        # The terminal does not sit next to the manifest. Packages live in
        # components/engine/<version>-<platform>-<backend>/.
        packaged = _fake_launcher(tmp_path / "terminal" / "strata", version="0.2.6")
        (packaged.parent / "engine-component.json").unlink()
        metal_dir = (
            tmp_path / "components" / "engine" / "0.1.50-macos-arm64-metal"
        )
        metal_dir.mkdir(parents=True)
        _write_engine_component(metal_dir, version="0.1.50", backend="metal")
        cuda_dir = tmp_path / "components" / "engine" / "0.1.60-linux-x86_64-cuda"
        cuda_dir.mkdir()
        _write_engine_component(cuda_dir, version="0.1.60", backend="cuda")
        found, reason = find_strata_binary(str(packaged))
        assert reason == ""
        assert found is not None
        assert (found.version, found.backend) == ("0.1.50", "metal")
        (metal_dir / "engine-component.json").unlink()
        _, reason = find_strata_binary(str(packaged))
        assert "CUDA" in reason and "Metal" in reason

    def test_pack_is_a_run_config_and_ple_shard(self, tmp_path, global_settings):
        binary = _fake_launcher(tmp_path / "strata")
        _write_engine_component(binary.parent)
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        folder = Path(entry.model_path)

        # A saved profile, a shared-settings file, and an ngram-named note
        # are not a pack Strata will start.
        (folder / "profile.json").write_text(json.dumps({"engine": "fixture"}))
        shared = folder / "strata-flash.shared-settings.json"
        shared.write_text(
            json.dumps(
                {
                    "exe": str(folder / "engine"),
                    "tokenizer": str(folder / "tokenizer.model"),
                    "args": ["--model", "model.gguf"],
                }
            )
        )
        _gguf_with_tensor(folder / "model.gguf", "token_embd.weight")
        (folder / "ngram-table.bin").write_bytes(b"not-a-shard")
        _target, reason = pool.strata_target(entry.model_id, None)
        assert _target is None
        assert "run config" in reason or "n-gram" in reason

        config = _write_run_config(folder, exe=binary)
        # The name without `.weight` is not Strata's n-gram tensor.
        _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd")
        missed, missed_reason = pool.strata_target(entry.model_id, None)
        assert missed is None
        assert "n-gram" in missed_reason

        _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd.weight")
        target, reason = pool.strata_target(entry.model_id, None)
        assert reason == ""
        assert target is not None
        assert target.profile_path == config

    def test_install_root_run_config_is_a_pack(self, tmp_path, global_settings):
        """setup_mac.py writes strata-<tag>.json at the install root, not in the model folder."""
        root = tmp_path / "Strata"
        binary = _fake_launcher(root / "build-metal" / "strata")
        global_settings.integrations.strata_path = str(binary)
        model = tmp_path / "data" / "models" / "qwen"
        model.mkdir(parents=True)
        shard = model / "qwen.gguf"
        _gguf_with_tensor(shard, "per_layer_token_embd.weight")
        (root / "strata-other.json").write_text(
            json.dumps(
                {
                    "exe": str(binary),
                    "tokenizer": str(root / "tok" / "tokenizer.model"),
                    "cwd": str(root),
                    "args": ["--model", str(tmp_path / "elsewhere" / "other.gguf")],
                }
            )
        )
        config = root / "strata-qwen.json"
        config.write_text(
            json.dumps(
                {
                    "exe": str(binary),
                    "tokenizer": str(root / "tok" / "tokenizer.model"),
                    "cwd": str(root),
                    "backend": "metal",
                    "args": ["--ple-gguf", str(shard)],
                }
            )
        )
        pool, entry = _pool(tmp_path)
        entry.model_id = "qwen"
        entry.model_path = str(model)
        entry.model_format = "gguf"
        pool._entries.clear()
        pool._entries[entry.model_id] = entry

        target, reason = pool.strata_target(entry.model_id, None)
        assert reason == ""
        assert target is not None
        assert target.profile_path == config
        engine = pool._external_engine_for(entry, ModelSettings(), "batched")
        assert isinstance(engine, StrataEngine)

    def test_source_build_json_is_the_metal_engine(self, tmp_path, monkeypatch):
        """build-metal/strata plus BUILD.json is a from-source Mac engine."""
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        binary = _fake_launcher(tmp_path / "build-metal" / "strata", version="0.2.6")
        (binary.parent / "engine-component.json").unlink()
        _write_build_json(binary.parent, version="0.1.50", backend="metal")

        found, reason = find_strata_binary(str(binary))

        assert reason == ""
        assert found is not None
        assert (found.version, found.backend) == ("0.1.50", "metal")

    def test_source_cuda_build_is_refused_on_a_mac(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        monkeypatch.setattr(strata_mod.sys, "platform", "darwin")
        binary = _fake_launcher(tmp_path / "build-metal" / "strata", version="0.2.6")
        (binary.parent / "engine-component.json").unlink()
        _write_build_json(binary.parent, version="0.1.50", backend="cuda")

        found, reason = find_strata_binary(str(binary))

        assert found is None
        assert "CUDA" in reason and "Metal" in reason

    def test_literal_exe_does_not_inherit_a_newer_component(
        self, tmp_path, monkeypatch, global_settings
    ):
        """The run config's exe is the engine. A newer installed package does not replace it."""
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        terminal = _fake_launcher(tmp_path / "bin" / "strata", version="0.2.6")
        (terminal.parent / "engine-component.json").unlink()
        packaged = tmp_path / "components" / "engine" / "0.1.50-macos-arm64-metal"
        packaged.mkdir(parents=True)
        _write_engine_component(packaged, version="0.1.50", backend="metal")
        old = _fake_launcher(tmp_path / "old-engine" / "strata", version="0.2.6")
        (old.parent / "engine-component.json").unlink()
        _write_build_json(old.parent, version="0.1.49", backend="metal")
        global_settings.integrations.strata_path = str(terminal)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        folder = Path(entry.model_path)
        _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd.weight")
        _write_run_config(folder, exe=old)

        target, reason = pool.strata_target(entry.model_id, None)
        assert target is None
        assert "too old" in reason
        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "splash"

        bare = _fake_launcher(tmp_path / "bare-engine" / "strata", version="0.2.6")
        (bare.parent / "engine-component.json").unlink()
        (bare.parent / "BUILD.json").unlink(missing_ok=True)
        _write_run_config(folder, exe=bare)
        target, reason = pool.strata_target(entry.model_id, None)
        assert target is None
        assert "engine-component.json" in reason
        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "splash"

    def test_component_reference_uses_the_engine_package(
        self, tmp_path, monkeypatch, global_settings
    ):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        monkeypatch.setattr(strata_mod.sys, "platform", "darwin")
        terminal = _fake_launcher(tmp_path / "bin" / "strata", version="0.2.6")
        # The launcher's own manifest is too old. The component reference must
        # still select the package, and must not inherit this file.
        _write_engine_component(terminal.parent, version="0.1.49", backend="metal")
        metal = tmp_path / "components" / "engine" / "0.1.50-macos-arm64-metal"
        metal.mkdir(parents=True)
        _write_engine_component(metal, version="0.1.50", backend="metal")
        _fake_launcher(metal / "strata", version="0.2.6")
        _write_engine_component(metal, version="0.1.50", backend="metal")
        cuda = tmp_path / "components" / "engine" / "0.1.60-linux-x86_64-cuda"
        cuda.mkdir(parents=True)
        _fake_launcher(cuda / "strata", version="0.2.6")
        _write_engine_component(cuda, version="0.1.60", backend="cuda")
        global_settings.integrations.strata_path = str(terminal)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        folder = Path(entry.model_path)
        _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd.weight")
        path = folder / "strata-flash.json"
        path.write_text(
            json.dumps(
                {
                    "exe": "component:strata-engine",
                    "tokenizer": str(folder / "tokenizer.model"),
                    "backend": "metal",
                    "args": ["--model", str(folder / "model.gguf")],
                }
            )
        )

        engine = pool._external_engine_for(entry, ModelSettings(), "batched")
        assert isinstance(engine, StrataEngine)
        assert engine._target.binary.path == terminal
        assert (engine._target.binary.version, engine._target.binary.backend) == (
            "0.1.50",
            "metal",
        )


class TestDispatch:
    def test_splash_stays_on_the_fake_launcher(self, tmp_path, monkeypatch, global_settings):
        """The existing Splash fake launcher still selects Splash, not Strata."""
        from tests.test_splash_engine import _fake_launcher as splash_launcher
        from tests.test_splash_engine import _pool as splash_pool

        installed: dict[str, Path] = {}
        monkeypatch.setattr("omlx.engine.splash._find_launcher", installed.get)
        monkeypatch.setattr("omlx.engine.splash._is_m1_or_m2", lambda: False)
        installed["splash"] = splash_launcher(tmp_path / "bin" / "splash")

        pool, entry = splash_pool(tmp_path / "splash")
        entry.model_format = "gguf"
        engine = pool._external_engine_for(entry, ModelSettings(), "vlm")
        assert isinstance(engine, SplashEngine)

        with pytest.raises(ModelLoadingError, match="Strata cannot serve"):
            pool._external_engine_for(
                entry, ModelSettings(external_engine="strata"), "vlm"
            )

    def test_explicit_strata_does_not_build_splash(self, tmp_path, global_settings):
        binary = _fake_launcher(tmp_path / "strata")
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        _write_pack(Path(entry.model_path), exe=binary)

        engine = pool._external_engine_for(
            entry, ModelSettings(external_engine="strata"), "batched"
        )
        assert isinstance(engine, StrataEngine)
        assert not isinstance(engine, SplashEngine)

    def test_an_empty_setting_selects_strata_for_a_pack(
        self, tmp_path, monkeypatch, global_settings
    ):
        from tests.test_splash_engine import _fake_launcher as splash_launcher

        monkeypatch.setattr("omlx.engine.splash._find_launcher", lambda name: None)
        monkeypatch.setattr("omlx.engine.splash._is_m1_or_m2", lambda: False)
        splash = splash_launcher(tmp_path / "bin" / "splash")
        monkeypatch.setattr(
            "omlx.engine.splash._find_launcher",
            lambda name: splash if name == "splash" else None,
        )
        binary = _fake_launcher(tmp_path / "strata")
        _write_engine_component(binary.parent)
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        _gguf_with_tensor(
            Path(entry.model_path) / "model.gguf", "per_layer_token_embd.weight"
        )
        _write_run_config(Path(entry.model_path), exe=binary)

        engine = pool._external_engine_for(entry, ModelSettings(), "batched")
        assert isinstance(engine, StrataEngine)
        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "strata"

        from tests.test_splash_engine import _pool as splash_pool

        plain, plain_entry = splash_pool(tmp_path / "plain")
        plain_entry.model_format = "gguf"
        splash_engine = plain._external_engine_for(
            plain_entry, ModelSettings(), "vlm"
        )
        assert isinstance(splash_engine, SplashEngine)
        assert plain.external_engine_id(plain_entry.model_id, ModelSettings()) == "splash"

    def test_a_pack_without_a_usable_engine_stays_on_splash(
        self, tmp_path, monkeypatch, global_settings
    ):
        """Files alone must not hide a GGUF when the Metal engine cannot start."""
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        monkeypatch.setattr(strata_mod, "_components_dir", lambda: tmp_path / "components")
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        folder = Path(entry.model_path)
        _gguf_with_tensor(folder / "model.gguf", "per_layer_token_embd.weight")
        _write_run_config(folder)

        target, reason = pool.strata_target(entry.model_id, None)
        assert target is None
        assert "n-gram" not in reason
        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "splash"
        with pytest.raises(ModelLoadingError, match="Strata cannot serve"):
            pool._external_engine_for(
                entry, ModelSettings(external_engine="strata"), "batched"
            )

        binary = _fake_launcher(tmp_path / "strata")
        global_settings.integrations.strata_path = str(binary)
        _write_run_config(folder, exe=binary)
        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "strata"
        assert isinstance(
            pool._external_engine_for(entry, ModelSettings(), "batched"), StrataEngine
        )

    def test_plain_gguf_does_not_parse_shard_headers(
        self, tmp_path, monkeypatch, global_settings
    ):
        def refuse(path: Path) -> set[str]:
            raise AssertionError(f"parsed {path}")

        monkeypatch.setattr(strata_mod, "_gguf_tensor_names", refuse)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        folder = Path(entry.model_path)
        nested = folder / "shards"
        nested.mkdir()
        (nested / "model-00001-of-00002.gguf").write_bytes(b"GGUF")
        (folder / "weights.gguf").write_bytes(b"GGUF")

        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "splash"
        _target, reason = pool.strata_target(entry.model_id, None)
        assert "run config" in reason

        seen: list[Path] = []

        def record(path: Path) -> set[str]:
            seen.append(Path(path))
            return set()

        monkeypatch.setattr(strata_mod, "_gguf_tensor_names", record)
        _gguf_with_tensor(
            nested / "ple.gguf", "per_layer_token_embd.weight"
        )
        _write_run_config(folder, exe=tmp_path / "missing-engine")
        missed, missed_reason = pool.strata_target(entry.model_id, None)
        assert missed is None
        assert "n-gram" in missed_reason
        assert seen
        assert all(path.parent == folder for path in seen)

    async def test_v1_models_lists_a_pack_splash_refuses(
        self, tmp_path, global_settings
    ):
        import omlx.server as server_module

        binary = _fake_launcher(tmp_path / "strata")
        _write_engine_component(binary.parent)
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        _gguf_with_tensor(
            Path(entry.model_path) / "model.gguf", "per_layer_token_embd.weight"
        )
        _write_run_config(Path(entry.model_path), exe=binary)
        plain = tmp_path / "models" / "plain"
        plain.mkdir()
        (plain / "weights.gguf").write_bytes(b"GGUF")
        pool._entries["plain"] = EngineEntry(
            model_id="plain",
            model_path=str(plain),
            model_type="llm",
            engine_type="batched",
            estimated_size=GIB,
            model_format="gguf",
        )
        original_pool = server_module._server_state.engine_pool
        original_settings = server_module._server_state.global_settings
        server_module._server_state.engine_pool = pool
        server_module._server_state.global_settings = global_settings
        try:
            response = await server_module.list_models(True)
        finally:
            server_module._server_state.engine_pool = original_pool
            server_module._server_state.global_settings = original_settings

        ids = {model["id"] for model in json.loads(bytes(response.body))["data"]}
        assert "flash" in ids
        assert "plain" not in ids

    def test_model_settings_can_pick_strata_for_a_pack(self):
        root = Path(__file__).resolve().parents[1]
        html = (
            root / "apps/omlx-web/omlx_web/templates/dashboard/_modal_model_settings.html"
        ).read_text()
        script = (root / "apps/omlx-web/omlx_web/static/js/dashboard.js").read_text()
        view = (
            root
            / "apps/omlx-mac/Sources/AppView/Screens/ModelSettingsScreen.swift"
        ).read_text()
        vm = (
            root
            / "apps/omlx-mac/Sources/AppView/ViewModels/ModelSettingsScreenVM.swift"
        ).read_text()
        assert "canPickExternalEngine" in html
        assert "canPickExternalEngine" in script
        assert "id === 'strata' && e.compatible" in script
        assert "canPickExternalEngine" in view
        assert 'id == "strata" && $0.compatible' in vm

    async def test_a_plain_gguf_names_splash_and_strata_when_neither_can_serve(
        self, tmp_path, monkeypatch, global_settings
    ):
        """The load refusal for a plain GGUF says why Splash and Strata each cannot serve it."""
        monkeypatch.setattr("omlx.engine.splash._find_launcher", lambda name: None)
        monkeypatch.setattr("omlx.engine.splash._is_m1_or_m2", lambda: False)
        binary = _fake_launcher(tmp_path / "strata")
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        entry.model_format = "gguf"
        (Path(entry.model_path) / "weights.gguf").write_bytes(b"GGUF")

        assert pool.external_engine_id(entry.model_id, ModelSettings()) == "splash"
        with pytest.raises(ModelLoadingError) as caught:
            pool._external_engine_for(entry, ModelSettings(), "batched")
        message = str(caught.value)
        assert "Splash cannot serve" in message
        assert "Splash is not installed" in message
        assert "Strata cannot serve" in message
        assert "missing run config" in message

        problem = await pool.external_engine_problem(entry.model_id, ModelSettings())
        assert "Splash cannot serve" in problem
        assert "Strata cannot serve" in problem
        assert "missing run config" in problem

        explicit = ModelSettings(external_engine="splash")
        assert pool.external_engine_id(entry.model_id, explicit) == "splash"
        with pytest.raises(ModelLoadingError) as splash_choice:
            pool._external_engine_for(entry, explicit, "batched")
        assert "Strata cannot serve" in str(splash_choice.value)
        assert "missing run config" in str(splash_choice.value)

        chosen = ModelSettings(external_engine="strata")
        assert pool.external_engine_id(entry.model_id, chosen) == "strata"
        with pytest.raises(ModelLoadingError) as strata_choice:
            pool._external_engine_for(entry, chosen, "batched")
        text = str(strata_choice.value)
        assert "Strata cannot serve" in text
        assert "Splash cannot serve" in text

    def test_gguf_settings_show_strata_even_when_it_cannot_serve(self):
        """The engine control lists Strata for a GGUF, with the reason it cannot be selected."""
        root = Path(__file__).resolve().parents[1]
        html = (
            root / "apps/omlx-web/omlx_web/templates/dashboard/_modal_model_settings.html"
        ).read_text()
        script = (root / "apps/omlx-web/omlx_web/static/js/dashboard.js").read_text()
        view = (
            root
            / "apps/omlx-mac/Sources/AppView/Screens/ModelSettingsScreen.swift"
        ).read_text()
        en = json.loads(
            (root / "apps/omlx-web/omlx_web/i18n/en.json").read_text()
        )
        assert 'x-show="canPickExternalEngine() || isGgufModel()"' in html
        select = html.split("external_engine_select", 1)[1].split("</select>", 1)[0]
        assert "engine.reason" in select
        reason = script.split("externalEngineReason()", 1)[1].split(
            "saveSplashSize", 1
        )[0]
        assert "isGgufModel()" in reason
        assert "!e.compatible" in reason
        assert "e.label" in reason
        hint = en["modal.model_settings.external_engine_select_hint"]
        assert "Splash" in hint and "Strata" in hint
        assert "https://github.com/Niko1221/Strata" in hint
        assert "vm.canPickExternalEngine || isGGUF" in view
        assert r"\(engine.label): \(engine.reason" in view

    def test_splash_only_controls_use_the_engine_badge(self):
        """KV cache, memory, SSD cache, Neural Engine prefill, and the draft download wear the Splash badge."""
        root = Path(__file__).resolve().parents[1]
        settings = (
            root / "apps/omlx-web/omlx_web/templates/dashboard/_settings.html"
        ).read_text()
        modal = (
            root / "apps/omlx-web/omlx_web/templates/dashboard/_modal_model_settings.html"
        ).read_text()
        script = (root / "apps/omlx-web/omlx_web/static/js/dashboard.js").read_text()
        integrations = (
            root / "apps/omlx-mac/Sources/AppView/Screens/IntegrationsScreen.swift"
        ).read_text()
        model_settings = (
            root / "apps/omlx-mac/Sources/AppView/Screens/ModelSettingsScreen.swift"
        ).read_text()
        badge = (
            'class="badge bg-neutral-100 text-neutral-600 border-neutral-200"'
        )
        splash_rows = settings.split("settings.integrations.strata.binary", 1)[0]
        for key in (
            "settings.integrations.splash.kv_format",
            "settings.integrations.splash.max_memory",
            "settings.integrations.splash.ssd_cache",
            "settings.integrations.splash.ane_prefill",
        ):
            row = splash_rows.split(key, 1)[1].split("</div>", 1)[0]
            assert badge in row
            assert "external_engines.splash.label" in row
        draft = modal.split("externalEngineDraftRepo()", 1)[1].split(
            "trust_remote_code", 1
        )[0]
        assert badge in draft
        assert "externalEngineDraftEngine().label" in draft
        assert "externalEngineDraftEngine()" in script
        assert integrations.count("badge: splashBadge") == 4
        assert "label: draftEngine.label" in model_settings

    def test_process_memory_counts_strata(self, tmp_path):
        pool, entry = _pool(tmp_path)
        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        engine = StrataEngine(
            model_name=entry.model_path,
            model_id=entry.model_id,
            target=StrataTarget(
                StrataBinary(tmp_path / "strata", "0.2.6", True), profile
            ),
            log_dir=tmp_path / "logs",
        )
        engine.process_memory_bytes = lambda: 4 * GIB
        entry.engine = engine
        assert pool.external_process_memory_bytes() == 4 * GIB
        assert pool.get_status()["models"][0]["external_engine"] == "strata"

    async def test_strata_path_is_checked_only_when_it_changes(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        settings = GlobalSettings(base_path=tmp_path)

        async def post(**fields):
            with patch.object(admin_routes, "_get_global_settings", lambda: settings):
                return await admin_routes.update_global_settings(
                    request=admin_routes.GlobalSettingsRequest(**fields), is_admin=True
                )

        with pytest.raises(HTTPException):
            await post(strata_path=str(tmp_path / "missing"))
        assert settings.integrations.strata_path is None

        custom = _fake_launcher(tmp_path / "custom" / "strata")
        await post(strata_path=f"  {custom}  ")
        assert settings.integrations.strata_path == str(custom)
        await post(strata_path="")
        assert settings.integrations.strata_path is None

    async def test_chat_is_refused(self, tmp_path):
        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        engine = StrataEngine(
            model_name=str(tmp_path),
            model_id="flash",
            target=StrataTarget(StrataBinary(tmp_path / "strata", "0.2.6", True), profile),
            log_dir=tmp_path / "logs",
        )
        with pytest.raises(InvalidRequestError, match="not implemented"):
            await engine.chat([{"role": "user", "content": "hi"}])

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_completion_refuses_instead_of_500(
        self, tmp_path, monkeypatch, stream
    ):
        """Chat dies as the not-implemented error, not a missing-method 500.

        The admin page counts tokens before ``chat()``. A Strata engine
        without that method used to raise AttributeError, which the
        unhandled-exception middleware turned into "Internal server error".
        """
        from fastapi.testclient import TestClient

        import omlx.server as srv

        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        engine = StrataEngine(
            model_name=str(tmp_path),
            model_id="flash",
            target=StrataTarget(
                StrataBinary(tmp_path / "strata", "0.2.6", True), profile
            ),
            log_dir=tmp_path / "logs",
        )
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
        monkeypatch.setattr(srv, "get_model_settings_for_request", lambda name: None)
        monkeypatch.setitem(
            srv.app.dependency_overrides, srv.verify_inference_api_key, lambda: True
        )

        with TestClient(srv.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "flash",
                    "stream": stream,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert response.status_code == 400, response.text
        assert response.json()["error"] == {
            "message": (
                "Strata cannot serve this request yet: "
                "chat translation is not implemented"
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
        assert engine._client is None


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


FAKE_SERVER = textwrap.dedent("""\
    import os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    args = sys.argv[1:]
    assert args[0] == "serve"
    profile = args[args.index("--profile") + 1]
    port = int(args[args.index("--port") + 1])
    host = args[args.index("--host") + 1]
    assert host == "127.0.0.1"
    assert os.path.isfile(profile)
    key = os.environ["STRATA_API_KEY"]
    assert key and key not in " ".join(args)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Authorization") != "Bearer " + key:
                self.send_response(401); self.end_headers(); return
            if self.path != "/v1/models":
                self.send_response(404); self.end_headers(); return
            self.send_response(200); self.end_headers()
            self.wfile.write(b'{"data":[{"id":"fixture"}]}')
        def log_message(self, *a):
            pass
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
    """)


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(strata_mod, "STARTUP_POLL_SECONDS", 0.05)
    profile = tmp_path / "fixture-profile.json"
    profile.write_text(json.dumps({"engine": "fixture"}))

    def install(body: str) -> StrataEngine:
        launcher = _fake_launcher(tmp_path / "bin" / "strata", body=body)
        return StrataEngine(
            model_name=str(tmp_path / "model"),
            model_id="flash",
            target=StrataTarget(StrataBinary(launcher, "0.2.6", True), profile),
            log_dir=tmp_path / "logs",
        )

    return install


class TestLifecycle:
    async def test_start_serves_on_loopback_and_stop_ends_the_group(
        self, tmp_path, lifecycle
    ):
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        engine = lifecycle(f'exec "{sys.executable}" "{script}" "$@"\n')

        await engine.start()
        group = engine._process.pid
        try:
            assert engine.get_stats()["running"] is True
            assert os.getpgid(group) == group
            command = " ".join(engine._serve_command(1))
            assert "--profile" in command and "127.0.0.1" in command
            assert engine._api_key not in command
            await engine.start()
            assert engine._process.pid == group
        finally:
            await engine.stop()

        assert not _group_alive(group)
        assert engine.get_stats()["running"] is False

    async def test_stop_reaps_an_out_of_group_engine(
        self, tmp_path, lifecycle, monkeypatch
    ):
        """lifeline moves the Metal engine to its own process group."""
        pid_path = tmp_path / "engine.pid"
        monkeypatch.setenv("STRATA_TEST_ENGINE_PID", str(pid_path))
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        body = (
            f'"{sys.executable}" -c \'import os, time; os.setpgid(0, 0); '
            'open(os.environ["STRATA_TEST_ENGINE_PID"], "w").write(str(os.getpid())); '
            "time.sleep(120)' &\n"
            f'exec "{sys.executable}" "{script}" "$@"\n'
        )
        engine = lifecycle(body)
        pid = None
        try:
            await engine.start()
            pid = int(pid_path.read_text())
            assert os.getpgid(pid) == pid
            assert os.getpgid(pid) != engine._process.pid
            await engine.stop()
            for _ in range(50):
                if not _pid_running(pid):
                    break
                await asyncio.sleep(0.1)
            assert not _pid_running(pid)
        finally:
            if engine._process is not None and engine._process.poll() is None:
                await engine.stop()
            if pid is not None and _pid_running(pid):
                os.kill(pid, signal.SIGKILL)

    async def test_stop_signals_the_engine_after_the_server_exits(
        self, tmp_path, lifecycle, monkeypatch
    ):
        """The server is allowed to stop the engine itself before we signal it.

        ``ps`` has to run off the event loop.
        """
        pid_path = tmp_path / "engine.pid"
        flag_path = tmp_path / "flag"
        monkeypatch.setenv("STRATA_TEST_ENGINE_PID", str(pid_path))
        monkeypatch.setenv("STRATA_TEST_FLAG", str(flag_path))
        child = tmp_path / "engine_child.py"
        child.write_text(
            textwrap.dedent(
                """
                import os
                import signal
                import time

                os.setpgid(0, 0)
                flag = os.environ["STRATA_TEST_FLAG"]
                open(os.environ["STRATA_TEST_ENGINE_PID"], "w").write(str(os.getpid()))

                def handler(signum, frame):
                    state = "dead" if os.getppid() == 1 else "alive"
                    open(flag, "w").write(state)
                    os._exit(0)

                signal.signal(signal.SIGTERM, handler)
                while True:
                    time.sleep(0.05)
                """
            )
        )
        script = tmp_path / "server.py"
        script.write_text(FAKE_SERVER)
        engine = lifecycle(
            f'"{sys.executable}" "{child}" &\n'
            f'exec "{sys.executable}" "{script}" "$@"\n'
        )
        loop_thread = threading.get_ident()
        ps_threads: list[int] = []
        real_run = subprocess.run

        def spy_run(*args, **kwargs):
            argv = args[0] if args else kwargs.get("args")
            if isinstance(argv, (list, tuple)) and argv and argv[0] == "ps":
                ps_threads.append(threading.get_ident())
            return real_run(*args, **kwargs)

        monkeypatch.setattr(strata_mod.subprocess, "run", spy_run)
        pid = None
        try:
            await engine.start()
            pid = int(pid_path.read_text())
            assert os.getpgid(pid) == pid
            await engine.stop()
            assert flag_path.read_text() == "dead"
            assert ps_threads
            assert all(ident != loop_thread for ident in ps_threads)
            assert not _pid_running(pid)
        finally:
            if engine._process is not None and engine._process.poll() is None:
                await engine.stop()
            if pid is not None and _pid_running(pid):
                os.kill(pid, signal.SIGKILL)

    async def test_exit_during_start_reports_the_log(self, tmp_path, lifecycle):
        engine = lifecycle('echo "error: the pack is missing" >&2\nexit 1\n')
        with pytest.raises(RuntimeError, match="pack is missing") as error:
            await engine.start()
        assert engine._process is None
        assert "log:" in str(error.value)

    async def test_stop_kills_a_stuck_server(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "STOP_TIMEOUT_SECONDS", 1)
        stuck = [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.1)",
        ]
        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        engine = StrataEngine(
            model_name=str(tmp_path),
            model_id="flash",
            target=StrataTarget(StrataBinary(tmp_path / "strata", "0.2.6", True), profile),
            log_dir=tmp_path / "logs",
        )
        engine._process = subprocess.Popen(
            strata_mod._watchdog_command(stuck), start_new_session=True
        )
        group = engine._process.pid
        await asyncio.sleep(0.5)

        await engine.stop()

        assert not _group_alive(group)

    async def test_dead_server_is_reported_and_not_relaunched(self, tmp_path):
        profile = tmp_path / "profile.json"
        profile.write_text("{}")
        engine = StrataEngine(
            model_name=str(tmp_path),
            model_id="flash",
            target=StrataTarget(StrataBinary(tmp_path / "strata", "0.2.6", True), profile),
            log_dir=tmp_path / "logs",
        )
        engine._process = subprocess.Popen(["/bin/sh", "-c", "exit 3"])
        engine._process.wait()

        assert "exited with code 3" in engine.runtime_failed_reason
        with pytest.raises(RuntimeError, match="exited with code 3"):
            await engine.start()

    def test_watchdog_stops_strata_when_its_parent_dies(self):
        dead_parent = os.fork()
        if dead_parent == 0:
            os._exit(0)
        os.waitpid(dead_parent, 0)
        command = strata_mod._watchdog_command(["/bin/sleep", "60"])
        command[command.index(str(os.getpid()))] = str(dead_parent)
        process = subprocess.Popen(command, start_new_session=True)
        try:
            assert process.wait(timeout=10) is not None
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
        assert not _group_alive(process.pid)

    async def test_load_skips_the_mlx_barrier(self, tmp_path, global_settings):
        binary = _fake_launcher(tmp_path / "strata")
        global_settings.integrations.strata_path = str(binary)
        pool, entry = _pool(tmp_path)
        _write_pack(Path(entry.model_path), exe=binary)
        pool._get_final_ceiling = lambda: 0
        manager = MagicMock()
        manager.get_settings.return_value = ModelSettings(external_engine="strata")
        pool._settings_manager = manager

        async def fake_start(self):
            return None

        with (
            patch.object(StrataEngine, "start", fake_start),
            patch.object(StrataEngine, "stop", AsyncMock()) as stop,
        ):
            await pool._load_engine(entry.model_id)
            assert isinstance(entry.engine, StrataEngine)
            with patch("omlx.engine_pool.mx") as mx:
                await pool._unload_engine(entry.model_id)

        stop.assert_awaited_once()
        mx.get_active_memory.assert_not_called()
        assert entry.engine is None
