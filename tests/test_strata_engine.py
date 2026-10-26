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
import subprocess
import sys
import textwrap
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
    return path


def _write_pack(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    profile = folder / "profile.json"
    profile.write_text(json.dumps({"engine": "fixture"}))
    (folder / "model.gguf").write_bytes(b"gguf")
    (folder / "ngram-table.bin").write_bytes(b"ngram")
    return profile


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
    def test_version_comes_from_build_json_or_the_cli(self, tmp_path, monkeypatch):
        monkeypatch.setattr(strata_mod, "_find_launcher", lambda name: None)
        binary = _fake_launcher(tmp_path / "strata", body="echo should-not-run; exit 1\n")
        (binary.parent / "BUILD.json").write_text(json.dumps({"version": "0.1.50", "metal": True}))

        found, reason = find_strata_binary(str(binary))

        assert reason == ""
        assert found is not None
        assert (found.version, found.backend) == ("0.1.50", "metal")

        binary2 = _fake_launcher(tmp_path / "bin" / "strata", version="0.2.6")
        found, reason = find_strata_binary(str(binary2))
        assert reason == ""
        assert found is not None
        assert found.version == "0.2.6"

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
        (old.parent / "BUILD.json").write_text(json.dumps({"version": "0.1.49"}))
        _, reason = find_strata_binary(str(old))
        assert "too old" in reason

        cuda = _fake_launcher(tmp_path / "cuda" / "strata")
        (cuda.parent / "BUILD.json").write_text(
            json.dumps({"version": "0.1.50", "cuda": "13.0"})
        )
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
        assert "missing profile, GGUF, n-gram table" in reason

        _write_pack(Path(entry.model_path))
        engine = pool._external_engine_for(entry, settings, "batched")
        assert isinstance(engine, StrataEngine)
        assert engine._target.profile_path.name == "profile.json"

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
        _write_pack(Path(entry.model_path))

        engine = pool._external_engine_for(
            entry, ModelSettings(external_engine="strata"), "batched"
        )
        assert isinstance(engine, StrataEngine)
        assert not isinstance(engine, SplashEngine)

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
        _write_pack(Path(entry.model_path))
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
