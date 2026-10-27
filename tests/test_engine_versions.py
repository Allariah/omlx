# SPDX-License-Identifier: Apache-2.0
"""Engine Versions panel: the stats payload lists every engine."""

import asyncio
from unittest.mock import MagicMock, patch

import omlx.admin.routes as admin_routes
import omlx.server  # noqa: F401 — set_admin_getters before the test patches
from omlx.engine.splash import EXTERNAL_ENGINES


def test_stats_payload_lists_splash_and_strata_from_integrations_status():
    """Splash and Strata follow the MLX packages, using integrations status.

    A missing engine stays in the list with no version, so the panel's
    existing chip prints unknown. The chip text comes from that status:
    its version, and a second version string on the same object when the
    status carries both. The strings are supplied here so a second probe
    cannot pass by reading the machine.
    """
    settings = MagicMock()
    settings.server.host = "127.0.0.1"
    settings.server.port = 8000
    settings.auth.api_key = ""
    statuses = {
        "splash": {
            "available": False,
            "version": None,
            "label": None,
        },
        "strata": {
            "available": True,
            "version": "0.1.50",
            "terminal_version": "0.2.6",
            "label": "Strata Metal",
        },
    }

    def external_status(got_settings, engine_id):
        assert got_settings is settings
        return statuses[engine_id]

    metrics = MagicMock()
    metrics.get_snapshot.return_value = {
        "total_prompt_tokens": 0,
        "total_cached_tokens": 0,
        "cache_efficiency": 0,
        "avg_prefill_tps": 0,
        "avg_generation_tps": 0,
        "total_requests": 0,
    }

    with (
        patch.object(admin_routes, "_get_global_settings", return_value=settings),
        patch("omlx.server_metrics.get_server_metrics", return_value=metrics),
        patch.object(
            admin_routes, "_external_engine_status", side_effect=external_status
        ),
        patch.object(admin_routes, "find_splash_binary") as splash_find,
        patch.object(admin_routes, "find_strata_binary") as strata_find,
        patch.object(
            admin_routes, "_build_active_models_data", return_value={"models": []}
        ),
        patch.object(
            admin_routes,
            "_build_runtime_cache_observability",
            return_value={"models": []},
        ),
    ):
        payload = asyncio.run(admin_routes.get_server_stats(is_admin=True))

    engines = payload["engines"]
    mlx_packages = ["mlx", "mlx-lm", "mlx-vlm", "mlx-embeddings", "mlx-audio"]
    assert list(engines)[: len(mlx_packages)] == mlx_packages
    assert list(engines)[-len(EXTERNAL_ENGINES) :] == list(EXTERNAL_ENGINES)
    assert engines["splash"] == {
        "name": "Splash",
        "version": None,
        "commit": None,
        "url": None,
    }
    assert engines["strata"] == {
        "name": "Strata",
        "version": "0.1.50 0.2.6",
        "commit": None,
        "url": None,
    }
    splash_find.assert_not_called()
    strata_find.assert_not_called()
