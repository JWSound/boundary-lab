"""BackendHealthController behaviour.

The controller has no widget dependencies, so these drive it directly with a
preferences callable — no MainWindow, and no fixture beyond a QApplication for
the QObject parent and signal machinery.
"""

from __future__ import annotations

import pytest

import blab.ui.main_window.backend_health as backend_health_module  # noqa: E402
from blab.ui.main_window.backend_health import BackendHealthController  # noqa: E402
from blab.ui.settings import GuiPreferences  # noqa: E402

SERVER_URL = "https://solver.example.test"
SYMMETRY_PAYLOAD = {"capabilities": {"mesh_symmetry": True}}


@pytest.fixture
def controller(qapp):
    prefs = GuiPreferences(solve_backend="server", solve_server_url=SERVER_URL)
    holder = {"prefs": prefs}
    ctrl = BackendHealthController(None, preferences=lambda: holder["prefs"])
    ctrl.holder = holder  # type: ignore[attr-defined]
    return ctrl


def test_cache_is_not_trusted_until_it_matches_the_configured_server(controller) -> None:
    assert controller.matches_preferences() is False, "empty cache must not match"

    controller.cache(SYMMETRY_PAYLOAD, SERVER_URL)
    assert controller.matches_preferences() is True

    controller.holder["prefs"] = GuiPreferences(solve_backend="server", solve_server_url="https://other.example.test")
    assert controller.matches_preferences() is False, "cache from a different server must not match"


def test_trailing_slashes_do_not_defeat_the_cache_match(controller) -> None:
    controller.cache(SYMMETRY_PAYLOAD, SERVER_URL + "/")

    assert controller.matches_preferences() is True


def test_a_local_backend_answers_from_the_registry_not_the_cache(controller, monkeypatch) -> None:
    seen: list[str] = []

    class Caps:
        supports_symmetry = True

    monkeypatch.setattr(
        backend_health_module,
        "backend_info",
        lambda backend_id: seen.append(backend_id) or type("I", (), {"capabilities": Caps})(),
    )

    assert controller.supports_symmetry("bempp_cpu") is True
    assert seen == ["bempp_cpu"], "non-server backends must consult the registry"


def test_server_symmetry_is_false_without_confirmed_health(controller) -> None:
    assert controller.supports_symmetry("server") is False


def test_server_symmetry_reads_an_explicitly_supplied_payload(controller, monkeypatch) -> None:
    monkeypatch.setattr(backend_health_module, "server_health_supports_symmetry", lambda payload: payload["ok"])

    assert controller.supports_symmetry("server", server_health_payload={"ok": True}) is True
    assert controller.supports_symmetry("server", server_health_payload={"ok": False}) is False


def test_symmetry_is_downgraded_when_the_backend_cannot_honour_it(controller) -> None:
    prefs = controller.holder["prefs"]

    # No confirmed server health, so the server backend cannot do symmetry.
    assert controller.effective_symmetry("xy", prefs) == "off"


def test_symmetry_is_preserved_when_the_backend_supports_it(controller, monkeypatch) -> None:
    monkeypatch.setattr(backend_health_module, "server_health_supports_symmetry", lambda _payload: True)
    controller.cache(SYMMETRY_PAYLOAD, SERVER_URL)

    assert controller.effective_symmetry("xy", controller.holder["prefs"]) == "xy"


def test_off_is_always_left_alone(controller) -> None:
    assert controller.effective_symmetry("off", controller.holder["prefs"]) == "off"


def test_startup_probe_is_skipped_for_non_server_backends(controller, monkeypatch) -> None:
    started: list[str] = []
    monkeypatch.setattr(
        backend_health_module,
        "ServerHealthCheckWorker",
        lambda *a, **k: started.append("built"),
    )
    controller.holder["prefs"] = GuiPreferences(solve_backend="bempp_cpu")

    controller.check_on_startup()

    assert started == [], "a local backend must not probe a server"


def test_probe_result_for_a_stale_url_is_discarded(controller) -> None:
    emitted: list[str] = []
    controller.capability_changed.connect(emitted.append)

    controller._on_succeeded("https://stale.example.test", SYMMETRY_PAYLOAD)

    assert controller.payload is None, "a reply from a server we no longer target must be dropped"
    assert emitted == []


def test_capability_change_is_announced_once_health_confirms_symmetry(controller, monkeypatch) -> None:
    monkeypatch.setattr(backend_health_module, "server_health_supports_symmetry", lambda _payload: True)
    emitted: list[str] = []
    controller.capability_changed.connect(emitted.append)

    controller._on_succeeded(SERVER_URL, SYMMETRY_PAYLOAD)

    assert controller.payload == SYMMETRY_PAYLOAD
    assert emitted == ["server_health_checked"], "a newly gained capability must invalidate derived state"


def test_no_announcement_when_the_answer_did_not_change(controller, monkeypatch) -> None:
    monkeypatch.setattr(backend_health_module, "server_health_supports_symmetry", lambda _payload: False)
    emitted: list[str] = []
    controller.capability_changed.connect(emitted.append)

    controller._on_succeeded(SERVER_URL, {"capabilities": {}})

    assert emitted == [], "an unchanged capability must not churn dependent state"
