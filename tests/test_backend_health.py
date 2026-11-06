"""Only BEAT physical-system capabilities are exposed to the GUI."""

import pytest

from blab.ui.main_window.backend_health import BackendHealthController
from blab.ui.settings import GuiPreferences


@pytest.mark.parametrize("backend", ["beat_cpu", "beat_cuda", "beat_rocm"])
def test_physical_backend_preserves_symmetry(qapp, backend):
    preferences = GuiPreferences(solve_backend=backend)
    controller = BackendHealthController(None, preferences=lambda: preferences)
    assert controller.selected_backend_supports_symmetry()
    assert controller.effective_symmetry("xy", preferences) == "xy"


def test_retired_backend_preferences_migrate_without_losing_symmetry(qapp):
    preferences = GuiPreferences(solve_backend="server")
    controller = BackendHealthController(None, preferences=lambda: preferences)
    assert preferences.solve_backend == "beat_cpu"
    assert controller.effective_symmetry("x", preferences) == "x"


@pytest.mark.parametrize("code", [0, 1])
def test_metal_check_is_async_and_only_reports_failure(qapp, monkeypatch, code):
    import sys
    import time

    from PySide6.QtTest import QTest

    import blab.ui.main_window.backend_health as health

    monkeypatch.setattr(health.sys, "platform", "darwin")
    monkeypatch.setenv("BLAB_PACKAGED_BACKENDS", "cpu,metal")
    monkeypatch.setattr(health, "metal_probe_command", lambda: (sys.executable, ["-c", f"raise SystemExit({code})"]))
    preferences = GuiPreferences(solve_backend="beat_metal")
    controller = BackendHealthController(None, preferences=lambda: preferences)
    messages = []
    controller.availability_failed.connect(messages.append)
    try:
        controller.check_selected_backend()
        assert controller._probe is not None
        deadline = time.monotonic() + 5
        while controller._probe is not None and time.monotonic() < deadline:
            QTest.qWait(10)
        assert controller._probe is None
        assert controller._metal_available == (code == 0)
        assert len(messages) == code
    finally:
        controller.close()


def test_cpu_selection_does_not_start_metal_probe(qapp, monkeypatch):
    import blab.ui.main_window.backend_health as health

    monkeypatch.setattr(health, "metal_probe_command", lambda: pytest.fail("unexpected probe"))
    controller = BackendHealthController(None, preferences=GuiPreferences)
    controller.check_selected_backend()
    assert controller._probe is None


@pytest.mark.parametrize("timeout", [False, True])
def test_metal_probe_shutdown_and_timeout(qapp, monkeypatch, timeout):
    import sys

    import blab.ui.main_window.backend_health as health

    monkeypatch.setattr(health.sys, "platform", "darwin")
    monkeypatch.setenv("BLAB_PACKAGED_BACKENDS", "cpu,metal")
    monkeypatch.setattr(health, "metal_probe_command", lambda: (sys.executable, ["-c", "import time; time.sleep(30)"]))
    controller = BackendHealthController(None, preferences=lambda: GuiPreferences(solve_backend="beat_metal"))
    messages = []
    controller.availability_failed.connect(messages.append)
    controller.check_selected_backend()
    if timeout:
        controller._timeout()
    else:
        controller.close()
    assert controller._probe is None
    assert len(messages) == int(timeout)
