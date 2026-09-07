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
