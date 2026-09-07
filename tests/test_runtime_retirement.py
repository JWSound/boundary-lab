"""Guard the physical-only application runtime and legacy settings migration."""

import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from blab.startup_checks import DEPENDENCY_NAMES
from blab.ui.settings import GuiPreferences, load_gui_preferences, save_gui_preferences


@pytest.mark.parametrize(
    "old_backend",
    [
        "local",
        "bempp",
        "bempp_cpu",
        "bempp_local",
        "local_bempp",
        "local_bempp_cl",
        "server",
        "bempp_server",
        "http_server",
    ],
)
def test_saved_retired_backend_migrates_and_persists_as_beat_cpu(old_backend):
    class Settings:
        def __init__(self):
            self.values = {"preferences/solve_backend": old_backend}

        def contains(self, key):
            return key in self.values

        def value(self, key, default=None):
            return self.values.get(key, default)

        def setValue(self, key, value):
            self.values[key] = value

    settings = Settings()
    preferences = load_gui_preferences(settings)
    assert preferences.solve_backend == "beat_cpu"
    save_gui_preferences(settings, preferences)
    assert settings.values["preferences/solve_backend"] == "beat_cpu"


@pytest.mark.parametrize("backend", ["beat_cpu", "beat_cuda", "beat_rocm"])
def test_existing_beat_selection_is_preserved(backend):
    assert GuiPreferences(solve_backend=backend).solve_backend == backend


def test_headless_imports_do_not_require_legacy_runtime_or_qt():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class RejectLegacyAndQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'bempp_cl', 'pyopencl', 'PySide6'}:
            raise AssertionError(f'Unexpected runtime dependency: {fullname}')

sys.meta_path.insert(0, RejectLegacyAndQt())
import blab.headless
import blab.project_cli
import blab.system_solve
import blab.solvers.coupled_backend
import blab.solvers.coupled_field
import blab.solver
import blab.server
""",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    dependencies = tomllib.loads((root / "pyproject.toml").read_text())["project"]["dependencies"]
    assert not {"bempp-cl", "pyopencl"}.intersection(dependencies)
    assert not {"bempp-cl", "pyopencl"}.intersection(DEPENDENCY_NAMES)


def test_preferences_offer_only_beat_backends(qapp):
    from blab.ui.dialogs import PreferencesDialog

    dialog = PreferencesDialog(GuiPreferences(solve_backend="bempp_cpu"))
    try:
        assert set(dialog.solve_backend_options.values()) == {"beat_cpu", "beat_cuda", "beat_rocm"}
        assert dialog.preferences().solve_backend == "beat_cpu"
        assert not hasattr(dialog, "check_server_button")
    finally:
        dialog.close()
        dialog.deleteLater()
