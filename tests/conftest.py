"""Shared Qt test harness.

Provides a session ``QApplication`` and a fully constructed :class:`MainWindow`
with the VTK-backed mesh preview stubbed out, so window behaviour can be tested
headlessly. ``MeshPreview`` is imported lazily inside ``MainWindow.__init__``,
which is what makes the substitution possible.

Nothing here imports Qt at module scope. A missing GUI stack has to be reported
by :mod:`gui_stack` as a loud error against the *test module* that needs it — if
this file blew up on import instead, the failure would take the whole suite with
it, including the tests that need no GUI at all.
"""

from __future__ import annotations

import os
import pathlib

import pytest

from gui_stack import (
    PYSIDE6_ERROR,
    gui_test_modules,
    missing_stack_error,
    require_pyvistaqt,
    skip_requested,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Resolved once, here, because pytest imports conftest before any test module.
# A broken GUI stack is an environment failure: it stops the run rather than
# quietly deleting a third of it. Setting BLAB_SKIP_GUI_TESTS=1 drops just the
# Qt-dependent modules from collection, which is visible in the summary as a
# smaller collected count rather than as a silent pass.
collect_ignore: list[str] = []
if PYSIDE6_ERROR is not None:
    if skip_requested():
        collect_ignore = gui_test_modules(pathlib.Path(__file__).resolve().parent)
    else:
        raise missing_stack_error("PySide6", PYSIDE6_ERROR)


@pytest.fixture(scope="session", autouse=True)
def _isolate_qsettings(tmp_path_factory) -> None:
    """Keep tests away from the developer's real Boundary Lab preferences."""
    if PYSIDE6_ERROR is not None:
        return
    from PySide6.QtCore import QSettings

    root = tmp_path_factory.mktemp("qsettings")
    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(root))
    QSettings.setPath(QSettings.IniFormat, QSettings.SystemScope, str(root))


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _stub_mesh_preview_class():
    """Stand-in for the VTK mesh preview, which cannot get a GL context offscreen."""
    from PySide6.QtWidgets import QWidget

    class StubMeshPreview(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.region_visibility_mode: str | None = None
            self.loaded: list[tuple] = []
            self.clear_count = 0

        def clear(self) -> None:
            self.clear_count += 1
            self.loaded.clear()

        def load_mesh_configs(self, *args, **kwargs) -> None:
            self.loaded.append((args, kwargs))

        def set_region_visibility_mode(self, mode) -> None:
            self.region_visibility_mode = mode

    return StubMeshPreview


@pytest.fixture
def main_window(qapp, monkeypatch):
    """A fully constructed MainWindow with a stubbed mesh preview."""
    import blab.ui.mesh_preview as mesh_preview_module

    monkeypatch.setattr(mesh_preview_module, "MeshPreview", _stub_mesh_preview_class())

    from blab.ui.main_window import MainWindow

    window = MainWindow()
    try:
        yield window
    finally:
        # A dirtied project would otherwise raise a modal save prompt during teardown.
        window.project_workflow.confirm_unsaved_project_changes = lambda _action: True
        window.close()
        window.deleteLater()
        qapp.processEvents()


def _stub_qt_interactor_class():
    """Stand-in for the pyvistaqt VTK plotter, which needs a GL context.

    ``BalloonPlotWindow.__init__`` imports ``QtInteractor`` lazily, which is
    what lets this be substituted; every render call is recorded and dropped.
    """
    from PySide6.QtWidgets import QWidget

    class StubQtInteractor(QWidget):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.interactor = QWidget()
            self.camera_position = "iso"
            self.renderer = None
            self.calls: list[str] = []

        # QWidget already defines render(), so __getattr__ never fires for it and
        # the real one raises on a no-argument call. Any plotter method whose name
        # collides with QWidget has to be overridden explicitly like this.
        def render(self, *args, **kwargs) -> None:
            self.calls.append("render")

        def __getattr__(self, name):
            def record(*args, **kwargs):
                self.__dict__.setdefault("calls", []).append(name)
                return None

            return record

    return StubQtInteractor


def balloon_raw_bundle(*, point_count: int = 64, freqs=(100.0, 400.0, 1000.0)) -> dict:
    """A raw balloon bundle shaped like LiveSolveDataset.as_balloon_raw_bundle()."""
    import numpy as np

    indices = np.arange(point_count, dtype=float)
    z = 1.0 - 2.0 * (indices + 0.5) / point_count
    rng = np.random.default_rng(0)
    return {
        "freq_hz": np.asarray(freqs, dtype=np.float32),
        # One radius per spherical sample, not a scalar.
        "r_distance_m": np.full(point_count, 1.0, dtype=np.float32),
        "theta_polar_rad": np.arccos(z).astype(np.float32),
        "phi_azimuth_rad": (indices * np.pi * (3.0 - np.sqrt(5.0))).astype(np.float32),
        "spl_norm": rng.normal(80.0, 3.0, (len(freqs), point_count)).astype(np.float32),
    }


@pytest.fixture
def balloon_window(qapp, monkeypatch):
    """A fully constructed BalloonPlotWindow with the VTK plotter stubbed."""
    require_pyvistaqt()
    import pyvistaqt
    from PySide6.QtCore import QSettings

    from blab.ui.settings import SETTINGS_APP, SETTINGS_ORG

    monkeypatch.setattr(pyvistaqt, "QtInteractor", _stub_qt_interactor_class())

    from blab.ui.balloon import BalloonPlotWindow

    # Closing the window saves its dock layout, so without this each test would
    # restore whatever layout the previous one happened to leave behind.
    settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
    settings.remove("balloon_window")
    settings.sync()

    window = BalloonPlotWindow(balloon_raw_bundle(), min_db=-30.0, max_db=6.0)
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()
