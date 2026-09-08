from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve
from blab.system_contract import QuantityResult, SystemFrequencyResult
from blab.ui.dialogs import PreferencesDialog
from blab.ui.settings import GuiPreferences, preferences_require_solve_invalidation
from blab.ui.system_solve import SystemSolveWorker


def test_preferences_server_fields_round_trip(qapp):
    preferences = GuiPreferences(
        solve_backend="beat_remote",
        solve_server_backend="beat_rocm",
        solve_server_url="https://solver.example:8765",
        solve_server_token_env="LAB_TOKEN",
        solve_server_ca="lab.pem",
    )
    dialog = PreferencesDialog(preferences)
    assert dialog.server_preferences.isEnabled()
    saved = dialog.preferences()
    for name in (
        "solve_backend",
        "solve_server_url",
        "solve_server_backend",
        "solve_server_token_env",
        "solve_server_ca",
    ):
        assert getattr(saved, name) == getattr(preferences, name)
    dialog.solve_backend_combo.setCurrentText("BEAT Engine (CPU)")
    assert not dialog.server_preferences.isEnabled()
    assert dialog.preferences().solve_server_url == preferences.solve_server_url
    dialog.close()


def test_remote_settings_invalidate_only_remote_solves():
    local = GuiPreferences()
    assert not preferences_require_solve_invalidation(local, replace(local, solve_server_backend="beat_cuda"))
    remote = replace(local, solve_backend="beat_remote")
    assert preferences_require_solve_invalidation(remote, replace(remote, solve_server_backend="beat_cuda"))


def test_gui_remote_worker_streams_complex_results(qapp, monkeypatch):
    project = load_headless_project(Path(__file__).parent / "fixtures/remote-exterior.blab.json")
    prepared = prepare_headless_solve(project, HeadlessSolveSpec(frequencies_hz=(500.0,)), backend_id="beat_remote")
    prepared = replace(prepared, remote_options={"url": "http://127.0.0.1:8765", "backend": "beat_cuda"})
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def select_backend(self, requested, **kwargs):
            calls.append(requested)
            return requested

        def create_system_session(self, request):
            output = next(item for item in request.outputs if item.quantity == "exterior_pressure")
            result = SystemFrequencyResult(
                freq_hz=500.0,
                excitation_port_ids=request.excitation_port_ids,
                quantities=(
                    QuantityResult(
                        id=output.id,
                        quantity=output.quantity,
                        unit="Pa",
                        values=np.full(
                            (len(request.excitation_port_ids), len(output.options["points_m"])),
                            1 + 2j,
                            dtype=np.complex64,
                        ),
                        axes=("excitation", "observation"),
                    ),
                ),
                diagnostics={"engine_provenance": {"engine": {"version": "test"}}},
            )
            return SimpleNamespace(solve_stream=lambda **kwargs: iter([result]))

    monkeypatch.setattr("blab.remote.RemoteBackend", Backend)
    worker = SystemSolveWorker(prepared)
    errors, results, live = [], [], []
    worker.failed.connect(errors.append)
    worker.system_result_ready.connect(results.append)
    worker.result_ready.connect(live.append)
    worker.run()
    assert not errors
    assert calls == ["beat_cuda"]
    assert len(results) == len(live) == 1
    assert results[0].diagnostics["engine_provenance"]["engine"]["version"] == "test"
    np.testing.assert_array_equal(live[0].horizontal_pressure, 1 + 2j)


def test_connection_check_ignores_stale_results(qapp):
    dialog = PreferencesDialog(GuiPreferences(solve_backend="beat_remote"))
    widget = dialog.server_preferences
    old_revision = widget._revision
    widget.url.setText("https://new.example")
    widget._queue.put((old_revision, "Connected to old server"))
    widget.poll()
    assert "old server" not in widget.status.text()
    dialog.close()


def test_saved_server_settings_exclude_token_value(qapp, tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from blab.ui.settings import load_gui_preferences, save_gui_preferences

    monkeypatch.setenv("LAB_TOKEN", "secret-value-not-for-settings")
    path = tmp_path / "preferences.ini"
    settings = QSettings(str(path), QSettings.IniFormat)
    original = GuiPreferences(
        solve_backend="beat_remote",
        solve_server_url="https://lab.example",
        solve_server_token_env="LAB_TOKEN",
        solve_server_ca="lab.pem",
        solve_server_backend="beat_rocm",
    )
    save_gui_preferences(settings, original)
    settings.sync()
    loaded = load_gui_preferences(settings)
    for field in (
        "solve_backend",
        "solve_server_url",
        "solve_server_backend",
        "solve_server_ca",
        "solve_server_token_env",
    ):
        assert getattr(loaded, field) == getattr(original, field)
    assert "secret-value-not-for-settings" not in path.read_text()


def test_remote_results_do_not_enable_observation_planes():
    from blab.ui.observation_plane_results import (
        exterior_field_results_from_solved_system,
        interior_field_results_from_solved_system,
    )

    solved = SimpleNamespace(provenance=SimpleNamespace(backend_id="beat_remote", solve_kind="coupled_bem_fem"))
    assert interior_field_results_from_solved_system(solved) is None
    assert exterior_field_results_from_solved_system(solved) is None


def test_connection_check_runs_off_ui_thread(qapp, monkeypatch):
    import threading

    from PySide6.QtTest import QTest

    entered, release = threading.Event(), threading.Event()

    def capabilities(_self):
        entered.set()
        assert release.wait(3)
        return {"backend_ids": ["beat_cpu"]}

    monkeypatch.setattr("blab.remote.RemoteBackend.check_capabilities", capabilities)
    dialog = PreferencesDialog(GuiPreferences(solve_backend="beat_remote"))
    widget = dialog.server_preferences
    widget.check.click()
    assert entered.wait(2)
    assert not widget.check.isEnabled()
    release.set()
    for _ in range(30):
        QTest.qWait(50)
        if widget.check.isEnabled():
            break
    assert "CPU: available" in widget.status.text()
    dialog.close()
