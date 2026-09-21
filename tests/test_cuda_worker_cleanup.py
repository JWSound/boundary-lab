import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import blab.solvers.coupled_backend as backend
from blab.system_contract import SystemSolveRequest


@pytest.mark.parametrize(
    "selected,enabled,runtime,policies,operations,expected",
    [
        ("cuda", True, True, ["cuda_reuse"], ["reclaim"], True),
        ("cpu", True, True, ["cuda_reuse"], ["reclaim"], False),
        ("cuda", False, True, ["cuda_reuse"], ["reclaim"], False),
        ("cuda", True, False, ["cuda_reuse"], ["reclaim"], False),
        ("cuda", True, True, [], ["reclaim"], False),
        ("cuda", True, True, ["cuda_reuse"], [], False),
    ],
)
def test_cleanup_requires_opt_in_actual_cuda_and_both_capabilities(
    monkeypatch, selected, enabled, runtime, policies, operations, expected
):
    payloads = []
    cleanup = {"policy": "cuda_reuse", "reason": "reuse", "seconds": 0.001}

    def submit(path, **kwargs):
        payloads.append(json.loads(path.read_text()))
        return iter([{"type": "completed", "worker_cleanup": cleanup}])

    worker = SimpleNamespace(
        submit=submit,
        ensure_started=Mock(),
        worker_info={"worker_cleanup_policies": policies, "operations": operations},
    )
    if runtime:
        worker.configure_idle_cleanup = Mock()
    monkeypatch.setattr(backend, "get_beat_engine_worker", lambda **kwargs: worker)
    monkeypatch.setattr(
        backend, "system_solve_request_to_dict", lambda request: {"solver_options": request.solver_options}
    )
    request = SystemSolveRequest(SimpleNamespace(meshes=()), (1000,), (), solver_options={"bem_backend": selected})
    session = backend.CoupledSession.__new__(backend.CoupledSession)
    session.request = request
    session.cuda_worker_reuse = enabled
    session.julia_executable = "unused"
    session.solver_script = Path("unused")
    session.julia_project = None
    session.julia_threads = 1
    session._stop = False
    assert list(session._solve_stream_persistent(stop_requested=None)) == []
    if expected:
        assert payloads[0]["solver_options"]["worker_cleanup"] == {
            "policy": "cuda_reuse",
            "max_requests": 8,
            "min_free_fraction": 0.2,
        }
        worker.configure_idle_cleanup.assert_called_once_with(5000)
    else:
        assert "worker_cleanup" not in payloads[0]["solver_options"]
        if runtime:
            worker.configure_idle_cleanup.assert_not_called()
    assert session.request is request
    assert session.worker_cleanup == cleanup


def test_cleanup_preference_roundtrips_without_invalidating_results():
    from dataclasses import replace

    from blab.ui.settings import (
        GuiPreferences,
        load_gui_preferences,
        preferences_require_solve_invalidation,
        preferences_require_visualization_refresh,
        save_gui_preferences,
    )
    from test_ui_settings import MemorySettings

    settings = MemorySettings()
    original = GuiPreferences()
    assert not original.cuda_worker_reuse
    updated = replace(original, cuda_worker_reuse=True)
    save_gui_preferences(settings, updated)
    assert load_gui_preferences(settings).cuda_worker_reuse
    assert not preferences_require_solve_invalidation(original, updated)
    assert not preferences_require_visualization_refresh(original, updated)


def test_preferences_dialog_exposes_opt_in(qapp):
    from blab.ui.dialogs import PreferencesDialog
    from blab.ui.settings import GuiPreferences

    dialog = PreferencesDialog(GuiPreferences())
    try:
        assert not dialog.cuda_worker_reuse_check.isChecked()
        dialog.cuda_worker_reuse_check.setChecked(True)
        assert dialog.preferences().cuda_worker_reuse
    finally:
        dialog.close()


def test_worker_abandoned_before_run_releases_reservation(qapp, monkeypatch):
    from PySide6.QtCore import QCoreApplication, QEvent

    import blab.ui.system_solve as ui_worker

    released = []
    monkeypatch.setattr(ui_worker, "hold_beat_engine_idle_cleanup", lambda: lambda: released.append(True))
    worker = ui_worker.SystemSolveWorker(SimpleNamespace())
    worker.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert released == [True]
