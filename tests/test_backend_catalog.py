import importlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from beat_engine import BackendInfo, backend_catalog, engine_paths

from blab.headless import HeadlessSolveSpec, load_headless_project, prepare_headless_solve, resolve_headless_backend
from blab.solvers import registry
from blab.solvers.beat_engine_runtime import default_beat_engine_project, julia_process_env
from blab.solvers.coupled_backend import PhysicalSystemProductionBackend
from blab.ui.dialogs import PreferencesDialog
from blab.ui.settings import GuiPreferences
from blab.ui.system_solve import SystemSolveWorker


@pytest.mark.parametrize("info", backend_catalog(), ids=lambda info: info.backend_id)
def test_catalog_backends_round_trip_without_starting_workers(qapp, monkeypatch, info):
    def no_process(*args, **kwargs):
        raise AssertionError("Choosing a backend must not start a process")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    backend_id = f"beat_{info.backend_id}"
    dialog = PreferencesDialog(GuiPreferences(solve_backend=backend_id))
    try:
        assert dialog.solve_backend_combo.currentText() == info.label
        assert dialog.preferences().solve_backend == backend_id
        assert dialog.solve_backend_options[info.label] == backend_id
        source = registry.create_backend(backend_id)
        system = PhysicalSystemProductionBackend(bem_backend=info.backend_id)
        assert source.beat_engine_backend == system.bem_backend == info.backend_id
        assert source.julia_project == system.julia_project == engine_paths(info.backend_id).project
        assert resolve_headless_backend(info.backend_id) == backend_id
    finally:
        dialog.close()


def test_future_engine_backend_needs_no_application_list_update(monkeypatch):
    from beat_engine import backends

    future = BackendInfo("future", "Future engine backend", "julia_future", ("linux",))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(backends, "_BACKENDS", (*backend_catalog(), future))
            importlib.reload(registry)
            assert registry.backend_label_to_id()[future.label] == "beat_future"
            # Registry metadata/factory definitions are obtained from the engine.
            assert registry.supports_physical_system_solves("beat_future")
    finally:
        importlib.reload(registry)


def test_metal_project_and_environment_are_not_redirected():
    paths = engine_paths("metal")
    assert default_beat_engine_project("beat_metal") == paths.project
    assert julia_process_env(2, paths.project)["BLAB_BEAT_ENGINE_GPU_BACKEND"] == "metal"
    for create in (default_beat_engine_project, lambda value: PhysicalSystemProductionBackend(bem_backend=value)):
        with pytest.raises(ValueError, match="Unknown BEAT Engine backend"):
            create("typo")


def test_backend_failure_is_emitted_only_when_solve_runs(qapp, monkeypatch):
    from blab.solvers import coupled_backend

    attempts = []

    def submit(*args, **kwargs):
        attempts.append(True)
        raise RuntimeError("Runtime/device or required solver libraries are not functional.")

    monkeypatch.setattr(coupled_backend, "get_beat_engine_worker",
                        lambda **kwargs: SimpleNamespace(submit=submit, worker_info={}))
    project = load_headless_project(Path(__file__).parent / "fixtures/remote-exterior.blab.json")
    prepared = prepare_headless_solve(project, HeadlessSolveSpec(frequencies_hz=(500.0,)), backend_id="beat_metal")
    worker = SystemSolveWorker(prepared)
    errors = []
    worker.failed.connect(errors.append)
    assert attempts == []
    assert errors == []
    worker.run()
    assert attempts == [True]
    assert len(errors) == 1
    assert "BEAT Engine (Apple Metal) could not run the solve" in errors[0]
    assert "Runtime/device" in errors[0]
