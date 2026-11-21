from pathlib import Path
from zipfile import ZipFile

import numpy as np

from blab.config import ChannelConfig
from blab.live import LiveSolveDataset
from blab.project.io import PROJECT_SCHEMA_VERSION, read_project_file
from blab.project.package import read_package, write_package
from blab.solve_results.model import SolvedSystem, SolveProvenance
from blab.solvers.base import FrequencyResult
from blab.ui.main_window.solve_session import SolveSession


def test_portable_project_and_resynthesis_roundtrip(tmp_path):
    mesh = tmp_path / "original.msh"
    mesh.write_bytes(b"mesh fixture")
    path = tmp_path / "design.blab"
    payload = {"schema_version": PROJECT_SCHEMA_VERSION, "imported_meshes": [{"source_file": str(mesh)}]}
    live = LiveSolveDataset(np.array([0.0, 90.0]), channel_configs=(ChannelConfig("main"),))
    basis = np.array([[1 + 2j, -2j]], dtype=np.complex64)
    live.add(
        FrequencyResult(
            200.0,
            np.zeros(2),
            np.zeros(2),
            np.zeros((1, 2)),
            channel_names=np.array(["main"]),
            horizontal_pressure=basis,
            vertical_pressure=basis,
        )
    )
    solved = SolvedSystem(
        "test",
        SolveProvenance("beat_cpu", "exterior_bem"),
        np.array([200.0]),
        ("port",),
        {},
        {},
        np.array([True]),
        ({},),
        "complete",
    )
    session = SolveSession(live_dataset=live, solved_system=solved)
    write_package(path, payload, True, session)
    mesh.unlink()
    assert Path(read_project_file(path)["imported_meshes"][0]["source_file"]).read_bytes() == b"mesh fixture"
    _, include, restored = read_package(path, with_results=True)
    assert include and restored["solved_system"].complete
    copy = restored["live_dataset"]
    for dataset in (live, copy):
        dataset.set_channel_synthesis((ChannelConfig("main", level_db=-6, delay_ms=2),))
    for left, right in zip(live.as_raw_polar_arrays(), copy.as_raw_polar_arrays(), strict=True):
        np.testing.assert_array_equal(left, right)
    write_package(path, {}, False, session)
    with ZipFile(path) as archive:
        assert "results.json" not in archive.namelist()


def test_save_as_and_open_restore_desktop_results(main_window, monkeypatch, tmp_path):
    workflow = main_window.project_workflow
    path = tmp_path / "sample.blab"
    live = LiveSolveDataset(np.array([0.0, 90.0]), channel_configs=(ChannelConfig("main"),))
    pressure = np.array([[1 + 2j, 3 - 1j]], dtype=np.complex64)
    live.add(
        FrequencyResult(
            100.0,
            np.zeros(2),
            np.zeros(2),
            np.zeros((1, 2)),
            channel_names=np.array(["main"]),
            horizontal_pressure=pressure,
            vertical_pressure=pressure,
        )
    )
    main_window.solve_session.live_dataset = live
    monkeypatch.setattr(main_window, "choose_project_save", lambda *args: (path, True))
    assert workflow.save_project_as()
    workflow.new_project()
    workflow._preparations = None
    workflow.open_recent_project(path)
    restored = main_window.solve_session.live_dataset
    assert restored is not None and restored.supports_channel_resynthesis
    np.testing.assert_array_equal(restored.results[100.0].horizontal_pressure, pressure)
    assert not workflow.has_unsaved_project_changes()
    monkeypatch.setattr(main_window, "choose_project_save", lambda *args: (path, False))
    assert workflow.save_project_as()
    assert workflow.save_project()
    assert read_package(path, with_results=True)[1:] == (False, None)
