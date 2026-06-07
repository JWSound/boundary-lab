from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from blab.ath import AthRunResult
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.main_window import MainWindow, STITCHED_MESH_NAME, STITCH_FAILURE_MESSAGE
from blab.ui.project_state import AthScriptState


def _write_triangle_mesh(path: Path, tag: int = 2) -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.0],
            ],
            dtype=float,
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([tag], dtype=np.int32)]},
        field_data={"SD1D1001": np.array([tag, 2], dtype=np.int32)},
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


def test_xy_stitch_candidates_use_reduced_ath_mesh_before_stitching(tmp_path: Path) -> None:
    raw_msh = tmp_path / "ath_case.msh"
    expanded_clean_msh = tmp_path / "ath_case_clean.msh"
    imported_clean_msh = tmp_path / "external_clean.msh"
    _write_triangle_mesh(raw_msh)
    _write_triangle_mesh(expanded_clean_msh)
    _write_triangle_mesh(imported_clean_msh, tag=3)

    script = AthScriptState(id="script1", name="ath", config_text="")
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=raw_msh,
        config_path=tmp_path / "ath_case.cfg",
        driven_tag=2,
        radiators=(),
        cleaned_msh_path=expanded_clean_msh,
    )

    window = MainWindow.__new__(MainWindow)
    window.symmetry = "xy"
    window.ath_scripts = (script,)
    window.ath_results_by_script_id = {script.id: result}
    window.imported_meshes = (
        MeshDialogEntry(
            name="external",
            source_file=str(imported_clean_msh),
            cleaned_file=str(imported_clean_msh),
        ),
    )

    configs = window._stitch_candidate_mesh_configs()
    reduced_msh = tmp_path / "ath_case_clean_reduced.msh"

    assert [config.name for config in configs] == ["ath", "external"]
    assert configs[0].file == str(reduced_msh)
    assert reduced_msh.exists()
    assert configs[1].file == str(imported_clean_msh)


def test_preview_falls_back_to_unstitched_meshes_when_preview_stitching_fails(tmp_path: Path) -> None:
    mesh_path = tmp_path / "quarter.msh"
    _write_triangle_mesh(mesh_path)
    loaded = {}

    class PreviewStub:
        def clear(self) -> None:
            loaded["cleared"] = True

        def load_mesh_configs(self, meshes, **kwargs) -> None:
            loaded["meshes"] = meshes
            loaded["kwargs"] = kwargs

    class StatusStub:
        def setText(self, text: str) -> None:
            loaded["status"] = text

    window = MainWindow.__new__(MainWindow)
    window.symmetry = "xy"
    window.stitch_imported_meshes = True
    window.preview = PreviewStub()
    window.status_label = StatusStub()
    window._has_solver_meshes = lambda: True
    window._solver_mesh_configs = lambda: (_ for _ in ()).throw(RuntimeError(STITCH_FAILURE_MESSAGE))
    window._stitch_candidate_mesh_configs = lambda: (
        MeshConfig(name="ath", file=str(mesh_path), scale_factor=0.001),
    )
    window._all_radiators = lambda: ()

    window._refresh_mesh_preview()

    assert loaded["meshes"][0].name == "ath"
    assert loaded["kwargs"]["symmetry"] == "xy"
    assert loaded["status"] == "Mesh preview showing unstitched meshes; stitching failed"
    assert "cleared" not in loaded


def test_stitched_solver_radiators_reference_stitched_mesh(tmp_path: Path) -> None:
    ath_msh = tmp_path / "ath_clean.msh"
    imported_msh = tmp_path / "external_clean.msh"
    _write_triangle_mesh(ath_msh, tag=2)
    _write_triangle_mesh(imported_msh, tag=2)

    script = AthScriptState(id="script1", name="ath", config_text="")
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=ath_msh,
        config_path=tmp_path / "ath_case.cfg",
        driven_tag=2,
        radiators=(RadiatorConfig(name="ath:SD1D1001", mesh="ath", tag=2),),
        cleaned_msh_path=ath_msh,
    )

    window = MainWindow.__new__(MainWindow)
    window.symmetry = "off"
    window.ath_scripts = (script,)
    window.ath_results_by_script_id = {script.id: result}
    window.imported_radiators = ()
    window.imported_meshes = (
        MeshDialogEntry(
            name="external",
            source_file=str(imported_msh),
            cleaned_file=str(imported_msh),
        ),
    )

    radiators = window._radiators_for_solver_meshes(
        (MeshConfig(name=STITCHED_MESH_NAME, file=str(tmp_path / "stitched.msh")),),
        (
            *window._all_radiators(),
            RadiatorConfig(name="external:SD1D1001", mesh="external", tag=2),
        ),
    )

    assert [(radiator.name, radiator.mesh, radiator.tag) for radiator in radiators] == [
        ("stitched:SD1D1001", "stitched", 2),
        ("stitched:SD1D1001_mesh2", "stitched", 1),
    ]


def test_solver_channels_include_radiator_default_channel_when_missing() -> None:
    window = MainWindow.__new__(MainWindow)
    window._channel_configs = lambda: (ChannelConfig(name="HF"),)

    channels = window._channels_for_solver_radiators(
        (RadiatorConfig(name="stitched:SD1D1001", mesh="stitched", tag=2, channel="main"),)
    )

    assert [channel.name for channel in channels] == ["HF", "main"]


def test_channel_dialog_channels_include_existing_radiator_channels() -> None:
    window = MainWindow.__new__(MainWindow)
    window._channel_configs = lambda: (ChannelConfig(name="HF", polarity=-1),)
    window._all_radiators = lambda: (
        RadiatorConfig(name="stitched:SD1D1001", mesh="stitched", tag=2, channel="main"),
    )

    channels = window._channel_configs_for_current_radiators()

    assert [channel.name for channel in channels] == ["HF", "main"]
    assert channels[0].polarity == -1


def test_discard_channel_config_dialog_closes_stale_dialog() -> None:
    closed = {}

    class DialogStub:
        def close(self) -> None:
            closed["called"] = True

    window = MainWindow.__new__(MainWindow)
    window.channel_config_dialog = DialogStub()

    window._discard_channel_config_dialog()

    assert closed["called"] is True
    assert window.channel_config_dialog is None
