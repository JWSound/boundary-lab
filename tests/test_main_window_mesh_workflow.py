from pathlib import Path
from types import SimpleNamespace

import meshio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from blab.ath import AthRunResult
from blab.config import MeshConfig
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.main_window import MainWindow, STITCH_FAILURE_MESSAGE
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
