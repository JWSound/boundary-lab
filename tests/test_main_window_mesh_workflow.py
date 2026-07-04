from pathlib import Path

import meshio
import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QByteArray

import blab.ui.main_window as main_window_module
from blab.ath import AthRunResult
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.main_window import (
    DEFAULT_DOCK_STATE_B64,
    OBSOLETE_DOCK_OBJECT_NAMES,
    STITCH_FAILURE_MESSAGE,
    STITCHED_MESH_NAME,
    MainWindow,
    _dock_state_has_obsolete_object_names,
)
from blab.ui.project_state import AthScriptState, scripts_from_payload
from blab.ui.settings import GuiPreferences


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


def test_default_dock_state_does_not_reference_removed_plots_dock() -> None:
    dock_state = QByteArray.fromBase64(DEFAULT_DOCK_STATE_B64.encode("ascii"))

    assert not _dock_state_has_obsolete_object_names(dock_state)


def test_obsolete_dock_state_detector_finds_legacy_plots_dock() -> None:
    legacy_state = QByteArray(b"prefix" + OBSOLETE_DOCK_OBJECT_NAMES[0].encode("utf-16-be") + b"suffix")

    assert _dock_state_has_obsolete_object_names(legacy_state)


def test_restore_window_state_skips_legacy_plots_dock_layout() -> None:
    legacy_state = QByteArray(b"prefix" + OBSOLETE_DOCK_OBJECT_NAMES[0].encode("utf-16-be") + b"suffix")
    removed_keys = []
    restore_calls = []

    class Settings:
        def value(self, key):
            return legacy_state if key == "window/dock_state" else None

        def remove(self, key):
            removed_keys.append(key)

    class Workspace:
        def restoreState(self, state):  # noqa: N802 - Qt API shape
            restore_calls.append(state)
            return True

    window = MainWindow.__new__(MainWindow)
    window.settings = Settings()
    window.workspace = Workspace()
    window.plot_entries = ()
    window.restoreGeometry = lambda _geometry: None
    window._sync_panel_view_action = lambda _dock_id: None
    window._sync_plot_view_action = lambda _plot_id: None

    MainWindow._restore_window_state(window)

    assert removed_keys == ["window/dock_state"]
    assert restore_calls == []


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
    window._stitch_candidate_mesh_configs = lambda: (MeshConfig(name="ath", file=str(mesh_path), scale_factor=0.001),)
    window._all_radiators = lambda: ()

    window._refresh_mesh_preview()

    assert loaded["meshes"][0].name == "ath"
    assert loaded["kwargs"]["symmetry"] == "xy"
    assert loaded["status"] == "Mesh preview showing unstitched meshes; stitching failed"
    assert "cleared" not in loaded


def test_preview_refresh_reports_non_stitch_failures() -> None:
    loaded = {}

    class PreviewStub:
        def clear(self) -> None:
            loaded["cleared"] = True

    class StatusStub:
        def setText(self, text: str) -> None:
            loaded["status"] = text

    window = MainWindow.__new__(MainWindow)
    window.stitch_imported_meshes = False
    window.preview = PreviewStub()
    window.status_label = StatusStub()
    window._last_mesh_preview_error = None
    window._has_solver_meshes = lambda: True
    window._solver_mesh_configs = lambda: (_ for _ in ()).throw(ValueError("bad mesh"))

    window._refresh_mesh_preview()

    assert loaded["cleared"] is True
    assert window._last_mesh_preview_error == "bad mesh"
    assert loaded["status"] == "Mesh preview failed: bad mesh"


def test_generated_result_updates_script_mesh_scale(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    script = AthScriptState(id="script1", name="ath", config_text="")
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "case.cfg",
        driven_tag=2,
        radiators=(),
    )
    emitted = {}

    class SignalStub:
        def emit(self, reason: str) -> None:
            emitted["reason"] = reason

    class StatusStub:
        def setText(self, text: str) -> None:
            emitted["status"] = text

    window = MainWindow.__new__(MainWindow)
    window.ath_generation_script_id = script.id
    window.ath_generation_mesh_name = script.mesh_name
    window.ath_results_by_script_id = {}
    window.ath_scripts = (script,)
    window.mesh_state_changed = SignalStub()
    window.status_label = StatusStub()
    window._last_mesh_preview_error = None
    window._apply_saved_source_config_to_result = lambda generated_result, _mesh_name: generated_result
    window._show_mesh_quality_warning = lambda _result: None

    window._on_ath_generated(result)

    assert window.ath_scripts[0].mesh_scale_factor == 0.001
    assert emitted["reason"] == "ath_mesh_generated"


def test_generation_start_clears_stale_generated_mesh_result(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    script = AthScriptState(
        id="script1",
        name="ath",
        config_text="",
        output_dir=str(tmp_path),
        msh_path=str(mesh_path),
        cleaned_msh_path=str(mesh_path),
        config_path=str(tmp_path / "case.cfg"),
    )
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "case.cfg",
        driven_tag=2,
        radiators=(),
        cleaned_msh_path=mesh_path,
    )
    emitted = {}

    class SignalStub:
        def emit(self, reason: str) -> None:
            emitted["reason"] = reason

    window = MainWindow.__new__(MainWindow)
    window.ath_scripts = (script,)
    window.ath_results_by_script_id = {script.id: result}
    window.mesh_state_changed = SignalStub()

    window._clear_generated_result_for_script(script.id, "geometry_generation_started")

    assert window.ath_results_by_script_id == {}
    assert window.ath_scripts[0].output_dir is None
    assert window.ath_scripts[0].msh_path is None
    assert window.ath_scripts[0].cleaned_msh_path is None
    assert window.ath_scripts[0].config_path is None
    assert emitted["reason"] == "geometry_generation_started"


def test_saved_ath_result_restores_script_mesh_scale(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    script = AthScriptState(
        id="script1",
        name="ath",
        config_text="",
        output_dir=str(tmp_path),
        msh_path=str(mesh_path),
        config_path=str(tmp_path / "case.cfg"),
        mesh_scale_factor=0.001,
    )
    window = MainWindow.__new__(MainWindow)

    result = window._result_from_script_state(script)

    assert result is not None


def test_legacy_saved_ath_scale_is_migrated_before_building_solver_meshes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    script = scripts_from_payload(
        [
            {
                "id": "script1",
                "name": "ath",
                "config_text": "",
                "mesh_scale_factor": 1.0,
            }
        ]
    )[0]
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "case.cfg",
        driven_tag=2,
        radiators=(),
    )
    window = MainWindow.__new__(MainWindow)
    window.symmetry = "off"
    window.ath_scripts = (script,)
    window.ath_results_by_script_id = {script.id: result}

    mesh_configs = window._ath_solver_mesh_configs()

    assert mesh_configs[0].scale_factor == 0.001


def test_stale_in_memory_ath_scale_is_repaired_before_building_solver_meshes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    script = AthScriptState(id="script1", name="ath", config_text="", mesh_scale_factor=1000.0)
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "case.cfg",
        driven_tag=2,
        radiators=(),
    )
    window = MainWindow.__new__(MainWindow)
    window.symmetry = "off"
    window.ath_scripts = (script,)
    window.ath_results_by_script_id = {script.id: result}

    mesh_configs = window._ath_solver_mesh_configs()

    assert mesh_configs[0].scale_factor == 0.001


def test_import_config_applies_abec_solve_metadata_to_gui_controls(tmp_path: Path) -> None:
    config_path = tmp_path / "case.cfg"
    config_path.write_text(
        """
        ABEC.NumFrequencies = 40
        ABEC.Polars:SPL_H = {
        Distance = 2
        MapAngleRange = 0,180,37
        NormAngle = 0
        }
        ABEC.Polars:SPL_V = {
        Distance = 2
        Inclination = 90
        MapAngleRange = 0,180,37
        NormAngle = 0
        }
        ABEC.f1 = 100
        ABEC.f2 = 20000
        """,
        encoding="utf-8",
    )
    script = AthScriptState(id="script1", name="ath", config_text="")
    saved_preferences = []
    status = {}

    class SpinStub:
        def __init__(self, minimum: int, maximum: int, value: int):
            self._minimum = minimum
            self._maximum = maximum
            self._value = value

        def minimum(self) -> int:
            return self._minimum

        def maximum(self) -> int:
            return self._maximum

        def setValue(self, value: int) -> None:  # noqa: N802 - Qt API shape
            self._value = int(value)

        def value(self) -> int:
            return self._value

    class StatusStub:
        def setText(self, text: str) -> None:
            status["text"] = text

    window = MainWindow.__new__(MainWindow)
    window.ath_scripts = (script,)
    window.active_ath_script_id = script.id
    window.freq_min_spin = SpinStub(20, 20000, 500)
    window.freq_max_spin = SpinStub(20, 20000, 1000)
    window.freq_count_spin = SpinStub(3, 200, 3)
    window.preferences = GuiPreferences(
        polar_angle_step_deg=30.0,
        polar_observation_distance_m=1.0,
        horizontal_normalization_angle=10.0,
        vertical_normalization_angle=10.0,
    )
    window._save_preferences = lambda: saved_preferences.append(window.preferences)
    window._rebuild_ath_script_tabs = lambda: None
    window.status_label = StatusStub()

    window._import_config_path(config_path)

    assert window.freq_min_spin.value() == 100
    assert window.freq_max_spin.value() == 20000
    assert window.freq_count_spin.value() == 40
    assert window.preferences.polar_angle_step_deg == pytest.approx(5.0)
    assert window.preferences.polar_observation_distance_m == 2.0
    assert window.preferences.horizontal_normalization_angle == 0.0
    assert window.preferences.vertical_normalization_angle == 0.0
    assert saved_preferences
    assert status["text"] == f"Imported {config_path}"


def test_import_config_into_script_makes_that_script_active(tmp_path: Path) -> None:
    config_path = tmp_path / "case.cfg"
    config_path.write_text("ABEC.NumFrequencies = 40\n", encoding="utf-8")
    script_a = AthScriptState(id="script1", name="empty", config_text="")
    script_b = AthScriptState(id="script2", name="loaded", config_text="")

    class SpinStub:
        def __init__(self, value: int):
            self._value = value

        def minimum(self) -> int:
            return 1

        def maximum(self) -> int:
            return 200

        def setValue(self, value: int) -> None:  # noqa: N802 - Qt API shape
            self._value = int(value)

    class StatusStub:
        def setText(self, _text: str) -> None:
            pass

    window = MainWindow.__new__(MainWindow)
    window.ath_scripts = (script_a, script_b)
    window.active_ath_script_id = script_a.id
    window.freq_min_spin = SpinStub(500)
    window.freq_max_spin = SpinStub(1000)
    window.freq_count_spin = SpinStub(3)
    window.preferences = GuiPreferences()
    window._save_preferences = lambda: None
    window._rebuild_ath_script_tabs = lambda: None
    window.status_label = StatusStub()

    window._import_config_path(config_path, script_id=script_b.id)

    assert window.active_ath_script_id == script_b.id
    assert window.ath_scripts[1].config_text == "ABEC.NumFrequencies = 40\n"


def test_solver_solve_settings_fall_back_to_enabled_ath_mesh_config(tmp_path: Path) -> None:
    mesh_path = tmp_path / "case.msh"
    _write_triangle_mesh(mesh_path)
    active_script = AthScriptState(id="script1", name="blank", config_text="")
    mesh_script = AthScriptState(
        id="script2",
        name="generated",
        config_text="""
        ABEC.NumFrequencies = 40
        ABEC.f1 = 100
        ABEC.f2 = 20000
        ABEC.Polars:SPL_H = {
        Distance = 2
        MapAngleRange = 0,180,37
        NormAngle = 0
        }
        """,
    )
    result = AthRunResult(
        output_dir=tmp_path,
        msh_path=mesh_path,
        config_path=tmp_path / "case.cfg",
        driven_tag=2,
        radiators=(),
    )

    window = MainWindow.__new__(MainWindow)
    window.ath_scripts = (active_script, mesh_script)
    window.active_ath_script_id = active_script.id
    window.ath_results_by_script_id = {mesh_script.id: result}

    settings = window._solver_ath_solve_settings()

    assert settings.freq_min_hz == 100.0
    assert settings.freq_max_hz == 20000.0
    assert settings.freq_count == 40
    assert settings.polar_step_deg == pytest.approx(5.0)


def test_import_config_links_adjacent_existing_ath_mesh_output(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    mesh_dir = case_dir / "ABEC_FreeStanding"
    mesh_dir.mkdir(parents=True)
    config_path = case_dir / "config.txt"
    config_path.write_text("ABEC.NumFrequencies = 40\n", encoding="utf-8")
    msh_path = mesh_dir / "case.msh"
    _write_triangle_mesh(msh_path)

    script = AthScriptState(
        id="script1",
        name="case",
        config_text="old",
        output_dir="old",
        msh_path="old.msh",
        config_path="old.cfg",
    )
    emitted = {}

    class SpinStub:
        def __init__(self, value: int):
            self._value = value

        def minimum(self) -> int:
            return 1

        def maximum(self) -> int:
            return 200

        def setValue(self, value: int) -> None:  # noqa: N802 - Qt API shape
            self._value = int(value)

    class StatusStub:
        def setText(self, text: str) -> None:
            emitted["status"] = text

    class SignalStub:
        def emit(self, reason: str) -> None:
            emitted["reason"] = reason

    window = MainWindow.__new__(MainWindow)
    window.ath_scripts = (script,)
    window.active_ath_script_id = script.id
    window.ath_results_by_script_id = {}
    window.freq_min_spin = SpinStub(500)
    window.freq_max_spin = SpinStub(1000)
    window.freq_count_spin = SpinStub(3)
    window.preferences = GuiPreferences()
    window._save_preferences = lambda: None
    window._rebuild_ath_script_tabs = lambda: None
    window._apply_saved_source_config_to_result = lambda result, _mesh_name: result
    window.status_label = StatusStub()
    window.mesh_state_changed = SignalStub()

    window._import_config_path(config_path)

    result = window.ath_results_by_script_id[script.id]
    assert result.output_dir == case_dir
    assert result.msh_path == msh_path
    assert window.ath_scripts[0].output_dir == str(case_dir)
    assert window.ath_scripts[0].msh_path == str(msh_path)
    assert window.ath_scripts[0].config_path == str(config_path)
    assert window.ath_scripts[0].mesh_scale_factor == 0.001
    assert emitted["reason"] == "ath_config_imported_with_existing_mesh"


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


def test_primary_solver_mesh_scale_uses_primary_mesh_scale() -> None:
    # The advertised top-level scale must equal the primary mesh's own scale, not
    # the millimetre default. A metre-unit mesh (scale 1.0) must keep 1.0, or
    # single-file backends that scale mesh_file by it collapse the model to a
    # point source (flat directivity at all angles).
    mesh_configs = (
        MeshConfig(name="wg", file="wg.msh", scale_factor=1.0),
        MeshConfig(name="other", file="other.msh", scale_factor=0.001),
    )
    assert MainWindow._primary_solver_mesh_scale(mesh_configs) == 1.0


def test_primary_solver_mesh_scale_falls_back_to_default() -> None:
    assert MainWindow._primary_solver_mesh_scale(()) == 0.001
    assert MainWindow._primary_solver_mesh_scale((MeshConfig(name="wg", file="wg.msh"),)) == 0.001


def test_ensure_ath_runtime_config_creates_missing_companion_cfg(tmp_path: Path, monkeypatch) -> None:
    ath_dir = tmp_path / "ath"
    ath_dir.mkdir()
    (ath_dir / "ath.exe").write_text("", encoding="utf-8")
    gmsh_exe = tmp_path / "gmsh" / "gmsh.exe"
    gmsh_exe.parent.mkdir()
    gmsh_exe.write_text("", encoding="utf-8")
    output_root = tmp_path / "runs" / "ath_output"

    monkeypatch.setattr(main_window_module, "ATH_BUNDLE_DIR", ath_dir)
    monkeypatch.setattr(main_window_module, "GMSH_BUNDLE_EXE", gmsh_exe)
    monkeypatch.setattr(main_window_module, "ATH_OUTPUT_ROOT", output_root)

    window = MainWindow.__new__(MainWindow)
    window._ensure_ath_runtime_config()

    cfg_text = (ath_dir / "ath.cfg").read_text(encoding="utf-8")
    assert f'OutputRootDir = "{output_root.resolve()}"' in cfg_text
    assert f'MeshCmd = "{gmsh_exe.resolve()} %f -"' in cfg_text


def test_native_open_edge_check_stays_strict_for_closed_generated_meshes() -> None:
    window = MainWindow.__new__(MainWindow)
    window.symmetry = "xy"
    window._enabled_ath_results = lambda: (
        (AthScriptState(id="script1", name="ath", config_text="mode = freestanding\n"), object()),
    )

    assert window._native_check_open_edges_for_solver() is True


def test_native_open_edge_check_opts_out_for_bare_generated_meshes() -> None:
    window = MainWindow.__new__(MainWindow)
    window.symmetry = "xy"
    window._enabled_ath_results = lambda: (
        (AthScriptState(id="script1", name="ath", config_text="Mesh.Mode = bare\n"), object()),
    )

    assert window._native_check_open_edges_for_solver() is False


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
    window._all_radiators = lambda: (RadiatorConfig(name="stitched:SD1D1001", mesh="stitched", tag=2, channel="main"),)

    channels = window._channel_configs_for_current_radiators()

    assert [channel.name for channel in channels] == ["HF", "main"]
    assert channels[0].polarity == -1


def test_discard_channel_config_dialog_deletes_stale_dialog() -> None:
    deleted = {}

    class DialogStub:
        def deleteLater(self) -> None:
            deleted["dialog"] = True

    window = MainWindow.__new__(MainWindow)
    window.channel_config_dialog = DialogStub()

    window._discard_channel_config_dialog()

    assert deleted["dialog"] is True
    assert window.channel_config_dialog is None


def test_channel_config_uses_bottom_button_and_modeless_dialog() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    open_channel_config = source[source.index("def open_channel_config") : source.index("def _set_panel_visible")]

    assert 'self.channel_config_button = QPushButton("Channel Config")' in source
    assert "controls_layout.addWidget(self.mesh_config_button)" in source
    assert "controls_layout.addWidget(self.channel_config_button)" in source
    assert "controls_layout.addWidget(self.source_config_button)" in source
    assert "self.channel_config_button.clicked.connect(self.open_channel_config)" in source
    assert '("channel_config", "Channel Config Panel")' not in source
    assert "ChannelConfigDialog(self._channel_configs_for_current_radiators(), self)" in open_channel_config
    assert "dialog.show()" in open_channel_config
    assert "dialog.activateWindow()" in open_channel_config
    assert "_make_panel_dock" not in open_channel_config
    assert "addDockWidget" not in open_channel_config


def test_project_dirty_state_ignores_generated_ath_mesh_paths() -> None:
    payload = {
        "ath_scripts": [
            {
                "id": "script1",
                "config_text": "",
                "output_dir": "",
                "msh_path": "",
                "cleaned_msh_path": "",
                "config_path": "",
            }
        ]
    }
    window = MainWindow.__new__(MainWindow)
    window._project_clean_payload = None
    window._project_payload = lambda: payload

    assert not window._has_unsaved_project_changes()

    window._mark_project_clean()

    assert not window._has_unsaved_project_changes()

    payload["ath_scripts"][0]["output_dir"] = "runs/ath_output/example"
    payload["ath_scripts"][0]["msh_path"] = "runs/ath_output/example/example.msh"
    payload["ath_scripts"][0]["cleaned_msh_path"] = "runs/ath_output/example/example_clean.msh"
    payload["ath_scripts"][0]["config_path"] = "runs/ath_output/example.cfg"

    assert not window._has_unsaved_project_changes()

    payload["ath_scripts"][0]["mesh_scale_factor"] = 0.002

    assert window._has_unsaved_project_changes()
