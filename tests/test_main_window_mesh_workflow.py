from pathlib import Path

import meshio
import numpy as np
import pytest

pytest.importorskip("PySide6")

import blab.ui.main_window as main_window_module
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.generators.ath import ath_source
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
)
from blab.ui.dialogs import MeshDialogEntry
from blab.ui.main_window import (
    STITCH_FAILURE_MESSAGE,
    STITCHED_MESH_NAME,
    MainWindow,
    _mesh_entries_with_file_overrides,
    _physical_system_preview_metadata,
)
from blab.ui.project_state import ProjectPreferencesState


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


def test_system_interface_mesh_override_is_persisted_as_imported_cleaned_file(tmp_path: Path) -> None:
    source_path = tmp_path / "exterior.msh"
    conformed_path = tmp_path / "exterior_interface_conformed.msh"
    imported_meshes = (
        MeshDialogEntry(
            name="Exterior",
            source_file=str(source_path),
        ),
    )

    updated = _mesh_entries_with_file_overrides(
        imported_meshes,
        {"Exterior": str(conformed_path)},
    )

    assert updated[0].source_file == str(source_path)
    assert updated[0].cleaned_file == str(conformed_path)


def test_physical_system_preview_metadata_identifies_interface_surfaces_and_mesh_regions() -> None:
    system = PhysicalSystem(
        id="system",
        name="System",
        meshes=(
            MeshResource("fem", "Interior mesh", "interior.msh", MeshPurpose.FEM_VOLUME),
            MeshResource("bem", "Exterior mesh", "exterior.msh", MeshPurpose.BEM_SURFACE),
        ),
        regions=(
            AcousticRegion("interior", "Interior", AcousticRegionKind.BOUNDED_AIR, ("fem",)),
            AcousticRegion("exterior", "Exterior", AcousticRegionKind.UNBOUNDED_AIR, ("bem",)),
        ),
        boundaries=(
            Boundary(
                "fem-interface",
                "Interior interface",
                "interior",
                PhysicalGroupRef("fem", 2, name="Interface"),
                BoundaryKind.INTERFACE,
            ),
            Boundary(
                "bem-interface",
                "Exterior interface",
                "exterior",
                PhysicalGroupRef("bem", 2, tag=7),
                BoundaryKind.INTERFACE,
            ),
            Boundary(
                "driver",
                "Driver",
                "interior",
                PhysicalGroupRef("fem", 2, name="Driver"),
                BoundaryKind.MOVING,
            ),
        ),
        components=(
            PhysicalComponent(
                "component",
                "Driver",
                ComponentKind.IDEAL_VELOCITY_SOURCE,
                ("driver",),
            ),
        ),
    )

    interfaces, component_surfaces, mesh_regions, has_interior = _physical_system_preview_metadata(
        system,
        {
            "Interior mesh": {"Interface": 4, "Driver": 9},
            "Exterior mesh": {"Interface": 7},
        },
    )

    assert interfaces == {("Interior mesh", 4), ("Exterior mesh", 7)}
    assert component_surfaces == {("Interior mesh", 9)}
    assert mesh_regions == {"Interior mesh": "interior", "Exterior mesh": "exterior"}
    assert has_interior is True


def test_mesh_preview_dock_exposes_theme_aware_region_filter_actions() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")

    assert 'QAction("Show interior regions", self)' in source
    assert 'QAction("Show exterior region", self)' in source
    assert "FEMTetra_dark.ico" in source
    assert "FEMTetra_light.ico" in source
    assert "BEMTri_dark.ico" in source
    assert "BEMTri_light.ico" in source
    assert "tool_actions=(self.show_interior_regions_action, self.show_exterior_region_action)" in source
    assert "self.show_interior_regions_action.setEnabled(has_interior_region)" in source
    assert "self.show_exterior_region_action.setEnabled(has_interior_region)" in source


def test_xy_stitch_candidates_use_reduced_generated_mesh_before_stitching(tmp_path: Path) -> None:
    raw_msh = tmp_path / "ath_case.msh"
    expanded_clean_msh = tmp_path / "ath_case_clean.msh"
    imported_clean_msh = tmp_path / "external_clean.msh"
    _write_triangle_mesh(raw_msh)
    _write_triangle_mesh(expanded_clean_msh)
    _write_triangle_mesh(imported_clean_msh, tag=3)

    document = GeneratorDocument(
        id="design1",
        name="waveguide",
        provider_id="ath",
        provider_schema_version=1,
        source=ath_source(""),
    )
    result = GeneratedGeometry(
        provider_id="ath",
        output_dir=tmp_path,
        mesh_path=raw_msh,
        source_path=tmp_path / "ath_case.cfg",
        radiators=(),
        cleaned_mesh_path=expanded_clean_msh,
    )

    window = MainWindow.__new__(MainWindow)
    window.symmetry = "xy"
    window.generator_documents = (document,)
    window.generated_geometry_by_document_id = {document.id: result}
    window.imported_meshes = (
        MeshDialogEntry(
            name="external",
            source_file=str(imported_clean_msh),
            cleaned_file=str(imported_clean_msh),
        ),
    )

    configs = window._stitch_candidate_mesh_configs()
    reduced_msh = tmp_path / "ath_case_clean_reduced.msh"

    assert [config.name for config in configs] == ["waveguide", "external"]
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
    window._prepare_mesh_assembly = lambda _radiators: (_ for _ in ()).throw(RuntimeError(STITCH_FAILURE_MESSAGE))
    window._stitch_candidate_mesh_configs = lambda: (MeshConfig(name="ath", file=str(mesh_path), scale_factor=0.001),)
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

    document = GeneratorDocument(
        id="design1",
        name="waveguide",
        provider_id="ath",
        provider_schema_version=1,
        source=ath_source(""),
    )
    result = GeneratedGeometry(
        provider_id="ath",
        output_dir=tmp_path,
        mesh_path=ath_msh,
        source_path=tmp_path / "ath_case.cfg",
        radiators=(RadiatorConfig(name="waveguide:SD1D1001", mesh="waveguide", tag=2),),
        cleaned_mesh_path=ath_msh,
    )

    window = MainWindow.__new__(MainWindow)
    window.symmetry = "off"
    window.generator_documents = (document,)
    window.generated_geometry_by_document_id = {document.id: result}
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


def test_system_and_channel_config_use_bottom_buttons() -> None:
    source = Path("src/blab/ui/main_window.py").read_text(encoding="utf-8")
    open_channel_config = source[source.index("def open_channel_config") : source.index("def _set_panel_visible")]

    assert 'self.mesh_config_button = QPushButton("Meshes")' in source
    assert 'self.system_config_button = QPushButton("System")' in source
    assert 'self.channel_config_button = QPushButton("Channels")' in source
    assert "controls_layout.addWidget(self.mesh_config_button)" in source
    assert "controls_layout.addWidget(self.system_config_button)" in source
    assert "controls_layout.addWidget(self.channel_config_button)" in source
    assert "controls_layout.addWidget(self.source_config_button)" not in source
    assert "self.system_config_button.clicked.connect(self.open_system_config)" in source
    assert "self.channel_config_button.clicked.connect(self.open_channel_config)" in source
    assert '("channel_config", "Channel Config Panel")' not in source
    assert "ChannelConfigDialog(self._channel_configs_for_current_radiators(), self)" in open_channel_config
    assert "dialog.show()" in open_channel_config
    assert "dialog.activateWindow()" in open_channel_config
    assert "_make_panel_dock" not in open_channel_config
    assert "addDockWidget" not in open_channel_config


def test_project_dirty_state_ignores_generated_geometry_artifacts() -> None:
    payload = {
        "generator_documents": [
            {
                "id": "design1",
                "source": {"format": "ath_cfg", "text": ""},
                "mesh_scale_factor": 0.001,
                "artifact": None,
            }
        ]
    }
    window = MainWindow.__new__(MainWindow)
    window._project_clean_payload = None
    window._project_payload = lambda: payload

    assert not window._has_unsaved_project_changes()

    window._mark_project_clean()

    assert not window._has_unsaved_project_changes()

    payload["generator_documents"][0]["artifact"] = {
        "output_dir": "runs/ath_output/example",
        "mesh_path": "runs/ath_output/example/example.msh",
        "cleaned_mesh_path": "runs/ath_output/example/example_clean.msh",
        "source_path": "runs/ath_output/example.cfg",
    }

    assert not window._has_unsaved_project_changes()

    payload["generator_documents"][0]["mesh_scale_factor"] = 0.002

    assert window._has_unsaved_project_changes()


def test_project_preference_prompt_only_appears_for_differences(monkeypatch) -> None:
    current = ProjectPreferencesState()
    window = MainWindow.__new__(MainWindow)
    window._current_project_preferences = lambda: current
    questions = []

    class MessageBoxStub:
        Yes = 1
        No = 2

        @staticmethod
        def question(*args):
            questions.append(args)
            return MessageBoxStub.Yes

    monkeypatch.setattr(main_window_module, "QMessageBox", MessageBoxStub)

    assert window._confirm_apply_project_preferences(None) is False
    assert window._confirm_apply_project_preferences(current) is False
    assert questions == []

    different = ProjectPreferencesState(horizontal_normalization_angle=15.0)
    assert window._confirm_apply_project_preferences(different) is True
    assert "unique application preferences" in questions[0][2]
