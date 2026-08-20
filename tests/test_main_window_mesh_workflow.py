import os
from dataclasses import replace
from pathlib import Path

import meshio
import numpy as np
import pytest

import blab.ui.main_window.mesh_workflow as mesh_workflow_module
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.generators.ath import ath_source
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.physical_model import (
    AcousticInterface,
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
from blab.ui.main_window.project_session import ProjectSession
from blab.ui.main_window.project_workflow import ProjectWorkflowController
from blab.ui.project_state import ProjectPreferencesState
from blab.ui.system_config import InterfaceRebuildResult
from repo_paths import MAIN_WINDOW_PKG


def main_window_source(*stems: str) -> str:
    """Concatenated source of the main_window package.

    ``main_window.py`` was split into a package of mixins; these remaining
    assertions still describe wiring rather than behaviour, so they read the
    package as one blob. Pass module stems to narrow the read when a test
    slices between two markers that must stay in the same module.
    """
    paths = [MAIN_WINDOW_PKG / f"{stem}.py" for stem in stems] if stems else sorted(MAIN_WINDOW_PKG.glob("*.py"))
    return "\n".join(path.read_text(encoding="utf-8") for path in paths)


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


def _write_tetrahedral_mesh(path: Path) -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        cells=[("tetra", np.array([[0, 1, 2, 3]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([1], dtype=np.int32)]},
        field_data={"Air": np.array([1, 3], dtype=np.int32)},
    )
    meshio.write(path, mesh, file_format="gmsh22", binary=False)


@pytest.mark.parametrize("volume_mesh", [False, True], ids=["bem-surface", "fem-volume"])
def test_focus_reload_detects_changed_surface_and_volume_meshes(
    main_window,
    tmp_path: Path,
    volume_mesh: bool,
) -> None:
    source_path = tmp_path / ("interior.msh" if volume_mesh else "exterior.msh")
    if volume_mesh:
        _write_tetrahedral_mesh(source_path)
    else:
        _write_triangle_mesh(source_path)

    entry = MeshDialogEntry(name="Interior" if volume_mesh else "Exterior", source_file=str(source_path))
    main_window.imported_meshes = main_window._clean_imported_meshes((entry,))
    main_window._record_imported_mesh_source_fingerprints()
    mesh_reasons = []
    invalidation_reasons = []
    main_window.mesh_state_changed.connect(mesh_reasons.append)
    main_window.solve_results_invalidated.connect(invalidation_reasons.append)

    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    if not volume_mesh:
        cleaned_path = Path(main_window.imported_meshes[0].cleaned_file)
        newer = cleaned_path.stat().st_mtime + 2.0
        os.utime(source_path, (newer, newer))

    assert main_window._updated_imported_mesh_names() == (entry.name,)

    main_window._last_imported_mesh_focus_check_at = 0.0
    main_window._reload_updated_imported_meshes_on_focus()

    assert mesh_reasons == ["imported_mesh_files_reloaded"]
    assert invalidation_reasons == ["imported_mesh_files_reloaded"]
    assert main_window._updated_imported_mesh_names() == ()
    assert (main_window.imported_meshes[0].cleaned_file is None) is volume_mesh


def test_failed_focus_reload_keeps_changed_fingerprint_for_retry(main_window, tmp_path: Path, monkeypatch) -> None:
    source_path = tmp_path / "interior.msh"
    _write_tetrahedral_mesh(source_path)
    main_window.imported_meshes = (MeshDialogEntry(name="Interior", source_file=str(source_path)),)
    main_window._record_imported_mesh_source_fingerprints()
    source_path.write_text(source_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    emitted = []
    invalidated = []
    main_window.mesh_state_changed.connect(emitted.append)
    main_window.solve_results_invalidated.connect(invalidated.append)
    monkeypatch.setattr(
        main_window,
        "_clean_imported_meshes",
        lambda _meshes: (_ for _ in ()).throw(ValueError("mesh is still being written")),
    )

    main_window._last_imported_mesh_focus_check_at = 0.0
    main_window._reload_updated_imported_meshes_on_focus()

    assert emitted == []
    assert invalidated == ["imported_mesh_files_reloaded"]
    assert main_window._updated_imported_mesh_names() == ("Interior",)


def test_focus_reload_rebuilds_configured_interface_for_changed_fem_mesh(
    main_window,
    tmp_path: Path,
    monkeypatch,
) -> None:
    fem_path = tmp_path / "interior.msh"
    bem_path = tmp_path / "exterior.msh"
    rebuilt_path = tmp_path / "exterior_interface_conformed.msh"
    _write_tetrahedral_mesh(fem_path)
    _write_triangle_mesh(bem_path)
    _write_triangle_mesh(rebuilt_path)
    entries = (
        MeshDialogEntry(name="Interior", source_file=str(fem_path)),
        MeshDialogEntry(name="Exterior", source_file=str(bem_path)),
    )
    main_window.imported_meshes = main_window._clean_imported_meshes(entries)
    system = PhysicalSystem(
        id="system",
        name="System",
        meshes=(
            MeshResource("fem", "Interior", str(fem_path), MeshPurpose.FEM_VOLUME),
            MeshResource("bem", "Exterior", str(bem_path), MeshPurpose.BEM_SURFACE),
        ),
        regions=(
            AcousticRegion("interior", "Interior", AcousticRegionKind.BOUNDED_AIR, ("fem",)),
            AcousticRegion("exterior", "Exterior", AcousticRegionKind.UNBOUNDED_AIR, ("bem",)),
        ),
        boundaries=(
            Boundary(
                "fem-interface",
                "Interface",
                "interior",
                PhysicalGroupRef("fem", 2, tag=1),
                BoundaryKind.INTERFACE,
            ),
            Boundary(
                "bem-interface",
                "Interface",
                "exterior",
                PhysicalGroupRef("bem", 2, tag=2),
                BoundaryKind.INTERFACE,
            ),
        ),
        interfaces=(AcousticInterface("interface", "Interface", "fem-interface", "bem-interface"),),
    )
    rebuilt_system = replace(
        system,
        meshes=tuple(
            replace(mesh, file=str(rebuilt_path)) if mesh.name == "Exterior" else mesh for mesh in system.meshes
        ),
    )
    main_window.project.physical_system = system
    main_window._record_imported_mesh_source_fingerprints()
    fem_path.write_text(fem_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    calls = []
    warnings = []

    def rebuild_interfaces(current_system, available_meshes, **options):
        calls.append((current_system, available_meshes, options))
        return InterfaceRebuildResult(
            system=rebuilt_system,
            mesh_file_overrides_by_name={"Exterior": str(rebuilt_path)},
            rebuilt_interface_ids=("interface",),
            quality_warning_interface_ids=("interface",),
        )

    monkeypatch.setattr(mesh_workflow_module, "rebuild_configured_interfaces", rebuild_interfaces)
    monkeypatch.setattr(
        mesh_workflow_module.QMessageBox,
        "warning",
        lambda _parent, title, message: warnings.append((title, message)),
    )

    main_window._last_imported_mesh_focus_check_at = 0.0
    main_window._reload_updated_imported_meshes_on_focus()

    assert len(calls) == 1
    assert calls[0][2]["changed_mesh_names"] == {"Interior"}
    assert main_window.project.physical_system == rebuilt_system
    exterior = next(mesh for mesh in main_window.imported_meshes if mesh.name == "Exterior")
    assert exterior.cleaned_file == str(rebuilt_path)
    assert len(warnings) == 1
    assert warnings[0][0] == "Inspect Simplified Interface"
    assert "Visually inspect the conformed interface" in warnings[0][1]


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


def test_physical_system_solve_entries_use_reduced_generated_mesh_for_symmetry(tmp_path: Path) -> None:
    raw_msh = tmp_path / "ath_case.msh"
    expanded_clean_msh = tmp_path / "ath_case_clean.msh"
    imported_clean_msh = tmp_path / "external_clean.msh"
    _write_triangle_mesh(raw_msh)
    _write_triangle_mesh(expanded_clean_msh)
    _write_triangle_mesh(imported_clean_msh, tag=3)

    document = GeneratorDocument(
        id="design1",
        name="ath",
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
    window.generator_documents = (document,)
    window.generated_geometry_by_document_id = {document.id: result}
    window.imported_meshes = (
        MeshDialogEntry(
            name="external",
            source_file=str(imported_clean_msh),
            cleaned_file=str(imported_clean_msh),
        ),
    )

    editor_entries = window._mesh_config_dialog_entries()
    solve_entries = window.mesh_entries_for_symmetry("xy")
    reduced_msh = tmp_path / "ath_case_clean_reduced.msh"

    assert editor_entries[0].source_file == str(expanded_clean_msh)
    assert solve_entries[0].source_file == str(reduced_msh)
    assert reduced_msh.exists()
    assert solve_entries[1].source_file == str(imported_clean_msh)


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
    window.has_solver_meshes = lambda: True
    window.prepare_mesh_assembly = lambda _radiators: (_ for _ in ()).throw(RuntimeError(STITCH_FAILURE_MESSAGE))
    window._stitch_candidate_mesh_configs = lambda: (MeshConfig(name="ath", file=str(mesh_path), scale_factor=0.001),)
    window.all_radiators = lambda: ()

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
            *window.all_radiators(),
            RadiatorConfig(name="external:SD1D1001", mesh="external", tag=2),
        ),
    )

    assert [(radiator.name, radiator.mesh, radiator.tag) for radiator in radiators] == [
        ("stitched:SD1D1001", "stitched", 2),
        ("stitched:SD1D1001_mesh2", "stitched", 1),
    ]


def test_solver_channels_include_radiator_default_channel_when_missing() -> None:
    window = MainWindow.__new__(MainWindow)
    window.channel_configs = lambda: (ChannelConfig(name="HF"),)

    channels = window.solver_channel_configs(
        (RadiatorConfig(name="stitched:SD1D1001", mesh="stitched", tag=2, channel="main"),)
    )

    assert [channel.name for channel in channels] == ["HF", "main"]


def test_channel_dialog_channels_include_existing_radiator_channels() -> None:
    window = MainWindow.__new__(MainWindow)
    window.channel_configs = lambda: (ChannelConfig(name="HF", polarity=-1),)
    window.all_radiators = lambda: (RadiatorConfig(name="stitched:SD1D1001", mesh="stitched", tag=2, channel="main"),)

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

    window.discard_channel_config_dialog()

    assert deleted["dialog"] is True
    assert window.channel_config_dialog is None


def test_system_and_channel_config_use_bottom_buttons() -> None:
    source = main_window_source()
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


def test_system_apply_coalesces_interface_and_project_preview_refresh() -> None:
    # _apply_system_config is the last method in the dialog_actions mixin, so the
    # slice runs to end-of-module rather than to a marker in a different file.
    source = main_window_source("dialog_actions")
    apply_system = source[source.index("def _apply_system_config") :]

    assert 'reason = "system_interface_mesh_built" if mesh_file_overrides else "system_config_changed"' in apply_system
    assert "self.project_state_changed.emit(reason)" in apply_system
    assert "self.mesh_state_changed.emit" not in apply_system


def _project_controller(monkeypatch, *, payload=None, project_preferences=None):
    """A ProjectWorkflowController with no window behind it.

    Both rules below used to be reached through a half-built ``MainWindow``
    created with ``__new__``; the controller takes its collaborators as
    arguments, so they can be driven directly.
    """

    class FakeView:
        def __init__(self) -> None:
            self.confirmed: list[tuple[str, str]] = []
            self.answer = True

        def confirm(self, title, message):
            self.confirmed.append((title, message))
            return self.answer

    session = ProjectSession()
    controller = ProjectWorkflowController.__new__(ProjectWorkflowController)
    controller._session = session
    controller._view = FakeView()
    if payload is not None:
        monkeypatch.setattr(controller, "project_payload", lambda: payload)
    if project_preferences is not None:
        monkeypatch.setattr(controller, "current_project_preferences", lambda: project_preferences)
    return controller


def test_project_dirty_state_ignores_generated_geometry_artifacts(monkeypatch) -> None:
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
    controller = _project_controller(monkeypatch, payload=payload)

    assert not controller.has_unsaved_project_changes()

    controller.mark_project_clean()

    assert not controller.has_unsaved_project_changes()

    payload["generator_documents"][0]["artifact"] = {
        "output_dir": "runs/ath_output/example",
        "mesh_path": "runs/ath_output/example/example.msh",
        "cleaned_mesh_path": "runs/ath_output/example/example_clean.msh",
        "source_path": "runs/ath_output/example.cfg",
    }

    assert not controller.has_unsaved_project_changes()

    payload["generator_documents"][0]["mesh_scale_factor"] = 0.002

    assert controller.has_unsaved_project_changes()


def test_project_preference_prompt_only_appears_for_differences(monkeypatch) -> None:
    current = ProjectPreferencesState()
    controller = _project_controller(monkeypatch, project_preferences=current)

    assert controller._confirm_apply_project_preferences(None) is False
    assert controller._confirm_apply_project_preferences(current) is False
    assert controller._view.confirmed == []

    different = ProjectPreferencesState(horizontal_normalization_angle=15.0)
    assert controller._confirm_apply_project_preferences(different) is True
    assert "unique application preferences" in controller._view.confirmed[0][1]
