from dataclasses import replace

import pytest
from PySide6.QtWidgets import QInputDialog

from blab.generators import GenerationRequest, GenerationResponse
from blab.generators.base import complete_generation
from blab.generators.configuration import project_revision
from blab.physical_compiler import PhysicalSystemCompiler
from blab.ui.dialogs import MeshConfigDialog
from test_generator_assemblies import assembly_case as assembly_case


def accept(main_window, assembly_case):
    project, geometry, patch = assembly_case
    main_window.project = project
    snapshot = main_window._generation_project_snapshot()
    document = snapshot.generator_documents[0]
    request = GenerationRequest(
        document.provider_id,
        document.id,
        document.name,
        document.source,
        geometry.output_dir,
        "test",
        project_revision=project_revision(snapshot),
    )
    main_window.accept_generation(complete_generation(request, GenerationResponse(request.request_id, geometry, patch)))
    return geometry


def test_gui_assembly_mesh_controls_are_linked_and_roundtrip(main_window, assembly_case):
    accept(main_window, assembly_case)
    entries = main_window._mesh_config_dialog_entries()
    assert len(entries) == 2 and all(entry.locked for entry in entries)
    dialog = MeshConfigDialog(entries)
    dialog.x_widgets[0].setValue(25)
    dialog.scale_widgets[1].setValue(0.002)
    dialog.enabled_widgets[0].setChecked(False)
    updated = dialog.meshes()
    assert all(
        entry.translation_mm == (25, 0, 0) and entry.scale_factor == 0.002 and not entry.enabled for entry in updated
    )
    main_window._apply_mesh_config_dialog_entries(updated)
    assert not main_window.project.imported_meshes
    assert len(main_window._mesh_config_dialog_entries()) == 2
    assert not main_window._enabled_generated_geometry()
    dialog.enabled_widgets[1].setChecked(True)
    main_window._apply_mesh_config_dialog_entries(dialog.meshes())
    configs = main_window._generated_solver_mesh_configs()
    assert len(configs) == 2
    assert all(config.translation_m == (0.025, 0, 0) for config in configs)
    dialog.deleteLater()


def test_gui_assembly_rename_and_remove_preserve_identity(main_window, assembly_case, monkeypatch):
    accept(main_window, assembly_case)
    ids = tuple(mesh.id for mesh in main_window.project.physical_system.meshes)
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_kw: ("renamed", True))
    main_window.rename_active_generator_document()
    assert tuple(mesh.id for mesh in main_window.project.physical_system.meshes) == ids
    assert {mesh.name for mesh in main_window.project.physical_system.meshes} == {"renamed__fem", "renamed__bem"}
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1
    main_window._remove_generator_document_at(0)
    assert not main_window.project.physical_system.meshes
    assert not main_window.project.physical_system.interfaces
    assert not main_window.generated_geometry_by_document_id


def test_gui_rejected_regeneration_keeps_previous_assembly(main_window, assembly_case):
    geometry = accept(main_window, assembly_case)
    project = main_window.project
    previous = main_window.generated_geometry_by_document_id.copy()
    snapshot = main_window._generation_project_snapshot()
    document = snapshot.generator_documents[0]
    request = GenerationRequest(
        document.provider_id,
        document.id,
        document.name,
        document.source,
        geometry.output_dir,
        "bad",
        project_revision=project_revision(snapshot),
    )
    incomplete = replace(geometry, meshes=(geometry.meshes[0],))
    with pytest.raises(ValueError):
        main_window.accept_generation(complete_generation(request, GenerationResponse(request.request_id, incomplete)))
    assert main_window.project is project
    assert main_window.generated_geometry_by_document_id == previous


def test_gui_reopens_assembly_without_installed_plugin(main_window, assembly_case):
    accept(main_window, assembly_case)
    payload = main_window.project_workflow.project_payload()
    main_window.project_workflow._apply_project_payload(payload)
    assert len(main_window._mesh_config_dialog_entries()) == 2
    assert not main_window.imported_meshes
    assert len(main_window.generated_geometry_by_document_id["design"].meshes) == 2
    assert "test.assembly" in main_window.editor_tabs.currentWidget().toPlainText()
    assert len(PhysicalSystemCompiler().compile(main_window.project.physical_system).interfaces) == 1


def test_assembly_rename_collision_does_not_change_project(main_window, assembly_case, monkeypatch):
    from blab.project.model import ImportedMeshState

    accept(main_window, assembly_case)
    main_window.project.imported_meshes = (ImportedMeshState("renamed__fem", "other.msh"),)
    before = main_window.active_generator_document()
    errors = []
    monkeypatch.setattr(main_window, "show_error", lambda *args: errors.append(args))
    monkeypatch.setattr(QInputDialog, "getText", lambda *_a, **_kw: ("renamed", True))
    main_window.rename_active_generator_document()
    assert errors
    assert main_window.active_generator_document() == before
    assert {mesh.name for mesh in main_window.project.physical_system.meshes} == {"cabinet__fem", "cabinet__bem"}
