"""Exterior axial sources: authoring, contract, geometry and optional engine parity."""

import os
from dataclasses import replace

import numpy as np
import pytest

from blab.acoustic_impedance import normalization_records
from blab.mesh_data import MeshData
from blab.physical_compiler import PhysicalModelCompileError, PhysicalSystemCompiler
from blab.physical_model import (
    AcousticRegion,
    AcousticRegionKind,
    Boundary,
    BoundaryKind,
    ComponentKind,
    ExcitationPort,
    ExcitationPortKind,
    MeshPurpose,
    MeshResource,
    PhysicalComponent,
    PhysicalGroupRef,
    PhysicalSystem,
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.solvers.coupled_backend import CoupledProductionBackend
from blab.source_motion import prescribed_source_parameters
from blab.system_contract import (
    OutputRequest,
    SystemSolveRequest,
    compiled_system_from_dict,
    compiled_system_to_dict,
)
from blab.system_solve import supports_exterior_system_protocol


def source_system(parameters, *, flat=False):
    mesh = MeshData(
        points=[[0, 0, 0 if flat else 0.05], [0.1, 0, 0], [0, 0.1, 0], [-0.1, 0, 0], [0, -0.1, 0], [0, 0, -0.05]],
        cells=[("triangle", [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]])],
        physical_tags=[[1, 1, 1, 1, 2, 2, 2, 2]],
        physical_names={"Drive": (1, 2), "Wall": (2, 2)},
    )
    return PhysicalSystem(
        id="system",
        name="Axial fixture",
        meshes=(MeshResource(id="mesh", name="Mesh", file="", purpose=MeshPurpose.BEM_SURFACE, mesh_data=mesh),),
        regions=(AcousticRegion(id="air", name="Air", kind=AcousticRegionKind.UNBOUNDED_AIR, mesh_ids=("mesh",)),),
        boundaries=tuple(
            Boundary(
                id=name,
                name=name,
                region_id="air",
                kind=kind,
                group=PhysicalGroupRef(mesh_id="mesh", dimension=2, name=name),
            )
            for name, kind in (("Drive", BoundaryKind.MOVING), ("Wall", BoundaryKind.RIGID))
        ),
        components=(
            PhysicalComponent(
                id="source",
                name="Source",
                kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
                boundary_ids=("Drive",),
                parameters=parameters,
            ),
        ),
        excitation_ports=(
            ExcitationPort(id="input", name="Input", component_id="source", kind=ExcitationPortKind.NORMAL_VELOCITY),
        ),
    )


def axial(axis=(0, 0, 1), weight=1):
    return {
        "motion_profile": "rigid_translation",
        "motion_axis": list(axis),
        "boundary_motion_weights": {"Drive": weight},
    }


def test_axial_compilation_roundtrip_area_and_backend():
    system = source_system(axial((0, 0, 10), 2))
    system = physical_system_from_dict(physical_system_to_dict(system))
    compiled = PhysicalSystemCompiler().compile(system)
    assert compiled.contract_version == 2
    assert compiled.components[0].parameters["motion_axis"] == [0, 0, 1]
    wire = compiled_system_to_dict(compiled)
    assert compiled_system_to_dict(compiled_system_from_dict(wire)) == wire
    record = normalization_records(compiled.metadata)["source"]
    assert record.effective_area_m2 == pytest.approx(0.04)
    assert record.area_kind == "weighted_projected_surface"
    assert supports_exterior_system_protocol(system, backend_id="beat_cpu", stitch_exterior_meshes=False)
    request = SystemSolveRequest(compiled_system=compiled, frequencies_hz=(500,), excitation_port_ids=("input",))
    CoupledProductionBackend(bem_backend="cpu").create_system_session(request)


def test_legacy_and_tangential_source_compilation():
    normal = PhysicalSystemCompiler().compile(source_system({"motion_profile": "uniform"}))
    assert normal.contract_version == 1
    assert normal.components[0].parameters == {}
    assert normalization_records(normal.metadata)["source"].effective_area_m2 == pytest.approx(0.02 * np.sqrt(1.5))
    tangential = PhysicalSystemCompiler().compile(source_system(axial((1, 0, 0)), flat=True))
    assert tangential.contract_version == 2
    assert normalization_records(tangential.metadata) == {}


@pytest.mark.parametrize("axis", [[0, 0, 0], [1, 2], [True, 0, 1], [float("inf"), 0, 1], None])
def test_invalid_source_axes(axis):
    with pytest.raises(ValueError, match="motion_axis"):
        prescribed_source_parameters({"motion_profile": "rigid_translation", "motion_axis": axis})


def test_source_symmetry_and_interior_restrictions():
    for symmetry in ("x", "xy"):
        with pytest.raises(PhysicalModelCompileError, match="symmetry planes"):
            PhysicalSystemCompiler().compile(source_system(axial((1, 0, 1))), symmetry_mode=symmetry)
        assert prescribed_source_parameters(axial((0, 0, 1e300)), symmetry=symmetry)["motion_axis"] == [0, 0, 1]
    with pytest.raises(ValueError, match="exterior-only"):
        prescribed_source_parameters(axial(), exterior=False)
    with pytest.raises(ValueError, match="requires motion_profile"):
        prescribed_source_parameters({"motion_axis": [0, 0, 1]})


def test_editor_source_modes_and_flip():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from blab.ui.component_editor import _ComponentDraft, _ComponentEditorDialog

    app = QApplication.instance() or QApplication([])
    system = source_system({"motion_profile": "uniform"})
    draft = _ComponentDraft(
        id="source",
        name="Source",
        kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
        boundary_ids=("Drive",),
        channel="main",
        parameters={"motion_profile": "uniform"},
        motion_axis_mode="automatic",
    )
    options = dict(
        boundaries=system.boundaries[:1],
        resources_by_id={"mesh": system.meshes[0]},
        region_names={"air": "Air"},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="off",
        mesh_cache={},
    )
    legacy = _ComponentEditorDialog(draft, **options)
    assert legacy.component_draft().parameters["motion_profile"] == "uniform"
    assert legacy.motion_group.isHidden()
    assert not legacy.source_motion_note.isHidden()
    editor = _ComponentEditorDialog(replace(draft, id="", parameters={}, motion_axis_mode="manual"), **options)
    assert not hasattr(editor, "motion_profile_combo")
    assert not editor.motion_group.isHidden()
    automatic = editor.component_draft()
    assert automatic.parameters["motion_axis"] == pytest.approx([0, 0, 1])
    assert automatic.motion_axis_mode == "automatic"
    editor._flip_axis()
    assert editor.component_draft().parameters["motion_axis"] == pytest.approx([0, 0, -1])
    editor.axis_mode_combo.setCurrentIndex(editor.axis_mode_combo.findData("manual"))
    for spin, value in zip(editor.axis_spins, (1, 0, 1)):
        spin.setValue(value)
    manual = editor.component_draft()
    assert manual.parameters["motion_axis"] == pytest.approx([2**-0.5, 0, 2**-0.5])
    reopened = _ComponentEditorDialog(manual, **options)
    assert reopened.component_draft().parameters == manual.parameters
    bounded = _ComponentEditorDialog(replace(draft, id="", parameters={}), exterior_only=False, **options)
    assert bounded.component_draft().parameters["motion_profile"] == "uniform"
    assert bounded.motion_group.isHidden()
    assert app is not None
    for dialog in (editor, reopened, bounded, legacy):
        dialog.close()


def test_editor_rejects_ambiguous_automatic_axis():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from blab.ui.component_editor import _ComponentDraft, _ComponentEditorDialog

    app = QApplication.instance() or QApplication([])
    system = source_system(axial())
    mesh = system.meshes[0].mesh_data.to_meshio()
    # Equal normal energy along X, Y and Z gives no preferred axis.
    mesh.points[0, 2] = 0.1
    resource = replace(system.meshes[0], mesh_data=MeshData.from_meshio(mesh))
    draft = _ComponentDraft(
        id="source",
        name="Source",
        kind=ComponentKind.IDEAL_VELOCITY_SOURCE,
        boundary_ids=("Drive",),
        channel="main",
        parameters=axial(),
        motion_axis_mode="automatic",
    )
    editor = _ComponentEditorDialog(
        draft,
        boundaries=system.boundaries[:1],
        resources_by_id={"mesh": resource},
        region_names={"air": "Air"},
        channel_names=("main",),
        unavailable_boundary_ids=set(),
        symmetry_mode="off",
        mesh_cache={},
    )
    with pytest.raises(ValueError, match="confidence is low"):
        editor.component_draft()
    editor.axis_mode_combo.setCurrentIndex(editor.axis_mode_combo.findData("manual"))
    assert editor.component_draft().parameters["motion_profile"] == "rigid_translation"
    assert app is not None
    editor.close()


@pytest.mark.skipif(
    os.environ.get("BLAB_RUN_EXTERIOR_CPU") != "1", reason="Set BLAB_RUN_EXTERIOR_CPU=1 for Julia parity"
)
@pytest.mark.parametrize("flat", [False, True])
def test_engine_axial_complex_pressure_and_neumann_parity(flat):
    backend = CoupledProductionBackend(bem_backend="cpu")

    def solve(parameters):
        compiled = PhysicalSystemCompiler().compile(source_system(parameters, flat=flat))
        request = SystemSolveRequest(
            compiled_system=compiled,
            frequencies_hz=(500,),
            excitation_port_ids=("input",),
            outputs=(
                OutputRequest(id="q", quantity="bem_boundary_neumann"),
                OutputRequest(id="p", quantity="exterior_pressure", options={"points_m": [[0, 0, 1]]}),
                OutputRequest(id="z", quantity="radiation_impedance"),
            ),
            solver_options={"quadrature_order": 2, "singular_order": 2},
        )
        (result,) = backend.create_system_session(request).solve_stream()
        return {q.id: q.values for q in result.quantities}

    normal = solve({})
    forward = solve(axial((0, 0, 10)))
    reverse = solve(axial((0, 0, -1)))
    factor = 1 if flat else 1 / np.sqrt(1.5)
    for quantity in ("p", "q"):
        np.testing.assert_allclose(forward[quantity], factor * normal[quantity], rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(reverse[quantity], -forward[quantity], rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(forward["z"], factor**2 * normal["z"], rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(reverse["z"], forward["z"], rtol=2e-5, atol=1e-6)
    if flat:
        tangent = solve(axial((1, 0, 0)))
        for values in tangent.values():
            np.testing.assert_allclose(values, 0, atol=1e-7)
