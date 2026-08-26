from __future__ import annotations

from pathlib import Path

import meshio
import numpy as np

from blab.physical_compiler import PhysicalSystemCompiler
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
)
from blab.speaker_symmetry import expand_speaker_system_for_export


def _write_fem(
    path: Path,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    moving_face=(0, 1, 2),
) -> None:
    points = np.asarray(
        [
            [offset_x + 0.0, offset_y + 0.0, 0.0],
            [offset_x + 1.0, offset_y + 0.0, 0.0],
            [offset_x + 0.0, offset_y + 1.0, 0.0],
            [offset_x + 0.0, offset_y + 0.0, 1.0],
        ]
    )
    faces = [(0, 2, 3), tuple(moving_face), (0, 1, 3), (1, 2, 3)]
    tags = np.asarray([4, 2, 3, 3], dtype=np.int32)
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("tetra", np.asarray([[0, 1, 2, 3]])), ("triangle", np.asarray(faces))],
            cell_data={
                "gmsh:physical": [np.asarray([1], dtype=np.int32), tags],
                "gmsh:geometrical": [np.asarray([1], dtype=np.int32), tags],
            },
            field_data={
                "air": np.asarray([1, 3]),
                "rear": np.asarray([2, 2]),
                "walls": np.asarray([3, 2]),
                "cut_x": np.asarray([4, 2]),
            },
        ),
        file_format="gmsh22",
        binary=False,
    )


def _write_bem(
    path: Path,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    moving_face=(0, 1, 2),
) -> None:
    points = np.asarray(
        [
            [offset_x + 0.0, offset_y + 0.0, 0.0],
            [offset_x + 1.0, offset_y + 0.0, 0.0],
            [offset_x + 0.0, offset_y + 1.0, 0.0],
            [offset_x + 0.0, offset_y + 0.0, 1.0],
        ]
    )
    faces = [tuple(moving_face), (0, 1, 3), (1, 2, 3)]
    tags = np.asarray([12, 13, 13], dtype=np.int32)
    meshio.write(
        path,
        meshio.Mesh(
            points,
            [("triangle", np.asarray(faces))],
            cell_data={
                "gmsh:physical": [tags],
                "gmsh:geometrical": [tags],
            },
            field_data={"front": np.asarray([12, 2]), "cabinet": np.asarray([13, 2])},
        ),
        file_format="gmsh22",
        binary=False,
    )


def _system(fem_path: Path, bem_path: Path, *, motion_axis=(0.0, 0.0, 1.0)) -> PhysicalSystem:
    fem_id = "mesh:fem"
    bem_id = "mesh:bem"
    bounded_id = "region:inside"
    exterior_id = "region:outside"
    boundaries = (
        Boundary(
            "boundary:rear",
            "Rear diaphragm",
            bounded_id,
            PhysicalGroupRef(fem_id, 2, "rear", 2),
            BoundaryKind.MOVING,
        ),
        Boundary(
            "boundary:walls",
            "Interior walls",
            bounded_id,
            PhysicalGroupRef(fem_id, 2, "walls", 3),
            BoundaryKind.RIGID,
        ),
        Boundary(
            "boundary:cut",
            "Symmetry cut",
            bounded_id,
            PhysicalGroupRef(fem_id, 2, "cut_x", 4),
            BoundaryKind.RIGID,
        ),
        Boundary(
            "boundary:front",
            "Front diaphragm",
            exterior_id,
            PhysicalGroupRef(bem_id, 2, "front", 12),
            BoundaryKind.MOVING,
        ),
        Boundary(
            "boundary:cabinet",
            "Cabinet",
            exterior_id,
            PhysicalGroupRef(bem_id, 2, "cabinet", 13),
            BoundaryKind.RIGID,
        ),
    )
    component = PhysicalComponent(
        "component:driver",
        "Driver",
        ComponentKind.ELECTRODYNAMIC_TRANSDUCER,
        ("boundary:rear", "boundary:front"),
        {
            "re_ohm": 6.0,
            "le_h": 0.0005,
            "bl_n_per_a": 7.0,
            "mmd_kg": 0.015,
            "cms_m_per_n": 0.0005,
            "rms_n_s_per_m": 1.0,
            "motion_axis": list(motion_axis),
            "motion_profile": "rigid_translation",
        },
    )
    return PhysicalSystem(
        id="system:test",
        name="Test",
        meshes=(
            MeshResource(fem_id, "FEM", str(fem_path), MeshPurpose.FEM_VOLUME),
            MeshResource(bem_id, "BEM", str(bem_path), MeshPurpose.BEM_SURFACE),
        ),
        regions=(
            AcousticRegion(
                bounded_id,
                "Inside",
                AcousticRegionKind.BOUNDED_AIR,
                (fem_id,),
                (PhysicalGroupRef(fem_id, 3, "air", 1),),
            ),
            AcousticRegion(exterior_id, "Outside", AcousticRegionKind.UNBOUNDED_AIR, (bem_id,)),
        ),
        boundaries=boundaries,
        components=(component,),
        excitation_ports=(ExcitationPort("port:driver", "Driver voltage", component.id, ExcitationPortKind.VOLTAGE),),
    )


def test_x_expansion_welds_cut_nodes_removes_cut_faces_and_recompiles(tmp_path: Path) -> None:
    fem_path = tmp_path / "fem.msh"
    bem_path = tmp_path / "bem.msh"
    _write_fem(fem_path)
    _write_bem(bem_path)

    expanded = expand_speaker_system_for_export(
        _system(fem_path, bem_path),
        symmetry="x",
        output_dir=tmp_path / "expanded",
    )

    assert len(expanded.system.components) == 1
    assert len(expanded.system.excitation_ports) == 1
    assert "boundary:cut" not in {boundary.id for boundary in expanded.system.boundaries}
    fem_resource = next(mesh for mesh in expanded.system.meshes if mesh.purpose == MeshPurpose.FEM_VOLUME)
    fem = meshio.read(fem_resource.file)
    tetrahedra = np.vstack([block.data for block in fem.cells if block.type == "tetra"])
    triangles = np.vstack([block.data for block in fem.cells if block.type == "triangle"])
    assert len(fem.points) == 5
    assert tetrahedra.shape == (2, 4)
    assert triangles.shape == (6, 3)
    for tetrahedron in tetrahedra:
        vertices = fem.points[tetrahedron]
        assert np.linalg.det(np.column_stack((vertices[1:] - vertices[0]))) > 0.0

    compiled = PhysicalSystemCompiler().compile(expanded.system, symmetry_mode="off")
    assert len(compiled.components) == 1
    assert compiled.components[0].parameters["surface_completion_factor"] == 1
    assert compiled.components[0].parameters["physical_driver_orbit_count"] == 1


def test_expansion_prefers_retained_full_domain_meshes(tmp_path: Path) -> None:
    fem_path = tmp_path / "fem.msh"
    bem_path = tmp_path / "bem.msh"
    _write_fem(fem_path)
    _write_bem(bem_path)
    system = _system(fem_path, bem_path)
    reflected = expand_speaker_system_for_export(
        system,
        symmetry="x",
        output_dir=tmp_path / "reflected",
    )
    retained = {resource.name: resource.file for resource in reflected.system.meshes}

    expanded = expand_speaker_system_for_export(
        system,
        symmetry="x",
        output_dir=tmp_path / "preferred",
        preferred_full_mesh_by_name=retained,
    )

    assert expanded.preferred_full_mesh_names == ("BEM", "FEM")
    compiled = PhysicalSystemCompiler().compile(expanded.system, symmetry_mode="off")
    assert len(compiled.components) == 1


def test_complete_off_axis_driver_gets_independent_components_and_ports(tmp_path: Path) -> None:
    fem_path = tmp_path / "fem-offset.msh"
    bem_path = tmp_path / "bem-offset.msh"
    _write_fem(fem_path, offset_x=2.0, moving_face=(1, 2, 3))
    _write_bem(bem_path, offset_x=2.0, moving_face=(1, 2, 3))

    expanded = expand_speaker_system_for_export(
        _system(fem_path, bem_path, motion_axis=(1.0, 0.0, 0.0)),
        symmetry="x",
        output_dir=tmp_path / "expanded-offset",
    )

    assert len(expanded.system.components) == 2
    assert len(expanded.system.excitation_ports) == 2
    assert set(expanded.component_source_ids.values()) == {"component:driver"}
    expansion_metadata = expanded.system.metadata["speaker_export_symmetry_expansion"]
    assert set(expansion_metadata["excitation_port_source_ids"].values()) == {"port:driver"}
    axes = {tuple(component.parameters["motion_axis"]) for component in expanded.system.components}
    assert axes == {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)}
    assert len({component.boundary_ids for component in expanded.system.components}) == 2
    assert all(component.parameters["physical_driver_orbit_count"] == 1 for component in expanded.system.components)

    compiled = PhysicalSystemCompiler().compile(expanded.system, symmetry_mode="off")
    assert len(compiled.components) == 2
    moving_tags = {
        boundary.group.tag for boundary in compiled.boundaries if boundary.kind == BoundaryKind.MOVING
    }
    assert len(moving_tags) == 4


def test_xy_expansion_creates_all_four_orientated_component_images(tmp_path: Path) -> None:
    fem_path = tmp_path / "fem-quarter.msh"
    bem_path = tmp_path / "bem-quarter.msh"
    _write_fem(fem_path, offset_x=2.0, offset_y=2.0, moving_face=(1, 2, 3))
    _write_bem(bem_path, offset_x=2.0, offset_y=2.0, moving_face=(1, 2, 3))

    expanded = expand_speaker_system_for_export(
        _system(fem_path, bem_path, motion_axis=(1.0, 1.0, 0.0)),
        symmetry="xy",
        output_dir=tmp_path / "expanded-quarter",
    )

    assert len(expanded.system.components) == 4
    assert len(expanded.system.excitation_ports) == 4
    assert {tuple(component.parameters["motion_axis"]) for component in expanded.system.components} == {
        (1.0, 1.0, 0.0),
        (-1.0, 1.0, 0.0),
        (1.0, -1.0, 0.0),
        (-1.0, -1.0, 0.0),
    }
    fem_resource = next(mesh for mesh in expanded.system.meshes if mesh.purpose == MeshPurpose.FEM_VOLUME)
    fem = meshio.read(fem_resource.file)
    tetrahedra = np.vstack([block.data for block in fem.cells if block.type == "tetra"])
    assert tetrahedra.shape == (4, 4)
    for tetrahedron in tetrahedra:
        vertices = fem.points[tetrahedron]
        assert np.linalg.det(np.column_stack((vertices[1:] - vertices[0]))) > 0.0

    compiled = PhysicalSystemCompiler().compile(expanded.system, symmetry_mode="off")
    assert len(compiled.components) == 4
