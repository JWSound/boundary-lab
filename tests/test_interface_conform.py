import copy
from pathlib import Path

import meshio
import numpy as np
import pytest

from blab.interface_conform import (
    InterfaceConformError,
    _best_fit_plane,
    _boundary_loops,
    _matching_discrete_paths,
    _physical_surface_tag,
    _triangle_data,
    conform_bem_interface_to_fem,
    validate_conforming_interfaces,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEM_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "femvolume.msh"
BEM_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "exterior.msh"
CONFORMING_BEM_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "exterior_conforming.msh"
SAWMOD_FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "SAWMOD"
CURVED_FEM_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "curvedinterfaceFEM.msh"
CURVED_BEM_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "curvedinterfaceBEM.msh"
NONPLANAR_MULTISURFACE_FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "nonplanar_multisurface_interface"


def test_direct_seam_match_requires_identical_edge_connectivity() -> None:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        )
    )

    reversed_match, reversed_deviation = _matching_discrete_paths(
        points,
        np.asarray((0, 1, 2, 3)),
        points,
        np.asarray((0, 3, 2, 1)),
        tolerance=1e-8,
        closed=True,
    )
    mismatched_edges, _deviation = _matching_discrete_paths(
        points,
        np.asarray((0, 1, 2, 3)),
        points,
        np.asarray((0, 2, 1, 3)),
        tolerance=1e-8,
        closed=True,
    )

    assert reversed_match
    assert reversed_deviation == 0.0
    assert not mismatched_edges


def test_fixture_interfaces_are_made_connectivity_identical_and_watertight() -> None:
    fem_mesh = meshio.read(FEM_FIXTURE)
    bem_mesh = meshio.read(BEM_FIXTURE)

    conformed_mesh, result = conform_bem_interface_to_fem(fem_mesh, bem_mesh)

    assert result.fem_interface_triangles == 180
    assert result.original_bem_interface_triangles == 196
    assert result.original_adjacent_triangles == 670
    assert result.remeshed_adjacent_triangles > 0
    assert result.identity.interface_triangles == 180
    assert result.identity.interface_vertices == 106
    assert result.identity.max_coordinate_error <= 1e-9
    assert result.identity.fem_facets_on_tetra_boundary == 180
    assert result.identity.bem_boundary_edges == 0
    assert conformed_mesh.field_data.keys() == bem_mesh.field_data.keys()


def test_original_fixture_interfaces_are_rejected_as_nonconforming() -> None:
    fem_mesh = meshio.read(FEM_FIXTURE)
    bem_mesh = meshio.read(BEM_FIXTURE)

    with pytest.raises(InterfaceConformError, match="Interface triangle counts differ"):
        validate_conforming_interfaces(fem_mesh, bem_mesh)


def test_generated_fixture_round_trips_as_a_conforming_gmsh_mesh() -> None:
    fem_mesh = meshio.read(FEM_FIXTURE)
    bem_mesh = meshio.read(CONFORMING_BEM_FIXTURE)

    report = validate_conforming_interfaces(fem_mesh, bem_mesh)

    assert report.interface_triangles == 180
    assert report.interface_vertices == 106
    assert report.max_coordinate_error <= 1e-9
    assert report.bem_boundary_edges == 0


def test_closed_interface_away_from_global_symmetry_planes_is_conformed_as_closed() -> None:
    fem_mesh = copy.deepcopy(meshio.read(FEM_FIXTURE))
    bem_mesh = copy.deepcopy(meshio.read(BEM_FIXTURE))
    fem_mesh.points[:, :2] += 200.0
    bem_mesh.points[:, :2] += 200.0

    conformed_mesh, result = conform_bem_interface_to_fem(
        fem_mesh,
        bem_mesh,
        symmetry_mode="xy",
    )
    report = validate_conforming_interfaces(
        fem_mesh,
        conformed_mesh,
        symmetry_mode="xy",
    )

    assert result.fem_interface_triangles == 180
    assert result.original_bem_interface_triangles == 196
    assert result.remeshed_adjacent_triangles > 0
    assert report.interface_triangles == 180
    assert report.interface_vertices == 106
    assert report.max_coordinate_error <= 1e-9
    assert report.bem_boundary_edges == 0


def test_curved_interface_with_disconnected_perimeter_and_split_geometry_is_conformed() -> None:
    fem_mesh = copy.deepcopy(meshio.read(FEM_FIXTURE))
    bem_mesh = copy.deepcopy(meshio.read(BEM_FIXTURE))

    fem_data = _triangle_data(fem_mesh, require_geometrical=False)
    fem_tag = _physical_surface_tag(fem_mesh, "Interface")
    fem_interface = fem_data.triangles[fem_data.physical_tags == fem_tag]
    fem_loop = _boundary_loops(fem_interface)[0]
    fem_vertices = np.unique(fem_interface)
    fem_interior = np.setdiff1d(fem_vertices, fem_loop)
    _origin, fem_plane_normal = _best_fit_plane(fem_mesh.points[fem_loop])
    fem_mesh.points[fem_interior] += fem_plane_normal * 1.0

    bem_data = _triangle_data(bem_mesh, require_geometrical=True)
    bem_tag = _physical_surface_tag(bem_mesh, "Interface")
    bem_interface_mask = bem_data.physical_tags == bem_tag
    bem_interface = bem_data.triangles[bem_interface_mask]
    bem_loop = _boundary_loops(bem_interface)[0]

    duplicated_start = len(bem_mesh.points)
    duplicated_indices = np.arange(duplicated_start, duplicated_start + len(bem_loop))
    duplicate_map = dict(zip(map(int, bem_loop), map(int, duplicated_indices), strict=True))
    duplicated_points = bem_mesh.points[bem_loop].copy()
    duplicated_points[:, 0] += 0.01
    bem_mesh.points = np.vstack((bem_mesh.points, duplicated_points))

    triangles = bem_mesh.cells[0].data.copy()
    for triangle_index in np.flatnonzero(bem_interface_mask):
        triangles[triangle_index] = [
            duplicate_map.get(int(vertex), int(vertex)) for vertex in triangles[triangle_index]
        ]
    bem_mesh.cells[0].data = triangles

    geometrical = bem_mesh.cell_data["gmsh:geometrical"][0].copy()
    exterior_indices = np.flatnonzero(~bem_interface_mask)
    geometrical[exterior_indices] += np.arange(len(exterior_indices), dtype=geometrical.dtype) % 2
    bem_mesh.cell_data["gmsh:geometrical"][0] = geometrical

    conformed_mesh, result = conform_bem_interface_to_fem(fem_mesh, bem_mesh)
    identity = validate_conforming_interfaces(fem_mesh, conformed_mesh)

    assert result.identity.bem_boundary_edges == 0
    assert identity.interface_triangles == len(fem_interface)
    assert identity.max_coordinate_error <= 1e-9

    conformed_data = _triangle_data(conformed_mesh, require_geometrical=True)
    conformed_interface = conformed_data.triangles[conformed_data.physical_tags == bem_tag]
    conformed_loop = _boundary_loops(conformed_interface)[0]
    plane_origin, plane_normal = _best_fit_plane(conformed_mesh.points[conformed_loop])
    interface_deviation = np.max(
        np.abs((conformed_mesh.points[np.unique(conformed_interface)] - plane_origin) @ plane_normal)
    )
    assert interface_deviation > 0.5


def test_nonplanar_curved_interface_with_matching_discrete_seam_is_replaced_directly(
    tmp_path: Path,
) -> None:
    fem_mesh = meshio.read(CURVED_FEM_FIXTURE)
    bem_mesh = meshio.read(CURVED_BEM_FIXTURE)

    conformed_mesh, result = conform_bem_interface_to_fem(
        fem_mesh,
        bem_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name="INTERFACE",
        symmetry_mode="xy",
    )
    identity = validate_conforming_interfaces(
        fem_mesh,
        conformed_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name="INTERFACE",
        symmetry_mode="xy",
    )

    assert result.fem_interface_triangles == 2506
    assert result.original_bem_interface_triangles == 2702
    assert result.original_adjacent_triangles == 0
    assert result.remeshed_adjacent_triangles == 0
    assert result.max_original_boundary_deviation <= 1e-8
    assert identity.interface_triangles == 2506
    assert identity.interface_vertices == 1318
    assert identity.max_coordinate_error <= 1e-8
    assert identity.fem_facets_on_tetra_boundary == 2506
    assert identity.bem_boundary_edges == 141

    output_path = tmp_path / "curvedinterface_conforming.msh"
    meshio.write(
        output_path,
        conformed_mesh,
        file_format="gmsh22",
        binary=False,
    )
    round_trip_identity = validate_conforming_interfaces(
        fem_mesh,
        meshio.read(output_path),
        fem_interface_name="INTERFACE",
        bem_interface_name="INTERFACE",
        symmetry_mode="xy",
    )
    assert round_trip_identity == identity


def test_nonplanar_multisurface_interface_with_near_matching_seam_is_welded_directly() -> None:
    fem_mesh = meshio.read(NONPLANAR_MULTISURFACE_FIXTURE_ROOT / "Throat2.msh")
    bem_mesh = meshio.read(NONPLANAR_MULTISURFACE_FIXTURE_ROOT / "Exterior2.msh")
    fem_mesh.points = np.asarray(fem_mesh.points, dtype=float) * 0.001
    bem_mesh.points = np.asarray(bem_mesh.points, dtype=float) * 0.001

    bem_data = _triangle_data(bem_mesh, require_geometrical=True)
    bem_tag = _physical_surface_tag(bem_mesh, "INTERFACE")
    bem_interface_mask = bem_data.physical_tags == bem_tag
    bem_interface = bem_data.triangles[bem_interface_mask]
    assert len(np.unique(bem_data.geometrical_tags[bem_interface_mask])) == 2

    bem_loop = _boundary_loops(bem_interface)[0]
    plane_origin, plane_normal = _best_fit_plane(bem_mesh.points[bem_loop])
    interface_plane_deviation = float(
        np.max(np.abs((bem_mesh.points[np.unique(bem_interface)] - plane_origin) @ plane_normal))
    )
    assert interface_plane_deviation > 1e-3

    phase = np.linspace(0.0, 2.0 * np.pi, len(bem_loop), endpoint=False)
    seam_offsets = 3e-6 + 2e-6 * np.sin(phase)
    bem_mesh.points[bem_loop, 2] += seam_offsets

    conformed_mesh, result = conform_bem_interface_to_fem(
        fem_mesh,
        bem_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name="INTERFACE",
        symmetry_mode="xy",
    )
    identity = validate_conforming_interfaces(
        fem_mesh,
        conformed_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name="INTERFACE",
        symmetry_mode="xy",
    )

    assert result.fem_interface_triangles == 5883
    assert result.original_bem_interface_triangles == 6507
    assert result.original_adjacent_triangles == 0
    assert result.remeshed_adjacent_triangles == 0
    assert 1e-6 < result.max_original_boundary_deviation <= 5.1e-6
    assert result.output_triangles == 7937
    assert identity.interface_triangles == 5883
    assert identity.interface_vertices == 3040
    assert identity.max_coordinate_error <= 1e-9
    assert identity.fem_facets_on_tetra_boundary == 5883
    assert identity.bem_boundary_edges == 173


@pytest.mark.parametrize(
    ("symmetry_mode", "suffix", "bem_interface_name", "expected_triangles"),
    (
        ("x", "reduced_x", "BEMInterface (1)", 522),
        ("xy", "reduced_xy", "BEMInterface (1) (1)", 277),
    ),
)
def test_reduced_sawmod_interfaces_are_conformed_with_open_edges_only_on_symmetry_planes(
    symmetry_mode: str,
    suffix: str,
    bem_interface_name: str,
    expected_triangles: int,
) -> None:
    fem_mesh = meshio.read(SAWMOD_FIXTURE_ROOT / f"SMfemvolume_{suffix}.msh")
    bem_mesh = meshio.read(SAWMOD_FIXTURE_ROOT / f"SMexterior_{suffix}.msh")

    conformed_mesh, result = conform_bem_interface_to_fem(
        fem_mesh,
        bem_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name=bem_interface_name,
        symmetry_mode=symmetry_mode,
    )
    report = validate_conforming_interfaces(
        fem_mesh,
        conformed_mesh,
        fem_interface_name="INTERFACE",
        bem_interface_name=bem_interface_name,
        symmetry_mode=symmetry_mode,
    )

    assert result.fem_interface_triangles == expected_triangles
    assert report.interface_triangles == expected_triangles
    assert report.max_coordinate_error <= 1e-9
    assert report.bem_boundary_edges > 0
    with pytest.raises(InterfaceConformError, match="away from the active symmetry planes"):
        validate_conforming_interfaces(
            fem_mesh,
            conformed_mesh,
            fem_interface_name="INTERFACE",
            bem_interface_name=bem_interface_name,
        )
