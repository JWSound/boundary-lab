from pathlib import Path

import meshio
import pytest

from blab.interface_conform import (
    InterfaceConformError,
    conform_bem_interface_to_fem,
    validate_conforming_interfaces,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FEM_FIXTURE = REPOSITORY_ROOT / "/tests/fixtures/femvolume.msh"
BEM_FIXTURE = REPOSITORY_ROOT / "/tests/fixtures/exterior.msh"
CONFORMING_BEM_FIXTURE = REPOSITORY_ROOT / "/tests/fixtures/exterior_conforming.msh"


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
