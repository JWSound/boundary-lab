from pathlib import Path

import meshio
import numpy as np
import pytest

import blab.component_symmetry as component_symmetry_module
from blab.component_symmetry import (
    ComponentSymmetryInference,
    ComponentSymmetryInferenceError,
    ProjectedAreaGeometryCache,
    infer_component_symmetry,
    infer_projected_diaphragm_area,
    infer_weighted_surface_area,
)
from blab.physical_model import (
    Boundary,
    BoundaryKind,
    MeshPurpose,
    MeshResource,
    PhysicalGroupRef,
)
from repo_paths import REPO_ROOT

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
SKRAM_EXAMPLE_ROOT = REPO_ROOT / "examples" / "SKRAM"


@pytest.mark.parametrize(
    ("mode", "cut_axes", "component_count", "expected"),
    (
        (
            "x",
            ("x",),
            1,
            "Moving surface(s) sliced along the x axis. Detected 1 distinct component in the fully mirrored system.",
        ),
        (
            "xy",
            ("x", "y"),
            1,
            "Moving surface(s) sliced along the x and y axes. "
            "Detected 1 distinct component in the fully mirrored system.",
        ),
        (
            "xy",
            ("y",),
            2,
            "Moving surface(s) sliced along the y axis. Detected 2 distinct components in the fully mirrored system.",
        ),
        (
            "xy",
            (),
            4,
            "Moving surface(s) not sliced along the x and y axes. "
            "Detected 4 distinct components in the fully mirrored system.",
        ),
    ),
)
def test_component_symmetry_summary_describes_slices_and_fully_mirrored_count(
    mode: str,
    cut_axes: tuple[str, ...],
    component_count: int,
    expected: str,
) -> None:
    inference = ComponentSymmetryInference(
        symmetry_mode=mode,
        fractional_symmetry_axes=cut_axes,
        surface_completion_factor=2 ** len(cut_axes),
        physical_driver_orbit_count=component_count,
        surface_patch_count=1,
        perimeter_edge_count=1,
        plane_edge_counts=(),
    )

    assert inference.summary() == expected


def _resource(path: Path, resource_id: str = "mesh") -> MeshResource:
    return MeshResource(
        id=resource_id,
        name=path.stem,
        file=str(path),
        purpose=MeshPurpose.FEM_VOLUME,
        scale_to_m=0.001,
    )


def _boundary(resource: MeshResource, group_name: str, boundary_id: str | None = None) -> Boundary:
    return Boundary(
        id=boundary_id or f"boundary:{group_name.lower()}",
        name=group_name,
        region_id="region:interior",
        group=PhysicalGroupRef(resource.id, 2, name=group_name),
        kind=BoundaryKind.MOVING,
    )


@pytest.mark.parametrize(
    ("mesh_name", "group_name", "expected_axes", "completion", "orbit_count"),
    (
        ("SMfemvolume_reduced_x.msh", "HF", ("x",), 2, 1),
        ("SMfemvolume_reduced_x.msh", "MF", (), 1, 2),
        ("SMfemvolume_reduced_xy.msh", "HF", ("x", "y"), 4, 1),
        ("SMfemvolume_reduced_xy.msh", "MF", ("y",), 2, 2),
    ),
)
def test_sawmod_component_symmetry_is_inferred_from_surface_perimeter(
    mesh_name: str,
    group_name: str,
    expected_axes: tuple[str, ...],
    completion: int,
    orbit_count: int,
) -> None:
    mode = "x" if "_x." in mesh_name else "xy"
    resource = _resource(FIXTURE_ROOT / "SAWMOD" / mesh_name)

    inferred = infer_component_symmetry(
        (_boundary(resource, group_name),),
        {resource.id: resource},
        mode,
    )

    assert inferred.fractional_symmetry_axes == expected_axes
    assert inferred.surface_completion_factor == completion
    assert inferred.physical_driver_orbit_count == orbit_count


def test_front_and_rear_driver_surfaces_infer_one_shared_x_cut_driver() -> None:
    front = _resource(SKRAM_EXAMPLE_ROOT / "SkramFrontChamber.msh", "mesh:front")
    rear = _resource(SKRAM_EXAMPLE_ROOT / "SkramRearChamber.msh", "mesh:rear")

    inferred = infer_component_symmetry(
        (
            _boundary(front, "Diaphragm", "boundary:front"),
            _boundary(rear, "Diaphragm", "boundary:rear"),
        ),
        {front.id: front, rear.id: rear},
        "x",
    )

    assert inferred.fractional_symmetry_axes == ("x",)
    assert inferred.surface_completion_factor == 2
    assert inferred.physical_driver_orbit_count == 1
    assert inferred.surface_patch_count == 2


def test_projected_diaphragm_area_averages_opposing_sides_and_reports_mismatch() -> None:
    front_resource = MeshResource("mesh:front", "Front", "unused-front.msh", MeshPurpose.FEM_VOLUME)
    rear_resource = MeshResource("mesh:rear", "Rear", "unused-rear.msh", MeshPurpose.FEM_VOLUME)
    front_mesh = meshio.Mesh(
        points=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        cells=[("triangle", np.asarray(((0, 1, 2),), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((1,), dtype=np.int32)]},
        field_data={"Front": np.asarray((1, 2))},
    )
    rear_mesh = meshio.Mesh(
        points=np.asarray(((0.0, 0.0, 0.1), (0.0, 0.8, 0.1), (0.8, 0.0, 0.1))),
        cells=[("triangle", np.asarray(((0, 1, 2),), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((1,), dtype=np.int32)]},
        field_data={"Rear": np.asarray((1, 2))},
    )

    inferred = infer_projected_diaphragm_area(
        (
            _boundary(front_resource, "Front", "boundary:front"),
            _boundary(rear_resource, "Rear", "boundary:rear"),
        ),
        {front_resource.id: front_resource, rear_resource.id: rear_resource},
        (0.0, 0.0, 1.0),
        1,
        mesh_cache={front_resource.id: front_mesh, rear_resource.id: rear_mesh},
    )

    assert inferred.positive_side_area_m2 == pytest.approx(0.5)
    assert inferred.negative_side_area_m2 == pytest.approx(0.32)
    assert inferred.projected_area_m2 == pytest.approx(0.41)
    assert inferred.relative_side_mismatch == pytest.approx(0.36)


def test_projected_diaphragm_area_supports_a_front_only_model() -> None:
    resource = MeshResource("mesh:front", "Front", "unused-front.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(((0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.0, 0.1, 0.0))),
        cells=[("triangle", np.asarray(((0, 1, 2),), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((1,), dtype=np.int32)]},
        field_data={"Front": np.asarray((1, 2))},
    )

    inferred = infer_projected_diaphragm_area(
        (_boundary(resource, "Front", "boundary:front"),),
        {resource.id: resource},
        (0.0, 0.0, 1.0),
        2,
        mesh_cache={resource.id: mesh},
    )

    assert inferred.projected_area_m2 == pytest.approx(0.01)
    assert not inferred.has_opposing_sides
    assert inferred.relative_side_mismatch is None


def test_quadratic_moving_surface_supports_symmetry_and_outward_projected_area() -> None:
    resource = MeshResource("mesh:p2", "P2", "unused.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
             (0.5, 0, 0), (0.5, 0.5, 0), (0, 0.5, 0),
             (0, 0, 0.5), (0.5, 0, 0.5), (0, 0.5, 0.5)), dtype=float,
        ),
        cells=[
            ("triangle6", np.asarray(((0, 1, 2, 4, 5, 6),))),
            ("tetra10", np.asarray((tuple(range(10)),))),
        ],
        cell_data={"gmsh:physical": [np.asarray((1,)), np.asarray((2,))]},
        field_data={"Front": np.asarray((1, 2)), "Air": np.asarray((2, 3))},
    )
    boundary = _boundary(resource, "Front")
    cache = {resource.id: mesh}
    symmetry = infer_component_symmetry((boundary,), {resource.id: resource}, "xy", mesh_cache=cache)
    assert symmetry.surface_completion_factor == 4
    area = infer_projected_diaphragm_area(
        (boundary,), {resource.id: resource}, (0, 0, 1), symmetry.surface_completion_factor,
        boundary_motion_weights={boundary.id: 0.5},
        boundary_side_keys={boundary.id: boundary.region_id}, mesh_cache=cache,
    )
    assert area.projected_area_m2 == pytest.approx(1.0)
    assert area.negative_side_area_m2 == pytest.approx(1.0)
    assert area.positive_side_area_m2 == 0.0
    assert area.relative_side_mismatch is None


def test_projected_diaphragm_area_reuses_tetrahedron_orientation_geometry(monkeypatch) -> None:
    resource = MeshResource("mesh:fem", "FEM", "unused.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
        cells=[
            ("triangle", np.asarray(((0, 2, 1),), dtype=np.int64)),
            ("tetra", np.asarray(((0, 1, 2, 3),), dtype=np.int64)),
        ],
        cell_data={"gmsh:physical": [np.asarray((1,), dtype=np.int32), np.asarray((2,), dtype=np.int32)]},
        field_data={"Front": np.asarray((1, 2)), "Volume": np.asarray((2, 3))},
    )
    build_calls = 0
    original = component_symmetry_module._tetrahedron_opposite_vertex_by_face

    def count_builds(value):
        nonlocal build_calls
        build_calls += 1
        return original(value)

    monkeypatch.setattr(component_symmetry_module, "_tetrahedron_opposite_vertex_by_face", count_builds)
    geometry_cache = ProjectedAreaGeometryCache()
    boundary = _boundary(resource, "Front")
    for weight in (1.0, 0.5):
        infer_projected_diaphragm_area(
            (boundary,),
            {resource.id: resource},
            (0.0, 0.0, 1.0),
            1,
            boundary_motion_weights={boundary.id: weight},
            mesh_cache={resource.id: mesh},
            projected_geometry_cache=geometry_cache,
        )

    assert build_calls == 1
    assert len(geometry_cache.surface_geometry_by_resource_tag) == 1


@pytest.mark.parametrize("axis_sign", (1.0, -1.0))
@pytest.mark.parametrize("rear", (False, True))
def test_projected_area_uses_net_displacement_per_region_for_folded_faces(axis_sign, rear) -> None:
    resource = MeshResource("mesh:folded", "Folded", "unused.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(
            ((0, 0, 0), (2, 0, 0), (0, 1, 0),
             (0, 0, 1), (0, 0.5, 1), (1, 0, 1),
             (0, 0, 2), (0, 1, 2), (2, 0, 2)), dtype=float,
        ),
        cells=[("triangle", np.asarray(((0, 1, 2), (3, 4, 5), (6, 7, 8))))],
        cell_data={"gmsh:physical": [np.asarray((1, 2, 3))]},
        field_data={"Dome": np.asarray((1, 2)), "Return": np.asarray((2, 2)), "Rear": np.asarray((3, 2))},
    )
    dome = _boundary(resource, "Dome", "boundary:dome")
    folded = _boundary(resource, "Return", "boundary:return")
    back = _boundary(resource, "Rear", "boundary:rear")
    boundaries = (dome, folded, back) if rear else (dome, folded)
    inferred = infer_projected_diaphragm_area(
        boundaries, {resource.id: resource}, (0, 0, axis_sign), 4,
        boundary_motion_weights={folded.id: 0.5},
        boundary_side_keys={dome.id: "front", folded.id: "front", back.id: "rear"},
        mesh_cache={resource.id: mesh},
    )
    # Net front displacement: 4 * (1 - 0.5 * 0.25) = 3.5 m².
    assert inferred.projected_area_m2 == pytest.approx(3.75 if rear else 3.5)
    if rear:
        assert inferred.relative_side_mismatch == pytest.approx(0.125)
    else:
        assert inferred.relative_side_mismatch is None
        assert not inferred.has_opposing_sides


def test_weighted_surface_area_applies_motion_weights_and_symmetry_completion() -> None:
    resource = MeshResource("mesh", "Mesh", "unused.msh", MeshPurpose.BEM_SURFACE)
    mesh = meshio.Mesh(
        points=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 2.0, 0.0),
            )
        ),
        cells=[("triangle", np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((1, 2), dtype=np.int32)]},
        field_data={"First": np.asarray((1, 2)), "Second": np.asarray((2, 2))},
    )
    first = _boundary(resource, "First", "boundary:first")
    second = _boundary(resource, "Second", "boundary:second")

    area = infer_weighted_surface_area(
        (first, second),
        {resource.id: resource},
        2,
        boundary_motion_weights={first.id: 0.5, second.id: 0.25},
        mesh_cache={resource.id: mesh},
    )

    assert area == pytest.approx(1.0)


def test_adjacent_surface_groups_are_unioned_before_perimeter_classification() -> None:
    resource = MeshResource("mesh", "mesh", "unused.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))),
        cells=[("triangle", np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((2, 3), dtype=np.int32)]},
        field_data={"Dome": np.asarray((2, 2)), "Surround": np.asarray((3, 2))},
    )

    inferred = infer_component_symmetry(
        (_boundary(resource, "Dome"), _boundary(resource, "Surround")),
        {resource.id: resource},
        "xy",
        mesh_cache={resource.id: mesh},
    )

    assert inferred.fractional_symmetry_axes == ("x", "y")
    assert inferred.surface_patch_count == 1


def test_inconsistent_disconnected_surface_patches_are_rejected() -> None:
    resource = MeshResource("mesh", "mesh", "unused.msh", MeshPurpose.FEM_VOLUME)
    mesh = meshio.Mesh(
        points=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (2.0, 0.0, 0.0),
                (3.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
            )
        ),
        cells=[("triangle", np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int64))],
        cell_data={"gmsh:physical": [np.asarray((2, 2), dtype=np.int32)]},
        field_data={"Radiator": np.asarray((2, 2))},
    )

    with pytest.raises(ComponentSymmetryInferenceError, match="inconsistent symmetry cuts"):
        infer_component_symmetry(
            (_boundary(resource, "Radiator"),),
            {resource.id: resource},
            "x",
            mesh_cache={resource.id: mesh},
        )
