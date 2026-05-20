import meshio
import numpy as np

from blab.mesh_clean import clean_mesh, stitch_meshes


def test_clean_mesh_mirrors_across_x_and_preserves_surface_tags() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={
            "gmsh:physical": [np.array([7], dtype=np.int32)],
            "gmsh:geometrical": [np.array([11], dtype=np.int32)],
        },
        field_data={"SD1D1001": np.array([7, 2], dtype=np.int32)},
    )

    cleaned, changes, before, after = clean_mesh(mesh, merge_tol=0.0, area_tol=0.0)
    triangles = cleaned.cells_dict["triangle"]

    assert before.vertices == 3
    assert after.vertices == 6
    assert after.triangles == 2
    assert changes["mirrored_vertices"] == 3
    assert changes["mirrored_faces"] == 1
    assert cleaned.cell_data_dict["gmsh:physical"]["triangle"].tolist() == [7, 7]
    assert cleaned.cell_data_dict["gmsh:geometrical"]["triangle"].tolist() == [11, 11]
    assert np.allclose(cleaned.points[triangles[1]], [[-1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [-2.0, 0.0, 0.0]])


def test_clean_mesh_removes_mirrored_duplicate_faces_on_symmetry_plane() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([3], dtype=np.int32)]},
    )

    cleaned, changes, _before, after = clean_mesh(mesh, merge_tol=1e-9, area_tol=0.0)

    assert after.vertices == 3
    assert after.triangles == 1
    assert changes["merged_vertices"] == 3
    assert changes["removed_duplicate_faces"] == 1
    assert cleaned.cell_data_dict["gmsh:physical"]["triangle"].tolist() == [3]


def test_clean_mesh_can_skip_x_mirroring() -> None:
    mesh = meshio.Mesh(
        points=np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([5], dtype=np.int32)]},
    )

    cleaned, changes, _before, after = clean_mesh(mesh, merge_tol=0.0, area_tol=0.0, mirror_x=False)

    assert after.vertices == 3
    assert after.triangles == 1
    assert changes["mirrored_vertices"] == 0
    assert changes["mirrored_faces"] == 0
    assert cleaned.cell_data_dict["gmsh:physical"]["triangle"].tolist() == [5]


def test_stitch_meshes_splits_mismatched_boundary_loops_into_shared_seam() -> None:
    enclosure = meshio.Mesh(
        points=np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cells=[
            (
                "triangle",
                np.array(
                    [
                        [0, 1, 2],
                        [0, 2, 3],
                        [0, 4, 5],
                        [0, 5, 1],
                        [1, 5, 6],
                        [1, 6, 2],
                        [2, 6, 7],
                        [2, 7, 3],
                        [3, 7, 4],
                        [3, 4, 0],
                    ],
                    dtype=np.int64,
                ),
            )
        ],
        cell_data={"gmsh:physical": [np.full(10, 1, dtype=np.int32)]},
        field_data={"Enclosure": np.array([1, 2], dtype=np.int32)},
    )
    cap = meshio.Mesh(
        points=np.array(
            [
                [-1.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        cells=[
            (
                "triangle",
                np.array(
                    [
                        [8, 0, 1],
                        [8, 1, 2],
                        [8, 2, 3],
                        [8, 3, 4],
                        [8, 4, 5],
                        [8, 5, 6],
                        [8, 6, 7],
                        [8, 7, 0],
                    ],
                    dtype=np.int64,
                ),
            )
        ],
        cell_data={"gmsh:physical": [np.full(8, 2, dtype=np.int32)]},
        field_data={"Waveguide": np.array([2, 2], dtype=np.int32)},
    )

    stitched, result = stitch_meshes((enclosure, cap), stitch_tol=1e-9, area_tol=0.0)

    assert result.before.boundary_edges == 12
    assert result.after.boundary_edges == 0
    assert result.seam_vertices == 8
    assert result.split_boundary_edges == 12
    assert stitched.cell_data_dict["gmsh:physical"]["triangle"].tolist().count(1) == 14
    assert stitched.cell_data_dict["gmsh:physical"]["triangle"].tolist().count(2) == 8
    assert stitched.field_data["Enclosure"].tolist() == [1, 2]
    assert stitched.field_data["Waveguide"].tolist() == [2, 2]


def test_stitch_meshes_remaps_colliding_physical_surface_tags() -> None:
    mesh_a = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([1], dtype=np.int32)]},
        field_data={"surface_a": np.array([1, 2], dtype=np.int32)},
    )
    mesh_b = meshio.Mesh(
        points=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
        cells=[("triangle", np.array([[0, 1, 2]], dtype=np.int64))],
        cell_data={"gmsh:physical": [np.array([1], dtype=np.int32)]},
        field_data={"surface_b": np.array([1, 2], dtype=np.int32)},
    )

    stitched, _result = stitch_meshes((mesh_a, mesh_b), stitch_tol=1e-9, area_tol=0.0)

    physical_tags = stitched.cell_data_dict["gmsh:physical"]["triangle"].tolist()
    assert sorted(set(physical_tags)) == [1, 2]
    assert stitched.field_data["surface_a"].tolist() == [1, 2]
    assert stitched.field_data["surface_b"].tolist() == [2, 2]
