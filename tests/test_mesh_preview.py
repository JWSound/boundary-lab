import numpy as np
import pytest

pytest.importorskip("PySide6")

from blab.ui.mesh_preview import _mirrored_triangle_images_for_preview, _preview_axis_length, _surface_hover_label


def test_surface_hover_label_includes_mesh_tag_and_element_count() -> None:
    label = _surface_hover_label("waveguide", "throat", 2, 1234)

    assert label == "Mesh: waveguide | Surface: throat | Tag: 2 | Elements: 1,234"


def test_surface_hover_label_handles_untagged_single_mesh_preview() -> None:
    label = _surface_hover_label(None, "untagged", None, 12)

    assert label == "Surface: untagged | Tag: untagged | Elements: 12"


def test_preview_axis_length_scales_with_mesh_bounds() -> None:
    points = np.array(
        [
            [-2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )

    assert _preview_axis_length(points) > 3.0
    assert _preview_axis_length(np.empty((0, 3))) == 1.0


def test_mirrored_preview_skips_triangles_on_symmetry_plane() -> None:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
        ]
    )
    triangles = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )

    images = _mirrored_triangle_images_for_preview(points, triangles, "x")

    assert len(images) == 1
    label, mirror_points, mirror_triangles, source_indices = images[0]
    assert label == "X"
    assert mirror_triangles.tolist() == [[3, 5, 4]]
    assert source_indices.tolist() == [1]
    assert np.allclose(mirror_points[[3, 4, 5], 0], [-1.0, -1.0, -1.0])


def test_xy_mirrored_preview_adds_three_images_for_quadrant_triangle() -> None:
    points = np.array(
        [
            [1.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
        ]
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)

    images = _mirrored_triangle_images_for_preview(points, triangles, "xy")

    assert [label for label, _points, _triangles, _indices in images] == ["X", "Y", "XY"]
    assert sum(len(triangles) for _label, _points, triangles, _indices in images) == 3
    assert images[0][2].tolist() == [[0, 2, 1]]
    assert images[1][2].tolist() == [[0, 2, 1]]
    assert images[2][2].tolist() == [[0, 1, 2]]


def test_xy_mirrored_preview_suppresses_duplicate_axis_images() -> None:
    points = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)

    images = _mirrored_triangle_images_for_preview(points, triangles, "xy")

    assert [label for label, _points, _triangles, _indices in images] == ["Y"]
    assert images[0][2].tolist() == [[0, 2, 1]]
