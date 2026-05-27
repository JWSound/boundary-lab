import pytest

pytest.importorskip("PySide6")

import numpy as np

from blab.ui.mesh_preview import _preview_axis_length, _surface_hover_label


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
