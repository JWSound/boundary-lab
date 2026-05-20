import pytest

pytest.importorskip("PySide6")

from blab.ui.mesh_preview import _surface_hover_label


def test_surface_hover_label_includes_mesh_tag_and_element_count() -> None:
    label = _surface_hover_label("waveguide", "throat", 2, 1234)

    assert label == "Mesh: waveguide | Surface: throat | Tag: 2 | Elements: 1,234"


def test_surface_hover_label_handles_untagged_single_mesh_preview() -> None:
    label = _surface_hover_label(None, "untagged", None, 12)

    assert label == "Surface: untagged | Tag: untagged | Elements: 12"
