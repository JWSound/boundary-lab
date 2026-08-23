import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem, QWidget

from blab.ui import mesh_preview as mesh_preview_module
from blab.ui.mesh_preview import (
    AXIS_LABELS,
    DRIVEN_COLOR,
    DRIVEN_EDGE_COLOR,
    DRIVEN_MIRROR_COLOR,
    DRIVEN_MIRROR_EDGE_COLOR,
    INTERFACE_COLOR,
    INTERFACE_EDGE_COLOR,
    INTERFACE_MIRROR_COLOR,
    INTERFACE_MIRROR_EDGE_COLOR,
    PREVIEW_HOME_CAMERA_DIRECTION,
    PREVIEW_HOME_VIEW_UP,
    PREVIEW_HOME_ZOOM,
    MeshPreview,
    _dimensions_lwh_mm,
    _line_segments_with_symmetry_images,
    _mesh_stats_label,
    _mirrored_triangle_images_for_preview,
    _preview_axis_label_points,
    _preview_axis_length,
    _preview_points_with_images,
    _surface_hover_label,
    _surface_preview_colors,
    _visibility_check_state,
    _visible_tree_row_count,
)
from repo_paths import source_text


def test_surface_hover_label_includes_mesh_tag_and_element_count() -> None:
    label = _surface_hover_label("waveguide", "throat", 2, 1234)

    assert label == "Mesh: waveguide | Surface: throat | Tag: 2 | Elements: 1,234"


def test_surface_hover_label_handles_untagged_single_mesh_preview() -> None:
    label = _surface_hover_label(None, "untagged", None, 12)

    assert label == "Surface: untagged | Tag: untagged | Elements: 12"


def test_preview_status_labels_do_not_force_panel_width() -> None:
    source = source_text("ui", "mesh_preview.py")

    assert "QSizePolicy" in source
    assert "self.hover_label.setMinimumWidth(0)" in source
    assert "self.hover_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)" in source
    assert "self.total_elements_label.setMinimumWidth(0)" in source
    assert "self.total_elements_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)" in source


def test_preview_background_tracks_the_application_theme() -> None:
    source = source_text("ui", "mesh_preview.py")

    assert "self._refresh_viewer_theme()" in source
    assert "viewer.set_background(themed_content_background(self.palette()))" in source
    assert "QEvent.Type.PaletteChange" in source


def test_driven_source_elements_use_high_contrast_blue() -> None:
    assert _surface_preview_colors(is_driven=True, is_interface=False, mirrored=False) == (
        DRIVEN_COLOR,
        DRIVEN_EDGE_COLOR,
    )
    assert _surface_preview_colors(is_driven=True, is_interface=False, mirrored=True) == (
        DRIVEN_MIRROR_COLOR,
        DRIVEN_MIRROR_EDGE_COLOR,
    )


def test_interface_elements_use_requested_green_colors_and_take_precedence() -> None:
    assert INTERFACE_COLOR == "#1cad0c"
    assert INTERFACE_MIRROR_COLOR == "#116b07"
    assert _surface_preview_colors(is_driven=True, is_interface=True, mirrored=False) == (
        INTERFACE_COLOR,
        INTERFACE_EDGE_COLOR,
    )
    assert _surface_preview_colors(is_driven=True, is_interface=True, mirrored=True) == (
        INTERFACE_MIRROR_COLOR,
        INTERFACE_MIRROR_EDGE_COLOR,
    )


def test_body_tree_parent_check_state_tracks_descendant_visibility() -> None:
    keys = (("exterior", 1), ("exterior", 2))

    assert _visibility_check_state(keys, {keys[0]: True, keys[1]: True}) == Qt.CheckState.Checked
    assert _visibility_check_state(keys, {keys[0]: False, keys[1]: False}) == Qt.CheckState.Unchecked
    assert _visibility_check_state(keys, {keys[0]: True, keys[1]: False}) == Qt.CheckState.PartiallyChecked


def test_body_tree_is_a_translucent_viewport_overlay_with_collapsed_project_root() -> None:
    source = source_text("ui", "mesh_preview.py")

    assert 'self.body_tree_overlay.setObjectName("mesh_body_tree_overlay")' in source
    assert "background-color: rgba(38, 44, 55, 120)" in source
    assert "scene_layout.addWidget(self.viewer, 0, 0)" in source
    assert '"Project",' in source
    assert "project_item.setExpanded(False)" in source
    assert "QSplitter" not in source


def test_body_tree_overlay_height_counts_only_visible_rows(qapp) -> None:
    del qapp
    tree = QTreeWidget()
    project = QTreeWidgetItem(["Project"])
    region = QTreeWidgetItem(["Exterior"])
    mesh = QTreeWidgetItem(["Mesh"])
    tree.addTopLevelItem(project)
    project.addChild(region)
    region.addChild(mesh)

    assert _visible_tree_row_count(tree) == 1
    project.setExpanded(True)
    assert _visible_tree_row_count(tree) == 2
    region.setExpanded(True)
    assert _visible_tree_row_count(tree) == 3


def test_mesh_preview_constructs_overlay_above_viewer_with_project_collapsed(qapp, monkeypatch) -> None:
    class StubViewer(QWidget):
        def set_background(self, _color) -> None:
            pass

    class StubSignal:
        def connect(self, _slot) -> None:
            pass

    class StubObservationPlaneViewport:
        def __init__(self, *_args) -> None:
            self.newPlaneRequested = StubSignal()
            self.planeChanged = StubSignal()
            self.propertiesRequested = StubSignal()
            self.deleteRequested = StubSignal()
            self.clipStateChanged = StubSignal()
            self.exteriorFieldRequested = StubSignal()

    monkeypatch.setattr(mesh_preview_module, "QtInteractor", StubViewer)
    monkeypatch.setattr(mesh_preview_module, "ObservationPlaneViewport", StubObservationPlaneViewport)
    monkeypatch.setattr(MeshPreview, "_install_hover_picker", lambda _self: None)

    preview = MeshPreview()
    qapp.processEvents()
    project = preview.body_tree.topLevelItem(0)

    assert project.text(0) == "Project"
    assert not project.isExpanded()
    assert preview.viewer.parentWidget() is preview.body_tree_overlay.parentWidget()
    assert preview.body_tree_overlay.height() < preview.body_tree_overlay.width()
    preview.close()


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


def test_preview_home_camera_projects_axes_for_speaker_forward_orientation() -> None:
    view_direction = -PREVIEW_HOME_CAMERA_DIRECTION
    screen_right = np.cross(view_direction, PREVIEW_HOME_VIEW_UP)
    screen_right = screen_right / np.linalg.norm(screen_right)
    screen_up = PREVIEW_HOME_VIEW_UP - np.dot(PREVIEW_HOME_VIEW_UP, view_direction) * view_direction
    screen_up = screen_up / np.linalg.norm(screen_up)

    x_axis = np.array([1.0, 0.0, 0.0])
    y_axis = np.array([0.0, 1.0, 0.0])
    z_axis = np.array([0.0, 0.0, 1.0])

    assert np.dot(y_axis, screen_right) == pytest.approx(0.0)
    assert np.dot(y_axis, screen_up) > 0.0
    assert np.dot(x_axis, screen_right) > 0.0
    assert np.dot(x_axis, screen_up) > 0.0
    assert np.dot(z_axis, screen_right) > 0.0
    assert np.dot(z_axis, screen_up) < 0.0


def test_preview_home_camera_uses_a_tighter_default_zoom() -> None:
    source = source_text("ui", "mesh_preview.py")

    assert PREVIEW_HOME_ZOOM == 1.2
    assert "camera.zoom(PREVIEW_HOME_ZOOM)" in source


def test_preview_orientation_guides_match_balloon_axis_labels() -> None:
    label_points = _preview_axis_label_points(10.0)

    assert AXIS_LABELS == ("Horizontal", "Vertical", "On Axis")
    np.testing.assert_allclose(
        label_points,
        np.array(
            [
                [10.6, 0.0, 0.0],
                [0.0, 10.6, 0.0],
                [0.0, 0.0, 11.6],
            ]
        ),
    )

    source = source_text("ui", "mesh_preview.py")
    assert "self.viewer.add_point_labels(" in source
    assert "list(AXIS_LABELS)" in source
    assert 'text_color="white"' in source
    assert "always_visible=True" in source


def test_mesh_stats_label_includes_mirrored_state_and_dimensions() -> None:
    assert _mesh_stats_label(1234, mirrored=True, dimensions_mm=(300, 200, 100)) == (
        "Total elements: 1,234 (Mirrored) | 300mm x 200mm x 100mm (LWH)"
    )
    assert _mesh_stats_label(0, mirrored=True, dimensions_mm=(0, 0, 0)) == ""


def test_dimensions_lwh_mm_maps_z_x_y_extents() -> None:
    points = np.array(
        [
            [-0.050, -0.010, -0.300],
            [0.150, 0.090, 0.100],
        ]
    )

    assert _dimensions_lwh_mm(points) == (400, 200, 100)
    assert _dimensions_lwh_mm(np.empty((0, 3))) == (0, 0, 0)


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


def test_mirrored_preview_dimensions_use_displayed_images_without_inflating_count() -> None:
    points = np.array(
        [
            [0.050, 0.0, 0.0],
            [0.100, 0.0, 0.0],
            [0.050, 0.020, 0.030],
        ]
    )
    triangles = np.array([[0, 1, 2]], dtype=np.int64)

    images = _mirrored_triangle_images_for_preview(points, triangles, "x")
    display_points = _preview_points_with_images(points, images)

    assert int(triangles.shape[0]) == 1
    assert _dimensions_lwh_mm(display_points) == (30, 200, 20)


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


def test_topology_issue_segments_are_mirrored_and_plane_duplicates_are_suppressed() -> None:
    segments = np.array(
        [
            [[1.0, 1.0, 0.0], [2.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
        ]
    )

    displayed = _line_segments_with_symmetry_images(segments, "x")

    assert displayed.shape == (3, 2, 3)
    assert np.sum(np.all(displayed == segments[1], axis=(1, 2))) == 1
    assert np.any(np.all(displayed == np.array([[-1.0, 1.0, 0.0], [-2.0, 1.0, 0.0]]), axis=(1, 2)))
