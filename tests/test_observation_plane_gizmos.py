"""Production gizmo regressions; no OpenGL context or preview utility required."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from blab.observation_planes import ObservationPlane, rotate_observation_plane
from blab.ui.observation_plane_gizmos import (
    ROTATION_RADIUS_PX,
    ROTATION_SNAP_TOLERANCE_PX,
    RotationRadar,
    hit_tick,
    radar_ticks,
    square_grip,
    unwrap_angle,
)
from blab.ui.observation_plane_viewport import MOVE_LENGTH_PX, RESIZE_HALF_SIZE_PX, ObservationPlaneViewport


@pytest.mark.parametrize("basis", [np.eye(2), np.array([[0.8, -0.3], [0.2, 0.6]])])
def test_radar_ticks_have_three_bands_with_finite_hit_segments(basis):
    center = np.array([300.0, 250.0])
    segments, metadata = radar_ticks(center, basis, ROTATION_RADIUS_PX)
    assert len(segments) == 8 + 36 + 72
    assert {step for angle, step in metadata} == {45, 10, 5}
    for index, segment in enumerate(segments):
        assert hit_tick(segment.mean(axis=0), segments) == index
    assert hit_tick(center, segments) is None
    assert hit_tick(center + [500, 0], segments) is None


def test_radar_snaps_within_ten_pixels_and_releases_outside_ticks():
    radar = RotationRadar(SimpleNamespace(), None)
    radar._center = lambda _plane: np.zeros(2)
    radar._basis = lambda _polar, _tangent: np.eye(2)
    plane = ObservationPlane(id="p", name="Plane")
    radar.begin(plane, 2, np.array([168, 0]))
    assert radar.snap_tolerance == ROTATION_SNAP_TOLERANCE_PX == 10
    _, _, segments, metadata = radar.geometry()
    for target, step in ((45, 45), (20, 10), (25, 5)):
        point = segments[metadata.index((target, step))].mean(axis=0)
        assert radar.angle_at(point) == target
        assert radar.drag.step == step
        angle = np.deg2rad(23.7)
        point = 1.8 * radar.radius * np.array([np.cos(angle), np.sin(angle)])
        assert radar.angle_at(point) == pytest.approx(23.7)
        assert radar.drag.tick is None
    point = segments[0].mean(axis=0)
    assert hit_tick(point + [0, 9.9], segments) == 0
    assert hit_tick(point + [0, 10.1], segments) is None
    assert unwrap_angle(179, -179) == 181
    assert unwrap_angle(-179, 179) == -181
    assert unwrap_angle(719, 0) == 720


def test_edge_on_radar_uses_invertible_screen_basis():
    radar = RotationRadar(SimpleNamespace(), None)
    radar._center = lambda _plane: np.zeros(2)
    radar._basis = lambda _polar, _tangent: np.array([[1, 0], [0, 0]])
    radar.begin(ObservationPlane(id="p", name="Plane"), 0, np.array([80, 80]))
    assert radar.drag.face_on
    _, basis, segments, metadata = radar.geometry()
    assert np.linalg.det(basis) == pytest.approx(1)
    assert radar.angle_at(segments[metadata.index((45, 45))].mean(axis=0)) == 45


@pytest.mark.parametrize("alt", [False, True])
def test_rotation_uses_radar_angle_without_modifier_rounding(alt):
    original = ObservationPlane(id="p", name="Plane")
    updated = []
    editor = SimpleNamespace(
        _radar=SimpleNamespace(angle_at=lambda _point: 23.7),
        _control_point=lambda position: position,
        _replace_selected=updated.append,
    )
    ObservationPlaneViewport._drag_rotate(
        editor, (0, 0), SimpleNamespace(original=original, control=1), alt_pressed=alt
    )
    assert updated == [rotate_observation_plane(original, 1, 23.7)]


def test_beveled_square_grips_have_colored_insets_on_both_sides():
    grip = square_grip("#c9cdd1", "#293134")
    assert grip.bounds[:4] == (-1, 1, -1, 1)
    assert grip.bounds[4] < 0 < grip.bounds[5]
    colors = grip.cell_data["tool_rgb"]
    assert colors.shape == (grip.n_cells, 3)
    assert np.count_nonzero(np.all(colors == [41, 49, 52], axis=1)) == 2


@pytest.mark.parametrize("ratio", [1.0, 2.0])
def test_screen_controls_keep_pixel_scale_and_world_anchors_without_rendering(qapp, ratio):
    import vtk
    from PySide6.QtWidgets import QWidget

    class Viewer(QWidget):
        def __init__(self):
            super().__init__()
            self.render_window = vtk.vtkRenderWindow()
            self.renderer = vtk.vtkRenderer()
            self.render_window.AddRenderer(self.renderer)

        def devicePixelRatioF(self):  # noqa: N802
            return ratio

    viewer = Viewer()
    editor = ObservationPlaneViewport(viewer, vtk, viewer)
    camera = viewer.renderer.GetActiveCamera()
    camera.SetPosition(2, -3, 4)
    camera.SetFocalPoint(0, 0, 0)
    plane = ObservationPlane(id="p", name="Plane", center_m=(0.1, 0.2, 0.3))
    editor._selected_id = plane.id
    try:
        for mode, count, size in (("move", 3, MOVE_LENGTH_PX), ("size", 4, RESIZE_HALF_SIZE_PX)):
            editor._mode = mode
            editor._tool_actor_by_control = {(mode, index): vtk.vtkActor() for index in range(count)}
            for parallel in (False, True):
                camera.SetParallelProjection(parallel)
                for width, height, zoom in ((640, 480, 0.5), (1200, 800, 2.0)):
                    viewer.render_window.SetSize(int(width * ratio), int(height * ratio))
                    camera.Zoom(zoom)
                    for dimensions in ((0.25, 0.25), (2.0, 3.0)):
                        current = replace(plane, width_m=dimensions[0], height_m=dimensions[1])
                        editor._planes = (current,)
                        editor._sync_controls()
                        for index in range(count):
                            matrix = editor._tool_actor_by_control[(mode, index)].GetUserMatrix()
                            basis = np.array([[matrix.GetElement(row, col) for col in range(3)] for row in range(3)])
                            np.testing.assert_allclose(np.linalg.norm(basis, axis=0), size * ratio)
                            anchor = current.center_m if mode == "move" else current.corners_m[index]
                            viewer.renderer.SetWorldPoint(*anchor, 1)
                            viewer.renderer.WorldToDisplay()
                            np.testing.assert_allclose(
                                [matrix.GetElement(0, 3), matrix.GetElement(1, 3)],
                                viewer.renderer.GetDisplayPoint()[:2],
                            )
        assert viewer.render_window.HasObserver("StartEvent")
        editor.close()
        editor.close()
        assert not viewer.render_window.HasObserver("StartEvent")
    finally:
        editor.close()
        viewer.close()
        viewer.deleteLater()
