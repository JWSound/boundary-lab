from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from blab.observation_planes import (
    InteriorRenderingMode,
    ObservationPlane,
    ObservationPlaneDisplay,
    ObservationPlaneType,
    new_observation_plane,
    observation_planes_from_payload,
    rotate_observation_plane,
)


def test_observation_plane_round_trip_preserves_authoring_and_display_state() -> None:
    plane = ObservationPlane(
        id="plane:test",
        name="Cabinet Slice",
        center_m=(0.1, -0.2, 0.3),
        orientation_wxyz=(2.0, 0.0, 0.0, 0.0),
        width_m=0.4,
        height_m=0.2,
        resolution_m=0.005,
        plane_type=ObservationPlaneType.COMBINED,
        display=ObservationPlaneDisplay.PHASE,
        interior_rendering=InteriorRenderingMode.ELEMENT_FIELD,
        invert_clip_side=True,
        response_id="channel:woofer",
        animation_speed_hz=1.5,
    ).validated()

    restored = ObservationPlane.from_payload(plane.to_payload())

    assert restored == plane
    assert restored.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert restored.sample_shape == (81, 41)
    assert restored.evaluation_point_count == 3321
    np.testing.assert_allclose(restored.corners_m[0], [-0.1, -0.3, 0.3])


def test_point_warning_starts_above_ten_thousand_evaluation_points() -> None:
    exactly_ten_thousand = replace(
        new_observation_plane("Plane"),
        width_m=0.99,
        height_m=0.99,
        resolution_m=0.01,
    )
    over_ten_thousand = replace(exactly_ten_thousand, width_m=1.0, height_m=1.0)

    assert exactly_ten_thousand.evaluation_point_count == 10_000
    assert not exactly_ten_thousand.warns_about_evaluation_points
    assert over_ten_thousand.evaluation_point_count == 10_201
    assert over_ten_thousand.warns_about_evaluation_points


def test_rotation_uses_plane_local_axes_and_preserves_dimensions() -> None:
    plane = new_observation_plane("Plane")

    rotated = rotate_observation_plane(plane, 0, 90.0)
    axis_u, axis_v, normal = rotated.local_axes

    np.testing.assert_allclose(axis_u, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(axis_v, [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(normal, [0.0, -1.0, 0.0], atol=1e-12)
    assert rotated.width_m == plane.width_m
    assert rotated.height_m == plane.height_m


def test_new_plane_faces_forward_along_positive_z() -> None:
    _axis_u, _axis_v, normal = new_observation_plane("Plane").local_axes

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])


def test_relative_rotation_overlay_reports_signed_angle() -> None:
    from blab.ui.observation_plane_viewport import _relative_rotation_text

    assert _relative_rotation_text(15.0) == "Relative rotation: +15.0°"
    assert _relative_rotation_text(-2.25) == "Relative rotation: -2.2°"


def test_payload_reader_drops_invalid_and_duplicate_planes() -> None:
    plane = new_observation_plane("Plane")

    restored = observation_planes_from_payload(
        [plane.to_payload(), plane.to_payload(), {"id": "broken", "width_m": -1.0}]
    )

    assert restored == (plane,)


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan")])
def test_plane_rejects_invalid_resolution(invalid: float) -> None:
    with pytest.raises(ValueError, match="resolution_m"):
        replace(new_observation_plane("Plane"), resolution_m=invalid).validated()


def test_properties_dialog_disables_result_controls_without_solved_data(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    dialog = ObservationPlanePropertiesDialog(new_observation_plane("Plane"))
    try:
        assert not dialog.frequency_slider.isEnabled()
        assert not dialog.response_combo.isEnabled()
        assert not dialog.animate_button.isEnabled()
        assert dialog.frequency_label.text() == "No solved plane data available"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_viewport_uses_a_shared_camera_foreground_renderer_for_gizmos(qapp) -> None:
    import vtk
    from PySide6.QtWidgets import QWidget

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    class ViewerStub(QWidget):
        def __init__(self):
            super().__init__()
            self.render_window = vtk.vtkRenderWindow()
            self.renderer = vtk.vtkRenderer()
            self.render_window.AddRenderer(self.renderer)
            self.cleared_keys = []
            self.key_callbacks = {}

        def clear_events_for_key(self, key) -> None:
            self.cleared_keys.append(key)

        def add_key_event(self, key, callback) -> None:
            self.key_callbacks[key] = callback

    viewer = ViewerStub()
    editor = ObservationPlaneViewport(viewer, vtk, viewer)
    try:
        foreground = editor._foreground_renderer
        assert viewer.render_window.GetNumberOfLayers() == 2
        assert foreground.GetLayer() == 1
        assert foreground.GetPreserveColorBuffer()
        assert not foreground.GetPreserveDepthBuffer()
        assert foreground.GetActiveCamera() is viewer.renderer.GetActiveCamera()
        assert viewer.cleared_keys == ["r"]
        assert "r" in viewer.key_callbacks

        class RotateKeyEvent:
            @staticmethod
            def key():
                from PySide6.QtCore import Qt

                return Qt.Key.Key_R

        assert editor._on_key_press(RotateKeyEvent())
    finally:
        viewer.close()
        viewer.deleteLater()


def test_properties_dialog_warns_above_point_budget_and_disables_resolution_for_element_field(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(new_observation_plane("Plane"), width_m=1.0, height_m=1.0, resolution_m=0.01)
    dialog = ObservationPlanePropertiesDialog(plane)
    try:
        assert "10,201" in dialog.point_count_label.text()
        assert "warning" in dialog.point_count_label.text()
        index = dialog.rendering_combo.findData(InteriorRenderingMode.ELEMENT_FIELD.value)
        dialog.rendering_combo.setCurrentIndex(index)
        assert not dialog.resolution_spin.isEnabled()
        assert dialog.point_count_label.text() == "Not used by Element Field"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_viewport_creation_is_persisted_and_marks_project_dirty(main_window) -> None:
    main_window.preview.newObservationPlaneRequested.emit(
        {
            "center_m": (0.1, 0.2, 0.3),
            "orientation_wxyz": (1.0, 0.0, 0.0, 0.0),
            "width_m": 0.4,
            "height_m": 0.2,
        }
    )

    assert len(main_window.project.observation_planes) == 1
    plane = main_window.project.observation_planes[0]
    assert plane.center_m == (0.1, 0.2, 0.3)
    assert main_window.preview.observation_planes == (plane,)
    assert main_window.preview.selected_observation_plane_id == plane.id
    assert main_window._has_unsaved_project_changes()


def test_viewport_transform_and_delete_update_project_model(main_window) -> None:
    main_window.preview.newObservationPlaneRequested.emit({})
    original = main_window.project.observation_planes[0]
    moved = replace(original, center_m=(1.0, 2.0, 3.0))

    main_window.preview.observationPlaneChanged.emit(moved)
    assert main_window.project.observation_planes[0].center_m == (1.0, 2.0, 3.0)

    main_window.preview.observationPlaneDeleteRequested.emit(original.id)
    assert main_window.project.observation_planes == ()
    assert main_window.preview.observation_planes == ()
