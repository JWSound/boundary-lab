from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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
        frequency_hz=1234.0,
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


def test_properties_dialog_selects_and_previews_solved_frequency(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(new_observation_plane("Plane"), frequency_hz=900.0)
    dialog = ObservationPlanePropertiesDialog(
        plane,
        solved_frequencies_hz=(100.0, 1000.0, 10_000.0),
        response_options=(("system", "System Response"), ("channel:main", "Channel: main")),
    )
    previews = []
    dialog.previewChanged.connect(lambda updated, animate: previews.append((updated, animate)))
    try:
        assert dialog.frequency_slider.value() == 1
        assert dialog.plane.frequency_hz == 1000.0
        assert dialog.animate_button.isEnabled()
        dialog.frequency_slider.setValue(2)
        assert previews[-1][0].frequency_hz == 10_000.0
        dialog.animate_button.setChecked(True)
        assert dialog.animation_speed_slider.isEnabled()
        assert previews[-1][1]
    finally:
        dialog.close()
        dialog.deleteLater()


def test_interior_field_results_synthesize_system_and_channel_responses() -> None:
    from blab.config import ChannelConfig
    from blab.solve_results import (
        FEM_NODAL_PRESSURE_ID,
        FEM_VOLUME_DOMAIN_ID,
        ResultDomain,
        SolvedQuantity,
        SolvedSystem,
        SolveProvenance,
    )
    from blab.ui.observation_plane_results import interior_field_results_from_solved_system

    pressure = SolvedQuantity(
        id=FEM_NODAL_PRESSURE_ID,
        quantity="fem_nodal_pressure",
        unit="Pa",
        dimensions=("frequency", "excitation", "fem_node"),
        values=np.asarray([[[1.0] * 4, [2.0] * 4]], dtype=np.complex64),
        domain_id=FEM_VOLUME_DOMAIN_ID,
        available_frequency_mask=np.asarray([True]),
    )
    domain = ResultDomain(
        id=FEM_VOLUME_DOMAIN_ID,
        kind="fem_volume",
        dimensions=("fem_node",),
        coordinates={"points_m": np.eye(4, 3)},
        topology={"tetrahedra": np.asarray([[0, 1, 2, 3]])},
    )
    compiled = SimpleNamespace(
        excitation_ports=(
            SimpleNamespace(id="port:woofer", component_id="component:woofer"),
            SimpleNamespace(id="port:tweeter", component_id="component:tweeter"),
        )
    )
    solved = SolvedSystem(
        run_id="run",
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        frequencies_hz=np.asarray([1000.0]),
        excitation_ids=("port:woofer", "port:tweeter"),
        domains={FEM_VOLUME_DOMAIN_ID: domain},
        quantities={FEM_NODAL_PRESSURE_ID: pressure},
        completion_mask=np.asarray([True]),
        diagnostics_by_frequency=({},),
        status="completed",
        compiled_system=compiled,
    )

    results = interior_field_results_from_solved_system(
        solved,
        component_channel_by_id={"component:woofer": "low", "component:tweeter": "high"},
        channel_configs=(ChannelConfig(name="low"), ChannelConfig(name="high", level_db=-6.0206)),
    )

    assert results is not None
    assert results.response_options == (
        ("system", "System Response"),
        ("channel:low", "Channel: low"),
        ("channel:high", "Channel: high"),
    )
    np.testing.assert_allclose(results.pressure(1000.0, "system"), 2.0, rtol=1e-5)
    np.testing.assert_allclose(results.pressure(1000.0, "channel:low"), 1.0)
    np.testing.assert_allclose(results.pressure(1000.0, "channel:high"), 1.0, rtol=1e-5)


def test_field_scalar_projection_uses_stable_ranges_and_animation_phase() -> None:
    from blab.ui.observation_plane_results import project_field_scalars

    pressure = np.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    normalized = project_field_scalars(pressure, ObservationPlaneDisplay.NORMALIZED_SPL)
    animated = project_field_scalars(
        pressure,
        ObservationPlaneDisplay.SPL,
        animation_phase_deg=90.0,
    )

    assert normalized.clim == (-40.0, 0.0)
    np.testing.assert_allclose(normalized.values, [0.0, 0.0])
    np.testing.assert_allclose(animated.values, [0.0, 1.0], atol=1e-12)
    assert animated.clim == (-1.0, 1.0)


def test_fem_field_expands_pressure_with_symmetry_images() -> None:
    from blab.ui.observation_plane_viewport import _expanded_fem_field

    points, tetrahedra, pressure = _expanded_fem_field(
        np.asarray([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]),
        np.asarray([[0, 1, 2, 3]]),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        "xy",
    )

    assert points.shape == (16, 3)
    assert tetrahedra.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    assert pressure.tolist() == [1.0, 2.0, 3.0, 4.0] * 4
    np.testing.assert_allclose(points[4], [-1.0, 2.0, 3.0])


def test_smooth_and_element_fields_interpolate_and_clip_a_tetrahedron() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    tetrahedra = np.asarray([[0, 1, 2, 3]])
    pressure = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.complex64)
    plane = replace(
        new_observation_plane("Slice"),
        center_m=(0.2, 0.2, 0.2),
        width_m=0.2,
        height_m=0.2,
        resolution_m=0.1,
    )
    editor_stub = SimpleNamespace(_field_generation=1, _field_cache={})

    sampled, sampled_pressure, clipped = ObservationPlaneViewport._smooth_field_mesh(
        editor_stub,
        plane,
        points,
        tetrahedra,
        pressure,
    )
    element_mesh, element_pressure = ObservationPlaneViewport._element_field_mesh(
        editor_stub,
        replace(plane, interior_rendering=InteriorRenderingMode.ELEMENT_FIELD),
        points,
        tetrahedra,
        pressure,
    )

    assert sampled.n_cells == 4
    assert sampled.n_points == 9
    np.testing.assert_allclose(sampled_pressure[4], 2.2, rtol=1e-6)
    assert clipped.n_cells > 0
    assert element_mesh.n_cells > 0
    np.testing.assert_allclose(element_pressure, 2.5)


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
