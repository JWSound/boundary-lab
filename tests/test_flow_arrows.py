from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from blab.fem_fields import flow_arrow_points, plane_flow_vectors, sample_pressure_velocity
from blab.observation_planes import (
    ObservationPlane,
    ObservationPlaneDisplay,
    ObservationPlaneType,
    new_observation_plane,
)
from blab.ui.observation_plane_results import InteriorFieldResults, tetrahedral_shape_gradients
from blab.ui.observation_plane_viewport import ObservationPlaneViewport


def test_active_intensity_rms_and_standing_wave():
    # Travelling wave: RMS p=2 Pa, v=0.01 m/s, I=0.02 W/m².
    vectors, reference = plane_flow_vectors([2, 2], [[0.01, 0, 8], [0.01j, 0, 8]], [0, 0, 1])
    np.testing.assert_allclose(vectors, [[0.02, 0, 0], [0, 0, 0]])
    assert reference == pytest.approx(0.02)
    # A common source phase rotation must not rotate active intensity.
    angle = np.exp(0.7j)
    shifted, _ = plane_flow_vectors(np.array([2]) * angle, np.array([[0.01, 0, 8]]) * angle, [0, 0, 1])
    np.testing.assert_allclose(shifted, vectors[:1])


def test_animation_projects_rotated_plane_and_preserves_phase_scale():
    velocity = np.array([[1, 1j, 10]])
    first, scale = plane_flow_vectors([1], velocity, [0, 0, 1], 0)
    quarter, scale_quarter = plane_flow_vectors([1], velocity, [0, 0, 1], 90)
    reverse, scale_reverse = plane_flow_vectors([1], velocity, [0, 0, 1], 180)
    np.testing.assert_allclose(first, [[np.sqrt(2), 0, 0]])
    np.testing.assert_allclose(quarter, [[0, -np.sqrt(2), 0]], atol=1e-14)
    np.testing.assert_allclose(reverse, -first, atol=1e-14)
    assert scale == scale_quarter == scale_reverse
    rotated, _ = plane_flow_vectors([1], [[1, 2, 3]], [1, 0, 0], 0)
    np.testing.assert_allclose(rotated, [[0, 2 * np.sqrt(2), 3 * np.sqrt(2)]])


def test_quadratic_sampling_uses_vtk_edge_order_and_exact_gradient():
    corners = np.eye(4, 3, k=-1)
    edges = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
    points = np.vstack((corners, [(corners[a] + corners[b]) / 2 for a, b in edges]))
    x, y, z = points.T
    pressure = (1 + 2j) * (x * x + 3 * y * z + 2 * z * z)
    bary = np.array([[0.4, 0.1, 0.2, 0.3], [0.1, 0.2, 0.3, 0.4]])
    values, velocity = sample_pressure_velocity(
        pressure,
        np.arange(10)[None, :],
        tetrahedral_shape_gradients(points, np.arange(4)[None, :]),
        [1.2],
        100,
        [0, 0],
        bary,
    )
    x, y, z = bary[:, 1:].T
    np.testing.assert_allclose(values, (1 + 2j) * (x * x + 3 * y * z + 2 * z * z))
    gradient = (1 + 2j) * np.column_stack((2 * x, 3 * z, 3 * y + 4 * z))
    np.testing.assert_allclose(velocity, gradient / (-1j * 2 * np.pi * 100 * 1.2))


def test_arrows_have_proportional_lengths_and_finite_zero_geometry():
    anchors = np.zeros((3, 3))
    vectors = np.array([[1, 0, 0], [0.5, 0, 0], [0, 0, 0]])
    points = flow_arrow_points(anchors, vectors, [0, 0, 1], 0.1, 1).reshape(3, 6, 3)
    np.testing.assert_allclose(points[:, 1, 0], [0.08, 0.04, 0])
    assert np.isfinite(points).all()


def test_flow_settings_roundtrip_and_dialog(qapp):
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(new_observation_plane("Flow"), display=ObservationPlaneDisplay.FLOW_ARROWS, arrow_density=31)
    assert ObservationPlane.from_payload(plane.to_payload()) == plane
    assert replace(plane, plane_type=ObservationPlaneType.EXTERIOR).validated().display == ObservationPlaneDisplay.SPL
    dialog = ObservationPlanePropertiesDialog(plane, solved_frequencies_hz=(100,))
    assert dialog.arrow_density_slider.isEnabled()
    assert not dialog.resolution_spin.isEnabled()
    dialog.arrow_density_slider.setValue(42)
    assert dialog.plane.arrow_density == 42
    assert dialog.velocity_scale_auto_check.isChecked()
    assert dialog.intensity_scale_auto_check.isChecked()
    assert not dialog.velocity_scale_spin.isEnabled()
    dialog.velocity_scale_auto_check.setChecked(False)
    dialog.velocity_scale_spin.setValue(0.025)
    dialog.intensity_scale_auto_check.setChecked(False)
    dialog.intensity_scale_spin.setValue(0.004)
    assert dialog.plane.velocity_scale_m_per_s == 0.03
    assert dialog.plane.intensity_scale_w_per_m2 == 0.01
    assert ObservationPlane.from_payload(dialog.plane.to_payload()) == dialog.plane
    dialog.animate_button.setChecked(True)
    assert dialog.plane.display == ObservationPlaneDisplay.FLOW_ARROWS
    dialog.close()


def test_flow_renderer_caches_sampling_and_updates_animation_in_place():
    import pyvista as pv

    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    cells = np.array([[0, 1, 2, 3]])
    # Mixed complex pressure produces nonzero active intensity and reversing velocity.
    results = InteriorFieldResults(
        run_id="flow",
        frequencies_hz=np.array([100]),
        frequency_indices=np.array([0]),
        points_m=points,
        tetrahedra=cells,
        tetrahedron_density_kg_per_m3=np.array([1.2]),
        pressure_shape_gradients=tetrahedral_shape_gradients(points, cells),
        pressure_by_frequency=np.array([[[1, 1 - 1j, 1, 1]]]),
        excitation_ids=("source",),
        channel_names_by_excitation=("source",),
        channel_configs=(),
        symmetry="x",
    )
    plane = replace(
        new_observation_plane("Flow"),
        display=ObservationPlaneDisplay.FLOW_ARROWS,
        center_m=(0, 0.2, 0.2),
        width_m=0.8,
        height_m=0.2,
        arrow_density=10,
    )
    viewer = pv.Plotter(off_screen=True, window_size=(600, 400))
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(interior=results),
        _field_generation=1,
        _field_cache={},
        _animation_plane_id=None,
        _animation_phase_deg=0,
        viewer=viewer,
        _text_color="white",
        _scalar_bar_titles=set(),
        _actor_names=set(),
        _active_field=None,
    )
    for name in ("_flow_geometry", "_fem_volume_geometry", "_update_flow_points", "_field_mesh_and_pressure"):
        setattr(editor, name, lambda *args, name=name: getattr(ObservationPlaneViewport, name)(editor, *args))
    try:
        geometry = editor._flow_geometry(plane)
        assert geometry.mesh.n_verts == 0
        assert editor._flow_geometry(plane) is geometry
        assert np.any(geometry.vector_signs[:, 0] == -1)
        mesh, fields, association, _ = editor._field_mesh_and_pressure(plane)
        ObservationPlaneViewport._add_colored_field_actor(
            editor, mesh, fields, plane, association=association, name="flow"
        )
        static_points = mesh.points.copy()
        assert "In-plane\nActive Intensity\n(W/m^2)" in editor._scalar_bar_titles
        actor = editor._active_field.actor
        editor._animation_phase_deg = 90
        assert ObservationPlaneViewport._update_animation_frame(editor, plane)
        assert editor._active_field.actor is actor
        assert not np.allclose(mesh.points, static_points)
        editor._animation_phase_deg = 270
        assert ObservationPlaneViewport._update_animation_frame(editor, plane)
        editor._animation_plane_id = None
        assert ObservationPlaneViewport._update_active_field(editor, plane)
        np.testing.assert_allclose(mesh.points, static_points)
        automatic_max = actor.GetMapper().GetScalarRange()[1]
        manual = replace(plane, intensity_scale_w_per_m2=2 * automatic_max)
        assert ObservationPlaneViewport._update_active_field(editor, manual)
        assert actor.GetMapper().GetScalarRange()[1] == pytest.approx(2 * automatic_max)
        tips = mesh.points.reshape(-1, 6, 3)[:, 1]
        static_tips = static_points.reshape(-1, 6, 3)[:, 1]
        np.testing.assert_allclose(tips - geometry.anchors, (static_tips - geometry.anchors) / 2)
        manual = replace(manual, velocity_scale_m_per_s=0.01)
        editor._animation_plane_id = plane.id
        assert ObservationPlaneViewport._update_active_field(editor, manual)
        assert actor.GetMapper().GetScalarRange()[1] == pytest.approx(0.01)
        assert editor._active_field.actor is actor
        denser = editor._flow_geometry(replace(plane, arrow_density=20))
        assert len(denser.anchors) > len(geometry.anchors)
    finally:
        viewer.close()


@pytest.mark.parametrize("field", ["velocity_scale_m_per_s", "intensity_scale_w_per_m2"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_flow_scales_require_positive_finite_values(field, value):
    with pytest.raises(ValueError):
        replace(new_observation_plane("Flow"), **{field: value}).validated()


def test_manual_arrow_scale_saturates_without_changing_direction():
    anchors = np.zeros((2, 3))
    points = flow_arrow_points(anchors, [[3, 4, 0], [0, 0.5, 0]], [0, 0, 1], 1, 1).reshape(2, 6, 3)
    np.testing.assert_allclose(points[:, 1], [[0.48, 0.64, 0], [0, 0.4, 0]])
