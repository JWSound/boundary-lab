from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from PySide6.QtTest import QTest

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
        pressure_color_limit_pa=0.25,
    ).validated()

    restored = ObservationPlane.from_payload(plane.to_payload())

    assert restored == plane
    assert restored.orientation_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert restored.sample_shape == (81, 41)
    assert restored.evaluation_point_count == 3321
    np.testing.assert_allclose(restored.corners_m[0], [-0.1, -0.3, 0.3])


def test_particle_velocity_display_round_trips_only_for_interior_planes() -> None:
    interior = replace(
        new_observation_plane("Velocity"),
        display=ObservationPlaneDisplay.PARTICLE_VELOCITY,
    )

    assert ObservationPlane.from_payload(interior.to_payload()) == interior
    assert (
        replace(interior, plane_type=ObservationPlaneType.EXTERIOR).validated().display == ObservationPlaneDisplay.SPL
    )
    assert (
        replace(interior, plane_type=ObservationPlaneType.COMBINED).validated().display == ObservationPlaneDisplay.SPL
    )


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
    plane = new_observation_plane("Plane")
    _axis_u, _axis_v, normal = plane.local_axes

    np.testing.assert_allclose(normal, [0.0, 0.0, 1.0])
    assert plane.invert_clip_side


def test_payload_defaults_to_the_inverted_clip_side_without_overriding_saved_choices() -> None:
    payload = new_observation_plane("Plane").to_payload()
    payload.pop("invert_clip_side")

    assert ObservationPlane.from_payload(payload).invert_clip_side
    assert not ObservationPlane.from_payload({**payload, "invert_clip_side": False}).invert_clip_side


def test_relative_rotation_overlay_reports_signed_angle() -> None:
    from blab.ui.observation_plane_viewport import _relative_rotation_text

    assert _relative_rotation_text(15.0) == "Relative rotation: +15.0°"
    assert _relative_rotation_text(-2.25) == "Relative rotation: -2.2°"


def test_observation_plane_help_is_compact_right_aligned_arial_text() -> None:
    from blab.ui.observation_plane_viewport import _observation_plane_help_text
    from repo_paths import source_text

    assert _observation_plane_help_text("Listening Plane", "rotate") == (
        "Listening Plane\nW - Move\nE - Rotate\nR - Scale\nMode: Rotate"
    )
    source = source_text("ui", "observation_plane_viewport.py")
    assert 'position="upper_right"' in source
    assert "font_size=8" in source
    assert source.count("font=OBSERVATION_PLANE_FONT") == 3
    assert '"font_family": OBSERVATION_PLANE_FONT' in source


def test_particle_velocity_scalar_bar_title_wraps_for_narrow_viewports() -> None:
    from blab.ui.observation_plane_viewport import _scalar_bar_position_x, _scalar_bar_title

    assert _scalar_bar_title("Particle Velocity Magnitude (m/s)") == ("Particle Velocity\nMagnitude (m/s)\n ")
    assert _scalar_bar_title("SPL (dB re 20 µPa)") == "SPL (dB re 20 µPa)"
    assert _scalar_bar_position_x(SimpleNamespace(width=lambda: 342)) == pytest.approx(0.70)
    assert _scalar_bar_position_x(SimpleNamespace(width=lambda: 754)) == pytest.approx(0.84)


def test_viewport_overlay_text_is_white_only_for_dark_palettes(qapp) -> None:
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QWidget

    from blab.ui.observation_plane_viewport import _viewport_text_color

    viewer = QWidget()
    palette = QPalette(viewer.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor("#20282b"))
    viewer.setPalette(palette)
    assert _viewport_text_color(viewer) == "white"

    palette.setColor(QPalette.ColorRole.Window, QColor("#f4f4f4"))
    viewer.setPalette(palette)
    assert _viewport_text_color(viewer) == "black"


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


@pytest.mark.parametrize("invalid", [0.0, -1.0, float("nan")])
def test_plane_rejects_invalid_manual_pressure_color_limit(invalid: float) -> None:
    with pytest.raises(ValueError, match="pressure_color_limit_pa"):
        replace(new_observation_plane("Plane"), pressure_color_limit_pa=invalid).validated()


def test_properties_dialog_disables_result_controls_without_solved_data(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    dialog = ObservationPlanePropertiesDialog(new_observation_plane("Plane"))
    try:
        assert not dialog.frequency_slider.isEnabled()
        assert not dialog.response_combo.isEnabled()
        assert not dialog.animate_button.isEnabled()
        assert dialog.frequency_label.text() == "No solved plane data available"
        assert not hasattr(dialog, "active_check")
    finally:
        dialog.close()
        dialog.deleteLater()


def test_combined_properties_use_a_single_smooth_rendering_mode(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(
        new_observation_plane("Combined"),
        plane_type=ObservationPlaneType.COMBINED,
        interior_rendering=InteriorRenderingMode.ELEMENT_FIELD,
    )
    dialog = ObservationPlanePropertiesDialog(plane)
    try:
        assert dialog.rendering_combo.currentData() == InteriorRenderingMode.SMOOTH_FIELD.value
        assert not dialog.rendering_combo.isEnabled()
        assert dialog.resolution_spin.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_properties_dialog_disables_particle_velocity_outside_interior_planes(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    dialog = ObservationPlanePropertiesDialog(new_observation_plane("Plane"))
    try:
        velocity_index = dialog.display_combo.findData(ObservationPlaneDisplay.PARTICLE_VELOCITY.value)
        velocity_item = dialog.display_combo.model().item(velocity_index)
        assert velocity_item.isEnabled()

        dialog.display_combo.setCurrentIndex(velocity_index)
        exterior_index = dialog.type_combo.findData(ObservationPlaneType.EXTERIOR.value)
        dialog.type_combo.setCurrentIndex(exterior_index)
        assert not velocity_item.isEnabled()
        assert dialog.display_combo.currentData() == ObservationPlaneDisplay.SPL.value

        combined_index = dialog.type_combo.findData(ObservationPlaneType.COMBINED.value)
        dialog.type_combo.setCurrentIndex(combined_index)
        assert not velocity_item.isEnabled()

        interior_index = dialog.type_combo.findData(ObservationPlaneType.INTERIOR.value)
        dialog.type_combo.setCurrentIndex(interior_index)
        assert velocity_item.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_properties_dialog_selects_and_previews_solved_frequency(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(new_observation_plane("Plane"), frequency_hz=900.0)
    dialog = ObservationPlanePropertiesDialog(
        plane,
        solved_frequencies_hz=(10_000.0, 100.0, 1000.0),
        response_options=(("system", "System Response"), ("channel:main", "Channel: main")),
    )
    previews = []
    dialog.previewChanged.connect(lambda updated, animate: previews.append((updated, animate)))
    try:
        assert dialog.frequency_slider.value() == 1
        assert dialog.plane.frequency_hz == 1000.0
        assert dialog.animate_button.isEnabled()
        dialog.frequency_slider.setValue(2)
        qapp.processEvents()
        QTest.qWait(20)
        qapp.processEvents()
        assert previews[-1][0].frequency_hz == 10_000.0
        previews.clear()
        dialog.frequency_slider.setValue(0)
        dialog.frequency_slider.setValue(1)
        dialog.frequency_slider.setValue(2)
        QTest.qWait(20)
        qapp.processEvents()
        assert len(previews) == 1
        assert previews[0][0].frequency_hz == 10_000.0
        dialog.animate_button.setChecked(True)
        assert dialog.animation_speed_slider.isEnabled()
        assert previews[-1][1]
        dialog.set_solved_results((500.0, 50.0), (("system", "System Response"),))
        assert dialog._solved_frequencies_hz == (50.0, 500.0)
        assert dialog.plane.frequency_hz == 500.0
        dialog.set_solved_results((), ())
        assert not dialog.frequency_slider.isEnabled()
        assert not dialog.animate_button.isEnabled()
        assert not dialog.animation_speed_slider.isEnabled()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_properties_dialog_edits_manual_pressure_color_range(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = replace(
        new_observation_plane("Pressure"),
        display=ObservationPlaneDisplay.REAL_PRESSURE,
    )
    dialog = ObservationPlanePropertiesDialog(plane)
    try:
        assert dialog.pressure_range_auto_check.isChecked()
        assert dialog.pressure_range_auto_check.isEnabled()
        assert not dialog.pressure_limit_spin.isEnabled()
        assert dialog.pressure_limit_spin.decimals() == 0
        assert dialog.pressure_limit_spin.value() == 10.0

        dialog.pressure_range_auto_check.setChecked(False)
        dialog.pressure_limit_spin.setValue(12.6)

        assert dialog.pressure_limit_spin.isEnabled()
        assert dialog.pressure_limit_spin.value() == 13.0
        assert dialog.plane.pressure_color_limit_pa == pytest.approx(13.0)

        spl_index = dialog.display_combo.findData(ObservationPlaneDisplay.SPL.value)
        dialog.display_combo.setCurrentIndex(spl_index)
        assert not dialog.pressure_range_auto_check.isEnabled()
        assert not dialog.pressure_limit_spin.isEnabled()
        assert dialog.plane.pressure_color_limit_pa == pytest.approx(13.0)
    finally:
        dialog.close()
        dialog.deleteLater()


def test_properties_dialog_has_no_active_control(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    plane = new_observation_plane("Plane")
    dialog = ObservationPlanePropertiesDialog(plane)
    try:
        assert not hasattr(dialog, "active_check")
        assert not hasattr(dialog, "activeChanged")
        assert dialog.plane == plane
    finally:
        dialog.close()
        dialog.deleteLater()


def test_frequency_sweep_oscillates_across_sorted_solved_frequencies(qapp, monkeypatch) -> None:
    import blab.ui.observation_plane_dialog as dialog_module
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    clock = [10.0]
    monkeypatch.setattr(dialog_module.time, "monotonic", lambda: clock[0])
    plane = replace(new_observation_plane("Sweep"), frequency_hz=100.0)
    dialog = ObservationPlanePropertiesDialog(
        plane,
        solved_frequencies_hz=(800.0, 200.0, 100.0, 400.0),
    )
    previews = []
    dialog.previewChanged.connect(lambda updated, animate: previews.append((updated, animate)))
    try:
        dialog.frequency_sweep_speed_slider.setValue(10)
        dialog.frequency_sweep_button.setChecked(True)
        assert dialog._frequency_sweep_timer.isActive()
        assert dialog.frequency_sweep_speed_slider.isEnabled()

        clock[0] = 10.1
        dialog._advance_frequency_sweep()
        assert dialog.frequency_slider.value() == 1
        assert dialog.frequency_label.text() == "200 Hz"
        QTest.qWait(20)
        qapp.processEvents()
        assert previews[-1][0].frequency_hz == 200.0

        clock[0] = 10.3
        dialog._advance_frequency_sweep()
        assert dialog.frequency_slider.value() == 3
        assert dialog._frequency_sweep_direction == -1.0

        clock[0] = 10.4
        dialog._advance_frequency_sweep()
        assert dialog.frequency_slider.value() == 2
        assert dialog.frequency_label.text() == "400 Hz"
    finally:
        dialog.close()
        dialog.deleteLater()


def test_frequency_sweep_and_phase_animation_are_mutually_exclusive(qapp) -> None:
    from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog

    dialog = ObservationPlanePropertiesDialog(
        new_observation_plane("Sweep"),
        solved_frequencies_hz=(100.0, 200.0),
    )
    try:
        dialog.frequency_sweep_button.setChecked(True)
        dialog.animate_button.setChecked(True)
        assert dialog.animate_button.isChecked()
        assert not dialog.frequency_sweep_button.isChecked()
        assert not dialog._frequency_sweep_timer.isActive()

        dialog.frequency_sweep_button.setChecked(True)
        assert dialog.frequency_sweep_button.isChecked()
        assert not dialog.animate_button.isChecked()
        assert dialog._frequency_sweep_timer.isActive()
    finally:
        dialog.close()
        dialog.deleteLater()


def test_interior_field_results_synthesize_system_and_channel_responses() -> None:
    from blab.config import ChannelConfig
    from blab.physical_model import ExcitationPortKind
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
        values=np.asarray(
            [
                [[1.0] * 4, [2.0] * 4],
                [[10.0] * 4, [20.0] * 4],
            ],
            dtype=np.complex64,
        ),
        domain_id=FEM_VOLUME_DOMAIN_ID,
        available_frequency_mask=np.asarray([True, True]),
    )
    domain = ResultDomain(
        id=FEM_VOLUME_DOMAIN_ID,
        kind="fem_volume",
        dimensions=("fem_node",),
        coordinates={"points_m": np.eye(4, 3)},
        topology={"tetrahedra": np.asarray([[0, 1, 2, 3]])},
        metadata={"density_kg_per_m3": [1.21]},
    )
    compiled = SimpleNamespace(
        excitation_ports=(
            SimpleNamespace(
                id="port:woofer", component_id="component:woofer", kind=ExcitationPortKind.VOLTAGE
            ),
            SimpleNamespace(
                id="port:tweeter", component_id="component:tweeter", kind=ExcitationPortKind.VOLTAGE
            ),
        )
    )
    solved = SolvedSystem(
        run_id="run",
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        frequencies_hz=np.asarray([1000.0, 100.0]),
        excitation_ids=("port:woofer", "port:tweeter"),
        domains={FEM_VOLUME_DOMAIN_ID: domain},
        quantities={FEM_NODAL_PRESSURE_ID: pressure},
        completion_mask=np.asarray([True, True]),
        diagnostics_by_frequency=({}, {}),
        status="completed",
        compiled_system=compiled,
    )

    results = interior_field_results_from_solved_system(
        solved,
        component_channel_by_id={"component:woofer": "low", "component:tweeter": "high"},
        channel_configs=(ChannelConfig(name="low"), ChannelConfig(name="high", level_db=-6.0206)),
    )

    assert results is not None
    assert results.frequencies_hz.tolist() == [100.0, 1000.0]
    assert results.frequency_indices.tolist() == [1, 0]
    assert results.response_options == (
        ("system", "System Response"),
        ("channel:low", "Channel: low"),
        ("channel:high", "Channel: high"),
    )
    np.testing.assert_allclose(results.pressure(1000.0, "system"), 2.0, rtol=1e-5)
    np.testing.assert_allclose(results.pressure(1000.0, "channel:low"), 1.0)
    np.testing.assert_allclose(results.pressure(1000.0, "channel:high"), 1.0, rtol=1e-5)
    np.testing.assert_allclose(results.pressure(100.0, "system"), 20.0, rtol=1e-5)

    voltage_results = interior_field_results_from_solved_system(
        solved,
        component_channel_by_id={"component:woofer": "low", "component:tweeter": "high"},
        channel_configs=(ChannelConfig(name="low", voltage_v=5.66), ChannelConfig(name="high", level_db=-6.0206)),
    )
    np.testing.assert_allclose(voltage_results.pressure(1000.0, "channel:low"), 2.0)


def test_interior_particle_velocity_uses_exact_p1_pressure_gradient() -> None:
    from blab.solve_results import (
        FEM_NODAL_PRESSURE_ID,
        FEM_VOLUME_DOMAIN_ID,
        ResultDomain,
        SolvedQuantity,
        SolvedSystem,
        SolveProvenance,
    )
    from blab.ui.observation_plane_results import interior_field_results_from_solved_system

    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    pressure = SolvedQuantity(
        id=FEM_NODAL_PRESSURE_ID,
        quantity="fem_nodal_pressure",
        unit="Pa",
        dimensions=("frequency", "excitation", "fem_node"),
        values=np.asarray([[[0.0, 2.0, 3.0, 4.0]]], dtype=np.complex64),
        domain_id=FEM_VOLUME_DOMAIN_ID,
        available_frequency_mask=np.asarray([True]),
    )
    domain = ResultDomain(
        id=FEM_VOLUME_DOMAIN_ID,
        kind="fem_volume",
        dimensions=("fem_node",),
        coordinates={"points_m": points},
        topology={
            "tetrahedra": np.asarray([[0, 1, 2, 3]]),
            "region_index": np.asarray([0]),
        },
        metadata={"density_kg_per_m3": [2.0]},
    )
    solved = SolvedSystem(
        run_id="run:velocity",
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="coupled_bem_fem"),
        frequencies_hz=np.asarray([100.0]),
        excitation_ids=("port:source",),
        domains={FEM_VOLUME_DOMAIN_ID: domain},
        quantities={FEM_NODAL_PRESSURE_ID: pressure},
        completion_mask=np.asarray([True]),
        diagnostics_by_frequency=({},),
        status="completed",
    )

    results = interior_field_results_from_solved_system(solved)

    assert results is not None
    expected = np.asarray([2.0, 3.0, 4.0]) / (1j * 2.0 * np.pi * 100.0 * 2.0)
    np.testing.assert_allclose(results.particle_velocity(100.0, "system")[0], expected, rtol=1e-6)


def test_observation_field_results_use_only_common_combined_frequencies() -> None:
    from blab.ui.observation_plane_results import ObservationFieldResults

    results = ObservationFieldResults(
        interior=SimpleNamespace(frequencies_hz=np.asarray([100.0, 200.0, 400.0])),
        exterior=SimpleNamespace(frequencies_hz=np.asarray([100.0, 400.0, 800.0])),
    )

    assert results.frequencies_for(ObservationPlaneType.COMBINED).tolist() == [100.0, 400.0]
    assert results.frequencies_for(ObservationPlaneType.INTERIOR).tolist() == [100.0, 200.0, 400.0]
    assert results.frequencies_for(ObservationPlaneType.EXTERIOR).tolist() == [100.0, 400.0, 800.0]


def test_channel_synthesis_update_reuses_field_geometry_and_refreshes_active_scalars() -> None:
    from blab.config import ChannelConfig
    from blab.ui.observation_plane_results import ObservationFieldResults
    from blab.ui.observation_plane_viewport import (
        ObservationPlaneViewport,
        _observation_field_results_signature,
        _observation_field_source_signature,
    )

    def results(level_db: float) -> ObservationFieldResults:
        return ObservationFieldResults(
            interior=SimpleNamespace(
                run_id="run",
                frequencies_hz=np.asarray([100.0, 1000.0]),
                frequency_indices=np.asarray([0, 1]),
                points_m=np.zeros((4, 3)),
                tetrahedra=np.asarray([[0, 1, 2, 3]]),
                pressure_by_frequency=np.ones((2, 1, 4), dtype=np.complex64),
                excitation_ids=("port:main",),
                channel_names_by_excitation=("main",),
                channel_configs=(ChannelConfig(name="main", level_db=level_db),),
                flat_target_enabled=False,
                flat_target_reference_angle_deg=0.0,
                symmetry="off",
            )
        )

    original = results(0.0)
    updated = results(-6.0)
    plane = replace(new_observation_plane("Interior"), frequency_hz=1000.0)
    geometry = object()
    updates = []
    stopped = []
    editor = SimpleNamespace(
        _field_results=original,
        _field_source_signature=_observation_field_source_signature(original),
        _field_results_signature=_observation_field_results_signature(original),
        _field_generation=4,
        _field_synthesis_generation=9,
        _field_cache={("smooth", plane.id): geometry},
        _exterior_results=OrderedDict((("old", np.ones(2, dtype=np.complex64)),)),
        _exterior_result_bytes=16,
        _exterior_pending={("pending",)},
        _exterior_failures={("failed",): "old"},
        _exterior_request_meshes={("pending",): object()},
        _exterior_refresh_timer=SimpleNamespace(stop=lambda: stopped.append("exterior")),
        _interior_refresh_timer=SimpleNamespace(stop=lambda: stopped.append("interior")),
        _field_plane=lambda: plane,
        _update_active_field=lambda candidate: updates.append(candidate) or True,
        _render=lambda: pytest.fail("a synthesis-only update must not rebuild the scene"),
    )

    ObservationPlaneViewport.set_field_results(editor, updated)

    assert editor._field_results is updated
    assert editor._field_generation == 4
    assert editor._field_synthesis_generation == 10
    assert editor._field_cache[("smooth", plane.id)] is geometry
    assert editor._exterior_results == {}
    assert editor._exterior_result_bytes == 0
    assert editor._exterior_pending == set()
    assert editor._exterior_failures == {}
    assert editor._exterior_request_meshes == {}
    assert updates == [plane]
    assert stopped == ["exterior", "interior"]


def test_exterior_field_results_synthesize_retained_boundary_traces() -> None:
    from blab.config import ChannelConfig
    from blab.physical_model import AcousticRegionKind, ExcitationPortKind
    from blab.solve_results import (
        BEM_BOUNDARY_DOMAIN_ID,
        BEM_BOUNDARY_NEUMANN_ID,
        BEM_BOUNDARY_PRESSURE_ID,
        ResultDomain,
        SolvedQuantity,
        SolvedSystem,
        SolveProvenance,
    )
    from blab.ui.observation_plane_results import exterior_field_results_from_solved_system

    domain = ResultDomain(
        id=BEM_BOUNDARY_DOMAIN_ID,
        kind="bem_boundary",
        dimensions=("bem_node", "bem_face"),
        coordinates={"points_m": np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])},
        topology={"triangles": np.asarray([[0, 1, 2]])},
        metadata={"symmetry": "x"},
    )
    pressure = np.asarray(
        [
            [[1.0] * 3, [2.0] * 3],
            [[10.0] * 3, [20.0] * 3],
        ],
        dtype=np.complex64,
    )
    normal = np.asarray([[[3.0], [4.0]], [[30.0], [40.0]]], dtype=np.complex64)
    available = np.asarray([True, True])
    compiled = SimpleNamespace(
        regions=(SimpleNamespace(kind=AcousticRegionKind.UNBOUNDED_AIR, sound_speed_m_per_s=344.0),),
        excitation_ports=(
            SimpleNamespace(
                id="port:woofer", component_id="component:woofer", kind=ExcitationPortKind.VOLTAGE
            ),
            SimpleNamespace(
                id="port:tweeter", component_id="component:tweeter", kind=ExcitationPortKind.VOLTAGE
            ),
        ),
    )
    solved = SolvedSystem(
        run_id="run",
        provenance=SolveProvenance(backend_id="beat_cpu", solve_kind="exterior_bem"),
        frequencies_hz=np.asarray([1000.0, 100.0]),
        excitation_ids=("port:woofer", "port:tweeter"),
        domains={BEM_BOUNDARY_DOMAIN_ID: domain},
        quantities={
            BEM_BOUNDARY_PRESSURE_ID: SolvedQuantity(
                id=BEM_BOUNDARY_PRESSURE_ID,
                quantity="bem_boundary_pressure",
                unit="Pa",
                dimensions=("frequency", "excitation", "bem_node"),
                values=pressure,
                domain_id=BEM_BOUNDARY_DOMAIN_ID,
                available_frequency_mask=available,
            ),
            BEM_BOUNDARY_NEUMANN_ID: SolvedQuantity(
                id=BEM_BOUNDARY_NEUMANN_ID,
                quantity="bem_boundary_neumann",
                unit="Pa/m",
                dimensions=("frequency", "excitation", "bem_face"),
                values=normal,
                domain_id=BEM_BOUNDARY_DOMAIN_ID,
                available_frequency_mask=available,
            ),
        },
        completion_mask=available,
        diagnostics_by_frequency=({}, {}),
        status="completed",
        compiled_system=compiled,
    )

    results = exterior_field_results_from_solved_system(
        solved,
        component_channel_by_id={"component:woofer": "low", "component:tweeter": "high"},
        channel_configs=(ChannelConfig(name="low"), ChannelConfig(name="high")),
    )

    assert results is not None
    assert results.frequencies_hz.tolist() == [100.0, 1000.0]
    assert results.sound_speed_m_per_s == 344.0
    assert results.traces.symmetry == "x"
    frequency, system_pressure, system_normal = results.boundary_response(100.0, "system")
    assert frequency == 100.0
    np.testing.assert_allclose(system_pressure, 30.0)
    np.testing.assert_allclose(system_normal, 70.0)
    _frequency, channel_pressure, channel_normal = results.boundary_response(1000.0, "channel:high")
    np.testing.assert_allclose(channel_pressure, 2.0)
    np.testing.assert_allclose(channel_normal, 4.0)

    voltage_results = exterior_field_results_from_solved_system(
        solved,
        component_channel_by_id={"component:woofer": "low", "component:tweeter": "high"},
        channel_configs=(ChannelConfig(name="low"), ChannelConfig(name="high", voltage_v=5.66)),
    )
    _frequency, channel_pressure, channel_normal = voltage_results.boundary_response(1000.0, "channel:high")
    np.testing.assert_allclose(channel_pressure, 4.0)
    np.testing.assert_allclose(channel_normal, 8.0)


def test_exterior_viewport_requests_then_reuses_an_evaluated_field() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(
        new_observation_plane("Exterior"),
        plane_type=ObservationPlaneType.EXTERIOR,
        frequency_hz=1000.0,
    )
    traces = SimpleNamespace(
        points_m=np.eye(3),
        triangles=np.asarray([[0, 1, 2]]),
        symmetry="off",
    )
    exterior = SimpleNamespace(
        run_id="run",
        backend_id="beat_cpu",
        sound_speed_m_per_s=343.0,
        traces=traces,
        resolved_frequency=lambda _frequency: 1000.0,
        boundary_response=lambda *_args: (
            1000.0,
            np.ones(3, dtype=np.complex64),
            np.ones(1, dtype=np.complex64),
        ),
    )
    source_points = np.zeros((4, 3))
    mesh = SimpleNamespace(points=source_points, n_points=4)
    requested = []
    rendered = []
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(exterior=exterior),
        _drag=None,
        _field_generation=7,
        _field_synthesis_generation=7,
        _exterior_results={},
        _exterior_failures={},
        _exterior_pending=set(),
        _exterior_request_meshes={},
        _active_field=None,
        _field_message=None,
        _exterior_plane_mesh=lambda _plane: mesh,
        _exterior_pressure_for_display=lambda _plane, _mesh, pressure: pressure,
        _add_colored_field_actor=lambda *args, **kwargs: rendered.append((args, kwargs)),
        exteriorFieldRequested=SimpleNamespace(emit=lambda task: requested.append(task)),
    )
    editor._exterior_key_for_plane = MethodType(ObservationPlaneViewport._exterior_key_for_plane, editor)
    editor._request_exterior_field_for_plane = MethodType(
        ObservationPlaneViewport._request_exterior_field_for_plane,
        editor,
    )

    assert not ObservationPlaneViewport._add_exterior_field(editor, plane)
    assert editor._field_message == "Evaluating exterior field..."
    assert len(requested) == 1
    task = requested[0]
    assert task.request.frequency_hz == 1000.0
    assert task.request.points_m.shape == (4, 3)
    assert task.request.points_m.dtype == np.float32
    assert task.request.points_m.flags.c_contiguous
    assert not np.shares_memory(task.request.points_m, source_points)
    source_points[:] = 12.0
    np.testing.assert_allclose(task.request.points_m, 0.0)

    editor._exterior_results[task.key] = np.arange(4, dtype=np.complex64)
    assert ObservationPlaneViewport._add_exterior_field(editor, plane)
    np.testing.assert_allclose(rendered[0][0][1], np.arange(4))


def test_combined_field_requests_only_points_classified_as_exterior() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _CombinedFieldGeometry

    plane = replace(
        new_observation_plane("Combined"),
        plane_type=ObservationPlaneType.COMBINED,
        frequency_hz=1000.0,
    )
    exterior = SimpleNamespace(
        run_id="run",
        backend_id="beat_cpu",
        sound_speed_m_per_s=343.0,
        traces=SimpleNamespace(
            points_m=np.eye(3),
            triangles=np.asarray([[0, 1, 2]]),
            symmetry="off",
        ),
        boundary_response=lambda *_args: (
            1000.0,
            np.ones(3, dtype=np.complex64),
            np.ones(1, dtype=np.complex64),
        ),
    )
    mesh = SimpleNamespace(points=np.arange(15, dtype=float).reshape(5, 3), n_points=5)
    geometry = _CombinedFieldGeometry(
        mesh=mesh,
        fem_point_ids=np.asarray([0, 2]),
        source_node_ids=np.empty((2, 4), dtype=np.int64),
        interpolation_weights=np.empty((2, 4)),
        exterior_point_ids=np.asarray([1, 4]),
        clipped=None,
    )
    requested = []
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(
            interior=object(),
            exterior=exterior,
            frequencies_for=lambda _plane_type: np.asarray([1000.0]),
        ),
        _field_generation=7,
        _field_synthesis_generation=7,
        _exterior_results={},
        _exterior_pending=set(),
        _exterior_request_meshes={},
        _active_field=None,
        exteriorFieldRequested=SimpleNamespace(emit=lambda task: requested.append(task)),
    )
    editor._exterior_key_for_plane = MethodType(ObservationPlaneViewport._exterior_key_for_plane, editor)

    key = ObservationPlaneViewport._request_exterior_field_for_plane(editor, plane, mesh=geometry)

    assert key is not None
    assert len(requested) == 1
    np.testing.assert_allclose(requested[0].request.points_m, mesh.points[[1, 4]])
    assert editor._exterior_request_meshes[key] is geometry


def test_exterior_frequency_update_uses_cached_field_without_interior_sampling() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _ActiveFieldState

    plane = replace(new_observation_plane("Exterior"), plane_type=ObservationPlaneType.EXTERIOR)
    key = ("field",)
    pressure = np.asarray([1.0 + 2.0j], dtype=np.complex64)
    applied = []
    editor = SimpleNamespace(
        _active_field=_ActiveFieldState(plane.id, SimpleNamespace(), np.zeros(1), "point"),
        _exterior_results=OrderedDict(((key, pressure),)),
        _exterior_key_for_plane=lambda _plane: key,
        _apply_exterior_field_result=lambda candidate, candidate_key, values: (
            applied.append((candidate, candidate_key, values)) or True
        ),
        _field_mesh_and_pressure=lambda _plane: pytest.fail("exterior updates must not sample the FEM field"),
    )

    assert ObservationPlaneViewport._update_active_field(editor, plane)
    assert applied == [(plane, key, pressure)]


def test_exterior_move_stream_applies_compatible_lagged_result_without_caching() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _DragState, _exterior_field_key

    original = replace(
        new_observation_plane("Exterior"),
        plane_type=ObservationPlaneType.EXTERIOR,
        frequency_hz=1000.0,
        width_m=0.1,
        height_m=0.1,
        resolution_m=0.1,
    )
    current = replace(original, center_m=(0.05, 0.0, 0.0))
    completed_key = _exterior_field_key(7, "run", original, 1000.0)
    current_key = _exterior_field_key(7, "run", current, 1000.0)
    requested_mesh = object()
    current_mesh = SimpleNamespace(n_points=4)
    pressure = np.arange(4, dtype=np.complex64)
    applied = []
    editor = SimpleNamespace(
        _field_generation=7,
        _field_synthesis_generation=7,
        _exterior_pending={completed_key},
        _exterior_failures={},
        _exterior_request_meshes={completed_key: requested_mesh},
        _drag=_DragState(original, "move", 0),
        _field_plane=lambda: current,
        _exterior_key_for_plane=lambda _plane: current_key,
        _exterior_plane_mesh=lambda _plane: current_mesh,
        _apply_exterior_field_result=lambda plane, key, values, *, mesh: (
            applied.append((plane, key, values, mesh)) or True
        ),
        _cache_exterior_field_result=lambda *_args: pytest.fail(
            "lagged interactive fields must not enter the exact-position cache"
        ),
    )
    editor._can_stream_exterior_field_result = MethodType(
        ObservationPlaneViewport._can_stream_exterior_field_result,
        editor,
    )

    ObservationPlaneViewport.set_exterior_field_result(editor, completed_key, pressure)

    assert editor._exterior_pending == set()
    assert editor._exterior_request_meshes == {}
    assert len(applied) == 1
    assert applied[0][0] is current
    assert applied[0][1] == completed_key
    assert applied[0][2] is pressure
    assert applied[0][3] is current_mesh


def test_exterior_result_from_prior_channel_synthesis_is_discarded() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    stale_key = (6, "run", "plane")
    editor = SimpleNamespace(
        _field_synthesis_generation=7,
        _exterior_pending={stale_key},
        _exterior_failures={stale_key: "stale"},
        _exterior_request_meshes={stale_key: object()},
    )

    ObservationPlaneViewport.set_exterior_field_result(
        editor,
        stale_key,
        np.ones(4, dtype=np.complex64),
    )

    assert editor._exterior_pending == set()
    assert editor._exterior_failures == {}
    assert editor._exterior_request_meshes == {}


def test_exterior_stream_compatibility_ignores_only_rigid_transform() -> None:
    from blab.ui.observation_plane_viewport import _exterior_field_key, _exterior_field_keys_stream_compatible

    original = replace(
        new_observation_plane("Exterior"),
        plane_type=ObservationPlaneType.EXTERIOR,
        frequency_hz=1000.0,
    )
    moved = replace(original, center_m=(1.0, 2.0, 3.0))
    resized = replace(moved, width_m=moved.width_m * 2.0)
    changed_response = replace(moved, response_id="channel:high")
    changed_type = replace(moved, plane_type=ObservationPlaneType.COMBINED)
    original_key = _exterior_field_key(3, "run", original, 1000.0)

    assert _exterior_field_keys_stream_compatible(
        original_key,
        _exterior_field_key(3, "run", moved, 1000.0),
    )
    assert not _exterior_field_keys_stream_compatible(
        original_key,
        _exterior_field_key(3, "run", resized, 1000.0),
    )
    assert not _exterior_field_keys_stream_compatible(
        original_key,
        _exterior_field_key(3, "run", changed_response, 1000.0),
    )
    assert not _exterior_field_keys_stream_compatible(
        original_key,
        _exterior_field_key(3, "run", changed_type, 1000.0),
    )


def test_exterior_geometry_preview_moves_existing_field_in_place() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _ActiveFieldState

    original = replace(
        new_observation_plane("Exterior"),
        plane_type=ObservationPlaneType.EXTERIOR,
        width_m=0.1,
        height_m=0.1,
        resolution_m=0.1,
    )
    outline = pv.PolyData(original.corners_m, np.asarray((4, 0, 1, 2, 3)))
    field = pv.PolyData(original.corners_m)
    outline_points_address = outline.GetPoints().GetData().GetAddressAsString("")
    field_points_address = field.GetPoints().GetData().GetAddressAsString("")
    pressure = np.arange(4, dtype=np.complex64)
    renders = []
    state = _ActiveFieldState(
        original.id,
        field,
        pressure,
        "point",
        plane=original,
        sample_shape=(2, 2),
    )
    editor = SimpleNamespace(
        _active_field=state,
        _plane_meshes={original.id: outline},
        _rotation_angle_deg=None,
        _refresh_interactive_tools=lambda _plane: None,
        viewer=SimpleNamespace(render=lambda: renders.append(True)),
    )

    updated = original
    for index in range(50):
        updated = replace(original, center_m=(0.004 * index, -0.002 * index, 0.006 * index))
        assert ObservationPlaneViewport._preview_exterior_geometry(editor, updated)

    np.testing.assert_allclose(outline.points, updated.corners_m)
    np.testing.assert_allclose(np.mean(field.points, axis=0), updated.center_m)
    assert outline.GetPoints().GetData().GetAddressAsString("") == outline_points_address
    assert field.GetPoints().GetData().GetAddressAsString("") == field_points_address
    assert state.pressure is pressure
    assert state.plane is updated
    assert len(renders) == 50


def test_interior_field_remains_visible_while_its_plane_is_dragged() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _DragState

    plane = new_observation_plane("Interior")
    rendered = []
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(interior=object()),
        _drag=_DragState(plane, "move", 0),
        _field_message=None,
        _field_mesh_and_pressure=lambda _plane: (object(), np.ones(4), "point", None),
        _add_colored_field_actor=lambda *args, **kwargs: rendered.append((args, kwargs)),
    )

    assert ObservationPlaneViewport._add_interior_field(editor, plane)
    assert len(rendered) == 1


def test_interior_drag_schedules_live_slice_without_rebuilding_on_every_mouse_event() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _DragState

    original = new_observation_plane("Interior")
    updated = replace(original, center_m=(0.0, 0.0, 0.05))
    calls = []
    editor = SimpleNamespace(
        _planes=(original,),
        _field_cache={
            ("smooth", original.id): object(),
            ("element", original.id): object(),
            ("combined", original.id): object(),
        },
        _field_results=SimpleNamespace(interior=object()),
        _drag=_DragState(original, "move", 2),
        _field_plane_id=lambda: original.id,
        _preview_interior_geometry=lambda plane: calls.append(("preview", plane.center_m)) or True,
        _schedule_interior_refresh=lambda: calls.append(("schedule", None)),
        _render=lambda: pytest.fail("raw drag events must use the throttled interior preview path"),
    )

    ObservationPlaneViewport._replace_selected(editor, updated)

    assert editor._planes == (updated,)
    assert editor._field_cache == {}
    assert calls == [("preview", updated.center_m), ("schedule", None)]


def test_exterior_boundary_mask_is_cached_with_viewport_geometry() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(
        new_observation_plane("Exterior"),
        plane_type=ObservationPlaneType.EXTERIOR,
        resolution_m=0.01,
    )
    plane_mesh = pv.PolyData(np.asarray([[0.25, 0.25, 0.0], [0.25, 0.25, 1.0]]))
    results = SimpleNamespace(
        run_id="run",
        traces=SimpleNamespace(
            points_m=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            triangles=np.asarray([[0, 1, 2]]),
            symmetry="off",
        ),
    )
    editor = SimpleNamespace(_field_generation=3, _field_cache={})
    editor._exterior_boundary_mesh = MethodType(ObservationPlaneViewport._exterior_boundary_mesh, editor)

    first = ObservationPlaneViewport._exterior_boundary_mask(editor, plane, plane_mesh, results)
    second = ObservationPlaneViewport._exterior_boundary_mask(editor, plane, plane_mesh, results)

    assert first.tolist() == [True, False]
    assert second is first


def test_exterior_boundary_mask_is_deferred_until_interaction_finishes() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _DragState

    plane = replace(new_observation_plane("Exterior"), plane_type=ObservationPlaneType.EXTERIOR)
    pressure = np.asarray([1.0 + 2.0j, 3.0 + 4.0j], dtype=np.complex64)
    masked = np.asarray([np.nan + 1j * np.nan, 3.0 + 4.0j], dtype=np.complex64)
    calls = []
    editor = SimpleNamespace(
        _drag=_DragState(plane, "move", 0),
        _masked_exterior_pressure=lambda candidate, mesh, values: calls.append((candidate, mesh, values)) or masked,
    )

    interactive = ObservationPlaneViewport._exterior_pressure_for_display(
        editor,
        plane,
        object(),
        pressure,
    )
    assert interactive is pressure
    assert calls == []

    editor._drag = None
    final = ObservationPlaneViewport._exterior_pressure_for_display(
        editor,
        plane,
        object(),
        pressure,
    )
    assert final is masked
    assert len(calls) == 1


def test_field_preferences_resize_result_cache_and_translation_timer() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    class Timer:
        interval_ms = 0
        active = False
        starts = 0

        def setInterval(self, interval_ms) -> None:  # noqa: N802 - Qt-compatible stub
            self.interval_ms = interval_ms

        def interval(self) -> int:
            return self.interval_ms

        def isActive(self) -> bool:  # noqa: N802 - Qt-compatible stub
            return self.active

        def start(self) -> None:
            self.starts += 1
            self.active = True

    item_size = 10 * 1024 * 1024
    timer = Timer()
    interior_timer = Timer()
    editor = SimpleNamespace(
        _exterior_results=OrderedDict(
            (
                (("old",), SimpleNamespace(nbytes=item_size)),
                (("new",), SimpleNamespace(nbytes=item_size)),
            )
        ),
        _exterior_result_bytes=2 * item_size,
        _exterior_result_cache_max_bytes=128 * 1024 * 1024,
        _exterior_refresh_timer=timer,
        _interior_refresh_timer=interior_timer,
    )
    editor._trim_exterior_result_cache = MethodType(ObservationPlaneViewport._trim_exterior_result_cache, editor)

    ObservationPlaneViewport.set_field_preferences(
        editor,
        cache_size_mb=16,
        translation_target_fps=25.0,
    )

    assert tuple(editor._exterior_results) == (("new",),)
    assert editor._exterior_result_bytes == item_size
    assert editor._exterior_result_cache_max_bytes == 16 * 1024 * 1024
    assert timer.interval_ms == 40
    assert interior_timer.interval_ms == 40

    ObservationPlaneViewport._schedule_exterior_refresh(editor)
    ObservationPlaneViewport._schedule_exterior_refresh(editor)
    assert timer.starts == 1
    timer.active = False
    ObservationPlaneViewport._schedule_exterior_refresh(editor)
    assert timer.starts == 2


def test_field_scalar_projection_uses_stable_ranges_and_animation_phase() -> None:
    from blab.ui.observation_plane_results import normalized_spl_reference_db, project_field_scalars

    pressure = np.asarray([1.0 + 0.0j, 0.0 + 1.0j])
    normalized = project_field_scalars(pressure, ObservationPlaneDisplay.NORMALIZED_SPL)
    animated = project_field_scalars(
        pressure,
        ObservationPlaneDisplay.SPL,
        animation_phase_deg=90.0,
    )
    phase = project_field_scalars(pressure, ObservationPlaneDisplay.PHASE)

    assert normalized.clim == (-40.0, 0.0)
    np.testing.assert_allclose(normalized.values, [0.0, 0.0])
    np.testing.assert_allclose(animated.values, [0.0, 1.0], atol=1e-12)
    assert animated.clim == (-1.0, 1.0)
    np.testing.assert_allclose(phase.values, [0.0, -90.0], atol=1e-12)

    stronger_pressure = np.asarray([2.0 + 0.0j, 0.0 + 1.0j])
    relative_to_initial = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.NORMALIZED_SPL,
        normalized_reference_db=normalized_spl_reference_db(pressure),
    )
    renormalized_at_final = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.NORMALIZED_SPL,
    )
    np.testing.assert_allclose(relative_to_initial.values, [20.0 * np.log10(2.0), 0.0])
    np.testing.assert_allclose(renormalized_at_final.values, [0.0, -20.0 * np.log10(2.0)])
    assert relative_to_initial.clim == renormalized_at_final.clim == (-40.0, 0.0)

    at_zero = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.SPL,
        animation_phase_deg=0.0,
    )
    at_ninety = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.SPL,
        animation_phase_deg=90.0,
    )
    assert at_zero.clim == at_ninety.clim == (-2.0, 2.0)

    manual = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.REAL_PRESSURE,
        pressure_color_limit_pa=0.25,
    )
    animated_manual = project_field_scalars(
        stronger_pressure,
        ObservationPlaneDisplay.SPL,
        animation_phase_deg=90.0,
        pressure_color_limit_pa=0.5,
    )
    assert manual.clim == (-0.25, 0.25)
    assert animated_manual.clim == (-0.5, 0.5)

    velocity = np.asarray([[3.0, 4.0, 0.0], [3.0j, 4.0j, 0.0]], dtype=np.complex64)
    velocity_magnitude = project_field_scalars(
        velocity,
        ObservationPlaneDisplay.PARTICLE_VELOCITY,
    )
    velocity_at_zero = project_field_scalars(
        velocity,
        ObservationPlaneDisplay.PARTICLE_VELOCITY,
        animation_phase_deg=0.0,
    )
    velocity_at_ninety = project_field_scalars(
        velocity,
        ObservationPlaneDisplay.PARTICLE_VELOCITY,
        animation_phase_deg=90.0,
    )
    np.testing.assert_allclose(velocity_magnitude.values, [5.0, 5.0])
    np.testing.assert_allclose(velocity_at_zero.values, [5.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(velocity_at_ninety.values, [0.0, 5.0], atol=1e-6)
    assert velocity_magnitude.clim == velocity_at_zero.clim == velocity_at_ninety.clim == (0.0, 35.0)
    assert velocity_magnitude.title == "Particle Velocity Magnitude (m/s)"


def test_move_drag_freezes_then_releases_normalized_spl_reference() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _ActiveFieldState

    plane = replace(
        new_observation_plane("Normalized"),
        display=ObservationPlaneDisplay.NORMALIZED_SPL,
    )
    initial_pressure = np.asarray([1.0 + 0.0j, 1.0 + 0.0j])
    moved_pressure = np.asarray([2.0 + 0.0j, 1.0 + 0.0j])
    position = SimpleNamespace(x=lambda: 10.0, y=lambda: 20.0)
    editor = SimpleNamespace(
        _active_field=_ActiveFieldState(plane.id, object(), initial_pressure, "point"),
        _selected_plane=lambda: plane,
        _axis_parameter=lambda *_args: 0.0,
        _world_per_display_pixel=lambda _center: 0.001,
    )
    editor._capture_normalized_spl_reference = MethodType(
        ObservationPlaneViewport._capture_normalized_spl_reference,
        editor,
    )

    ObservationPlaneViewport._begin_drag(editor, ("move", 0), position)

    assert editor._drag is not None
    initial_reference = editor._drag.normalized_reference_db
    assert initial_reference is not None
    during_drag = ObservationPlaneViewport._project_field_scalars(editor, moved_pressure, plane)
    np.testing.assert_allclose(during_drag.values, [20.0 * np.log10(2.0), 0.0])

    editor._drag = None
    after_release = ObservationPlaneViewport._project_field_scalars(editor, moved_pressure, plane)
    np.testing.assert_allclose(after_release.values, [0.0, -20.0 * np.log10(2.0)])


@pytest.mark.parametrize("association", ["point", "cell"])
def test_animation_frame_updates_scalars_without_rebuilding_scene(association: str) -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import (
        FIELD_SCALAR_NAME,
        ObservationPlaneViewport,
        _ActiveFieldState,
    )

    plane = new_observation_plane("Animated")
    pressure = np.asarray([2.0 + 0.0j, 0.0 + 1.0j])
    mesh = pv.PolyData(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    attributes = mesh.cell_data if association == "cell" else mesh.point_data
    attributes[FIELD_SCALAR_NAME] = np.asarray([2.0, 0.0])
    viewer = SimpleNamespace(render_calls=0)
    viewer.render = lambda: setattr(viewer, "render_calls", viewer.render_calls + 1)
    editor = SimpleNamespace(
        _active_field=_ActiveFieldState(plane.id, mesh, pressure, association),
        _animation_phase_deg=90.0,
        viewer=viewer,
    )

    updated = ObservationPlaneViewport._update_animation_frame(editor, plane)

    assert updated
    np.testing.assert_allclose(attributes[FIELD_SCALAR_NAME], [0.0, 1.0], atol=1e-12)
    assert viewer.render_calls == 1


def test_frequency_update_reuses_active_field_actor_and_updates_scalar_range() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import (
        FIELD_SCALAR_NAME,
        ObservationPlaneViewport,
        _ActiveFieldState,
    )

    class Mapper:
        scalar_range = None

        def SetScalarRange(self, minimum, maximum) -> None:  # noqa: N802 - VTK-compatible stub
            self.scalar_range = (minimum, maximum)

    mapper = Mapper()
    actor = SimpleNamespace(GetMapper=lambda: mapper)
    plane = replace(
        new_observation_plane("Frequency"),
        display=ObservationPlaneDisplay.REAL_PRESSURE,
        pressure_color_limit_pa=0.25,
    )
    mesh = pv.PolyData(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    mesh.point_data[FIELD_SCALAR_NAME] = np.zeros(2)
    pressure = np.asarray([2.0 + 1.0j, -1.0 - 3.0j])
    viewer = SimpleNamespace(render_calls=0)
    viewer.render = lambda: setattr(viewer, "render_calls", viewer.render_calls + 1)
    editor = SimpleNamespace(
        _active_field=_ActiveFieldState(plane.id, mesh, np.zeros(2), "point", actor),
        _field_mesh_and_pressure=lambda _plane: (mesh, pressure, "point", None),
        _animation_plane_id=None,
        _animation_phase_deg=0.0,
        viewer=viewer,
    )

    updated = ObservationPlaneViewport._update_active_field(editor, plane)

    assert updated
    np.testing.assert_allclose(mesh.point_data[FIELD_SCALAR_NAME], [2.0, -1.0])
    assert mapper.scalar_range == (-0.25, 0.25)
    assert viewer.render_calls == 1


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


def test_particle_velocity_slice_reflects_vector_components_for_symmetry() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(
        new_observation_plane("Velocity"),
        display=ObservationPlaneDisplay.PARTICLE_VELOCITY,
        interior_rendering=InteriorRenderingMode.ELEMENT_FIELD,
    )
    mesh = object()
    geometry = SimpleNamespace(
        mesh=mesh,
        source_tetrahedron_ids=np.asarray([0, 0]),
        vector_signs=np.asarray([[1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]),
    )
    results = SimpleNamespace(
        points_m=np.empty((0, 3)),
        tetrahedra=np.asarray([[0, 1, 2, 3]]),
        symmetry="x",
        particle_velocity=lambda _frequency, _response: np.asarray([[1.0, 2.0, 3.0]], dtype=np.complex64),
        pressure=lambda *_args: pytest.fail("particle velocity must not use pressure interpolation"),
    )
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(interior=results),
        _element_field_mesh=lambda *_args: geometry,
    )

    result_mesh, velocity, association, clipped = ObservationPlaneViewport._field_mesh_and_pressure(
        editor,
        plane,
    )

    assert result_mesh is mesh
    assert association == "cell"
    assert clipped is None
    np.testing.assert_allclose(velocity, [[1.0, 2.0, 3.0], [-1.0, 2.0, 3.0]])


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

    smooth = ObservationPlaneViewport._smooth_field_mesh(
        editor_stub,
        plane,
        points,
        tetrahedra,
        "off",
    )
    element = ObservationPlaneViewport._element_field_mesh(
        editor_stub,
        replace(plane, interior_rendering=InteriorRenderingMode.ELEMENT_FIELD),
        points,
        tetrahedra,
        "off",
    )
    sampled_pressure = np.sum(
        pressure[smooth.source_node_ids] * smooth.interpolation_weights,
        axis=1,
    )
    source_pressure = np.mean(pressure[tetrahedra], axis=1)
    element_pressure = source_pressure[element.source_tetrahedron_ids]

    assert smooth.mesh.n_cells == 4
    assert smooth.mesh.n_points == 9
    np.testing.assert_allclose(sampled_pressure[4], 2.2, rtol=1e-6)
    assert smooth.clipped.n_cells > 0
    assert element.mesh.n_cells > 0
    np.testing.assert_allclose(element_pressure, 2.5)

    cached_smooth = ObservationPlaneViewport._smooth_field_mesh(
        editor_stub,
        replace(plane, frequency_hz=1000.0, response_id="channel:main"),
        points,
        tetrahedra,
        "off",
    )
    cached_element = ObservationPlaneViewport._element_field_mesh(
        editor_stub,
        replace(
            plane,
            interior_rendering=InteriorRenderingMode.ELEMENT_FIELD,
            frequency_hz=1000.0,
            response_id="channel:main",
        ),
        points,
        tetrahedra,
        "off",
    )
    assert cached_smooth is smooth
    assert cached_element is element


def test_combined_geometry_classifies_fem_exterior_and_solid_points() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(
        new_observation_plane("Combined"),
        plane_type=ObservationPlaneType.COMBINED,
        center_m=(0.0, 0.0, 0.0),
        width_m=3.0,
        height_m=3.0,
        resolution_m=0.25,
    )
    interior = SimpleNamespace(
        run_id="run",
        symmetry="off",
        points_m=np.asarray(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.0, 0.5, -0.5],
                [0.0, 0.0, 0.5],
            ]
        ),
        tetrahedra=np.asarray([[0, 1, 2, 3]]),
    )
    exterior = SimpleNamespace(run_id="run", traces=SimpleNamespace(symmetry="off"))
    boundary = pv.Cube(bounds=(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)).triangulate()
    editor = SimpleNamespace(
        _field_generation=1,
        _field_cache={},
        _exterior_boundary_mesh=lambda _results: boundary,
    )
    editor._combined_boundary_mesh = MethodType(ObservationPlaneViewport._combined_boundary_mesh, editor)

    geometry = ObservationPlaneViewport._combined_field_geometry(editor, plane, interior, exterior)

    acoustic_points = geometry.fem_point_ids.size + geometry.exterior_point_ids.size
    assert geometry.fem_point_ids.size > 0
    assert geometry.exterior_point_ids.size > 0
    assert acoustic_points < geometry.mesh.n_points
    assert 0 < geometry.mesh.n_cells < (plane.sample_shape[0] - 1) * (plane.sample_shape[1] - 1)
    assert ObservationPlaneViewport._combined_field_geometry(editor, plane, interior, exterior) is geometry


def test_combined_boundary_rejects_open_bem_geometry() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    boundary = pv.PolyData(
        np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        np.asarray([3, 0, 1, 2]),
    )
    results = SimpleNamespace(run_id="run", traces=SimpleNamespace(symmetry="off"))
    editor = SimpleNamespace(
        _field_generation=1,
        _field_cache={},
        _exterior_boundary_mesh=lambda _results: boundary,
    )

    with pytest.raises(ValueError, match="not closed and manifold"):
        ObservationPlaneViewport._combined_boundary_mesh(editor, results)


def test_combined_pressure_merges_complex_fem_and_exterior_values_before_projection() -> None:
    import pyvista as pv

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _CombinedFieldGeometry

    plane = replace(new_observation_plane("Combined"), plane_type=ObservationPlaneType.COMBINED)
    geometry = _CombinedFieldGeometry(
        mesh=pv.PolyData(np.zeros((4, 3))),
        fem_point_ids=np.asarray([0, 1]),
        source_node_ids=np.asarray([[0, 1, 2, 3], [0, 1, 2, 3]]),
        interpolation_weights=np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5]]),
        exterior_point_ids=np.asarray([2]),
        clipped=None,
    )
    interior = SimpleNamespace(
        pressure=lambda _frequency, _response: np.asarray([1.0 + 2.0j, 3.0, 5.0 + 1.0j, 7.0 + 3.0j])
    )
    editor = SimpleNamespace(
        _field_results=SimpleNamespace(interior=interior),
        _exterior_key_for_plane=lambda _plane: (None,) * 8 + (1000.0,),
    )

    pressure = ObservationPlaneViewport._combined_pressure(
        editor,
        plane,
        geometry,
        np.asarray([11.0 + 13.0j], dtype=np.complex64),
    )

    np.testing.assert_allclose(pressure[:3], [1.0 + 2.0j, 6.0 + 2.0j, 11.0 + 13.0j])
    assert np.isnan(pressure[3].real) and np.isnan(pressure[3].imag)


def test_result_selection_changes_use_fast_viewport_update() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    original = replace(new_observation_plane("Slice"), frequency_hz=100.0)
    updated = replace(original, frequency_hz=200.0)
    calls = []
    editor = SimpleNamespace(
        _planes=(original,),
        _selected_id=original.id,
        _active_id=None,
        _field_cache={},
        _mode=None,
        _selected_plane=lambda: updated,
        _field_plane=lambda: updated,
        _update_active_field=lambda plane: calls.append(("field", plane.frequency_hz)) or True,
        _render=lambda: calls.append(("render", None)),
    )

    ObservationPlaneViewport.set_planes(editor, (updated,), selected_id=updated.id)

    assert calls == [("field", 200.0)]

    speed_only = replace(updated, animation_speed_hz=2.0)
    ObservationPlaneViewport.set_planes(editor, (speed_only,), selected_id=speed_only.id)
    assert calls == [("field", 200.0)]

    manual_range = replace(speed_only, pressure_color_limit_pa=0.1)
    ObservationPlaneViewport.set_planes(editor, (manual_range,), selected_id=manual_range.id)
    assert calls == [("field", 200.0), ("field", 200.0)]

    active_update = replace(manual_range, frequency_hz=300.0)
    editor._selected_id = None
    editor._active_id = manual_range.id
    editor._field_plane = lambda: active_update
    ObservationPlaneViewport.set_planes(editor, (active_update,))
    assert calls == [("field", 200.0), ("field", 200.0), ("field", 300.0)]


def test_plane_sync_prunes_only_plane_owned_field_cache_entries() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = new_observation_plane("Slice")
    shared_volume = object()
    shared_boundary = object()
    retained_plane_geometry = object()
    editor = SimpleNamespace(
        _planes=(plane,),
        _selected_id=plane.id,
        _active_id=None,
        _mode=None,
        _field_cache={
            ("fem-volume", "off"): shared_volume,
            ("exterior-boundary", "run"): shared_boundary,
            ("smooth", plane.id): retained_plane_geometry,
            ("smooth", "deleted-plane"): object(),
        },
    )

    ObservationPlaneViewport.set_planes(editor, (plane,), selected_id=plane.id)

    assert editor._field_cache == {
        ("fem-volume", "off"): shared_volume,
        ("exterior-boundary", "run"): shared_boundary,
        ("smooth", plane.id): retained_plane_geometry,
    }


def test_active_plane_remains_field_plane_after_camera_deselect() -> None:
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = new_observation_plane("Pinned")
    renders = []
    editor = SimpleNamespace(
        _planes=(plane,),
        _selected_id=plane.id,
        _active_id=None,
        _mode=None,
        _drag=None,
        _rotation_angle_deg=None,
        _render=lambda: renders.append(True),
    )
    editor._field_plane_id = lambda: ObservationPlaneViewport._field_plane_id(editor)

    ObservationPlaneViewport.set_active_plane(editor, plane.id)
    assert not renders  # Pinning the already displayed plane needs no redraw.
    ObservationPlaneViewport._select(editor, None)

    assert editor._selected_id is None
    assert ObservationPlaneViewport._field_plane(editor) is plane
    assert len(renders) == 1


def test_active_plane_animation_continues_without_selection(monkeypatch) -> None:
    import blab.ui.observation_plane_viewport as viewport_module
    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(new_observation_plane("Pinned"), animation_speed_hz=2.0)
    updates = []
    editor = SimpleNamespace(
        _planes=(plane,),
        _selected_id=None,
        _active_id=plane.id,
        _animation_plane_id=plane.id,
        _field_results=object(),
        _animation_updated_at=10.0,
        _animation_phase_deg=0.0,
        _update_animation_frame=lambda candidate: updates.append(candidate.id) or True,
        set_animation=lambda *_args: pytest.fail("active animation should not stop"),
    )
    monkeypatch.setattr(viewport_module.time, "monotonic", lambda: 10.25)

    ObservationPlaneViewport._advance_animation(editor)

    assert updates == [plane.id]
    assert editor._animation_phase_deg == 180.0


def test_viewport_uses_maya_tool_shortcuts_and_a_shared_foreground_renderer(qapp) -> None:
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
        assert viewer.cleared_keys == ["w", "e", "r"]
        assert set(viewer.key_callbacks) == {"w", "e", "r"}
        assert viewer.key_callbacks["w"].__func__ is ObservationPlaneViewport._activate_move_mode
        assert viewer.key_callbacks["e"].__func__ is ObservationPlaneViewport._activate_rotate_mode
        assert viewer.key_callbacks["r"].__func__ is ObservationPlaneViewport._activate_scale_mode
    finally:
        viewer.close()
        viewer.deleteLater()


def test_qt_tool_shortcuts_select_move_rotate_and_scale_modes() -> None:
    from PySide6.QtCore import Qt

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    selected_modes = []
    editor = SimpleNamespace(
        _selected_id="plane",
        _activate_move_mode=lambda: selected_modes.append("move"),
        _activate_rotate_mode=lambda: selected_modes.append("rotate"),
        _activate_scale_mode=lambda: selected_modes.append("size"),
    )

    for key in (Qt.Key.Key_W, Qt.Key.Key_E, Qt.Key.Key_R):
        event = SimpleNamespace(key=lambda key=key: key)
        assert ObservationPlaneViewport._on_key_press(editor, event)

    assert selected_modes == ["move", "rotate", "size"]


def test_move_gizmo_refresh_reuses_actor_instances_and_updates_transforms() -> None:
    import vtk

    from blab.ui.observation_plane_viewport import ObservationPlaneViewport

    plane = replace(new_observation_plane("Plane"), center_m=(1.0, 2.0, 3.0))
    actors = {("move", index): vtk.vtkActor() for index in range(3)}
    editor = SimpleNamespace(
        _mode="move",
        _tool_actor_by_control=actors,
        vtk=vtk,
        _tool_scale=lambda _plane: 0.25,
        _clear_interactive_tools=lambda: pytest.fail("existing gizmos must not be rebuilt"),
        _add_move_gizmo=lambda _plane: pytest.fail("existing gizmos must not be rebuilt"),
    )
    editor._set_tool_actor_transform = MethodType(ObservationPlaneViewport._set_tool_actor_transform, editor)
    editor._update_move_gizmo = MethodType(ObservationPlaneViewport._update_move_gizmo, editor)

    original_actor_ids = {control: id(actor) for control, actor in actors.items()}
    ObservationPlaneViewport._refresh_interactive_tools(editor, plane)

    assert {control: id(actor) for control, actor in actors.items()} == original_actor_ids
    x_matrix = actors[("move", 0)].GetUserMatrix()
    np.testing.assert_allclose(
        [x_matrix.GetElement(row, 3) for row in range(3)],
        plane.center_m,
    )
    np.testing.assert_allclose(
        [x_matrix.GetElement(row, 0) for row in range(3)],
        np.asarray(plane.local_axes[0]) * 0.25,
    )

    moved = replace(plane, center_m=(-1.0, 0.5, 4.0))
    ObservationPlaneViewport._refresh_interactive_tools(editor, moved)
    assert {control: id(actor) for control, actor in actors.items()} == original_actor_ids
    x_matrix = actors[("move", 0)].GetUserMatrix()
    np.testing.assert_allclose(
        [x_matrix.GetElement(row, 3) for row in range(3)],
        moved.center_m,
    )


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


def test_properties_window_is_modeless_and_commits_or_reverts_live_preview(main_window, qapp) -> None:
    main_window.preview.newObservationPlaneRequested.emit({})
    original = main_window.project.observation_planes[0]
    controller = main_window.observation_plane_controller

    main_window.preview.observationPlanePropertiesRequested.emit(original.id)
    dialog = controller._dialogs[original.id]
    assert dialog.isVisible()
    assert not dialog.isModal()
    main_window.preview.observationPlanePropertiesRequested.emit(original.id)
    assert controller._dialogs[original.id] is dialog

    dialog.name_edit.setText("Accepted Slice")
    assert main_window.preview.observation_planes[0].name == "Accepted Slice"
    assert main_window.project.observation_planes[0].name == original.name
    dialog.accept()
    qapp.processEvents()
    assert main_window.project.observation_planes[0].name == "Accepted Slice"

    main_window.preview.observationPlanePropertiesRequested.emit(original.id)
    dialog = controller._dialogs[original.id]
    dialog.name_edit.setText("Rejected Slice")
    assert main_window.preview.observation_planes[0].name == "Rejected Slice"
    dialog.reject()
    qapp.processEvents()
    assert main_window.project.observation_planes[0].name == "Accepted Slice"
    assert main_window.preview.observation_planes[0].name == "Accepted Slice"


def test_result_sync_preserves_modeless_dialog_frequency_preview(main_window, qapp) -> None:
    main_window.preview.newObservationPlaneRequested.emit({})
    persisted = main_window.project.observation_planes[0]
    controller = main_window.observation_plane_controller
    results = SimpleNamespace(
        response_options=(("system", "System Response"),),
        frequencies_for=lambda _plane_type: np.asarray([100.0, 1000.0, 10_000.0]),
    )
    controller._field_results = lambda: results

    main_window.preview.observationPlanePropertiesRequested.emit(persisted.id)
    dialog = controller._dialogs[persisted.id]
    dialog.frequency_slider.setValue(2)
    QTest.qWait(20)
    qapp.processEvents()

    assert main_window.project.observation_planes[0].frequency_hz is None
    assert main_window.preview.observation_planes[0].frequency_hz == 10_000.0

    controller.sync_view()

    assert dialog.plane.frequency_hz == 10_000.0
    assert main_window.preview.observation_planes[0].frequency_hz == 10_000.0
    assert main_window.preview.active_observation_plane_id == persisted.id
    assert main_window.project.observation_planes[0].frequency_hz is None


def test_opening_properties_automatically_pins_one_plane_at_a_time(main_window) -> None:
    main_window.preview.newObservationPlaneRequested.emit({})
    main_window.preview.newObservationPlaneRequested.emit({})
    first, second = main_window.project.observation_planes
    controller = main_window.observation_plane_controller

    main_window.preview.observationPlanePropertiesRequested.emit(first.id)
    first_dialog = controller._dialogs[first.id]
    assert controller._active_plane_id == first.id
    assert main_window.preview.active_observation_plane_id == first.id
    first_dialog.name_edit.setText("Pinned Plane")
    assert main_window.preview.selected_observation_plane_id is None

    main_window.preview.observationPlanePropertiesRequested.emit(second.id)
    assert controller._dialogs[second.id].isVisible()
    assert controller._active_plane_id == second.id
    assert main_window.preview.active_observation_plane_id == second.id
    assert not hasattr(first_dialog, "active_check")
    assert not hasattr(first, "active")
