from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from blab.observation_planes import (
    InteriorRenderingMode,
    ObservationPlane,
    ObservationPlaneType,
    new_observation_plane,
)
from blab.observation_planes import (
    ObservationPlaneDisplay as Display,
)
from blab.ui.observation_plane_dialog import ObservationPlanePropertiesDialog
from blab.ui.observation_plane_results import project_field_scalars
from blab.ui.observation_plane_viewport import ObservationPlaneViewport, _result_selection_update


@pytest.mark.parametrize("display", list(Display))
def test_dialog_shows_only_applicable_rows(qapp, display):
    dialog = ObservationPlanePropertiesDialog(
        replace(new_observation_plane("Controls"), display=display),
        solved_frequencies_hz=(100, 200),
    )
    try:
        assert dialog.pressure_range_auto_check.isHidden() == (
            display not in {Display.REAL_PRESSURE, Display.IMAGINARY_PRESSURE}
        )
        assert dialog.spl_range_auto_check.isHidden() == (display != Display.SPL)
        assert dialog.normalized_range_spin.isHidden() == (display != Display.NORMALIZED_SPL)
        assert dialog.particle_velocity_scale_auto_check.isHidden() == (display != Display.PARTICLE_VELOCITY)
        assert dialog.intensity_scale_auto_check.isHidden() == (display != Display.FLOW_ARROWS)
        assert dialog.velocity_scale_auto_check.isHidden()
        assert dialog.arrow_density_slider.isHidden() == (display != Display.FLOW_ARROWS)
        assert dialog.animate_button.isHidden() == (not display.supports_cycle_animation)
        assert dialog.animate_button.isEnabled() == display.supports_cycle_animation
        assert dialog.animate_button.text() == "Animate Cycle"
        assert dialog.animation_speed_slider.isHidden()
        assert dialog.frequency_sweep_speed_slider.isHidden()
        if display.supports_cycle_animation:
            dialog.animate_button.setChecked(True)
            assert not dialog.animation_speed_slider.isHidden()
            if display == Display.FLOW_ARROWS:
                assert dialog.intensity_scale_auto_check.isHidden()
                assert not dialog.velocity_scale_auto_check.isHidden()
        dialog.frequency_sweep_button.setChecked(True)
        assert not dialog.frequency_sweep_speed_slider.isHidden()
        assert not dialog.animate_button.isChecked()
        assert dialog.animation_speed_slider.isHidden()
    finally:
        dialog.close()


@pytest.mark.parametrize("display", [Display.SPL, Display.NORMALIZED_SPL, Display.PHASE])
def test_switching_display_stops_cycle_and_keeps_frequency_sweep_available(qapp, display):
    dialog = ObservationPlanePropertiesDialog(
        replace(new_observation_plane("Controls"), display=Display.FLOW_ARROWS),
        solved_frequencies_hz=(100, 200),
    )
    previews = []
    dialog.previewChanged.connect(lambda plane, animate: previews.append((plane, animate)))
    try:
        dialog.animate_button.setChecked(True)
        dialog.display_combo.setCurrentIndex(dialog.display_combo.findData(display.value))
        assert not dialog.animate_button.isChecked()
        assert dialog.animate_button.isHidden()
        assert previews[-1][0].display == display
        assert previews[-1][1] is False
        assert dialog.frequency_sweep_button.isEnabled()
        dialog.set_solved_results((300, 400), ())
        assert not dialog.animate_button.isEnabled()
        assert dialog.animate_button.isHidden()
    finally:
        dialog.close()


def test_hidden_scales_keep_values_and_roundtrip(qapp):
    dialog = ObservationPlanePropertiesDialog(new_observation_plane("Controls"))
    try:
        dialog.spl_range_auto_check.setChecked(False)
        dialog.spl_min_spin.setValue(65)
        dialog.spl_max_spin.setValue(115)
        dialog.display_combo.setCurrentIndex(dialog.display_combo.findData(Display.PARTICLE_VELOCITY.value))
        dialog.particle_velocity_scale_auto_check.setChecked(False)
        dialog.particle_velocity_scale_spin.setValue(0.2)
        dialog.display_combo.setCurrentIndex(dialog.display_combo.findData(Display.NORMALIZED_SPL.value))
        dialog.normalized_range_spin.setValue(25)
        stored = dialog.plane
        assert stored.spl_min_db == 65 and stored.spl_max_db == 115
        assert stored.particle_velocity_limit_m_per_s == 0.2
        assert stored.normalized_spl_range_db == 25
        assert ObservationPlane.from_payload(stored.to_payload()) == stored
        dialog.display_combo.setCurrentIndex(dialog.display_combo.findData(Display.SPL.value))
        assert dialog.spl_min_spin.value() == 65
        assert dialog.spl_max_spin.value() == 115
        assert not dialog.spl_range_auto_check.isChecked()
    finally:
        dialog.close()


def test_geometry_controls_follow_plane_and_rendering_type(qapp):
    dialog = ObservationPlanePropertiesDialog(new_observation_plane("Controls"))
    try:
        dialog.rendering_combo.setCurrentIndex(
            dialog.rendering_combo.findData(InteriorRenderingMode.ELEMENT_FIELD.value)
        )
        assert dialog.resolution_spin.isHidden()
        assert not dialog.invert_clip_check.isHidden()
        assert dialog.invert_clip_check.isEnabled()
        original_side = dialog.plane.invert_clip_side
        dialog.invert_clip_check.setChecked(not original_side)
        assert dialog.plane.invert_clip_side is not original_side
        dialog.type_combo.setCurrentIndex(dialog.type_combo.findData(ObservationPlaneType.EXTERIOR.value))
        assert dialog.rendering_combo.isHidden()
        assert dialog.invert_clip_check.isHidden()
        assert not dialog.resolution_spin.isHidden()
        dialog.type_combo.setCurrentIndex(dialog.type_combo.findData(ObservationPlaneType.COMBINED.value))
        assert dialog.rendering_combo.isHidden()
        assert not dialog.invert_clip_check.isHidden()
    finally:
        dialog.close()


def test_scale_projection_and_rms_animation():
    pressure = np.array([2.0, 1j])
    spl = project_field_scalars(pressure, Display.SPL, spl_min_db=70, spl_max_db=110)
    assert spl.clim == (70, 110)
    normalized = project_field_scalars(pressure, Display.NORMALIZED_SPL, normalized_spl_range_db=25)
    assert normalized.clim == (-25, 0)
    np.testing.assert_allclose(normalized.values, [0, -20 * np.log10(2)], atol=1e-5)
    velocity = np.array([[0.1, 0.2j, 0]])
    static = project_field_scalars(velocity, Display.PARTICLE_VELOCITY)
    assert static.clim == pytest.approx((0, np.sqrt(0.05)))
    for phase in (0, 90, 180):
        animated = project_field_scalars(velocity, Display.PARTICLE_VELOCITY, animation_phase_deg=phase)
        assert animated.clim == pytest.approx((0, np.sqrt(0.1)))
        manual = project_field_scalars(
            velocity,
            Display.PARTICLE_VELOCITY,
            animation_phase_deg=phase,
            particle_velocity_limit_m_per_s=0.5,
        )
        assert manual.clim == (0, 0.5)
    pressure_frame = project_field_scalars(pressure, Display.REAL_PRESSURE, animation_phase_deg=0)
    np.testing.assert_allclose(pressure_frame.values, [2 * np.sqrt(2), 0])


@pytest.mark.parametrize("display", [Display.SPL, Display.NORMALIZED_SPL, Display.PHASE])
def test_unsupported_cycle_is_ignored_by_projection_and_rejected_by_viewport(display):
    pressure = np.array([2, 1j])
    static = project_field_scalars(pressure, display)
    attempted = project_field_scalars(pressure, display, animation_phase_deg=90)
    np.testing.assert_array_equal(static.values, attempted.values)
    assert static.clim == attempted.clim
    plane = replace(new_observation_plane("Controls"), display=display)
    editor = SimpleNamespace(
        _planes=(plane,),
        _animation_plane_id=None,
        _field_results=object(),
        _animation_timer=SimpleNamespace(isActive=lambda: False),
    )
    ObservationPlaneViewport.set_animation(editor, plane.id, True)
    assert editor._animation_plane_id is None


@pytest.mark.parametrize(
    "changes",
    [
        {"spl_min_db": 10},
        {"spl_min_db": 100, "spl_max_db": 50},
        {"spl_min_db": float("nan"), "spl_max_db": 50},
        {"normalized_spl_range_db": 0},
        {"particle_velocity_limit_m_per_s": -1},
    ],
)
def test_invalid_scale_settings_are_rejected(changes):
    with pytest.raises(ValueError):
        replace(new_observation_plane("Controls"), **changes).validated()


def test_new_scale_defaults_and_fast_updates():
    plane = new_observation_plane("Controls")
    legacy = plane.to_payload()
    for field in ("spl_min_db", "spl_max_db", "normalized_spl_range_db", "particle_velocity_limit_m_per_s"):
        del legacy[field]
    restored = ObservationPlane.from_payload(legacy)
    assert restored == plane
    updated = replace(plane, spl_min_db=70, spl_max_db=110)
    assert _result_selection_update((plane,), (updated,), plane.id, plane.id, None) == "field"


def test_viewport_stops_running_cycle_when_display_changes():
    original = replace(new_observation_plane("Controls"), display=Display.REAL_PRESSURE)
    updated = replace(original, display=Display.SPL)
    stopped = []
    editor = SimpleNamespace(
        _planes=(original,),
        _animation_plane_id=original.id,
        _animation_phase_deg=90,
        _animation_timer=SimpleNamespace(stop=lambda: stopped.append(True)),
        _selected_id=original.id,
        _active_id=None,
        _field_cache={},
        _render=lambda: None,
    )
    ObservationPlaneViewport.set_planes(editor, (updated,), selected_id=updated.id)
    assert stopped == [True]
    assert editor._animation_plane_id is None
    assert editor._animation_phase_deg == 0
