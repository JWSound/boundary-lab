"""Contour action gating. Pure logic: no Qt, no MainWindow."""

from __future__ import annotations

import pytest

from blab.ui.main_window.plot_controls import contour_controls

READY = dict(
    has_live_data=True,
    final_resolution_active=True,
    final_plots_rendered=True,
    plot_visible=True,
    has_captured_contours=False,
)


def test_capture_is_offered_once_a_final_render_is_on_screen() -> None:
    assert contour_controls(**READY).capture is True


@pytest.mark.parametrize(
    "missing",
    ["has_live_data", "final_resolution_active", "final_plots_rendered", "plot_visible"],
)
def test_every_precondition_is_required_for_capture(missing) -> None:
    assert contour_controls(**{**READY, missing: False}).capture is False, missing


def test_a_live_preview_cannot_be_captured_before_the_final_render() -> None:
    # Guards against freezing contours that do not match the finished solve.
    controls = contour_controls(**{**READY, "final_plots_rendered": False})

    assert controls.capture is False


def test_clear_follows_only_whether_something_was_captured() -> None:
    assert contour_controls(**{**READY, "has_captured_contours": True}).clear is True
    assert contour_controls(**READY).clear is False


def test_clear_stays_available_even_when_capture_is_gated() -> None:
    controls = contour_controls(
        has_live_data=False,
        final_resolution_active=False,
        final_plots_rendered=False,
        plot_visible=False,
        has_captured_contours=True,
    )

    assert controls.capture is False
    assert controls.clear is True, "captured contours remain clearable during a later solve"
