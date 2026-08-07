"""BalloonPlotWindow structure and behaviour, driven as a real window.

These replace roughly 180 assertions that read balloon.py as text and matched
strings like ``controls_layout.addWidget(self.frequency_slider, 0, 1)`` and
``subplots_adjust(left=0.18, right=0.74``. Those pinned how the widget tree was
typed rather than what it does, so any visual rework broke them wholesale while
catching no real regression.

``BalloonPlotWindow.__init__`` imports the VTK plotter lazily, so the
``balloon_window`` fixture can stub it and build the genuine window offscreen.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDockWidget  # noqa: E402

from blab.ui.balloon import WAVEFRONT_LEVEL_DB, _contour_levels  # noqa: E402
from conftest import balloon_raw_bundle  # noqa: E402

DOCK_IDS = {"balloon_3d", "radar_slicer", "forward_beam_shape", "isobar_angle_slice"}


def _docks(window) -> dict[str, QDockWidget]:
    return {dock.objectName(): dock for dock in window.workspace.findChildren(QDockWidget)}


# ------------------------------------------------------------------ workspace


def test_every_view_is_a_dock_in_the_nested_workspace(balloon_window) -> None:
    assert set(_docks(balloon_window)) == DOCK_IDS
    assert balloon_window.workspace.isAnimated() is not None  # workspace is a real QMainWindow
    assert balloon_window.workspace.dockOptions() & balloon_window.workspace.DockOption.AllowNestedDocks


def test_the_beam_shape_dock_starts_hidden(balloon_window) -> None:
    """It is the expensive one to render, so it is opt-in from the View menu.

    Only the default: `_restore_window_state` deliberately lets a saved layout
    bring it back for a user who left it open.
    """
    assert _docks(balloon_window)["forward_beam_shape"].isHidden()


def test_a_saved_layout_can_restore_the_beam_shape_dock(balloon_window, qapp) -> None:
    dock = _docks(balloon_window)["forward_beam_shape"]
    dock.show()
    balloon_window.close()

    from blab.ui.balloon import BalloonPlotWindow

    reopened = BalloonPlotWindow(balloon_raw_bundle(), min_db=-30.0, max_db=6.0)
    try:
        assert not _docks(reopened)["forward_beam_shape"].isHidden()
    finally:
        reopened.close()
        reopened.deleteLater()
        qapp.processEvents()


def test_every_dock_can_be_toggled_from_the_view_menu(balloon_window) -> None:
    view_action = next(a for a in balloon_window.menuBar().actions() if a.text() == "View")
    view_menu = view_action.menu()

    toggles = {action.text() for action in view_menu.actions()}

    for dock in _docks(balloon_window).values():
        assert dock.toggleViewAction().text() in toggles, dock.objectName()


def test_the_controls_live_below_the_workspace_not_inside_it(balloon_window) -> None:
    """The frequency and slice-angle controls are a bottom bar, not a dock."""
    slider_ancestors = set()
    widget = balloon_window.frequency_slider
    while widget is not None:
        slider_ancestors.add(widget)
        widget = widget.parentWidget()

    assert balloon_window.workspace not in slider_ancestors
    assert balloon_window.spl_legend in balloon_window.frequency_slider.parentWidget().findChildren(
        type(balloon_window.spl_legend)
    ), "the SPL legend shares the bottom control bar"


# -------------------------------------------------------------------- actions


@pytest.mark.parametrize(
    ("attribute", "dock_id"),
    [
        ("save_wavefront_shape_action", "forward_beam_shape"),
        ("hires_slice_action", "isobar_angle_slice"),
        ("save_slice_action", "isobar_angle_slice"),
    ],
)
def test_plot_tool_actions_exist_with_icons_and_tooltips(balloon_window, attribute, dock_id) -> None:
    action = getattr(balloon_window, attribute)

    assert action.toolTip(), f"{attribute} needs a tooltip; it renders as an icon-only button"
    assert not action.icon().isNull(), f"{attribute} icon failed to resolve from APP_ROOT"
    assert dock_id in _docks(balloon_window)


def test_the_high_resolution_action_re_renders_the_slice_at_final_quality(balloon_window) -> None:
    balloon_window._prepare_and_render_initial()
    assert balloon_window._slice_plot_high_res is False, "live rendering is the default"

    balloon_window.hires_slice_action.trigger()

    assert balloon_window._slice_plot_high_res is True


# ------------------------------------------------------------- window state


def test_closing_persists_geometry_and_dock_layout(balloon_window) -> None:
    balloon_window.close()

    assert balloon_window.settings.value("balloon_window/geometry") is not None
    assert balloon_window.settings.value("balloon_window/dock_state") is not None


def test_transient_view_state_is_deliberately_not_persisted(balloon_window) -> None:
    """Frequency, slice angle and camera reset each session by design."""
    balloon_window.close()

    for key in (
        "balloon_window/frequency_index",
        "balloon_window/radar_slicer_enabled",
        "balloon_window/protractor_angle_deg",
        "balloon_window/camera_position",
    ):
        assert balloon_window.settings.value(key) is None, key


# ----------------------------------------------------------- live refresh


def test_refreshing_without_a_provider_is_a_no_op(balloon_window) -> None:
    balloon_window._raw_balloon_data_provider = None

    balloon_window.refresh_from_latest_results()  # must not raise


def test_identical_results_do_not_trigger_a_re_render(balloon_window, monkeypatch) -> None:
    """The window refreshes on focus, so unchanged data must stay cheap."""
    same = balloon_window._raw_balloon_data
    balloon_window._raw_balloon_data_provider = lambda: same
    rendered = []
    monkeypatch.setattr(balloon_window, "_prepare_and_render", lambda **k: rendered.append(k))

    balloon_window.refresh_from_latest_results()

    assert rendered == []


def test_new_results_re_render_and_keep_the_chosen_frequency(balloon_window, monkeypatch) -> None:
    newer = balloon_raw_bundle(freqs=(100.0, 400.0, 1000.0, 2000.0))
    balloon_window._raw_balloon_data_provider = lambda: newer
    rendered = []
    monkeypatch.setattr(balloon_window, "_prepare_and_render", lambda **k: rendered.append(k))

    balloon_window.refresh_from_latest_results()

    assert rendered == [{"preserve_frequency": True}]
    assert balloon_window._raw_balloon_data is newer
    assert balloon_window._wavefront_shape_summary_cache is None


def test_a_provider_returning_nothing_leaves_the_view_alone(balloon_window, monkeypatch) -> None:
    balloon_window._raw_balloon_data_provider = lambda: None
    rendered = []
    monkeypatch.setattr(balloon_window, "_prepare_and_render", lambda **k: rendered.append(k))

    balloon_window.refresh_from_latest_results()

    assert rendered == []


# --------------------------------------------------------------- pure rules


def test_contour_levels_exclude_the_configured_endpoints() -> None:
    """A contour drawn exactly at the clip limit is an artefact of clipping."""
    levels = _contour_levels(-30.0, 6.0, 3.0)

    assert levels, "there should be interior levels between -30 and 6 dB"
    assert all(-30.0 < level < 6.0 for level in levels)


def test_the_beam_shape_is_fitted_at_the_minus_six_db_contour() -> None:
    assert WAVEFRONT_LEVEL_DB == -6.0
