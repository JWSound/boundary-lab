import os
from types import SimpleNamespace

import numpy as np
import pytest
from cycler import cycler
from matplotlib import rcParams
from matplotlib.backend_bases import MouseButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.balloon import SliceRadarCanvas, WavefrontShapeCanvas
from blab.ui.electrical_impedance_plot import ElectricalImpedanceCanvas
from blab.ui.excursion_plot import ExcursionCanvas
from blab.ui.group_delay_plot import GroupDelayCanvas
from blab.ui.main_window import MainWindow
from blab.ui.max_spl_plot import MaxSplCanvas
from blab.ui.plots import ImpedanceCanvas, IsobarCanvas, OnAxisResponseCanvas, SpinoramaCanvas
from blab.ui.result_projection import (
    ElectricalImpedanceProjection,
    ExcursionProjection,
    GroupDelayProjection,
    ImpedanceProjection,
    IsobarProjection,
    MaxSplProjection,
    PolarResponseProjection,
    VisualizationProjection,
)

_APP = QApplication.instance() or QApplication([])


def _mouse_event(axes, button, *, xdata=1000.0, ydata=0.0, dblclick=False):
    return SimpleNamespace(
        inaxes=axes,
        button=button,
        xdata=xdata,
        ydata=ydata,
        dblclick=dblclick,
    )


def _spinorama_curves(offset: float = 0.0) -> SpinoramaCurves:
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    return SpinoramaCurves(
        freq_hz=freqs,
        on_axis_db=np.asarray([0.0, 0.0, 0.0]) + offset,
        listening_window_db=np.asarray([-1.0, -1.0, -1.0]) + offset,
        early_reflections_db=np.asarray([-2.0, -2.0, -2.0]) + offset,
        sound_power_db=np.asarray([-3.0, -3.0, -3.0]) + offset,
        estimated_in_room_db=np.asarray([-2.5, -2.5, -2.5]) + offset,
        early_reflections_di_db=np.asarray([2.0, 2.0, 2.0]) + offset,
        sound_power_di_db=np.asarray([3.0, 3.0, 3.0]) + offset,
    )


class _FakeTimer:
    def __init__(self, *, active: bool = False):
        self.active = active
        self.start_count = 0

    def isActive(self) -> bool:
        return self.active

    def start(self) -> None:
        self.active = True
        self.start_count += 1


def test_live_plot_refresh_requests_coalesce_while_timer_is_active() -> None:
    timer = _FakeTimer()
    window = SimpleNamespace(_live_plot_refresh_dirty=False, _live_plot_refresh_timer=timer)

    MainWindow.request_live_refresh(window)
    MainWindow.request_live_refresh(window)

    assert window._live_plot_refresh_dirty is True
    assert timer.start_count == 1


def test_live_plot_refresh_flushes_only_actively_visible_entries() -> None:
    refresh_calls = []
    window = SimpleNamespace(
        _live_plot_refresh_dirty=True,
        refresh_plots=lambda **options: refresh_calls.append(options),
    )

    MainWindow.flush_live_refresh(window)

    assert window._live_plot_refresh_dirty is False
    assert refresh_calls == [{"active_only": True}]

    visible_widget = SimpleNamespace(visibleRegion=lambda: SimpleNamespace(isEmpty=lambda: False))
    background_widget = SimpleNamespace(visibleRegion=lambda: SimpleNamespace(isEmpty=lambda: True))
    dock = SimpleNamespace(isHidden=lambda: False)
    window = SimpleNamespace(plot_docks={"visible": dock, "background": dock})
    assert MainWindow._plot_entry_is_actively_visible(window, SimpleNamespace(plot_id="visible", widget=visible_widget))
    assert not MainWindow._plot_entry_is_actively_visible(
        window, SimpleNamespace(plot_id="background", widget=background_widget)
    )


def test_final_plot_refresh_skips_tab_covered_canvases() -> None:
    dataset = object()
    updates = []
    active = SimpleNamespace(plot_id="active", update=lambda value: updates.append(("active", value)))
    covered = SimpleNamespace(plot_id="covered", update=lambda value: updates.append(("covered", value)))
    dock = SimpleNamespace(isHidden=lambda: False)
    window = SimpleNamespace(
        plot_entries=(active, covered),
        plot_docks={"active": dock, "covered": dock},
        _plot_entry_is_actively_visible=lambda entry: entry.plot_id == "active",
        _use_final_isobar_resolution=True,
        prepared_live_dataset=lambda **_options: dataset,
    )

    result = MainWindow.refresh_plots(window, active_only=False)

    assert result is dataset
    assert updates == [("active", dataset)]


def test_activated_plot_refresh_uses_cached_solve_after_geometry_is_visible() -> None:
    refresh_calls = []
    contour_calls = []
    active = SimpleNamespace(plot_id="on_axis_frequency_response")
    covered = SimpleNamespace(plot_id="spinorama")
    window = SimpleNamespace(
        _plot_activation_refresh_pending=True,
        plot_entries=(active, covered),
        _plot_entry_is_actively_visible=lambda entry: entry is active,
        solve_controller=SimpleNamespace(active=False),
        preferences=SimpleNamespace(live_plot_streaming=True),
        refresh_plots=lambda **options: refresh_calls.append(options),
        request_live_refresh=lambda: None,
        _use_final_isobar_resolution=True,
        _final_isobar_plots_rendered=False,
        refresh_contour_controls=lambda: contour_calls.append(True),
    )

    MainWindow._refresh_visible_plots(window)

    assert window._plot_activation_refresh_pending is False
    assert refresh_calls == [{"active_only": True}]
    assert window._final_isobar_plots_rendered is False
    assert contour_calls == [True]


def test_isobar_canvas_reuses_mesh_colorbar_and_colormap() -> None:
    canvas = IsobarCanvas("Horizontal")
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    angles = np.asarray([-90.0, 0.0, 90.0], dtype=np.float32)
    values = np.arange(9, dtype=np.float32).reshape(3, 3) - 8.0

    canvas.update_plot(freqs, angles, values, -30.0, 0.0, shading="nearest", contour_step_db=3.0)
    first_mesh = canvas._mesh_artist
    first_colorbar = canvas._colorbar
    first_colormap = canvas._colormap

    canvas.update_plot(freqs, angles, values + 1.0, -30.0, 0.0, shading="nearest", contour_step_db=3.0)

    assert canvas._mesh_artist is first_mesh
    assert canvas._colorbar is first_colorbar
    assert canvas._colormap is first_colormap


def test_clearing_a_colorbar_canvas_does_not_need_a_spinorama_layout_hook() -> None:
    """``_draw_empty`` runs on every canvas at the start of every solve.

    It reaches ``_remove_colorbar``, which asks the canvas to re-apply its
    layout. Only ``SpinoramaCanvas`` defined that hook, so a second solve died
    here on an isobar plot before the solver was ever invoked.
    """
    canvas = IsobarCanvas("Horizontal")
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    angles = np.asarray([-90.0, 0.0, 90.0], dtype=np.float32)
    values = np.arange(9, dtype=np.float32).reshape(3, 3) - 8.0

    canvas.update_plot(freqs, angles, values, -30.0, 0.0, shading="nearest")
    assert canvas._colorbar is not None

    canvas._draw_empty()

    assert canvas._colorbar is None


def test_line_plot_canvases_reuse_existing_artists() -> None:
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)

    impedance = ImpedanceCanvas()
    impedance.update_plot(freqs, np.asarray(["HF"]), np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 1.0, 0.0]]))
    impedance_lines = tuple(impedance._lines)
    impedance.update_plot(freqs, np.asarray(["HF"]), np.asarray([[2.0, 3.0, 4.0]]), np.asarray([[1.0, 0.0, 1.0]]))
    assert tuple(impedance._lines) == impedance_lines

    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    on_axis = OnAxisResponseCanvas()
    on_axis.update_plot(freqs, angles, response)
    on_axis_lines = tuple(on_axis._lines)
    on_axis.update_plot(freqs, angles, response + 1.0)
    assert tuple(on_axis._lines) == on_axis_lines

    excursion = ExcursionCanvas()
    excursion.update_plot(freqs, np.asarray(["Woofer"]), np.asarray([[1.0, 2.0, 3.0]]))
    excursion_lines = dict(excursion._lines)
    excursion.update_plot(freqs, np.asarray(["Woofer"]), np.asarray([[2.0, 3.0, 4.0]]))
    assert excursion._lines == excursion_lines

    electrical = ElectricalImpedanceCanvas()
    electrical.update_plot(
        freqs,
        np.asarray(["LF"]),
        np.asarray([[4.0, 8.0, 6.0]]),
        np.asarray([[10.0, 20.0, 30.0]]),
    )
    electrical_lines = dict(electrical._magnitude_lines)
    electrical.update_plot(
        freqs,
        np.asarray(["LF"]),
        np.asarray([[5.0, 9.0, 7.0]]),
        np.asarray([[20.0, 30.0, 40.0]]),
    )
    assert electrical._magnitude_lines == electrical_lines

    group_delay = GroupDelayCanvas()
    group_delay.update_plot(
        freqs,
        np.asarray(["Sum", "LF"]),
        np.asarray([[1.0, 2.0, 1.0], [2.0, 3.0, 2.0]]),
    )
    group_delay_lines = dict(group_delay._lines)
    group_delay.update_plot(
        freqs,
        np.asarray(["Sum", "LF"]),
        np.asarray([[1.5, 2.5, 1.5], [2.5, 3.5, 2.5]]),
    )
    assert group_delay._lines == group_delay_lines

    spinorama = SpinoramaCanvas()
    curves = _spinorama_curves()
    spinorama.update_curves(curves)
    spl_lines = dict(spinorama._spl_lines)
    di_lines = dict(spinorama._di_lines)
    spinorama.update_curves(curves)
    assert spinorama._spl_lines == spl_lines
    assert spinorama._di_lines == di_lines


def test_on_axis_plot_uses_log_frequency_and_five_db_minor_grids() -> None:
    canvas = OnAxisResponseCanvas()
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    response = np.asarray(
        [[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]],
        dtype=np.float32,
    )

    canvas.update_plot(freqs, angles, response)
    canvas.figure.canvas.draw()

    visible_x_minor = {
        round(float(tick.get_loc()))
        for tick in canvas.axes.xaxis.get_minor_ticks()
        if tick.gridline.get_visible() and 20.0 <= tick.get_loc() <= 20_000.0
    }
    visible_y_minor = {
        round(float(tick.get_loc())) for tick in canvas.axes.yaxis.get_minor_ticks() if tick.gridline.get_visible()
    }

    assert {30, 40, 60, 70, 80, 90, 300, 3000}.issubset(visible_x_minor)
    assert {35, 45, 55, 65, 75, 85}.issubset(visible_y_minor)


def test_excursion_plot_uses_audio_frequency_axis_zero_baseline_and_trace_filter() -> None:
    canvas = ExcursionCanvas()
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    canvas.update_plot(
        freqs,
        np.asarray(["Woofer", "Passive radiator"]),
        np.asarray([[1.0, 2.0, 1.5], [0.5, 3.0, 2.0]], dtype=np.float32),
    )

    assert canvas.axes.get_xscale() == "log"
    assert canvas.axes.get_xlim() == (20.0, 20_000.0)
    assert canvas.axes.get_ylim()[0] == 0.0
    assert canvas.axes.get_ylabel() == "Excursion (mm)"
    canvas._series_actions["Passive radiator"].setChecked(False)
    assert not canvas._lines["Passive radiator"].get_visible()
    assert canvas._lines["Woofer"].get_visible()


def test_acoustic_impedance_trace_filter_hides_real_and_imaginary_pair() -> None:
    canvas = ImpedanceCanvas()
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    canvas.update_plot(
        freqs,
        np.asarray(["Woofer", "Tweeter"]),
        np.asarray([[2.0, 8.0, 4.0], [1.0, 3.0, 2.0]], dtype=np.float32),
        np.asarray([[0.0, 6.0, -2.0], [0.5, 1.0, -0.5]], dtype=np.float32),
    )

    assert canvas.axes.get_ylabel() == "Normalized Acoustic Impedance (Z / ρcSd)"
    assert set(canvas._series_actions) == {"Woofer", "Tweeter"}
    canvas._series_actions["Woofer"].setChecked(False)
    assert not canvas._lines[0].get_visible()
    assert not canvas._lines[1].get_visible()
    assert canvas._lines[2].get_visible()
    assert canvas._lines[3].get_visible()


def test_electrical_impedance_plot_uses_parallel_load_magnitude_and_phase_toggle() -> None:
    canvas = ElectricalImpedanceCanvas()
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    canvas.update_plot(
        freqs,
        np.asarray(["LF", "HF"]),
        np.asarray([[4.0, 20.0, 8.0], [8.0, 10.0, 12.0]], dtype=np.float32),
        np.asarray([[170.0, -170.0, -90.0], [10.0, 20.0, 30.0]], dtype=np.float32),
    )

    assert canvas.axes.get_xscale() == "log"
    assert canvas.axes.get_xlim() == (20.0, 20_000.0)
    assert canvas.axes.get_ylim()[0] == 0.0
    assert canvas.axes.get_ylabel() == "Impedance (Ω)"
    assert canvas.show_phase_action.isEnabled()
    assert not canvas.phase_axes.get_visible()
    assert all(line.get_linestyle() == ":" for line in canvas._phase_lines.values())
    assert all(
        canvas._phase_lines[label].get_color() == canvas._magnitude_lines[label].get_color()
        for label in canvas._series_labels
    )

    canvas.show_phase_action.setChecked(True)
    assert canvas.phase_axes.get_visible()
    assert canvas.phase_axes.get_ylim() == (-180.0, 600.0)
    assert np.isnan(np.asarray(canvas._phase_lines["LF"].get_ydata())).any()
    finite_phase = np.asarray(canvas._phase_lines["LF"].get_ydata(), dtype=float)
    finite_phase = finite_phase[np.isfinite(finite_phase)]
    assert np.all(finite_phase >= -180.0)
    assert np.all(finite_phase <= 180.0)

    canvas._series_actions["HF"].setChecked(False)
    assert not canvas._magnitude_lines["HF"].get_visible()
    assert not canvas._phase_lines["HF"].get_visible()


def test_group_delay_plot_uses_audio_axis_trace_filter_and_signed_milliseconds() -> None:
    canvas = GroupDelayCanvas()
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    canvas.update_plot(
        freqs,
        np.asarray(["Sum", "LF", "HF"]),
        np.asarray(
            [[0.5, 1.0, 0.25], [2.0, 3.0, 1.0], [-0.5, -1.0, -0.25]],
            dtype=np.float32,
        ),
    )

    assert canvas.axes.get_xscale() == "log"
    assert canvas.axes.get_xlim() == (20.0, 20_000.0)
    assert canvas.axes.get_ylabel() == "Group Delay (ms)"
    assert canvas.axes.get_ylim()[0] < -1.0
    assert canvas.axes.get_ylim()[1] > 3.0
    assert canvas._lines["Sum"].get_color() == "#000000"
    assert canvas._lines["Sum"].get_linewidth() > canvas._lines["LF"].get_linewidth()

    canvas._series_actions["HF"].setChecked(False)
    assert not canvas._lines["HF"].get_visible()
    assert canvas._lines["Sum"].get_visible()


def test_on_axis_plot_filters_solid_magnitudes_and_dotted_wrapped_phase(monkeypatch) -> None:
    monkeypatch.setitem(
        rcParams,
        "axes.prop_cycle",
        cycler(color=[(31 / 255, 119 / 255, 180 / 255), (1.0, 0.5, 0.0)]),
    )
    canvas = OnAxisResponseCanvas()
    freqs = np.asarray([100.0, 1000.0, 10_000.0], dtype=np.float32)
    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    channel_names = np.asarray(["LF", "HF"])
    channel_spl = np.asarray([[81.0, 82.0, 83.0], [72.0, 78.0, 84.0]])
    sum_phase = np.asarray([170.0, -170.0, -90.0])
    channel_phase = np.asarray([[10.0, 20.0, 30.0], [-10.0, -20.0, -30.0]])

    canvas.update_plot(
        freqs,
        angles,
        response,
        channel_names,
        channel_spl,
        sum_phase,
        channel_phase,
    )

    assert canvas._magnitude_lines["Sum"].get_color() == "#000000"
    assert all(line.get_linestyle() == "-" for line in canvas._magnitude_lines.values())
    assert all(line.get_linestyle() == ":" for line in canvas._phase_lines.values())
    assert all(
        canvas._phase_lines[label].get_color() == canvas._magnitude_lines[label].get_color()
        for label in canvas._series_labels
    )
    assert canvas.show_phase_action.isEnabled()
    assert not canvas.phase_axes.get_visible()

    canvas.show_phase_action.setChecked(True)
    assert canvas.phase_axes.get_visible()
    assert canvas.phase_axes.get_ylim() == (-180.0, 600.0)
    assert np.isnan(np.asarray(canvas._phase_lines["Sum"].get_ydata())).any()

    canvas._series_actions["HF"].setChecked(False)
    assert not canvas._magnitude_lines["HF"].get_visible()
    assert not canvas._phase_lines["HF"].get_visible()
    assert canvas._magnitude_lines["LF"].get_visible()


def test_line_plot_crosshairs_track_raw_coordinates_and_persist_until_double_click() -> None:
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    impedance = ImpedanceCanvas()
    impedance.update_plot(
        freqs,
        np.asarray(["HF"]),
        np.asarray([[1.0, 2.0, 3.0]]),
        np.asarray([[0.0, 1.0, 0.0]]),
    )
    on_axis = OnAxisResponseCanvas()
    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    on_axis.update_plot(freqs, angles, response)

    for canvas, y_value, expected_label in (
        (impedance, 2.25, "2.25"),
        (on_axis, 82.5, "82.5 dB"),
    ):
        canvas._on_crosshair_button_press(_mouse_event(canvas.axes, MouseButton.LEFT, ydata=y_value))
        canvas._on_crosshair_button_release(_mouse_event(canvas.axes, MouseButton.LEFT, ydata=y_value))

        assert canvas._crosshair_visible is True
        assert canvas._crosshair_dragging is False
        assert canvas._crosshair_freq_label.get_text() == "1 kHz"
        assert canvas._crosshair_y_label.get_text() == expected_label

        canvas._on_crosshair_button_press(_mouse_event(canvas.axes, MouseButton.LEFT, ydata=y_value, dblclick=True))
        assert canvas._crosshair_visible is False


def test_spinorama_crosshair_reports_both_raw_y_axes_and_restores_title() -> None:
    canvas = SpinoramaCanvas()
    canvas.update_curves(_spinorama_curves())

    assert canvas.axes.get_title() == "Spinorama"
    canvas._on_crosshair_button_press(_mouse_event(canvas.di_axes, MouseButton.LEFT, ydata=5.0))
    canvas._on_crosshair_button_release(_mouse_event(canvas.di_axes, MouseButton.LEFT, ydata=5.0))

    assert canvas._crosshair_freq_label.get_text() == "1 kHz"
    assert canvas._crosshair_y_label.get_text().endswith("dB SPL")
    assert canvas._crosshair_secondary_label.get_text() == "5.0 dB DI"


def test_isobar_comparison_restores_title_and_latest_deferred_live_update() -> None:
    canvas = IsobarCanvas("Horizontal Isobar")
    freqs = np.asarray([100.0, 1000.0], dtype=np.float32)
    angles = np.asarray([-90.0, 90.0], dtype=np.float32)
    current = np.asarray([[-3.0, -2.0], [-1.0, 0.0]], dtype=np.float32)
    previous = current - 3.0
    latest = current + 2.0
    canvas.update_plot(freqs, angles, current, -30.0, 0.0, shading="gouraud")
    canvas.set_comparison_plot(freqs, angles, previous, -30.0, 0.0, shading="gouraud")

    canvas._on_comparison_button_press(_mouse_event(canvas.axes, MouseButton.RIGHT))
    assert canvas.axes.get_title() == "Horizontal Isobar - Previous Solve"
    np.testing.assert_allclose(canvas._mesh_values_db, previous)

    canvas.update_plot(freqs, angles, latest, -30.0, 0.0, shading="gouraud")
    np.testing.assert_allclose(canvas._mesh_values_db, previous)
    canvas._on_comparison_button_release(_mouse_event(canvas.axes, MouseButton.RIGHT))

    assert canvas.axes.get_title() == "Horizontal Isobar"
    assert canvas._comparison_active is False
    np.testing.assert_allclose(canvas._mesh_values_db, np.clip(latest, -30.0, 0.0))


def test_remaining_plots_restore_current_solve_when_comparison_ends() -> None:
    freqs = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)
    names = np.asarray(["HF"])

    impedance = ImpedanceCanvas()
    impedance.update_plot(freqs, names, np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 1.0, 0.0]]))
    impedance.set_comparison_plot(
        freqs,
        names,
        np.asarray([[4.0, 5.0, 6.0]]),
        np.asarray([[3.0, 2.0, 1.0]]),
    )
    impedance._on_comparison_button_press(_mouse_event(impedance.axes, MouseButton.RIGHT))
    np.testing.assert_allclose(impedance._lines[0].get_ydata(), [4.0, 5.0, 6.0])
    impedance._on_comparison_button_release(_mouse_event(impedance.axes, MouseButton.RIGHT))
    assert impedance.axes.get_title() == "Acoustic Impedance"
    np.testing.assert_allclose(impedance._lines[0].get_ydata(), [1.0, 2.0, 3.0])

    electrical = ElectricalImpedanceCanvas()
    electrical.update_plot(
        freqs,
        names,
        np.asarray([[4.0, 8.0, 6.0]]),
        np.asarray([[10.0, 20.0, 30.0]]),
    )
    electrical.set_comparison_plot(
        freqs,
        names,
        np.asarray([[5.0, 9.0, 7.0]]),
        np.asarray([[20.0, 30.0, 40.0]]),
    )
    electrical._on_comparison_button_press(_mouse_event(electrical.axes, MouseButton.RIGHT))
    np.testing.assert_allclose(electrical._lines[0].get_ydata(), [5.0, 9.0, 7.0])
    electrical._on_comparison_button_release(_mouse_event(electrical.axes, MouseButton.RIGHT))
    assert electrical.axes.get_title() == "Electrical Impedance"
    np.testing.assert_allclose(electrical._lines[0].get_ydata(), [4.0, 8.0, 6.0])

    group_delay = GroupDelayCanvas()
    group_delay.update_plot(freqs, names, np.asarray([[1.0, 2.0, 3.0]]))
    group_delay.set_comparison_plot(freqs, names, np.asarray([[4.0, 5.0, 6.0]]))
    group_delay._on_comparison_button_press(_mouse_event(group_delay.axes, MouseButton.RIGHT))
    np.testing.assert_allclose(group_delay._lines["HF"].get_ydata(), [4.0, 5.0, 6.0])
    group_delay._on_comparison_button_release(_mouse_event(group_delay.axes, MouseButton.RIGHT))
    assert group_delay.axes.get_title() == "Group Delay"
    np.testing.assert_allclose(group_delay._lines["HF"].get_ydata(), [1.0, 2.0, 3.0])

    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    current_response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    on_axis = OnAxisResponseCanvas()
    on_axis.update_plot(freqs, angles, current_response)
    on_axis.set_comparison_plot(freqs, angles, current_response - 10.0)
    on_axis._on_comparison_button_press(_mouse_event(on_axis.axes, MouseButton.RIGHT))
    np.testing.assert_allclose(on_axis._lines[0].get_ydata(), [71.0, 73.0, 75.0])
    on_axis._on_figure_leave(SimpleNamespace())
    assert on_axis.axes.get_title() == "On-Axis Frequency Response"
    np.testing.assert_allclose(on_axis._lines[0].get_ydata(), [81.0, 83.0, 85.0])

    spinorama = SpinoramaCanvas()
    spinorama.update_curves(_spinorama_curves())
    spinorama._set_comparison_plot_state(_spinorama_curves(-5.0))
    spinorama._on_comparison_button_press(_mouse_event(spinorama.di_axes, MouseButton.RIGHT))
    assert spinorama.axes.get_title() == "Spinorama - Previous Solve"
    np.testing.assert_allclose(spinorama._spl_lines["On Axis"].get_ydata(), [-5.0, -5.0, -5.0])
    spinorama._on_comparison_button_release(_mouse_event(spinorama.di_axes, MouseButton.RIGHT))
    assert spinorama.axes.get_title() == "Spinorama"
    np.testing.assert_allclose(spinorama._spl_lines["On Axis"].get_ydata(), [0.0, 0.0, 0.0])


def test_main_window_distributes_previous_projection_to_every_plot() -> None:
    class PlotRecorder:
        def __init__(self):
            self.calls = []

        def set_comparison_plot(self, *args, **kwargs):
            self.calls.append((args, kwargs))

    freqs = np.asarray([100.0, 1000.0], dtype=np.float32)
    angles = np.asarray([-90.0, 90.0], dtype=np.float32)
    isobar = IsobarProjection(freqs, angles, np.zeros((2, 2)), np.ones((2, 2)), -30.0, 0.0)
    impedance = ImpedanceProjection(freqs, np.asarray(["HF"]), np.ones((1, 2)), np.zeros((1, 2)))
    response = PolarResponseProjection(
        freq_hz=freqs,
        angle_deg=angles,
        horizontal_spl_db=np.zeros((2, 2)),
        vertical_spl_db=np.ones((2, 2)),
        channel_on_axis_names=np.asarray(["main"]),
        channel_on_axis_spl_db=np.zeros((1, 2)),
        on_axis_phase_deg=np.zeros(2),
        channel_on_axis_phase_deg=np.zeros((1, 2)),
        spin_horizontal_reference_angle_deg=10.0,
        spin_vertical_reference_angle_deg=-5.0,
    )
    excursion = ExcursionProjection(freqs, np.asarray(["Woofer"]), np.ones((1, 2)))
    electrical = ElectricalImpedanceProjection(
        freqs,
        np.asarray(["main"]),
        np.ones((1, 2)),
        np.zeros((1, 2)),
    )
    group_delay = GroupDelayProjection(
        freqs,
        np.asarray(["Sum", "main"]),
        np.ones((2, 2)),
    )
    maximum = MaxSplProjection(freqs, np.asarray(["main"]), np.full((1, 2), 110.0))
    plots = [PlotRecorder() for _index in range(9)]
    window = SimpleNamespace(
        _last_completed_visualization_dataset=VisualizationProjection(
            isobar,
            impedance,
            response,
            excursion,
            electrical,
            group_delay,
            maximum,
        ),
        horizontal_plot=plots[0],
        vertical_plot=plots[1],
        impedance_plot=plots[2],
        electrical_impedance_plot=plots[3],
        on_axis_plot=plots[4],
        group_delay_plot=plots[5],
        excursion_plot=plots[6],
        max_spl_plot=plots[7],
        spinorama_plot=plots[8],
        preferences=SimpleNamespace(isobar_contour_step_db=3.0),
    )

    MainWindow.apply_last_completed_comparison(window)

    assert [len(plot.calls) for plot in plots] == [1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert plots[8].calls[0][1] == {
        "horizontal_reference_angle_deg": 10.0,
        "vertical_reference_angle_deg": -5.0,
    }
    assert len(plots[4].calls[0][0]) == 7


def test_chart_panels_share_layout_profiles_by_artist_requirements() -> None:
    single_axis = (
        ImpedanceCanvas(),
        GroupDelayCanvas(),
        ExcursionCanvas(),
        MaxSplCanvas(),
    )
    dual_axis = (ElectricalImpedanceCanvas(), OnAxisResponseCanvas())

    single_boxes = {canvas.axes.get_position().bounds for canvas in single_axis}
    dual_boxes = {canvas.axes.get_position().bounds for canvas in dual_axis}

    assert len(single_boxes) == 1
    assert len(dual_boxes) == 1
    assert single_axis[0].axes.get_position().x1 > dual_axis[0].axes.get_position().x1
    assert SpinoramaCanvas().axes.get_position().y0 > single_axis[0].axes.get_position().y0
    assert IsobarCanvas("Horizontal Isobar").axes.get_position().x1 < single_axis[0].axes.get_position().x1


def test_adaptive_layout_keeps_physical_margins_constant_as_a_plot_grows() -> None:
    canvas = ExcursionCanvas()

    for width, height in ((420, 260), (900, 520)):
        canvas.resize(width, height)
        _APP.processEvents()
        canvas.figure.canvas.draw()
        position = canvas.axes.get_position()
        width_pt, height_pt = canvas.figure.get_size_inches() * 72.0

        assert position.x0 * width_pt == pytest.approx(52.0, abs=1.0)
        assert (1.0 - position.x1) * width_pt == pytest.approx(14.0, abs=1.0)
        assert position.y0 * height_pt == pytest.approx(36.0, abs=1.0)
        assert (1.0 - position.y1) * height_pt == pytest.approx(22.0, abs=1.0)


def _assert_visible_axes_artists_fit_figure(canvas, *, tolerance_px: float = 1.5) -> None:
    canvas.figure.canvas.draw()
    renderer = canvas.figure.canvas.get_renderer()
    figure_bounds = canvas.figure.bbox

    for axes in canvas.figure.axes:
        if not axes.get_visible():
            continue
        bounds = axes.get_tightbbox(renderer)
        assert bounds.x0 >= figure_bounds.x0 - tolerance_px
        assert bounds.y0 >= figure_bounds.y0 - tolerance_px
        assert bounds.x1 <= figure_bounds.x1 + tolerance_px
        assert bounds.y1 <= figure_bounds.y1 + tolerance_px


def test_adaptive_layout_keeps_plot_text_colorbars_and_legends_visible() -> None:
    freqs = np.asarray([20.0, 1000.0, 20_000.0], dtype=np.float32)
    angles = np.asarray([-10.0, 0.0, 10.0], dtype=np.float32)
    response = np.asarray(
        [[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]],
        dtype=np.float32,
    )

    acoustic = ImpedanceCanvas()
    acoustic.update_plot(
        freqs,
        np.asarray(["Woofer"]),
        np.asarray([[2.0, 8.0, 4.0]], dtype=np.float32),
        np.asarray([[0.0, 6.0, -2.0]], dtype=np.float32),
    )

    electrical = ElectricalImpedanceCanvas()
    electrical.update_plot(
        freqs,
        np.asarray(["LF"]),
        np.asarray([[4.0, 20.0, 8.0]], dtype=np.float32),
        np.asarray([[170.0, -170.0, -90.0]], dtype=np.float32),
    )
    electrical.show_phase_action.setChecked(True)

    on_axis = OnAxisResponseCanvas()
    on_axis.update_plot(
        freqs,
        angles,
        response,
        np.asarray(["LF"]),
        np.asarray([[81.0, 82.0, 83.0]], dtype=np.float32),
        np.asarray([170.0, -170.0, -90.0], dtype=np.float32),
        np.asarray([[10.0, 20.0, 30.0]], dtype=np.float32),
    )
    on_axis.show_phase_action.setChecked(True)

    isobar = IsobarCanvas("Horizontal Isobar")
    isobar.update_plot(freqs, angles, response, -30.0, 6.0)

    spinorama = SpinoramaCanvas()
    spinorama.update_curves(_spinorama_curves())

    for canvas in (acoustic, electrical, on_axis, isobar):
        canvas.resize(420, 260)
        _APP.processEvents()
        _assert_visible_axes_artists_fit_figure(canvas)

    spinorama.resize(480, 300)
    _APP.processEvents()
    _assert_visible_axes_artists_fit_figure(spinorama)


def test_specialized_balloon_charts_use_resize_aware_layouts() -> None:
    frequencies = np.asarray([100.0, 1000.0, 10_000.0], dtype=np.float32)
    wavefront = WavefrontShapeCanvas()
    wavefront.update_plot(
        {
            "freq_hz": frequencies,
            "shape_exponent": np.asarray([1.0, 2.5, 5.0], dtype=np.float32),
            "fit_residual_percent": np.asarray([2.0, 6.0, 12.0], dtype=np.float32),
            "directivity_index_db": np.asarray([2.0, 8.0, 18.0], dtype=np.float32),
            "valid": np.ones(3, dtype=bool),
        }
    )
    radar = SliceRadarCanvas(-30.0, 6.0)
    radar.update_plot(
        np.arange(0.0, 360.0, 30.0),
        np.linspace(0.0, -24.0, 12, dtype=np.float32),
    )

    for canvas, size in ((wavefront, (420, 260)), (radar, (320, 320))):
        canvas.resize(*size)
        _APP.processEvents()
        _assert_visible_axes_artists_fit_figure(canvas)


def test_spinorama_legend_clears_the_x_label_at_panel_height() -> None:
    """The external legend and x label remain separate at a representative dock height."""
    canvas = SpinoramaCanvas()
    canvas.resize(520, 740)
    canvas.update_curves(_spinorama_curves())
    figure = canvas.figure
    figure.canvas.draw()

    to_figure = figure.transFigure.inverted()
    label_bottom = to_figure.transform_bbox(canvas.axes.xaxis.label.get_window_extent()).y0
    legend_top = to_figure.transform_bbox(canvas.axes.get_legend().get_window_extent()).y1

    assert legend_top < label_bottom, "legend overlaps the x label"
    assert legend_top > 0.0, "legend fell off the bottom of the figure"


def test_chart_titles_stand_off_the_axes() -> None:
    """pad=1 sat the title on the top spine."""
    from blab.ui.plots import PLOT_TITLE_PAD

    assert PLOT_TITLE_PAD >= 5

    canvas = OnAxisResponseCanvas()
    canvas.resize(520, 740)
    canvas.figure.canvas.draw()

    to_figure = canvas.figure.transFigure.inverted()
    title_bottom = to_figure.transform_bbox(canvas.axes.title.get_window_extent()).y0

    assert title_bottom > canvas.axes.get_position().y1
