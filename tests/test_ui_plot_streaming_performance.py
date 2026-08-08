import os
from types import SimpleNamespace

import numpy as np
from matplotlib.backend_bases import MouseButton

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.main_window import MainWindow
from blab.ui.plots import ImpedanceCanvas, IsobarCanvas, OnAxisResponseCanvas, SpinoramaCanvas
from blab.ui.result_projection import (
    ImpedanceProjection,
    IsobarProjection,
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

    spinorama = SpinoramaCanvas()
    curves = _spinorama_curves()
    spinorama.update_curves(curves)
    spl_lines = dict(spinorama._spl_lines)
    di_lines = dict(spinorama._di_lines)
    spinorama.update_curves(curves)
    assert spinorama._spl_lines == spl_lines
    assert spinorama._di_lines == di_lines


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
        spin_horizontal_reference_angle_deg=10.0,
        spin_vertical_reference_angle_deg=-5.0,
    )
    plots = [PlotRecorder() for _index in range(5)]
    window = SimpleNamespace(
        _last_completed_visualization_dataset=VisualizationProjection(isobar, impedance, response),
        horizontal_plot=plots[0],
        vertical_plot=plots[1],
        impedance_plot=plots[2],
        on_axis_plot=plots[3],
        spinorama_plot=plots[4],
        preferences=SimpleNamespace(isobar_contour_step_db=3.0),
    )

    MainWindow.apply_last_completed_comparison(window)

    assert [len(plot.calls) for plot in plots] == [1, 1, 1, 1, 1]
    assert plots[4].calls[0][1] == {
        "horizontal_reference_angle_deg": 10.0,
        "vertical_reference_angle_deg": -5.0,
    }


def test_every_chart_panel_gets_the_same_axes_box() -> None:
    """The spinorama used to reserve extra bottom margin for its legend."""
    canvases = (
        ImpedanceCanvas(),
        OnAxisResponseCanvas(),
        SpinoramaCanvas(),
        IsobarCanvas("Horizontal Isobar"),
    )
    boxes = {type(canvas).__name__: canvas.axes.get_position().bounds[1::2] for canvas in canvases}

    assert len(set(boxes.values())) == 1, f"axes boxes disagree: {boxes}"


def test_spinorama_legend_clears_the_x_label_at_panel_height() -> None:
    """Both are sized in points while the margin is a figure fraction, so
    they collide only on a short panel. Checked at the dock's real height.
    """
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
