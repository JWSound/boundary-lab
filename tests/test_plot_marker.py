import os

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from blab.ui.plot_marker import PlotMarker, PlotMarkerOverlay
from blab.ui.plots import ImpedanceCanvas, IsobarCanvas, OnAxisResponseCanvas, format_plot_frequency

_APP = QApplication.instance() or QApplication([])

# Mouse events land on whole pixels: about 1.6 % of frequency on a 3-decade axis.
_PIXEL_REL = 0.02
_FREQS = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)


def _impedance_canvas() -> ImpedanceCanvas:
    canvas = ImpedanceCanvas()
    canvas.update_plot(_FREQS, np.asarray(["HF"]), np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 1.0, 0.0]]))
    return canvas


def _on_axis_canvas() -> OnAxisResponseCanvas:
    canvas = OnAxisResponseCanvas()
    response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    canvas.update_plot(_FREQS, np.asarray([-10.0, 0.0, 10.0], dtype=np.float32), response)
    return canvas


def _isobar_canvas() -> IsobarCanvas:
    canvas = IsobarCanvas("Horizontal Isobar")
    angles = np.asarray([-90.0, 0.0, 90.0], dtype=np.float32)
    canvas.update_plot(_FREQS, angles, np.zeros((3, 3)), -30.0, 0.0, shading="gouraud")
    return canvas


def _shown(canvas):
    canvas.resize(520, 400)
    canvas.show()
    _APP.processEvents()
    canvas.draw()
    return canvas


@pytest.fixture
def canvases():
    created = []

    def make(factory):
        canvas = _shown(factory())
        created.append(canvas)
        return canvas

    yield make
    for canvas in created:
        canvas.close()
        canvas.deleteLater()


def _display_point(canvas, freq_hz: float) -> tuple[float, float]:
    axis_x = canvas._x_limits_from_user_domain((freq_hz, freq_hz))[0]
    y_mid = float(np.mean(canvas.axes.get_ylim()))
    return tuple(canvas.axes.transData.transform((axis_x, y_mid)))


def _hover(canvas, freq_hz: float) -> None:
    x, y = _display_point(canvas, freq_hz)
    MouseEvent("motion_notify_event", canvas, x, y)._process()


def _click(canvas, freq_hz: float, button=MouseButton.LEFT) -> None:
    x, y = _display_point(canvas, freq_hz)
    MouseEvent("button_press_event", canvas, x, y, button)._process()
    MouseEvent("button_release_event", canvas, x, y, button)._process()


def _marker_artist(canvas, gid: str):
    matches = [artist for artist in canvas.axes.get_children() if artist.get_gid() == gid]
    return matches[0] if matches else None


def _marker_line_frequency(canvas) -> float | None:
    line = _marker_artist(canvas, "plot_marker_line")
    if line is None or not line.get_visible():
        return None
    axis_x = float(np.asarray(line.get_xdata())[0])
    return canvas._x_limits_to_user_domain((axis_x, axis_x))[0]


def _marker_label_text(canvas) -> str | None:
    label = _marker_artist(canvas, "plot_marker_label")
    if label is None or not label.get_visible():
        return None
    return label.get_text()


def test_hovering_one_plot_moves_the_marker_on_every_plot(canvases) -> None:
    marker = PlotMarker()
    impedance = canvases(_impedance_canvas)
    on_axis = canvases(_on_axis_canvas)
    for canvas in (impedance, on_axis):
        PlotMarkerOverlay(canvas, marker)

    marker.action.setChecked(True)
    _hover(impedance, 1000.0)

    hovered = _marker_line_frequency(impedance)
    assert hovered == pytest.approx(1000.0, rel=_PIXEL_REL)
    assert _marker_line_frequency(on_axis) == pytest.approx(hovered)
    assert _marker_label_text(impedance) is None
    assert _marker_label_text(on_axis) is None


def test_click_locks_the_marker_and_a_second_click_unlocks_it(canvases) -> None:
    marker = PlotMarker()
    canvas = canvases(_impedance_canvas)
    PlotMarkerOverlay(canvas, marker)
    marker.action.setChecked(True)

    _click(canvas, 300.0)
    locked_at = _marker_line_frequency(canvas)
    _hover(canvas, 3000.0)
    assert _marker_line_frequency(canvas) == pytest.approx(locked_at)
    assert locked_at == pytest.approx(300.0, rel=_PIXEL_REL)

    _click(canvas, 3000.0)
    _hover(canvas, 500.0)
    assert _marker_line_frequency(canvas) == pytest.approx(500.0, rel=_PIXEL_REL)


def test_marker_is_clamped_to_each_plots_frequency_range(canvases) -> None:
    marker = PlotMarker()
    wide = canvases(_impedance_canvas)
    narrow = canvases(_on_axis_canvas)
    wide.axes.set_xlim(20.0, 20000.0)
    narrow.axes.set_xlim(200.0, 2000.0)
    for canvas in (wide, narrow):
        canvas.draw()
        PlotMarkerOverlay(canvas, marker)
    marker.action.setChecked(True)

    _hover(wide, 10000.0)

    assert _marker_line_frequency(wide) == pytest.approx(10000.0, rel=_PIXEL_REL)
    assert _marker_line_frequency(narrow) == pytest.approx(2000.0)
    assert _marker_label_text(narrow) is None


def test_isobar_plot_shows_the_marker_line_and_frequency(canvases) -> None:
    marker = PlotMarker()
    response = canvases(_on_axis_canvas)
    isobar = canvases(_isobar_canvas)
    for canvas in (response, isobar):
        PlotMarkerOverlay(canvas, marker)
    marker.action.setChecked(True)

    _hover(response, 1000.0)

    hovered = _marker_line_frequency(response)
    assert _marker_line_frequency(isobar) == pytest.approx(hovered)
    assert _marker_label_text(isobar) == format_plot_frequency(hovered)
    assert _marker_label_text(response) is None

    _hover(isobar, 300.0)
    assert _marker_line_frequency(response) == pytest.approx(300.0, rel=_PIXEL_REL)


def test_turning_the_marker_off_removes_it_and_hover_does_nothing(canvases) -> None:
    marker = PlotMarker()
    canvas = canvases(_impedance_canvas)
    PlotMarkerOverlay(canvas, marker)

    _hover(canvas, 1000.0)
    assert _marker_line_frequency(canvas) is None

    marker.action.setChecked(True)
    _hover(canvas, 1000.0)
    marker.action.setChecked(False)

    assert _marker_line_frequency(canvas) is None
    assert _marker_label_text(canvas) is None


def test_clearing_a_plot_for_a_new_solve_removes_its_marker(canvases) -> None:
    marker = PlotMarker()
    cleared = canvases(_impedance_canvas)
    kept = canvases(_on_axis_canvas)
    for canvas in (cleared, kept):
        PlotMarkerOverlay(canvas, marker)
    marker.action.setChecked(True)
    _hover(kept, 1000.0)

    cleared.clear_for_solve()
    cleared.draw()

    assert _marker_line_frequency(cleared) is None
    assert _marker_line_frequency(kept) is not None


def test_crosshair_is_hidden_and_inactive_while_the_marker_is_on(canvases) -> None:
    marker = PlotMarker()
    canvas = canvases(_impedance_canvas)
    PlotMarkerOverlay(canvas, marker)

    def crosshair_frequency_text() -> str | None:
        texts = [
            text.get_text()
            for text in canvas.axes.texts
            if text.get_visible() and text.get_gid() is None and text.get_text().endswith("Hz")
        ]
        return texts[0] if texts else None

    _click(canvas, 1000.0)
    crosshair_text = crosshair_frequency_text()
    assert crosshair_text is not None

    marker.action.setChecked(True)
    assert crosshair_frequency_text() is None
    _click(canvas, 300.0)
    assert crosshair_frequency_text() is None
    assert _marker_label_text(canvas) is None

    marker.action.setChecked(False)
    assert crosshair_frequency_text() == crosshair_text


def test_marker_toggle_and_position_are_shared_by_all_main_window_plots(main_window) -> None:
    def marker_button(plot_id: str):
        title_bar = main_window.plot_docks[plot_id].titleBarWidget()
        return next(button for button in title_bar.tool_buttons if button.text() == "Marker")

    spinorama_button = marker_button("spinorama")
    assert all(not marker_button(plot_id).isChecked() for plot_id in main_window.plot_docks)

    spinorama_button.click()

    assert all(marker_button(plot_id).isChecked() for plot_id in main_window.plot_docks)

    impedance = main_window.impedance_plot
    on_axis = main_window.on_axis_plot
    impedance.update_plot(_FREQS, np.asarray(["HF"]), np.asarray([[1.0, 2.0, 3.0]]), np.asarray([[0.0, 1.0, 0.0]]))
    on_axis.update_plot(
        _FREQS,
        np.asarray([-10.0, 0.0, 10.0], dtype=np.float32),
        np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]]),
    )
    impedance.draw()
    _hover(impedance, 1000.0)

    hovered = _marker_line_frequency(impedance)
    assert hovered == pytest.approx(1000.0, rel=_PIXEL_REL)
    assert _marker_line_frequency(on_axis) == pytest.approx(hovered)

    marker_button("max_spl").click()

    assert all(not marker_button(plot_id).isChecked() for plot_id in main_window.plot_docks)
    assert _marker_line_frequency(on_axis) is None
