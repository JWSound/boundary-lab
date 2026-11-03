import math
import os

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_hex

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.electrical_impedance_plot import ElectricalImpedanceCanvas
from blab.ui.marker_readout import interpolate_log_frequency
from blab.ui.plot_marker import PlotMarker, PlotMarkerOverlay
from blab.ui.plots import IsobarCanvas, OnAxisResponseCanvas, SpinoramaCanvas, format_plot_frequency

_APP = QApplication.instance() or QApplication([])
_FREQS = np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32)


def test_interpolates_linearly_over_log_frequency() -> None:
    # 316.2 Hz is halfway between 100 Hz and 1 kHz on a log axis.
    value = interpolate_log_frequency([100.0, 1000.0], [80.0, 90.0], math.sqrt(100.0 * 1000.0))
    assert value == pytest.approx(85.0)


def test_returns_the_sample_value_at_a_sample_frequency() -> None:
    assert interpolate_log_frequency([100.0, 1000.0, 10000.0], [1.0, 2.0, 3.0], 1000.0) == pytest.approx(2.0)


def test_clamps_to_the_curve_data_range() -> None:
    freqs = [100.0, 1000.0]
    values = [80.0, 90.0]
    assert interpolate_log_frequency(freqs, values, 20.0) == pytest.approx(80.0)
    assert interpolate_log_frequency(freqs, values, 20000.0) == pytest.approx(90.0)


def test_accepts_descending_frequencies() -> None:
    value = interpolate_log_frequency([1000.0, 100.0], [90.0, 80.0], math.sqrt(100.0 * 1000.0))
    assert value == pytest.approx(85.0)


def test_skips_missing_samples() -> None:
    freqs = [100.0, np.nan, 1000.0, 10000.0]
    values = [80.0, 85.0, np.nan, 100.0]
    # The 1 kHz sample is missing, so 1 kHz lies halfway between 100 Hz and 10 kHz.
    assert interpolate_log_frequency(freqs, values, 1000.0) == pytest.approx(90.0)


def test_returns_none_without_finite_data() -> None:
    assert interpolate_log_frequency([], [], 1000.0) is None
    assert interpolate_log_frequency([100.0, 1000.0], [np.nan, np.nan], 300.0) is None


def test_periodic_values_take_the_short_way_across_a_wrap() -> None:
    # Phase wraps from 170 deg to -170 deg: halfway is 180 deg, not 0 deg.
    value = interpolate_log_frequency([100.0, 1000.0], [170.0, -170.0], math.sqrt(100.0 * 1000.0), period=360.0)
    assert value == pytest.approx(180.0) or value == pytest.approx(-180.0)
    quarter = interpolate_log_frequency([100.0, 10000.0], [170.0, -150.0], 316.227766, period=360.0)
    assert quarter == pytest.approx(-180.0) or quarter == pytest.approx(180.0)


# Canvas level -------------------------------------------------------------------


def _on_axis_canvas() -> OnAxisResponseCanvas:
    # Sum SPL is 80, 90, 100 dB: 80 + 10 * log10(f / 100 Hz) between samples.
    canvas = OnAxisResponseCanvas()
    canvas.update_plot(
        _FREQS,
        np.asarray([-10.0, 0.0, 10.0], dtype=np.float32),
        np.asarray([[80.0] * 3, [90.0] * 3, [100.0] * 3]),
        np.asarray(["Woofer"]),
        np.asarray([[70.0, 70.0, 70.0]]),
        np.asarray([170.0, -170.0, -150.0]),
        np.asarray([[10.0, 20.0, 30.0]]),
    )
    return canvas


def _spinorama_canvas() -> SpinoramaCanvas:
    canvas = SpinoramaCanvas()
    canvas.update_curves(
        SpinoramaCurves(
            freq_hz=_FREQS,
            on_axis_db=np.asarray([0.0, 0.0, 0.0]),
            listening_window_db=np.asarray([-1.0, -1.0, -1.0]),
            early_reflections_db=np.asarray([-2.0, -2.0, -2.0]),
            sound_power_db=np.asarray([-3.0, -3.0, -3.0]),
            estimated_in_room_db=np.asarray([-2.5, -2.5, -2.5]),
            early_reflections_di_db=np.asarray([2.0, 2.0, 2.0]),
            sound_power_di_db=np.asarray([3.0, 3.0, 3.0]),
        )
    )
    return canvas


def _isobar_canvas() -> IsobarCanvas:
    canvas = IsobarCanvas("Horizontal Isobar")
    angles = np.asarray([-90.0, 0.0, 90.0], dtype=np.float32)
    canvas.update_plot(_FREQS, angles, np.zeros((3, 3)), -30.0, 0.0, shading="gouraud")
    return canvas


@pytest.fixture
def marked():
    created = []
    marker = PlotMarker()

    def make(factory):
        canvas = factory()
        canvas.resize(520, 400)
        canvas.show()
        _APP.processEvents()
        canvas.draw()
        PlotMarkerOverlay(canvas, marker)
        created.append(canvas)
        return canvas

    make.marker = marker
    yield make
    for canvas in created:
        canvas.close()
        canvas.deleteLater()


def _point(canvas, freq_hz: float) -> tuple[float, float]:
    y_mid = float(np.mean(canvas.axes.get_ylim()))
    return tuple(canvas.axes.transData.transform((freq_hz, y_mid)))


def _hover(canvas, freq_hz: float) -> None:
    MouseEvent("motion_notify_event", canvas, *_point(canvas, freq_hz))._process()


def _click(canvas, freq_hz: float) -> None:
    x, y = _point(canvas, freq_hz)
    MouseEvent("button_press_event", canvas, x, y, MouseButton.LEFT)._process()
    MouseEvent("button_release_event", canvas, x, y, MouseButton.LEFT)._process()


def _readout(canvas) -> dict[str, tuple[str, str | None]] | None:
    """Visible readout rows as {curve: (value text, swatch colour)}."""
    boxes = [artist for artist in canvas.axes.get_children() if artist.get_gid() == "plot_marker_readout"]
    if not boxes or not boxes[0].get_visible():
        return None
    rows = {}
    for handle, text in zip(boxes[0].legend_handles, boxes[0].get_texts(), strict=True):
        label, value = text.get_text().split(": ", 1)
        color = None if handle.get_linestyle() == "None" else to_hex(handle.get_color())
        rows[label] = (value, color)
    return rows


def _readout_legend(canvas):
    boxes = [artist for artist in canvas.axes.get_children() if artist.get_gid() == "plot_marker_readout"]
    if not boxes or not boxes[0].get_visible():
        return None
    return boxes[0]


def _sum_spl(freq_hz: float) -> str:
    return f"{80.0 + 10.0 * math.log10(freq_hz / 100.0):.1f} dB"


def test_curve_plot_lists_each_visible_curve_with_swatch_and_interpolated_value(marked) -> None:
    canvas = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)

    _hover(canvas, 400.0)

    rows = _readout(canvas)
    woofer_color = to_hex(canvas.axes.get_legend().legend_handles[1].get_color())
    assert rows == {
        "Sum": (_sum_spl(marked.marker.frequency_hz), "#000000"),
        "Woofer": ("70.0 dB", woofer_color),
        "Frequency": (format_plot_frequency(marked.marker.frequency_hz), None),
    }


def test_curve_readout_ends_with_frequency_without_swatch_and_replaces_axis_label(marked) -> None:
    canvas = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)

    _hover(canvas, 1250.0)

    legend = _readout_legend(canvas)
    assert [text.get_text() for text in legend.get_texts()] == [
        f"Sum: {_sum_spl(marked.marker.frequency_hz)}",
        "Woofer: 70.0 dB",
        f"Frequency: {format_plot_frequency(marked.marker.frequency_hz)}",
    ]
    assert legend.legend_handles[-1].get_linestyle() == "None"
    canvas.draw()
    text_lefts = [text.get_window_extent(canvas.get_renderer()).x0 for text in legend.get_texts()]
    assert text_lefts == pytest.approx([text_lefts[0]] * len(text_lefts))
    marker_labels = [artist for artist in canvas.axes.get_children() if artist.get_gid() == "plot_marker_label"]
    assert marker_labels and not marker_labels[0].get_visible()

    canvas.set_series_visible("Sum", False)
    canvas.set_series_visible("Woofer", False)
    canvas.draw()
    legend = _readout_legend(canvas)
    assert [text.get_text() for text in legend.get_texts()] == [
        f"Frequency: {format_plot_frequency(marked.marker.frequency_hz)}"
    ]

    _click(canvas, 2000.0)
    locked_frequency = marked.marker.frequency_hz
    _hover(canvas, 5000.0)
    assert marked.marker.frequency_hz == locked_frequency
    assert [text.get_text() for text in _readout_legend(canvas).get_texts()] == [
        f"Frequency: {format_plot_frequency(locked_frequency)}"
    ]


def test_readout_follows_the_marker_and_holds_while_locked(marked) -> None:
    canvas = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)

    _hover(canvas, 200.0)
    assert _readout(canvas)["Sum"][0] == _sum_spl(marked.marker.frequency_hz)
    _click(canvas, 2000.0)
    locked = _readout(canvas)["Sum"][0]
    _hover(canvas, 5000.0)

    assert locked == _sum_spl(marked.marker.frequency_hz)
    assert locked != _sum_spl(200.0)
    assert _readout(canvas)["Sum"][0] == locked


def test_hiding_a_curve_removes_it_from_the_readout_and_showing_restores_it(marked) -> None:
    canvas = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)
    _hover(canvas, 1000.0)

    canvas.set_series_visible("Woofer", False)
    canvas.draw()
    assert set(_readout(canvas)) == {"Sum", "Frequency"}

    canvas.set_series_visible("Woofer", True)
    canvas.draw()
    assert set(_readout(canvas)) == {"Sum", "Woofer", "Frequency"}


def test_phase_traces_are_listed_in_degrees_only_when_shown(marked) -> None:
    canvas = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)
    _hover(canvas, 300.0)
    assert set(_readout(canvas)) == {"Sum", "Woofer", "Frequency"}

    canvas.show_phase_action.setChecked(True)
    canvas.draw()

    rows = _readout(canvas)
    decades = math.log10(marked.marker.frequency_hz / 100.0)
    assert rows["Woofer phase"][0] == f"{10.0 + 10.0 * decades:.1f} deg"
    # Sum phase wraps from 170 deg to -170 deg: it passes through 180 deg, not 0 deg.
    expected_sum = (170.0 + 20.0 * decades + 180.0) % 360.0 - 180.0
    assert rows["Sum phase"][0] == f"{expected_sum:.1f} deg"


def test_spinorama_lists_spl_and_di_curves_with_their_units(marked) -> None:
    canvas = marked(_spinorama_canvas)
    marked.marker.action.setChecked(True)

    _hover(canvas, 1000.0)

    rows = _readout(canvas)
    assert rows["On Axis"][0] == "0.0 dB SPL"
    assert rows["Sound Power"][0] == "-3.0 dB SPL"
    assert rows["ERDI"][0] == "2.0 dB DI"
    assert rows["SPDI"][0] == "3.0 dB DI"

    canvas.set_series_visible("ERDI", False)
    canvas.draw()
    assert "ERDI" not in _readout(canvas)


def test_electrical_impedance_lists_magnitude_and_phase_units(marked) -> None:
    def factory() -> ElectricalImpedanceCanvas:
        canvas = ElectricalImpedanceCanvas()
        canvas.update_plot(_FREQS, np.asarray(["LF"]), np.asarray([[8.0, 8.0, 8.0]]), np.asarray([[45.0, 45.0, 45.0]]))
        canvas.show_phase_action.setChecked(True)
        return canvas

    canvas = marked(factory)
    marked.marker.action.setChecked(True)

    _hover(canvas, 1000.0)

    assert {label: value for label, (value, _color) in _readout(canvas).items()} == {
        "LF": "8 Ω",
        "LF phase": "45.0°",
        "Frequency": format_plot_frequency(marked.marker.frequency_hz),
    }


def test_isobar_plot_shows_no_readout(marked) -> None:
    curves = marked(_on_axis_canvas)
    isobar = marked(_isobar_canvas)
    marked.marker.action.setChecked(True)

    _hover(curves, 1000.0)

    assert _readout(curves) is not None
    assert _readout(isobar) is None


def test_readout_is_removed_on_solve_reset_and_when_the_marker_is_off(marked) -> None:
    cleared = marked(_on_axis_canvas)
    kept = marked(_on_axis_canvas)
    marked.marker.action.setChecked(True)
    _hover(kept, 1000.0)
    assert _readout(cleared) is not None

    cleared.clear_for_solve()
    cleared.draw()
    assert _readout(cleared) is None

    marked.marker.action.setChecked(False)
    assert _readout(kept) is None


class _LogDomainOnAxisCanvas(OnAxisResponseCanvas):
    """Keeps log10(frequency) as x data, like an isobar in log-image mode."""

    def _x_limits_to_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        return float(10.0 ** limits[0]), float(10.0 ** limits[1])

    def _x_limits_from_user_domain(self, limits: tuple[float, float]) -> tuple[float, float]:
        return float(np.log10(limits[0])), float(np.log10(limits[1]))


def test_readout_sits_beside_the_marker_when_x_data_is_not_in_hz() -> None:
    canvas = _LogDomainOnAxisCanvas()
    canvas.update_plot(_FREQS, np.asarray([-10.0, 0.0, 10.0], dtype=np.float32), np.asarray([[80.0, 90.0, 100.0]] * 3))
    canvas.axes.set_xscale("linear")
    canvas.axes.set_xlim(2.0, 4.0)

    (box,) = canvas.update_marker_readout(math.sqrt(10.0) * 100.0)

    # 316 Hz is log10 = 2.5: a quarter of the way across the 2..4 axis.
    anchor = box.get_bbox_to_anchor()
    anchor_axes_x = canvas.axes.transAxes.inverted().transform((anchor.x0, anchor.y0))[0]
    assert anchor_axes_x == pytest.approx(0.25)
    canvas.deleteLater()
