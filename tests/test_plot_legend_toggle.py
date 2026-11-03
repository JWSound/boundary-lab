"""Clicking a plot legend entry hides or shows that curve."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.colors import to_hex
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.electrical_impedance_plot import ElectricalImpedanceCanvas
from blab.ui.excursion_plot import ExcursionCanvas
from blab.ui.group_delay_plot import GroupDelayCanvas
from blab.ui.max_spl_plot import MaxSplCanvas
from blab.ui.plot_marker import PlotMarker, PlotMarkerOverlay
from blab.ui.plots import ImpedanceCanvas, OnAxisResponseCanvas, SpinoramaCanvas

_APP = QApplication.instance() or QApplication([])

FREQS = np.asarray([100.0, 1000.0, 10_000.0], dtype=np.float32)


@pytest.fixture
def max_spl():
    canvas = MaxSplCanvas()
    canvas.resize(640, 420)
    canvas.update_plot(
        FREQS,
        np.asarray(["LF", "HF"]),
        np.asarray([[100.0, 104.0, 102.0], [90.0, 95.0, 110.0]], dtype=np.float32),
    )
    canvas.draw()
    yield canvas
    canvas.close()
    canvas.deleteLater()


def _line(axes, label: str):
    return next(line for line in axes.get_lines() if line.get_label() == label)


def _legend_text(axes, label: str):
    legend = axes.get_legend()
    return next(text for text in legend.get_texts() if text.get_text() == label)


def _click_legend_entry(canvas, axes, label: str, button=MouseButton.LEFT) -> None:
    canvas.draw()
    box = _legend_text(axes, label).get_window_extent()
    x, y = (box.x0 + box.x1) / 2.0, (box.y0 + box.y1) / 2.0
    MouseEvent("button_press_event", canvas, x, y, button)._process()
    MouseEvent("button_release_event", canvas, x, y, button)._process()


def test_clicking_a_legend_entry_hides_its_curve_and_clicking_again_shows_it(max_spl) -> None:
    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert not _line(max_spl.axes, "HF").get_visible()
    assert _line(max_spl.axes, "LF").get_visible()

    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert _line(max_spl.axes, "HF").get_visible()


def _legend_handle(axes, label: str):
    legend = axes.get_legend()
    return next(
        handle
        for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=True)
        if text.get_text() == label
    )


def _is_greyed(axes, label: str) -> bool:
    text_alpha = _legend_text(axes, label).get_alpha()
    handle_alpha = _legend_handle(axes, label).get_alpha()
    return text_alpha is not None and text_alpha < 0.5 and handle_alpha is not None and handle_alpha < 0.5


def test_hidden_entries_stay_in_the_legend_greyed_out(max_spl) -> None:
    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert [text.get_text() for text in max_spl.axes.get_legend().get_texts()] == ["LF", "HF"]
    assert _is_greyed(max_spl.axes, "HF")
    assert not _is_greyed(max_spl.axes, "LF")

    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert not _is_greyed(max_spl.axes, "HF")


def _menu_action(canvas, label: str):
    return next(action for action in canvas.trace_filter_menu.actions() if action.text() == label)


def test_legend_click_and_traces_menu_share_one_state(max_spl) -> None:
    changes = []
    max_spl.trace_visibility_changed.connect(lambda: changes.append(max_spl.is_series_visible("HF")))

    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert not _menu_action(max_spl, "HF").isChecked()
    assert changes == [False]

    _menu_action(max_spl, "HF").setChecked(True)

    assert _line(max_spl.axes, "HF").get_visible()
    assert not _is_greyed(max_spl.axes, "HF")
    assert changes == [False, True]


def test_hidden_state_survives_a_data_update_with_the_same_series(max_spl) -> None:
    _click_legend_entry(max_spl, max_spl.axes, "HF")

    max_spl.update_plot(
        FREQS,
        np.asarray(["LF", "HF"]),
        np.asarray([[101.0, 105.0, 103.0], [91.0, 96.0, 111.0]], dtype=np.float32),
    )

    assert not _line(max_spl.axes, "HF").get_visible()
    assert _is_greyed(max_spl.axes, "HF")
    assert not _menu_action(max_spl, "HF").isChecked()
    _click_legend_entry(max_spl, max_spl.axes, "HF")
    assert _line(max_spl.axes, "HF").get_visible()


def test_legend_click_inside_the_axes_does_not_place_the_crosshair(max_spl) -> None:
    series_lines = set(max_spl.axes.get_lines())

    _click_legend_entry(max_spl, max_spl.axes, "LF")

    crosshair_lines = [line for line in max_spl.axes.get_lines() if line not in series_lines]
    assert not any(line.get_visible() for line in crosshair_lines)


def test_legend_click_inside_the_axes_does_not_lock_the_marker(max_spl) -> None:
    marker = PlotMarker()
    PlotMarkerOverlay(max_spl, marker)
    marker.action.setChecked(True)

    _click_legend_entry(max_spl, max_spl.axes, "HF")

    assert not marker.locked
    assert not _line(max_spl.axes, "HF").get_visible()


def _impedance():
    canvas = ImpedanceCanvas()
    canvas.update_plot(
        FREQS,
        np.asarray(["Woofer", "Tweeter"]),
        np.asarray([[2.0, 8.0, 4.0], [1.0, 3.0, 2.0]], dtype=np.float32),
        np.asarray([[0.0, 6.0, -2.0], [0.5, 1.0, -0.5]], dtype=np.float32),
    )
    return canvas, "Tweeter Z imag", "Tweeter"


def _on_axis():
    canvas = OnAxisResponseCanvas()
    response = np.asarray([[80.0, 81.0, 80.0], [82.0, 83.0, 82.0], [84.0, 85.0, 84.0]])
    canvas.update_plot(
        FREQS,
        np.asarray([-10.0, 0.0, 10.0], dtype=np.float32),
        response,
        np.asarray(["LF", "HF"]),
        np.asarray([[80.0, 70.0, 50.0], [60.0, 75.0, 82.0]]),
    )
    return canvas, "HF", "HF"


def _electrical():
    canvas = ElectricalImpedanceCanvas()
    canvas.update_plot(
        FREQS,
        np.asarray(["LF", "HF"]),
        np.asarray([[4.0, 20.0, 8.0], [8.0, 10.0, 12.0]], dtype=np.float32),
        np.asarray([[10.0, -20.0, -9.0], [10.0, 20.0, 30.0]], dtype=np.float32),
    )
    return canvas, "HF", "HF"


def _group_delay():
    canvas = GroupDelayCanvas()
    canvas.update_plot(FREQS, np.asarray(["LF", "HF"]), np.asarray([[1.0, 0.5, 0.1], [0.3, 0.2, 0.1]]))
    return canvas, "HF", "HF"


def _excursion():
    canvas = ExcursionCanvas()
    canvas.update_plot(FREQS, np.asarray(["Woofer", "Port"]), np.asarray([[1.0, 0.5, 0.1], [0.3, 0.2, 0.1]]))
    return canvas, "Port", "Port"


@pytest.mark.parametrize("factory", [_impedance, _on_axis, _electrical, _group_delay, _excursion])
def test_every_traces_menu_plot_toggles_series_from_its_legend(factory) -> None:
    canvas, legend_label, series_label = factory()
    canvas.resize(640, 420)
    try:
        series_lines = [
            line
            for line in canvas.axes.get_lines()
            if line.get_label() == series_label or line.get_label().startswith(f"{series_label} ")
        ]
        assert series_lines

        _click_legend_entry(canvas, canvas.axes, legend_label)

        assert not any(line.get_visible() for line in series_lines)
        assert not _menu_action(canvas, series_label).isChecked()
        assert not canvas.is_series_visible(series_label)
        assert _is_greyed(canvas.axes, legend_label)

        _menu_action(canvas, series_label).setChecked(True)

        assert all(line.get_visible() for line in series_lines)
        assert not _is_greyed(canvas.axes, legend_label)
    finally:
        canvas.close()
        canvas.deleteLater()


def _spinorama_curves(offset: float = 0.0, *, spherical: bool = False) -> SpinoramaCurves:
    curves = SpinoramaCurves(
        freq_hz=FREQS,
        on_axis_db=np.asarray([0.0, 0.0, 0.0]) + offset,
        listening_window_db=np.asarray([-1.0, -1.0, -1.0]) + offset,
        early_reflections_db=np.asarray([-2.0, -2.0, -2.0]) + offset,
        sound_power_db=np.asarray([-3.0, -3.0, -3.0]) + offset,
        estimated_in_room_db=np.asarray([-2.5, -2.5, -2.5]) + offset,
        early_reflections_di_db=np.asarray([2.0, 2.0, 2.0]) + offset,
        sound_power_di_db=np.asarray([3.0, 3.0, 3.0]) + offset,
    )
    return replace(curves, sound_power_di_label="Spherical DI") if spherical else curves


@pytest.fixture
def spinorama():
    canvas = SpinoramaCanvas()
    canvas.resize(640, 480)
    canvas.update_curves(_spinorama_curves())
    canvas.draw()
    yield canvas
    canvas.close()
    canvas.deleteLater()


def _visible_curves(canvas) -> set[str]:
    return {
        line.get_label()
        for axes in (canvas.axes, canvas.di_axes)
        for line in axes.get_lines()
        if line.get_visible() and not line.get_label().startswith("_")
    }


ALL_PLANAR_CURVES = {"On Axis", "Listen. Wind.", "Early Refl.", "Sound Power", "PIR", "ERDI", "SPDI"}


def test_spinorama_spl_and_di_curves_toggle_independently(spinorama) -> None:
    changes = []
    spinorama.trace_visibility_changed.connect(lambda: changes.append(True))

    _click_legend_entry(spinorama, spinorama.axes, "Early Refl.")
    _click_legend_entry(spinorama, spinorama.axes, "ERDI")

    assert _visible_curves(spinorama) == ALL_PLANAR_CURVES - {"Early Refl.", "ERDI"}
    assert _is_greyed(spinorama.axes, "Early Refl.")
    assert _is_greyed(spinorama.axes, "ERDI")
    assert not _is_greyed(spinorama.axes, "SPDI")
    assert len(changes) == 2

    _click_legend_entry(spinorama, spinorama.axes, "ERDI")

    assert _visible_curves(spinorama) == ALL_PLANAR_CURVES - {"Early Refl."}
    assert not _is_greyed(spinorama.axes, "ERDI")


def test_spinorama_visibility_drives_data_export_selection(spinorama) -> None:
    spinorama.set_series_visible("Sound Power", False)

    assert not spinorama.is_series_visible("Sound Power")
    assert spinorama.is_series_visible("On Axis")
    assert _is_greyed(spinorama.axes, "Sound Power")
    assert spinorama.has_selected_traces()

    for name in ALL_PLANAR_CURVES:
        spinorama.set_series_visible(name, False)

    assert not spinorama.has_selected_traces()
    assert _visible_curves(spinorama) == set()


def test_spinorama_hidden_curves_survive_a_data_update(spinorama) -> None:
    _click_legend_entry(spinorama, spinorama.axes, "PIR")

    spinorama.update_curves(_spinorama_curves(offset=1.0))

    assert _visible_curves(spinorama) == ALL_PLANAR_CURVES - {"PIR"}
    assert _is_greyed(spinorama.axes, "PIR")


def test_hidden_sound_power_di_stays_hidden_across_the_spherical_rename(spinorama) -> None:
    _click_legend_entry(spinorama, spinorama.axes, "SPDI")

    spinorama.update_curves(_spinorama_curves(spherical=True))

    assert not spinorama.is_series_visible("Spherical DI")
    assert "Spherical DI" not in _visible_curves(spinorama)
    assert _is_greyed(spinorama.axes, "Spherical DI")

    _click_legend_entry(spinorama, spinorama.axes, "Spherical DI")
    spinorama.update_curves(_spinorama_curves())

    assert spinorama.is_series_visible("SPDI")
    assert "SPDI" in _visible_curves(spinorama)


def test_spinorama_legend_follows_a_curve_colour_change_between_updates(spinorama) -> None:
    spinorama.axes.set_facecolor("#293134")

    spinorama.update_curves(_spinorama_curves(offset=1.0))

    assert to_hex(_legend_handle(spinorama.axes, "On Axis").get_color()) == to_hex(
        _line(spinorama.axes, "On Axis").get_color()
    )
