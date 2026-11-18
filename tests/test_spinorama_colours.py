import dataclasses
import os

import numpy as np
import pytest
from matplotlib.colors import to_hex

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves, SpinoramaSeriesId, spinorama_curve_colors
from blab.ui.plots import SpinoramaCanvas
from blab.ui.theme import apply_application_theme

_APP = QApplication.instance() or QApplication([])

BLACK = "#000000"
WHITE = "#ffffff"
GREEN = "#2ca02c"
BLUE = "#1f77b4"
RED = "#d62728"
YELLOW = "#e6b800"

CONVENTIONAL_SPL = {
    "On Axis": BLACK,
    "Listen. Wind.": GREEN,
    "Early Refl.": BLUE,
    "Sound Power": RED,
    "PIR": YELLOW,
}


def _curves(*, spherical: bool = False) -> SpinoramaCurves:
    curves = SpinoramaCurves(
        freq_hz=np.asarray([100.0, 1000.0, 10000.0], dtype=np.float32),
        on_axis_db=np.zeros(3),
        listening_window_db=np.full(3, -1.0),
        early_reflections_db=np.full(3, -2.0),
        sound_power_db=np.full(3, -3.0),
        estimated_in_room_db=np.full(3, -2.5),
        early_reflections_di_db=np.full(3, 2.0),
        sound_power_di_db=np.full(3, 3.0),
    )
    if spherical:
        curves = dataclasses.replace(curves, sound_power_di_label="Spherical DI")
    return curves


def _styles(axes, names) -> dict[str, tuple[str, str]]:
    return {
        line.get_label(): (to_hex(line.get_color()), line.get_linestyle())
        for line in axes.get_lines()
        if line.get_label() in names
    }


def _di_label(spherical: bool) -> str:
    return "Spherical DI" if spherical else "SPDI"


def _assert_conventional(canvas: SpinoramaCanvas, *, spherical: bool, on_axis: str = BLACK) -> None:
    expected_spl = {name: (color, "-") for name, color in CONVENTIONAL_SPL.items()}
    expected_spl["On Axis"] = (on_axis, "-")
    assert _styles(canvas.axes, expected_spl) == expected_spl
    expected_di = {"ERDI": (BLUE, "--"), _di_label(spherical): (RED, "--")}
    assert _styles(canvas.di_axes, expected_di) == expected_di


@pytest.mark.parametrize("spherical", [False, True])
def test_spinorama_curves_use_conventional_colours(spherical):
    canvas = SpinoramaCanvas()

    canvas.update_curves(_curves(spherical=spherical))

    _assert_conventional(canvas, spherical=spherical)


def test_toggling_spherical_spin_never_changes_colours():
    canvas = SpinoramaCanvas()

    for spherical in (False, True, False, True, False):
        canvas.update_curves(_curves(spherical=spherical))
        _assert_conventional(canvas, spherical=spherical)


@pytest.mark.parametrize("spherical", [False, True])
def test_on_axis_is_white_on_a_dark_plot_background(spherical):
    canvas = SpinoramaCanvas()
    canvas.figure.set_facecolor("#293134")
    canvas.axes.set_facecolor("#293134")

    canvas.update_curves(_curves(spherical=spherical))

    _assert_conventional(canvas, spherical=spherical, on_axis=WHITE)


def test_on_axis_follows_figure_background_when_axes_are_transparent():
    canvas = SpinoramaCanvas()
    canvas.figure.set_facecolor("#1e1e1e")
    canvas.axes.set_facecolor("none")

    canvas.update_curves(_curves())

    assert _styles(canvas.axes, {"On Axis"}) == {"On Axis": (WHITE, "-")}


def test_on_axis_turns_white_when_background_darkens_between_updates():
    canvas = SpinoramaCanvas()
    canvas.update_curves(_curves())
    canvas.axes.set_facecolor("#293134")

    canvas.update_curves(_curves())

    assert _styles(canvas.axes, {"On Axis"}) == {"On Axis": (WHITE, "-")}


def test_on_axis_is_white_when_the_application_uses_the_dark_theme():
    canvas = SpinoramaCanvas()
    try:
        apply_application_theme("dark")

        canvas.update_curves(_curves())

        assert _styles(canvas.axes, {"On Axis"}) == {"On Axis": (WHITE, "-")}
    finally:
        apply_application_theme("system")
        canvas.close()
        canvas.deleteLater()


def test_colour_table_is_keyed_by_stable_series_identity():
    colors = spinorama_curve_colors()

    assert colors[SpinoramaSeriesId.ON_AXIS] == BLACK
    assert colors[SpinoramaSeriesId.SOUND_POWER_DI] == RED


@pytest.mark.parametrize("spherical", [False, True])
def test_headless_spinorama_png_uses_the_same_conventional_colours(spherical, tmp_path, monkeypatch):
    from blab import plotting

    figures = []
    close = plotting.plt.close
    monkeypatch.setattr(plotting.plt, "close", figures.append)

    plotting._save_spinorama_plot(tmp_path / "spinorama.png", _curves(spherical=spherical), 6.0, 4.0, 50)
    close(figures[0])

    spl_axes, di_axes = figures[0].axes
    expected_spl = {name: (color, "-") for name, color in CONVENTIONAL_SPL.items()}
    assert _styles(spl_axes, expected_spl) == expected_spl
    expected_di = {"ERDI": (BLUE, "--"), _di_label(spherical): (RED, "--")}
    assert _styles(di_axes, expected_di) == expected_di
