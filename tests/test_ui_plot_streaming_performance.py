import os
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.main_window import MainWindow
from blab.ui.plots import ImpedanceCanvas, IsobarCanvas, OnAxisResponseCanvas, SpinoramaCanvas

_APP = QApplication.instance() or QApplication([])


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

    MainWindow._request_live_plot_refresh(window)
    MainWindow._request_live_plot_refresh(window)

    assert window._live_plot_refresh_dirty is True
    assert timer.start_count == 1


def test_live_plot_refresh_flushes_only_actively_visible_entries() -> None:
    refresh_calls = []
    window = SimpleNamespace(
        _live_plot_refresh_dirty=True,
        _refresh_plots=lambda **options: refresh_calls.append(options),
    )

    MainWindow._flush_live_plot_refresh(window)

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
    curves = SpinoramaCurves(
        freq_hz=freqs,
        on_axis_db=np.asarray([0.0, 0.0, 0.0]),
        listening_window_db=np.asarray([-1.0, -1.0, -1.0]),
        early_reflections_db=np.asarray([-2.0, -2.0, -2.0]),
        sound_power_db=np.asarray([-3.0, -3.0, -3.0]),
        estimated_in_room_db=np.asarray([-2.5, -2.5, -2.5]),
        early_reflections_di_db=np.asarray([2.0, 2.0, 2.0]),
        sound_power_di_db=np.asarray([3.0, 3.0, 3.0]),
    )
    spinorama.update_curves(curves)
    spl_lines = dict(spinorama._spl_lines)
    di_lines = dict(spinorama._di_lines)
    spinorama.update_curves(curves)
    assert spinorama._spl_lines == spl_lines
    assert spinorama._di_lines == di_lines
