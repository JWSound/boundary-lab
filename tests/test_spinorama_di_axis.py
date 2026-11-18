"""The Spinorama DI axis puts DI 0 on the frequency axis or on a major gridline."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blab.spinorama import SpinoramaCurves
from blab.ui.plots import PlotAxisLimits, SpinoramaCanvas, spinorama_di_limits

_APP = QApplication.instance() or QApplication([])


def test_non_negative_di_starts_the_axis_at_zero() -> None:
    assert spinorama_di_limits([np.asarray([0.0, 3.0, 12.0]), np.asarray([1.0, 7.5])]) == (0.0, 50.0)


def test_negative_di_rounds_down_to_the_next_multiple_of_ten() -> None:
    assert spinorama_di_limits([np.asarray([4.0, -0.5]), np.asarray([6.0])]) == (-10.0, 40.0)
    assert spinorama_di_limits([np.asarray([-10.0, 2.0])]) == (-10.0, 40.0)
    assert spinorama_di_limits([np.asarray([-23.0])]) == (-30.0, 20.0)


def test_no_visible_or_finite_di_starts_the_axis_at_zero() -> None:
    assert spinorama_di_limits([]) == (0.0, 50.0)
    assert spinorama_di_limits([np.asarray([np.nan, np.inf]), np.asarray([])]) == (0.0, 50.0)
    assert spinorama_di_limits([np.asarray([np.nan, -3.0])]) == (-10.0, 40.0)


# Canvas ----------------------------------------------------------------


def _curves(*, erdi: float, spdi: float) -> SpinoramaCurves:
    freqs = np.asarray([100.0, 1000.0, 10_000.0], dtype=np.float32)
    flat = np.zeros(3)
    return SpinoramaCurves(
        freq_hz=freqs,
        on_axis_db=flat,
        listening_window_db=flat - 1.0,
        early_reflections_db=flat - 2.0,
        sound_power_db=flat - 3.0,
        estimated_in_room_db=flat - 2.5,
        early_reflections_di_db=np.asarray([2.0, erdi, 4.0]),
        sound_power_di_db=np.asarray([3.0, spdi, 6.0]),
    )


@pytest.fixture
def spinorama():
    canvas = SpinoramaCanvas()
    canvas.resize(640, 420)
    yield canvas
    canvas.close()
    canvas.deleteLater()


def _major_gridline_fractions(axes) -> list[float]:
    low, high = axes.get_ylim()
    ticks = [tick for tick in axes.yaxis.get_majorticklocs() if low - 1e-9 <= tick <= high + 1e-9]
    return [round((tick - low) / (high - low), 6) for tick in ticks]


def test_canvas_puts_di_zero_on_the_frequency_axis_when_no_di_is_negative(spinorama) -> None:
    spinorama.update_curves(_curves(erdi=1.0, spdi=8.0))

    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))
    assert _major_gridline_fractions(spinorama.di_axes) == _major_gridline_fractions(spinorama.axes)


def test_canvas_puts_di_zero_on_a_major_gridline_when_di_is_negative(spinorama) -> None:
    spinorama.update_curves(_curves(erdi=1.0, spdi=-12.0))

    low, high = spinorama.di_axes.get_ylim()
    assert (low, high) == pytest.approx((-20.0, 30.0))
    assert 0.0 in spinorama.di_axes.yaxis.get_majorticklocs()
    assert _major_gridline_fractions(spinorama.di_axes) == _major_gridline_fractions(spinorama.axes)


def test_hiding_and_showing_di_curves_re_evaluates_the_di_axis(spinorama) -> None:
    spinorama.update_curves(_curves(erdi=-3.0, spdi=-12.0))

    spinorama.set_series_visible("SPDI", False)
    assert spinorama.di_axes.get_ylim() == pytest.approx((-10.0, 40.0))

    spinorama.set_series_visible("ERDI", False)
    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))

    spinorama.set_series_visible("SPDI", True)
    assert spinorama.di_axes.get_ylim() == pytest.approx((-20.0, 30.0))


def test_hidden_spherical_di_stays_out_of_the_di_axis_after_a_data_update(spinorama) -> None:
    spinorama.update_curves(_curves(erdi=1.0, spdi=-12.0))
    spinorama.set_series_visible("SPDI", False)

    spinorama.update_curves(replace(_curves(erdi=1.0, spdi=-25.0), sound_power_di_label="Spherical DI"))

    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))


def test_manual_limits_cover_the_primary_axis_only_while_the_automatic_di_rule_applies(spinorama) -> None:
    spinorama.set_axis_limits(PlotAxisLimits(x_min=50.0, x_max=5000.0, y_min=-30.0, y_max=0.0))
    spinorama.update_curves(_curves(erdi=-3.0, spdi=2.0))

    assert spinorama.axes.get_ylim() == pytest.approx((-30.0, 0.0))
    assert spinorama.di_axes.get_ylim() == pytest.approx((-10.0, 40.0))

    spinorama.set_series_visible("ERDI", False)
    assert spinorama.axes.get_ylim() == pytest.approx((-30.0, 0.0))
    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))

    # The public limits contract reports the primary frequency/SPL axes. A
    # separate manual DI range is deliberately outside this feature's scope.
    assert spinorama.displayed_axis_limits() == PlotAxisLimits(50.0, 5000.0, -30.0, 0.0)


def test_empty_canvas_and_solve_reset_show_a_zero_based_di_axis(spinorama) -> None:
    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))
    spinorama.update_curves(_curves(erdi=-3.0, spdi=2.0))

    spinorama.show()
    _APP.processEvents()
    spinorama.clear_for_solve()

    assert spinorama.di_axes.get_ylim() == pytest.approx((0.0, 50.0))
