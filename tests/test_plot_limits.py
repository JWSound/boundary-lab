from __future__ import annotations

import numpy as np
import pytest

from blab.ui.plot_limits_dialog import PlotLimitsDialog
from blab.ui.plots import ImpedanceCanvas, IsobarCanvas, PlotAxisLimits


def test_plot_limits_dialog_defaults_to_auto_and_disables_fields(qapp) -> None:
    del qapp
    dialog = PlotLimitsDialog(
        "Test Plot",
        PlotAxisLimits(20.0, 20_000.0, -10.0, 10.0),
        automatic=True,
    )

    assert dialog.auto_checkbox.isChecked()
    assert not dialog.limit_group.isEnabled()
    assert dialog.limits() is None

    dialog.auto_checkbox.setChecked(False)

    assert dialog.limit_group.isEnabled()
    assert dialog.limits() == PlotAxisLimits(20.0, 20_000.0, -10.0, 10.0)


def test_plot_limits_dialog_rejects_invalid_manual_ranges(qapp) -> None:
    del qapp
    dialog = PlotLimitsDialog(
        "Test Plot",
        PlotAxisLimits(20.0, 20_000.0, -10.0, 10.0),
        automatic=False,
    )
    dialog.x_min_edit.setText("200")
    dialog.x_max_edit.setText("100")

    with pytest.raises(ValueError, match="lower X"):
        dialog.limits()


def test_manual_limits_survive_plot_updates_and_auto_recalculates(qapp) -> None:
    del qapp
    canvas = ImpedanceCanvas()
    frequencies = np.asarray([20.0, 100.0, 1000.0, 20_000.0])
    names = np.asarray(["Driver"])
    real = np.asarray([[1.0, 2.0, 3.0, 4.0]])
    imaginary = np.asarray([[-2.0, -1.0, 1.0, 2.0]])
    manual = PlotAxisLimits(50.0, 5000.0, -5.0, 5.0)

    canvas.set_axis_limits(manual)
    canvas.update_plot(frequencies, names, real, imaginary)

    assert canvas.automatic_axis_limits is False
    np.testing.assert_allclose(canvas.axes.get_xlim(), (50.0, 5000.0))
    np.testing.assert_allclose(canvas.axes.get_ylim(), (-5.0, 5.0))

    canvas.set_axis_limits(None)

    assert canvas.automatic_axis_limits is True
    np.testing.assert_allclose(canvas.axes.get_xlim(), (20.0, 20_000.0))
    assert canvas.axes.get_ylim()[0] < -2.0
    assert canvas.axes.get_ylim()[1] > 4.0


def test_isobar_manual_frequency_limits_are_converted_for_image_rendering(qapp) -> None:
    del qapp
    canvas = IsobarCanvas("Horizontal Isobar")
    canvas._x_axis_mode = "log_image"
    canvas._configure_axes()

    canvas.set_axis_limits(PlotAxisLimits(100.0, 10_000.0, -90.0, 90.0))

    np.testing.assert_allclose(canvas.axes.get_xlim(), (2.0, 4.0))
    assert canvas.displayed_axis_limits() == PlotAxisLimits(100.0, 10_000.0, -90.0, 90.0)
