"""Canvas support for the per-plot "25 dB/dec" (IEC 60263) toolbar toggle."""

from __future__ import annotations

from matplotlib.axes import Axes
from PySide6.QtGui import QAction

from blab.ui.plot_scale import iec_axes_box


class IecScaleCanvasMixin:
    """Shrink the axes box to the IEC 60263 ratio while ``iec_scale_action`` is checked.

    Mix in before ``RawCoordinatePlotCanvas``. The adaptive layout still decides the
    available area (so decorations keep their margins); this mixin only fits and
    centres the primary and secondary axes inside it, before every draw and after
    every layout change, so the ratio follows limit changes and resizes.

    Host contract: ``RawCoordinatePlotCanvas`` provides ``axes`` and
    ``secondary_axes`` (``None`` on single-axis plots).
    """

    axes: Axes
    secondary_axes: Axes | None

    def __init__(self, *args, **kwargs) -> None:
        self._iec_scale_enabled = False
        super().__init__(*args, **kwargs)
        self.iec_scale_action = QAction("25 dB/dec", self)
        self.iec_scale_action.setToolTip("Scale the plot to 25 dB per frequency decade (IEC 60263)")
        self.iec_scale_action.setCheckable(True)
        self.iec_scale_action.toggled.connect(self._set_iec_scale_enabled)

    def _set_iec_scale_enabled(self, enabled: bool) -> None:
        self._iec_scale_enabled = bool(enabled)
        self._series_legend_entries = ()
        self._apply_series_visibility()
        self._apply_layout()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _series_legend_kwargs(self) -> dict:
        options = super()._series_legend_kwargs()
        if not self._iec_scale_enabled:
            return options
        return {
            **options,
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 1.02),
            "ncols": 4,
            "borderaxespad": 0.0,
            "frameon": False,
        }

    def _after_layout_profile_applied(self) -> None:
        super()._after_layout_profile_applied()
        self._fit_iec_axes_box()

    # A pending empty-plot rebuild resets the axes limits and layout. Run it before
    # fitting, so the box matches the limits about to be drawn; the base class then
    # finds nothing pending.

    def draw(self, *args, **kwargs):
        self._ensure_empty_plot()
        self._fit_iec_axes_box()
        return super().draw(*args, **kwargs)

    def print_figure(self, *args, **kwargs):
        self._ensure_empty_plot()
        self._fit_iec_axes_box()
        return super().print_figure(*args, **kwargs)

    def _fit_iec_axes_box(self) -> None:
        if not self._iec_scale_enabled:
            return
        figure = self.figure
        fig_width, fig_height = (float(value) for value in figure.get_size_inches())
        available = self.axes.get_subplotspec().get_position(figure)
        x0, y0, width, height = iec_axes_box(
            (
                available.x0 * fig_width,
                available.y0 * fig_height,
                available.width * fig_width,
                available.height * fig_height,
            ),
            self._x_limits_to_user_domain(self.axes.get_xlim()),
            self.axes.get_ylim(),
        )
        box = (x0 / fig_width, y0 / fig_height, width / fig_width, height / fig_height)
        for axes in (self.axes, self.secondary_axes):
            if axes is not None:
                axes.set_position(box)
