"""Per-curve value readout at the shared frequency Marker.

``interpolate_log_frequency`` reads a curve at any frequency. ``MarkerReadoutBox``
draws the box that lists each visible curve's value with a colour swatch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

MARKER_READOUT_GID = "plot_marker_readout"
# Phase in degrees wraps once per turn.
PHASE_PERIOD_DEG = 360.0


def interpolate_log_frequency(
    freqs_hz: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    freq_hz: float,
    *,
    period: float | None = None,
) -> float | None:
    """Value of a curve at ``freq_hz``, linear over log frequency.

    Samples with a missing frequency or value are skipped, and ``freq_hz`` is
    clamped to the remaining data range. With ``period`` (360 for phase), the
    value takes the short way across a wrap and is returned wrapped to
    ``[-period/2, period/2)``. Returns ``None`` when no finite sample is left.
    """
    freqs = np.asarray(freqs_hz, dtype=float).ravel()
    data = np.asarray(values, dtype=float).ravel()
    if freqs.shape != data.shape:
        raise ValueError("Frequencies and values must have the same length.")
    usable = np.isfinite(freqs) & np.isfinite(data) & (freqs > 0.0)
    if not np.any(usable):
        return None
    log_freqs = np.log10(freqs[usable])
    data = data[usable]
    order = np.argsort(log_freqs, kind="stable")
    log_freqs = log_freqs[order]
    data = data[order]
    if period is not None:
        # Unwrap so that neighbouring samples never differ by more than half a period.
        steps = np.diff(data)
        steps = (steps + period / 2.0) % period - period / 2.0
        data = np.concatenate(([data[0]], data[0] + np.cumsum(steps)))
    target = float(np.clip(np.log10(float(freq_hz)), log_freqs[0], log_freqs[-1]))
    value = float(np.interp(target, log_freqs, data))
    if period is not None:
        value = (value + period / 2.0) % period - period / 2.0
    return value


@dataclass(frozen=True)
class MarkerReadoutCurve:
    """A curve listed in the readout; ``period`` is set for wrapped values such as phase."""

    label: str
    line: Line2D
    period: float | None = None


def phase_readout_curves(labels: Sequence[str], phase_lines: Mapping[str, Line2D]) -> tuple[MarkerReadoutCurve, ...]:
    """One wrapped "<label> phase" readout curve per series, in ``labels`` order."""
    return tuple(MarkerReadoutCurve(f"{label} phase", phase_lines[label], period=PHASE_PERIOD_DEG) for label in labels)


@dataclass(frozen=True)
class MarkerReadoutRow:
    label: str
    color: Any
    text: str
    linestyle: str = "-"


class MarkerReadoutBox:
    """A legend-like box listing a colour swatch and value per curve, drawn by blitting."""

    def __init__(self, *, fontsize: float) -> None:
        self._fontsize = fontsize
        self._legend: Legend | None = None
        self._layout: tuple | None = None

    @property
    def legend(self) -> Legend | None:
        return self._legend

    def hide(self) -> None:
        if self._legend is not None:
            self._legend.set_visible(False)

    def update(self, axes, rows: Sequence[MarkerReadoutRow], *, marker_axes_x: float) -> tuple[object, ...]:
        """Show ``rows`` at the top, beside the marker on its roomier side; return the artist to blit.

        ``marker_axes_x`` is the marker position as a fraction of the axes width.
        """
        if not rows:
            self.hide()
            return ()
        marker_on_right = marker_axes_x > 0.5
        loc = "upper right" if marker_on_right else "upper left"
        layout = (loc, tuple((row.label, str(row.color), row.linestyle) for row in rows))
        legend = self._legend
        # Clearing the axes for new data drops the box; rebuild it on demand.
        if legend is None or legend not in axes.get_children() or layout != self._layout:
            if legend is not None and legend in axes.get_children():
                legend.remove()
            legend = self._build(axes, rows, loc)
            self._layout = layout
        else:
            for text, row in zip(legend.get_texts(), rows, strict=True):
                text.set_text(self._row_text(row))
        legend.set_bbox_to_anchor((marker_axes_x, 1.0), transform=axes.transAxes)
        legend.set_visible(True)
        return (legend,)

    def _build(self, axes, rows: Sequence[MarkerReadoutRow], loc: str) -> Legend:
        handles = [Line2D([], [], color=row.color, linestyle=row.linestyle, linewidth=2.0) for row in rows]
        legend = Legend(
            axes,
            handles,
            [self._row_text(row) for row in rows],
            loc=loc,
            fontsize=self._fontsize,
            framealpha=0.92,
            handlelength=1.6,
            handletextpad=0.5,
            borderpad=0.4,
            borderaxespad=0.6,
            labelspacing=0.25,
        )
        legend.set_gid(MARKER_READOUT_GID)
        legend.set_zorder(11)
        legend.set_animated(True)
        axes.add_artist(legend)
        self._legend = legend
        return legend

    @staticmethod
    def _row_text(row: MarkerReadoutRow) -> str:
        return f"{row.label}: {row.text}"


__all__ = [
    "MARKER_READOUT_GID",
    "MarkerReadoutBox",
    "MarkerReadoutCurve",
    "MarkerReadoutRow",
    "PHASE_PERIOD_DEG",
    "interpolate_log_frequency",
    "phase_readout_curves",
]
