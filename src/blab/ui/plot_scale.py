"""IEC 60263 plot scale: the axes-box ratio for dB versus log-frequency plots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

IEC_DB_PER_DECADE = 25.0

Box = tuple[float, float, float, float]


class SpinoramaLegendPlacement(StrEnum):
    """Supported legend positions for a Spinorama IEC layout."""

    RIGHT = "right"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class SpinoramaIecLayout:
    """Axes and legend placement for a Spinorama 25 dB/dec layout."""

    axes_box: Box
    legend_placement: SpinoramaLegendPlacement
    legend_anchor: tuple[float, float] | None


def iec_axes_box(
    available: Box,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    db_per_decade: float = IEC_DB_PER_DECADE,
) -> Box:
    """Return the largest box centred in ``available`` that shows ``db_per_decade``.

    ``available`` is ``(x0, y0, width, height)`` in physical units (any length unit,
    the same on both axes). The returned box keeps height / width equal to
    (y span / ``db_per_decade``) / (x decades). When the limits do not define a
    ratio, the available box is returned unchanged.
    """

    x0, y0, width, height = (float(value) for value in available)
    x_low, x_high = sorted(float(value) for value in x_limits)
    y_span = abs(float(y_limits[1]) - float(y_limits[0]))
    if width <= 0.0 or height <= 0.0 or x_low <= 0.0 or db_per_decade <= 0.0:
        return x0, y0, width, height
    decades = math.log10(x_high / x_low)
    if not (math.isfinite(decades) and decades > 0.0 and math.isfinite(y_span) and y_span > 0.0):
        return x0, y0, width, height
    ratio = (y_span / db_per_decade) / decades
    box_width = min(width, height / ratio)
    box_height = box_width * ratio
    return (
        x0 + 0.5 * (width - box_width),
        y0 + 0.5 * (height - box_height),
        box_width,
        box_height,
    )


def spinorama_iec_layout(
    *,
    right_available: Box,
    bottom_available: Box,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    legend_size: tuple[float, float],
    legend_gap: float,
) -> SpinoramaIecLayout:
    """Choose the right Spinorama legend when it fits beside the IEC axes box.

    All boxes and sizes use the same physical unit. The right-hand candidate is
    evaluated first. Its axes box and legend are centred horizontally as one
    group, with the legend top aligned to the axes box. A candidate that is too
    wide or taller than the axes falls back to the bottom-legend available area.
    """

    right_box = iec_axes_box(right_available, x_limits, y_limits)
    available_width = max(float(right_available[2]), 0.0)
    axes_width = max(float(right_box[2]), 0.0)
    axes_height = max(float(right_box[3]), 0.0)
    legend_width = max(float(legend_size[0]), 0.0)
    legend_height = max(float(legend_size[1]), 0.0)
    gap = max(float(legend_gap), 0.0)
    fits_right = legend_width + gap <= available_width - axes_width and legend_height <= axes_height
    if fits_right:
        group_width = axes_width + gap + legend_width
        axes_x = float(right_available[0]) + 0.5 * (available_width - group_width)
        axes_box = (axes_x, right_box[1], axes_width, axes_height)
        return SpinoramaIecLayout(
            axes_box=axes_box,
            legend_placement=SpinoramaLegendPlacement.RIGHT,
            legend_anchor=(axes_x + axes_width + gap, right_box[1] + axes_height),
        )

    return SpinoramaIecLayout(
        axes_box=iec_axes_box(bottom_available, x_limits, y_limits),
        legend_placement=SpinoramaLegendPlacement.BOTTOM,
        legend_anchor=None,
    )
