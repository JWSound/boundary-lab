"""When plot-derived actions are offered to the user.

The decision is pure and lives here; applying it to QActions lives in the view.
Contour capture in particular depends on five separate conditions, which is
exactly the kind of rule that should not be discoverable only by reading widget
code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContourControls:
    """Availability of the capture/clear contour actions for one isobar plot."""

    capture: bool
    clear: bool


def contour_controls(
    *,
    has_live_data: bool,
    final_resolution_active: bool,
    final_plots_rendered: bool,
    plot_visible: bool,
    has_captured_contours: bool,
) -> ContourControls:
    """Decide whether contours can be captured from, or cleared on, a plot.

    Capture is only meaningful once a solve has produced data *and* the plot has
    actually been re-rendered at final resolution — capturing from a coarse live
    preview would freeze contours that do not match the finished solve. It also
    requires the plot to be on screen, since capture reads what is drawn.

    Clearing only depends on there being something captured, so it stays
    available even while a later solve is running.
    """
    return ContourControls(
        capture=has_live_data and final_resolution_active and final_plots_rendered and plot_visible,
        clear=has_captured_contours,
    )
