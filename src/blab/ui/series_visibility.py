"""Shared per-series visibility for plot canvases: Traces menu, legend clicks and greying."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType

from matplotlib.backend_bases import MouseButton
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from PySide6.QtCore import QSignalBlocker
from PySide6.QtGui import QAction

HIDDEN_LEGEND_ENTRY_ALPHA = 0.3
LEGEND_HANDLE_PICK_RADIUS = 5.0


class SeriesVisibilityMixin:
    """One visibility state per series, shared by the Traces menu and a clickable legend.

    Host contract: canvases with series assign their own ``_series_labels``
    (displayed labels) and ``_series_visibility`` dict (keyed by
    :meth:`_series_visibility_key`); the read-only class defaults below stand in
    for canvases without series. A canvas with a Traces menu also has
    ``trace_filter_menu``/``trace_filter_action``. After a visibility change the
    mixin calls :meth:`_on_series_visibility_changed`, which re-applies line
    visibility and limits. The default :meth:`_apply_series_visibility` serves
    canvases that keep one line per series in a ``_lines`` dict; others override
    it. Canvases override :meth:`_update_visible_series_limits` to re-fit limits.
    """

    _series_labels: tuple[str, ...] = ()
    _series_visibility: Mapping[str, bool] = MappingProxyType({})
    _lines: Mapping[str, Line2D]

    def __init__(self, *args, **kwargs) -> None:
        self._series_actions: dict[str, QAction] = {}
        self._series_legend: Legend | None = None
        self._series_legend_entries: tuple[tuple[str, Line2D], ...] = ()
        self._series_legend_targets: dict[object, str] = {}
        self._series_legend_handles: list[tuple[object, Line2D]] = []
        self._last_series_legend_click = None
        super().__init__(*args, **kwargs)

    def _series_visibility_key(self, label: str) -> str:
        """Key that stays the same when a series' displayed label changes."""
        return str(label)

    def is_series_visible(self, label: str) -> bool:
        """Return the series selection, including series not rendered yet."""
        return self._series_visibility.get(self._series_visibility_key(label), True)

    def has_selected_traces(self) -> bool:
        labels = self._series_labels
        return not labels or any(self.is_series_visible(label) for label in labels)

    def set_series_visible(self, label: str, visible: bool) -> None:
        key = self._series_visibility_key(label)
        if key not in {self._series_visibility_key(name) for name in self._series_labels}:
            return
        visible = bool(visible)
        self._series_visibility[key] = visible
        for action_label, action in self._series_actions.items():
            if self._series_visibility_key(action_label) == key and action.isChecked() != visible:
                with QSignalBlocker(action):
                    action.setChecked(visible)
        self.trace_visibility_changed.emit()
        self._on_series_visibility_changed()

    def _on_series_visibility_changed(self) -> None:
        self._apply_series_visibility()
        self._update_visible_series_limits()
        self._apply_manual_axis_limits()
        self._invalidate_crosshair_background()
        self.draw_idle()

    def _apply_series_visibility(self) -> None:
        for label, line in self._lines.items():
            line.set_visible(self.is_series_visible(label))
        self._show_series_legend(self._lines.items())

    def _update_visible_series_limits(self) -> None:
        """Re-fit automatic limits to the visible series; fixed-range plots keep theirs."""

    def _sync_trace_filter_actions(self, labels: tuple[str, ...]) -> None:
        self.trace_filter_menu.clear()
        self._series_actions = {}
        for label in labels:
            action = self.trace_filter_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(self.is_series_visible(label))
            action.toggled.connect(lambda checked, series_label=label: self.set_series_visible(series_label, checked))
            self._series_actions[label] = action
        self.trace_filter_action.setEnabled(bool(labels))

    # Legend -------------------------------------------------------------

    def _show_series_legend(self, entries: Iterable[tuple[str, object]], **legend_kwargs) -> None:
        """Show every series line in the legend; hidden ones greyed, all clickable.

        ``entries`` pairs a series label with a line; several lines may share a
        label. The legend is rebuilt only when the entries change, so the layout
        stays put while series are toggled.
        """
        entries = tuple(entries)
        effective_kwargs = self._series_legend_kwargs()
        effective_kwargs.update(legend_kwargs)
        legend = self.axes.get_legend()
        if legend is None or legend is not self._series_legend or entries != self._series_legend_entries:
            if legend is not None:
                legend.remove()
            self._build_series_legend(entries, effective_kwargs)
        self._style_series_legend()

    def _series_legend_kwargs(self) -> dict:
        """Additional placement options supplied by canvas mixins."""
        return {}

    def _build_series_legend(self, entries: Sequence[tuple[str, object]], legend_kwargs: dict) -> None:
        self._series_legend_entries = entries
        self._series_legend_targets = {}
        self._series_legend_handles = []
        self._series_legend = None
        if not entries:
            return
        legend_kwargs.setdefault("loc", "best")
        lines = [line for _label, line in entries]
        legend = self.axes.legend(lines, [line.get_label() for line in lines], **legend_kwargs)
        for (label, line), handle, text in zip(entries, legend.legend_handles, legend.get_texts(), strict=True):
            handle.set_picker(LEGEND_HANDLE_PICK_RADIUS)
            text.set_picker(True)
            self._series_legend_targets[handle] = label
            self._series_legend_targets[text] = label
            self._series_legend_handles.append((handle, line))
        self._series_legend = legend

    def _style_series_legend(self) -> None:
        # Handles are copies made when the legend was built; follow later curve colour changes.
        for handle, line in self._series_legend_handles:
            handle.set_color(line.get_color())
        for artist, label in self._series_legend_targets.items():
            artist.set_alpha(None if self.is_series_visible(label) else HIDDEN_LEGEND_ENTRY_ALPHA)

    def _on_series_legend_pick(self, event) -> None:
        label = self._series_legend_targets.get(event.artist)
        mouse_event = event.mouseevent
        if label is None or mouse_event.button != MouseButton.LEFT:
            return
        # The handle and the text of one entry can both report the same click.
        if mouse_event is self._last_series_legend_click:
            return
        self._last_series_legend_click = mouse_event
        self.set_series_visible(label, not self.is_series_visible(label))

    def _event_hits_series_legend(self, event) -> bool:
        legend = self._series_legend
        x, y = getattr(event, "x", None), getattr(event, "y", None)
        if legend is None or x is None or y is None or legend is not self.axes.get_legend():
            return False
        return legend.get_window_extent().contains(x, y)


__all__ = ["HIDDEN_LEGEND_ENTRY_ALPHA", "SeriesVisibilityMixin"]
