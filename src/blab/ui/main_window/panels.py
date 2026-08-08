"""Dock and plot panel visibility, kept in sync with View-menu actions."""

from __future__ import annotations

from PySide6.QtCore import QSignalBlocker, QTimer


class PanelVisibilityMixin:
    """Dock and plot panel visibility, kept in sync with View-menu actions.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _set_panel_visible(self, dock_id: str, visible: bool) -> None:
        dock = self.editor_dock if dock_id == "editor" else self.preview_dock if dock_id == "preview" else None
        if dock is None:
            return
        if dock.isHidden() == visible:
            dock.setVisible(bool(visible))
        if visible:
            dock.raise_()

    def _sync_panel_view_action(self, dock_id: str) -> None:
        action = self.panel_view_actions.get(dock_id)
        if action is None:
            return
        dock = None
        if dock_id == "editor":
            dock = self.editor_dock
        elif dock_id == "preview":
            dock = self.preview_dock
        with QSignalBlocker(action):
            action.setChecked(dock is not None and not dock.isHidden())

    def _set_plot_visible(self, plot_id: str, visible: bool) -> None:
        for entry in self.plot_entries:
            if entry.plot_id != plot_id:
                continue
            dock = self.plot_docks.get(plot_id)
            if dock is not None and dock.isVisible() != visible:
                dock.setVisible(visible)
            if visible:
                if dock is not None:
                    dock.raise_()
                self._schedule_visible_plot_refresh()
            self.refresh_contour_controls()
            break

    def _sync_plot_view_action(self, plot_id: str) -> None:
        action = self.plot_view_actions.get(plot_id)
        dock = self.plot_docks.get(plot_id)
        if action is None or dock is None:
            return
        with QSignalBlocker(action):
            action.setChecked(not dock.isHidden())
        if not dock.isHidden():
            self._schedule_visible_plot_refresh()
        self.refresh_contour_controls()

    def _schedule_visible_plot_refresh(self) -> None:
        """Refresh selected plot tabs after Qt has assigned their geometry."""
        if self._plot_activation_refresh_pending:
            return
        self._plot_activation_refresh_pending = True
        QTimer.singleShot(0, self._refresh_visible_plots)

    def _refresh_visible_plots(self) -> None:
        self._plot_activation_refresh_pending = False
        active_entries = [entry for entry in self.plot_entries if self._plot_entry_is_actively_visible(entry)]
        if not active_entries:
            return
        if self.solve_controller.active:
            if self.preferences.live_plot_streaming:
                self.request_live_refresh()
        else:
            self.refresh_plots(active_only=True)
        if self._use_final_isobar_resolution and any(
            entry.plot_id in {"horizontal_isobar", "vertical_isobar"} for entry in active_entries
        ):
            self._final_isobar_plots_rendered = True
        self.refresh_contour_controls()
