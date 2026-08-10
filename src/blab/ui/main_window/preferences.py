"""Preferences, theme, persisted window state, and the recent-project list."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QByteArray, QTimer, Slot
from PySide6.QtGui import QAction, QIcon, QPalette
from PySide6.QtWidgets import QDialog

from blab.ui.dialogs import (
    PreferencesDialog,
)
from blab.ui.main_window.constants import (
    BEM_PREVIEW_DARK_ICON,
    BEM_PREVIEW_LIGHT_ICON,
    CAPTURE_CONTOURS_DARK_ICON,
    CAPTURE_CONTOURS_LIGHT_ICON,
    CLEAR_CONTOURS_DARK_ICON,
    CLEAR_CONTOURS_LIGHT_ICON,
    DEFAULT_DOCK_STATE_B64,
    FEM_PREVIEW_DARK_ICON,
    FEM_PREVIEW_LIGHT_ICON,
    SAVE_DARK_ICON,
    SAVE_LIGHT_ICON,
    SYNTAX_HIGHLIGHT_DARK_ICON,
    SYNTAX_HIGHLIGHT_LIGHT_ICON,
)
from blab.ui.project_history import (
    clear_recent_projects,
    load_recent_project_paths,
    remember_recent_project,
    remove_recent_project,
)
from blab.ui.settings import (
    GuiPreferences,
    load_gui_preferences,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
    save_gui_preferences,
)
from blab.ui.theme import apply_application_theme


class PreferencesMixin:
    """Preferences, theme, persisted window state, and the recent-project list.

    Mixed into :class:`~blab.ui.main_window.window.MainWindow`.
    """

    def _load_preferences(self) -> GuiPreferences:
        return load_gui_preferences(self.settings)

    def _save_preferences(self) -> None:
        save_gui_preferences(self.settings, self.preferences)

    def _apply_theme(self) -> None:
        apply_application_theme(self.preferences.theme)
        self._refresh_plot_export_icons()

    def _apply_field_preferences(self) -> None:
        preview = getattr(self, "preview", None)
        setter = getattr(preview, "set_observation_plane_field_preferences", None)
        if setter is not None:
            setter(
                cache_size_mb=self.preferences.field_cache_size_mb,
                translation_target_fps=self.preferences.field_translation_target_fps,
            )

    def _refresh_plot_export_icons(self) -> None:
        if not hasattr(self, "export_plot_actions") and not hasattr(self, "show_interior_regions_action"):
            return
        palette = self.palette()
        window_color = palette.color(QPalette.Window)
        light_theme = window_color.lightness() >= 128
        icon = QIcon(str(SAVE_LIGHT_ICON if light_theme else SAVE_DARK_ICON))
        capture_icon = QIcon(str(CAPTURE_CONTOURS_LIGHT_ICON if light_theme else CAPTURE_CONTOURS_DARK_ICON))
        clear_icon = QIcon(str(CLEAR_CONTOURS_LIGHT_ICON if light_theme else CLEAR_CONTOURS_DARK_ICON))
        fem_icon = QIcon(str(FEM_PREVIEW_LIGHT_ICON if light_theme else FEM_PREVIEW_DARK_ICON))
        bem_icon = QIcon(str(BEM_PREVIEW_LIGHT_ICON if light_theme else BEM_PREVIEW_DARK_ICON))
        syntax_icon = QIcon(str(SYNTAX_HIGHLIGHT_LIGHT_ICON if light_theme else SYNTAX_HIGHLIGHT_DARK_ICON))
        for action in getattr(self, "export_plot_actions", {}).values():
            action.setIcon(icon)
        for action in getattr(self, "capture_contour_actions", {}).values():
            action.setIcon(capture_icon)
        for action in getattr(self, "clear_contour_actions", {}).values():
            action.setIcon(clear_icon)
        if hasattr(self, "show_interior_regions_action"):
            self.show_interior_regions_action.setIcon(fem_icon)
        if hasattr(self, "show_exterior_region_action"):
            self.show_exterior_region_action.setIcon(bem_icon)
        if hasattr(self, "syntax_highlighting_action"):
            self.syntax_highlighting_action.setIcon(syntax_icon)

    @Slot()
    def _save_frequency_settings(self) -> None:
        frequencies = self.frequency_range()
        self.settings.setValue("solve/freq_min_hz", frequencies.min_hz)
        self.settings.setValue("solve/freq_max_hz", frequencies.max_hz)
        self.settings.setValue("solve/freq_count", frequencies.count)
        if hasattr(self, "project"):
            self.project.project_preferences = self._current_project_preferences()

    def _restore_window_state(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        dock_state = self.settings.value("window/dock_state")
        if dock_state is None:
            dock_state = QByteArray.fromBase64(DEFAULT_DOCK_STATE_B64.encode("ascii"))
        if dock_state is not None:
            self.workspace.restoreState(dock_state)
        for dock_id in ("editor", "preview"):
            self._sync_panel_view_action(dock_id)
        for entry in self.plot_entries:
            action = self.plot_view_actions.get(entry.plot_id)
            dock = self.plot_docks.get(entry.plot_id)
            if action is not None and dock is not None:
                self._sync_plot_view_action(entry.plot_id)

    def _save_window_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/dock_state", self.workspace.saveState())
        self.settings.sync()

    def _remember_recent_project(self, path: Path) -> None:
        remember_recent_project(self.settings, path)
        if hasattr(self, "open_recent_menu"):
            self._rebuild_open_recent_menu()

    def _remove_recent_project(self, path: Path) -> None:
        remove_recent_project(self.settings, path)
        if hasattr(self, "open_recent_menu"):
            self._rebuild_open_recent_menu()

    def _clear_recent_projects(self) -> None:
        clear_recent_projects(self.settings)
        self._rebuild_open_recent_menu()

    def _rebuild_open_recent_menu(self) -> None:
        self.open_recent_menu.clear()
        recent_paths = load_recent_project_paths(self.settings)
        if not recent_paths:
            empty_action = QAction("No Recent Projects", self)
            empty_action.setEnabled(False)
            self.open_recent_menu.addAction(empty_action)
            return

        for path in recent_paths:
            action = QAction(path.name or str(path), self)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _checked=False, project_path=path: self.open_recent_project(project_path))
            self.open_recent_menu.addAction(action)

        self.open_recent_menu.addSeparator()
        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(lambda _checked=False: self._clear_recent_projects())
        self.open_recent_menu.addAction(clear_action)

    @Slot()
    def open_preferences(self) -> None:
        previous_preferences = self.preferences
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() != QDialog.Accepted:
            return
        preferences = dialog.preferences()
        checked_server_health = None
        if (
            dialog.server_health_payload is not None
            and dialog.server_health_url == preferences.solve_server_url.rstrip("/")
            and dialog.server_health_access_token == dialog.server_access_token_edit.text().strip()
        ):
            checked_server_health = dialog.server_health_payload
        symmetry_will_be_disabled = (
            self.backend_health.effective_symmetry(
                self.symmetry,
                preferences,
                server_health_payload=checked_server_health,
            )
            != self.symmetry
        )
        requires_invalidation = symmetry_will_be_disabled or preferences_require_solve_invalidation(
            previous_preferences, preferences
        )
        if requires_invalidation and not self._confirm_clear_solved_data():
            dialog.deleteLater()
            return

        dialog.remember_server_access_token()
        dialog.deleteLater()
        self.preferences = preferences
        self._apply_field_preferences()
        if checked_server_health is not None and preferences.solve_backend == "server":
            self.backend_health.cache(checked_server_health, preferences.solve_server_url)
        elif (
            preferences.solve_backend != previous_preferences.solve_backend
            or preferences.solve_server_url != previous_preferences.solve_server_url
        ):
            self.backend_health.clear()
        self._save_preferences()
        self.project.project_preferences = self._current_project_preferences()
        symmetry_disabled = self.reconcile_symmetry_with_backend()
        QTimer.singleShot(0, self._apply_theme)
        self.mesh_state_changed.emit("preferences_changed")
        if symmetry_disabled or preferences_require_solve_invalidation(previous_preferences, self.preferences):
            self.solve_results_invalidated.emit("preferences_changed")
        elif preferences_require_visualization_refresh(previous_preferences, self.preferences):
            self.visualization_settings_changed.emit("preferences_changed")
        self.status_label.setText("Preferences updated")
