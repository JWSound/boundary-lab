"""Main Qt window and user workflow orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import meshio
import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from blab import __version__
from blab.ath import (
    AthRunResult,
    clean_ath_reduced_mesh_output,
    detect_ath_radiators,
    find_physical_tag_by_name,
    read_surface_physical_names,
    write_ath_gmsh_path,
    write_ath_output_root,
)
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.exporting import export_plot_png, export_polar_text_files
from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
    build_log_frequencies,
    order_frequencies_for_live_plotting,
)
from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file, stitch_meshes
from blab.plotting import VisualizerConfig
from blab.postprocess import PrepConfig
from blab.solvers.http_server import server_health_supports_symmetry
from blab.solvers.registry import backend_info
from blab.symmetry import SymmetryValidationError, validate_reduced_mesh_configs
from blab.ui.ath_worker import AthGenerationWorker
from blab.ui.diagnostics import DiagnosticsDialog
from blab.ui.dialogs import (
    ChannelConfigDialog,
    DonateDialog,
    MeshConfigDialog,
    MeshDialogEntry,
    PreferencesDialog,
    SourceConfigDialog,
)
from blab.ui.main_window_widgets import (
    AthScriptEditor,
    DockTitleBar,
    PlotEntry,
    format_frequency_solve_timings,
)
from blab.ui.plots import (
    AUDIO_FREQ_MAX_HZ,
    AUDIO_FREQ_MIN_HZ,
    FINAL_ISOBAR_ANGLE_SAMPLES,
    FINAL_ISOBAR_FREQ_SAMPLES,
    FINAL_ISOBAR_SHADING,
    FREQ_SLIDER_STEPS,
    LIVE_ISOBAR_SHADING,
    ImpedanceCanvas,
    IsobarCanvas,
    OnAxisResponseCanvas,
    SpinoramaCanvas,
    frequency_to_slider_value,
    slider_value_to_frequency,
)
from blab.ui.project_history import (
    clear_recent_projects,
    load_recent_project_paths,
    remember_recent_project,
    remove_recent_project,
)
from blab.ui.project_io import (
    PROJECT_DEFAULT_NAME,
    PROJECT_FILE_FILTER,
    build_project_payload,
    normalize_project_path,
    read_project_file,
    write_project_file,
)
from blab.ui.project_state import (
    AthScriptState,
    default_scripts,
    new_script,
    replace_script,
    script_to_payload,
    scripts_from_payload,
    unique_script_name,
)
from blab.ui.settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    GuiPreferences,
    balloon_sampling_points,
    live_plot_angle_samples,
    live_plot_freq_samples,
    load_gui_preferences,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
    save_gui_preferences,
    settings_int,
)
from blab.ui.solve_worker import SolveWorker
from blab.ui.source_channel_config import (
    apply_saved_imported_source_config,
    apply_saved_source_config_to_result,
    channel_configs,
    channels_for_solver_radiators,
    clear_source_channel_configs,
    load_channel_config_by_name,
    load_source_config_by_name,
    save_channel_config,
    save_channel_config_by_name,
    save_source_config,
    save_source_config_by_name,
)
from blab.ui.theme import apply_application_theme

ATH_MESH_NAME = "ath"
STITCHED_MESH_NAME = "stitched"
DEFAULT_MESH_SCALE_FACTOR = 0.001
STITCH_FAILURE_MESSAGE = (
    "Error - unable to stitch separate mesh entities. "
    "Refer to help documentation for more info on multi-mesh workflows."
)


APP_ROOT = Path(__file__).resolve().parents[3]
ATH_BUNDLE_DIR = APP_ROOT / "ath"
ATH_OUTPUT_ROOT = APP_ROOT / "runs" / "ath_output"
GMSH_BUNDLE_EXE = APP_ROOT / "gmsh" / "gmsh-4.15.2-Windows64" / "gmsh.exe"
HELP_GUIDE_PDF = APP_ROOT / "docs" / "Boundary Lab Guide.pdf"
SAVE_DARK_ICON = APP_ROOT / "assets" / "save_dark.ico"
SAVE_LIGHT_ICON = APP_ROOT / "assets" / "save_light.ico"
CAPTURE_CONTOURS_DARK_ICON = APP_ROOT / "assets" / "capturecontours_dark.ico"
CAPTURE_CONTOURS_LIGHT_ICON = APP_ROOT / "assets" / "capturecontours_light.ico"
CLEAR_CONTOURS_DARK_ICON = APP_ROOT / "assets" / "clearcontours_dark.ico"
CLEAR_CONTOURS_LIGHT_ICON = APP_ROOT / "assets" / "clearcontours_light.ico"
ADD_SCRIPT_TAB_LABEL = "+"
DEFAULT_DOCK_STATE_B64 = (
    "AAAA/wAAAAD9AAAAAQAAAAAAAAduAAADdvwCAAAAAfwAAAAAAAADdgAAAG4A/////AEAAAAG+wAAAB4AYQB0AGgAXwBl"
    "AGQAaQB0AG8AcgBfAGQAbwBjAGsBAAAAAAAAAdsAAACFAP////wAAAHfAAADRQAAAGoA/////AIAAAAC+wAAACIAbQBl"
    "AHMAaABfAHAAcgBlAHYAaQBlAHcAXwBkAG8AYwBrAQAAAAAAAAN2AAAANAD////7AAAAHABzAHAAaQBuAG8AcgBhAG0A"
    "YQBfAGQAbwBjAGsIAAAB4AAAAZYAAAAiAP////sAAAAUAHAAbABvAHQAcwBfAGQAbwBjAGsBAAAE9QAAAnkAAAAAAAAA"
    "APwAAAUoAAACRgAAAHsA/////AIAAAAC+wAAACwAaABvAHIAaQB6AG8AbgB0AGEAbABfAGkAcwBvAGIAYQByAF8AZABv"
    "AGMAawEAAAAAAAABugAAACIA////+wAAACgAdgBlAHIAdABpAGMAYQBsAF8AaQBzAG8AYgBhAHIAXwBkAG8AYwBrAQAA"
    "Ab4AAAG4AAAAIgD////7AAAALgBhAGMAbwB1AHMAdABpAGMAXwBpAG0AcABlAGQAYQBuAGMAZQBfAGQAbwBjAGsAAAAA"
    "AP////8AAACNAP////sAAAA+AG8AbgBfAGEAeABpAHMAXwBmAHIAZQBxAHUAZQBuAGMAeQBfAHIAZQBzAHAAbwBuAHMA"
    "ZQBfAGQAbwBjAGsIAAAF/AAAAXIAAAC6AP///wAAAAAAAAN2AAAABAAAAAQAAAAIAAAACPwAAAAA"
)


class MainWindow(QMainWindow):
    mesh_state_changed = Signal(str)
    source_config_changed = Signal(str)
    project_state_changed = Signal(str)
    solve_results_invalidated = Signal(str)
    visualization_settings_changed = Signal(str)

    def __init__(self, startup_status: Callable[[str], None] | None = None):
        super().__init__()

        def startup(stage: str) -> None:
            if startup_status is not None:
                startup_status(stage)

        startup("Loading saved settings...")
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle(f"Boundary Lab Beta {__version__}")
        self.resize(1500, 900)
        self.imported_meshes: tuple[MeshDialogEntry, ...] = ()
        self.stitch_imported_meshes = False
        self.symmetry = "off"
        self.preferences = self._load_preferences()
        self.server_health_payload: dict | None = None
        self.server_health_url: str | None = None
        self._apply_theme()
        self.ath_scripts: tuple[AthScriptState, ...] = default_scripts("")
        self.active_ath_script_id: str | None = self.ath_scripts[0].id if self.ath_scripts else None
        self.ath_results_by_script_id: dict[str, AthRunResult] = {}
        self.imported_radiators: tuple[RadiatorConfig, ...] = ()
        self.live_dataset: LiveSolveDataset | None = None
        self.balloon_window: QDialog | None = None
        self.channel_config_dialog: ChannelConfigDialog | None = None
        self.project_path: Path | None = None
        self._project_clean_payload: dict | None = None
        self.solve_thread: QThread | None = None
        self.solve_worker: SolveWorker | None = None
        self.ath_thread: QThread | None = None
        self.ath_worker: AthGenerationWorker | None = None
        self.ath_generation_script_id: str | None = None
        self.ath_generation_mesh_name: str | None = None
        self.ath_generation_cancel_requested = False
        self.solve_expected_count = 0
        self.solve_failed = False
        self.solve_started_at: float | None = None
        self.solve_cancel_requested = False
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self._last_imported_mesh_focus_check_at = 0.0
        self._plot_dpi_screen = None
        self._plot_dpi_window_handle = None
        self._plot_dpi_refresh_pending = False
        startup("Preparing Ath runtime config...")
        self._ensure_ath_runtime_config()

        startup("Building script editor...")
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.currentChanged.connect(self._on_active_ath_tab_changed)
        self.editor_tabs.tabCloseRequested.connect(self._remove_ath_script_at)
        self.editor_tabs.tabBar().installEventFilter(self)
        self._rebuild_ath_script_tabs()

        startup("Creating mesh preview...")
        from blab.ui.mesh_preview import MeshPreview

        self.preview = MeshPreview()
        if self._has_solver_meshes():
            startup("Loading mesh preview...")
            self._refresh_mesh_preview()

        self.generate_button = QPushButton("Generate (F7)")
        self.generate_button.setShortcut(QKeySequence("F7"))
        self.solve_button = QPushButton("Solve (F5)")
        self.solve_button.setShortcut(QKeySequence("F5"))
        self.cancel_button = QPushButton("Stop (Shift+F5)")
        self.cancel_button.setShortcut(QKeySequence("Shift+F5"))
        self.cancel_button.setEnabled(False)
        self.mesh_config_button = QPushButton("Mesh Config")
        self.channel_config_button = QPushButton("Channel Config")
        self.source_config_button = QPushButton("Source Config")
        self.source_config_button.setEnabled(self._has_solver_meshes())

        freq_min = min(max(settings_int(self.settings, "solve/freq_min_hz", 200), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ)
        freq_max = min(
            max(settings_int(self.settings, "solve/freq_max_hz", 20000), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ
        )
        freq_count = min(max(settings_int(self.settings, "solve/freq_count", 41), 3), 161)

        self.freq_min_slider = self._make_slider(0, FREQ_SLIDER_STEPS, frequency_to_slider_value(freq_min))
        self.freq_max_slider = self._make_slider(0, FREQ_SLIDER_STEPS, frequency_to_slider_value(freq_max))
        self.freq_count_slider = self._make_slider(3, 161, freq_count)
        self.freq_count_slider.setSingleStep(2)

        self.freq_min_spin = self._make_spin(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ, freq_min)
        self.freq_max_spin = self._make_spin(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ, freq_max)
        self.freq_count_spin = self._make_spin(3, 161, freq_count)

        self.status_label = QLabel("Ready")
        startup("Creating plot panels...")
        self.horizontal_plot = IsobarCanvas("Horizontal Isobar")
        self.vertical_plot = IsobarCanvas("Vertical Isobar")
        self.impedance_plot = ImpedanceCanvas()
        self.on_axis_plot = OnAxisResponseCanvas()
        self.spinorama_plot = SpinoramaCanvas()
        self.plot_entries = (
            PlotEntry(
                "horizontal_isobar",
                "Horizontal Isobar",
                "horizontal_isobar.png",
                self.horizontal_plot,
                self._update_horizontal_plot,
            ),
            PlotEntry(
                "vertical_isobar",
                "Vertical Isobar",
                "vertical_isobar.png",
                self.vertical_plot,
                self._update_vertical_plot,
            ),
            PlotEntry(
                "acoustic_impedance",
                "Acoustic Impedance",
                "acoustic_impedance.png",
                self.impedance_plot,
                self._update_impedance_plot,
            ),
            PlotEntry(
                "on_axis_frequency_response",
                "On-Axis Frequency Response",
                "on_axis_frequency_response.png",
                self.on_axis_plot,
                self._update_on_axis_plot,
            ),
            PlotEntry(
                "spinorama",
                "Spinorama",
                "spinorama.png",
                self.spinorama_plot,
                self._update_spinorama_plot,
            ),
        )
        self.plot_view_actions: dict[str, QAction] = {}
        self.export_plot_actions: dict[str, QAction] = {}
        self.panel_view_actions: dict[str, QAction] = {}
        self.plot_docks: dict[str, QDockWidget] = {}
        self.capture_contour_actions: dict[str, QAction] = {}
        self.clear_contour_actions: dict[str, QAction] = {}

        startup("Wiring controls...")
        self._wire_controls()
        startup("Building menus...")
        self._build_menu_bar()
        startup("Building main layout...")
        self._build_layout()
        self._connect_state_events()
        startup("Restoring window layout...")
        self._restore_window_state()
        startup("Starting new project...")
        self.new_project()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._reload_updated_imported_meshes_on_focus()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self._connect_plot_dpi_signals()

    def _connect_plot_dpi_signals(self) -> None:
        window = self.windowHandle()
        if window is None:
            QTimer.singleShot(0, self._connect_plot_dpi_signals)
            return
        if self._plot_dpi_window_handle is not window:
            if self._plot_dpi_window_handle is not None:
                try:
                    self._plot_dpi_window_handle.screenChanged.disconnect(self._on_plot_screen_changed)
                except (RuntimeError, TypeError):
                    pass
            window.screenChanged.connect(self._on_plot_screen_changed)
            self._plot_dpi_window_handle = window
        self._on_plot_screen_changed(window.screen())

    def _on_plot_screen_changed(self, screen) -> None:
        if screen is self._plot_dpi_screen:
            return
        self._disconnect_plot_dpi_screen()
        self._plot_dpi_screen = screen
        if screen is not None:
            screen.logicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)
            screen.physicalDotsPerInchChanged.connect(self._schedule_plot_canvas_dpi_refresh)
            screen.geometryChanged.connect(self._schedule_plot_canvas_dpi_refresh)
        self._schedule_plot_canvas_dpi_refresh()

    def _disconnect_plot_dpi_screen(self) -> None:
        screen = self._plot_dpi_screen
        self._plot_dpi_screen = None
        if screen is None:
            return
        for signal in (
            screen.logicalDotsPerInchChanged,
            screen.physicalDotsPerInchChanged,
            screen.geometryChanged,
        ):
            try:
                signal.disconnect(self._schedule_plot_canvas_dpi_refresh)
            except (RuntimeError, TypeError):
                pass

    def _schedule_plot_canvas_dpi_refresh(self, *_args) -> None:
        if self._plot_dpi_refresh_pending:
            return
        self._plot_dpi_refresh_pending = True
        QTimer.singleShot(0, self._refresh_plot_canvas_dpi)

    def _refresh_plot_canvas_dpi(self) -> None:
        self._plot_dpi_refresh_pending = False
        window = self.windowHandle()
        screen = None if window is None else window.screen()
        for entry in self.plot_entries:
            canvas = entry.widget
            if screen is not None and hasattr(canvas, "_update_screen"):
                canvas._update_screen(screen)
            if hasattr(canvas, "_update_pixel_ratio"):
                canvas._update_pixel_ratio()
            canvas.draw_idle()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
        if watched is self.editor_tabs.tabBar() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.editor_tabs.tabBar().tabAt(event.position().toPoint())
            if index == len(self.ath_scripts):
                self.add_ath_script()
                return True
        if watched is self.editor_tabs.tabBar() and event.type() == QEvent.Type.MouseButtonDblClick:
            index = self.editor_tabs.tabBar().tabAt(event.position().toPoint())
            if 0 <= index < len(self.ath_scripts):
                self.editor_tabs.setCurrentIndex(index)
                self.rename_active_ath_script()
                return True
        return super().eventFilter(watched, event)

    def _build_menu_bar(self) -> None:
        file_menu = self.menuBar().addMenu("File")

        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.new_project)
        file_menu.addAction(new_project_action)

        file_menu.addSeparator()

        save_project_action = QAction("Save Project", self)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        save_project_as_action = QAction("Save Project As", self)
        save_project_as_action.triggered.connect(self.save_project_as)
        file_menu.addAction(save_project_as_action)

        load_project_action = QAction("Open Project", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

        self.open_recent_menu = file_menu.addMenu("Open Recent")
        self._rebuild_open_recent_menu()

        file_menu.addSeparator()

        import_action = QAction("Import .cfg", self)
        import_action.triggered.connect(self.import_config)
        file_menu.addAction(import_action)

        export_cfg_action = QAction("Export .cfg", self)
        export_cfg_action.triggered.connect(self.export_config)
        file_menu.addAction(export_cfg_action)

        for entry in self.plot_entries:
            action = QAction(entry.title, self)
            action.setToolTip(f"Export {entry.title}")
            action.setEnabled(False)
            action.triggered.connect(lambda _checked=False, plot_id=entry.plot_id: self.export_plot(plot_id))
            self.export_plot_actions[entry.plot_id] = action
            if entry.plot_id in {"horizontal_isobar", "vertical_isobar"}:
                capture_action = QAction("Capture Contours", self)
                capture_action.setToolTip(f"Capture contours for {entry.title}")
                capture_action.setEnabled(False)
                capture_action.triggered.connect(
                    lambda _checked=False, plot_id=entry.plot_id: self.capture_isobar_contours(plot_id)
                )
                self.capture_contour_actions[entry.plot_id] = capture_action
                clear_action = QAction("Clear Contours", self)
                clear_action.setToolTip(f"Clear contours for {entry.title}")
                clear_action.setEnabled(False)
                clear_action.triggered.connect(
                    lambda _checked=False, plot_id=entry.plot_id: self.clear_isobar_contours(plot_id)
                )
                self.clear_contour_actions[entry.plot_id] = clear_action

        self.export_polar_data_action = QAction("Export Polar Data", self)
        self.export_polar_data_action.setEnabled(False)
        self.export_polar_data_action.triggered.connect(self.export_polar_data)
        file_menu.addAction(self.export_polar_data_action)

        view_menu = self.menuBar().addMenu("View")
        self.balloon_plot_action = QAction("Balloon Plot", self)
        self.balloon_plot_action.setEnabled(False)
        self.balloon_plot_action.triggered.connect(self.open_balloon_plot)
        view_menu.addAction(self.balloon_plot_action)
        view_menu.addSeparator()
        for dock_id, title in (
            ("editor", "Ath Editor Panel"),
            ("preview", "Mesh Preview Panel"),
        ):
            action = QAction(title, self)
            action.setCheckable(True)
            action.setChecked(True)
            view_menu.addAction(action)
            self.panel_view_actions[dock_id] = action
        view_menu.addSeparator()
        for entry in self.plot_entries:
            action = QAction(entry.title, self)
            action.setCheckable(True)
            action.setChecked(True)
            view_menu.addAction(action)
            self.plot_view_actions[entry.plot_id] = action

        edit_menu = self.menuBar().addMenu("Edit")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self.open_preferences)
        edit_menu.addAction(preferences_action)

        about_menu = self.menuBar().addMenu("About")
        diagnostics_action = QAction("Diagnostic Info", self)
        diagnostics_action.triggered.connect(self.open_diagnostics)
        about_menu.addAction(diagnostics_action)

        donate_action = QAction("Donate", self)
        donate_action.triggered.connect(self.open_donate)
        about_menu.addAction(donate_action)

        help_action = QAction("Help", self)
        help_action.triggered.connect(self.open_help)
        about_menu.addAction(help_action)

    def _make_panel_dock(
        self,
        object_name: str,
        title: str,
        widget: QWidget,
        *,
        save_action: QAction | None = None,
        tool_actions: tuple[QAction, ...] = (),
    ) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable | QDockWidget.DockWidgetClosable
        )
        dock.setTitleBarWidget(DockTitleBar(title, dock, save_action=save_action, tool_actions=tool_actions))
        return dock

    def _build_layout(self) -> None:
        self.editor_panel = QWidget()
        editor_layout = QVBoxLayout(self.editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.editor_tabs)

        self.editor_container = QWidget()
        editor_container_layout = QHBoxLayout(self.editor_container)
        editor_container_layout.setContentsMargins(0, 0, 0, 0)
        editor_container_layout.setSpacing(0)
        editor_container_layout.addWidget(self.editor_panel, 1)

        self.workspace = QMainWindow()
        self.workspace.setDockOptions(
            QMainWindow.AllowNestedDocks | QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks
        )
        self.editor_dock = self._make_panel_dock("ath_editor_dock", "Ath Editor", self.editor_container)
        self.preview_dock = self._make_panel_dock("mesh_preview_dock", "Mesh Preview", self.preview)
        self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.editor_dock)
        self.workspace.addDockWidget(Qt.LeftDockWidgetArea, self.preview_dock)
        self.workspace.splitDockWidget(self.editor_dock, self.preview_dock, Qt.Horizontal)
        previous_plot_dock = None
        for entry in self.plot_entries:
            dock = self._make_panel_dock(
                f"{entry.plot_id}_dock",
                entry.title,
                entry.widget,
                save_action=self.export_plot_actions.get(entry.plot_id),
                tool_actions=tuple(
                    action
                    for action in (
                        self.capture_contour_actions.get(entry.plot_id),
                        self.clear_contour_actions.get(entry.plot_id),
                    )
                    if action is not None
                ),
            )
            self.plot_docks[entry.plot_id] = dock
            self.workspace.addDockWidget(Qt.RightDockWidgetArea, dock)
            if previous_plot_dock is None:
                self.workspace.splitDockWidget(self.preview_dock, dock, Qt.Horizontal)
            else:
                self.workspace.tabifyDockWidget(previous_plot_dock, dock)
            previous_plot_dock = dock
        if previous_plot_dock is not None:
            previous_plot_dock.raise_()
        self.workspace.resizeDocks(
            [self.editor_dock, self.preview_dock, *self.plot_docks.values()],
            [420, 520, *([520] * len(self.plot_docks))],
            Qt.Horizontal,
        )
        for dock_id, dock in (
            ("editor", self.editor_dock),
            ("preview", self.preview_dock),
        ):
            action = self.panel_view_actions.get(dock_id)
            if action is not None:
                action.toggled.connect(lambda checked, dock_id=dock_id: self._set_panel_visible(dock_id, checked))
                dock.visibilityChanged.connect(lambda _visible, dock_id=dock_id: self._sync_panel_view_action(dock_id))
        for entry in self.plot_entries:
            dock = self.plot_docks[entry.plot_id]
            action = self.plot_view_actions.get(entry.plot_id)
            if action is not None:
                action.toggled.connect(lambda checked, plot_id=entry.plot_id: self._set_plot_visible(plot_id, checked))
                dock.visibilityChanged.connect(
                    lambda _visible, plot_id=entry.plot_id: self._sync_plot_view_action(plot_id)
                )

        controls = QFrame()
        controls_layout = QHBoxLayout(controls)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.solve_button)
        controls_layout.addWidget(self.cancel_button)
        controls_layout.addWidget(self.mesh_config_button)
        controls_layout.addWidget(self.channel_config_button)
        controls_layout.addWidget(self.source_config_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(QLabel("Min Hz"))
        controls_layout.addWidget(self.freq_min_slider)
        controls_layout.addWidget(self.freq_min_spin)
        controls_layout.addWidget(QLabel("Max Hz"))
        controls_layout.addWidget(self.freq_max_slider)
        controls_layout.addWidget(self.freq_max_spin)
        controls_layout.addWidget(QLabel("Frequencies"))
        controls_layout.addWidget(self.freq_count_slider)
        controls_layout.addWidget(self.freq_count_spin)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.workspace, stretch=1)
        layout.addWidget(controls)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)
        self._refresh_plot_export_icons()

    def _wire_controls(self) -> None:
        self.generate_button.clicked.connect(self.generate_geometry)
        self.solve_button.clicked.connect(self.start_solve)
        self.cancel_button.clicked.connect(self.cancel_current_operation)
        self.mesh_config_button.clicked.connect(self.open_mesh_config)
        self.channel_config_button.clicked.connect(self.open_channel_config)
        self.source_config_button.clicked.connect(self.open_source_config)

        self.freq_min_slider.valueChanged.connect(
            lambda value: self._sync_frequency_spin_from_slider(self.freq_min_spin, value)
        )
        self.freq_min_spin.valueChanged.connect(
            lambda value: self._sync_frequency_slider_from_spin(self.freq_min_slider, value)
        )
        self.freq_max_slider.valueChanged.connect(
            lambda value: self._sync_frequency_spin_from_slider(self.freq_max_spin, value)
        )
        self.freq_max_spin.valueChanged.connect(
            lambda value: self._sync_frequency_slider_from_spin(self.freq_max_slider, value)
        )
        self.freq_count_slider.valueChanged.connect(self.freq_count_spin.setValue)
        self.freq_count_spin.valueChanged.connect(self.freq_count_slider.setValue)
        self.freq_min_spin.valueChanged.connect(self._save_frequency_settings)
        self.freq_max_spin.valueChanged.connect(self._save_frequency_settings)
        self.freq_count_spin.valueChanged.connect(self._save_frequency_settings)

    def _connect_state_events(self) -> None:
        self.mesh_state_changed.connect(self._on_mesh_state_changed)
        self.source_config_changed.connect(self._on_source_config_changed)
        self.project_state_changed.connect(self._on_project_state_changed)
        self.solve_results_invalidated.connect(self._on_solve_results_invalidated)
        self.visualization_settings_changed.connect(self._on_visualization_settings_changed)

    @Slot(str)
    def _on_mesh_state_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()
        self.source_config_button.setEnabled(self._has_solver_meshes())

    @Slot(str)
    def _on_source_config_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()

    @Slot(str)
    def _on_project_state_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()
        self.source_config_button.setEnabled(self._has_solver_meshes())

    @Slot(str)
    def _on_solve_results_invalidated(self, _reason: str) -> None:
        self._clear_plots()

    def _has_solved_data(self) -> bool:
        return bool(self.live_dataset is not None and self.live_dataset.solved_count > 0)

    def _confirm_clear_solved_data(self) -> bool:
        if not self._has_solved_data():
            return True
        message = QMessageBox(
            QMessageBox.Warning,
            "Clear solved data?",
            "Applying this action will clear solved data",
            QMessageBox.NoButton,
            self,
        )
        continue_button = message.addButton("Continue", QMessageBox.AcceptRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        return message.clickedButton() is continue_button

    @Slot(str)
    def _on_visualization_settings_changed(self, _reason: str) -> None:
        self._refresh_plots()

    def _rebuild_ath_script_tabs(self) -> None:
        self.editor_tabs.blockSignals(True)
        self.editor_tabs.clear()
        for script in self.ath_scripts:
            editor = AthScriptEditor()
            editor.setFont(QFont("Consolas", 10))
            editor.setPlainText(script.config_text)
            editor.textChanged.connect(
                lambda script_id=script.id, editor=editor: self._update_script_text(script_id, editor)
            )
            editor.configDropped.connect(
                lambda path, script_id=script.id: self._import_config_path(Path(path), script_id=script_id)
            )
            self.editor_tabs.addTab(editor, script.name)
        add_tab = AthScriptEditor()
        add_tab.setReadOnly(True)
        add_tab.configDropped.connect(lambda path: self._import_config_path(Path(path)))
        add_index = self.editor_tabs.addTab(add_tab, ADD_SCRIPT_TAB_LABEL)
        self.editor_tabs.tabBar().setTabButton(add_index, QTabBar.ButtonPosition.RightSide, None)
        self.editor_tabs.tabBar().setTabToolTip(add_index, "Add Ath script")
        active_index = self._active_script_index()
        if active_index >= 0:
            self.editor_tabs.setCurrentIndex(active_index)
        self.editor_tabs.blockSignals(False)

    def _active_script_index(self) -> int:
        for index, script in enumerate(self.ath_scripts):
            if script.id == self.active_ath_script_id:
                return index
        return 0 if self.ath_scripts else -1

    def _active_script(self) -> AthScriptState | None:
        if not self.ath_scripts:
            return None
        index = self._active_script_index()
        return self.ath_scripts[index] if index >= 0 else None

    def _update_script_text(self, script_id: str, editor: QPlainTextEdit) -> None:
        self.ath_scripts = replace_script(self.ath_scripts, script_id, config_text=editor.toPlainText())

    def _on_active_ath_tab_changed(self, index: int) -> None:
        if index == len(self.ath_scripts):
            self.add_ath_script()
            return
        if 0 <= index < len(self.ath_scripts):
            self.active_ath_script_id = self.ath_scripts[index].id

    @Slot()
    def add_ath_script(self) -> None:
        name = unique_script_name("ath", self.ath_scripts)
        script = new_script(name, "")
        self.ath_scripts = (*self.ath_scripts, script)
        self.active_ath_script_id = script.id
        self._rebuild_ath_script_tabs()

    @Slot()
    def rename_active_ath_script(self) -> None:
        script = self._active_script()
        if script is None:
            return
        name, accepted = QInputDialog.getText(self, "Rename Ath Script", "Script name:", text=script.name)
        if not accepted:
            return
        name = name.strip()
        if not name:
            return
        self.ath_scripts = replace_script(
            self.ath_scripts,
            script.id,
            name=unique_script_name(name, tuple(s for s in self.ath_scripts if s.id != script.id)),
        )
        self._rebuild_ath_script_tabs()
        self.mesh_state_changed.emit("ath_script_renamed")
        self.solve_results_invalidated.emit("ath_script_renamed")

    def _remove_ath_script_at(self, index: int) -> None:
        if not (0 <= index < len(self.ath_scripts)):
            return
        script = self.ath_scripts[index]
        self.ath_scripts = tuple(item for item in self.ath_scripts if item.id != script.id)
        self.ath_results_by_script_id.pop(script.id, None)
        self.active_ath_script_id = (
            self.ath_scripts[min(index, len(self.ath_scripts) - 1)].id if self.ath_scripts else None
        )
        self._rebuild_ath_script_tabs()
        self.mesh_state_changed.emit("ath_script_removed")
        self.solve_results_invalidated.emit("ath_script_removed")

    def _sync_frequency_spin_from_slider(self, spin: QSpinBox, slider_value: int) -> None:
        with QSignalBlocker(spin):
            spin.setValue(slider_value_to_frequency(slider_value))

    def _sync_frequency_slider_from_spin(self, slider: QSlider, freq_hz: int) -> None:
        with QSignalBlocker(slider):
            slider.setValue(frequency_to_slider_value(freq_hz))

    def _make_slider(self, minimum: int, maximum: int, value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        return slider

    def _make_spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _load_preferences(self) -> GuiPreferences:
        return load_gui_preferences(self.settings)

    def _save_preferences(self) -> None:
        save_gui_preferences(self.settings, self.preferences)

    def _apply_theme(self) -> None:
        apply_application_theme(self.preferences.theme)
        self._refresh_plot_export_icons()

    def _refresh_plot_export_icons(self) -> None:
        if not hasattr(self, "export_plot_actions"):
            return
        palette = self.palette()
        window_color = palette.color(QPalette.Window)
        light_theme = window_color.lightness() >= 128
        icon = QIcon(str(SAVE_LIGHT_ICON if light_theme else SAVE_DARK_ICON))
        capture_icon = QIcon(str(CAPTURE_CONTOURS_LIGHT_ICON if light_theme else CAPTURE_CONTOURS_DARK_ICON))
        clear_icon = QIcon(str(CLEAR_CONTOURS_LIGHT_ICON if light_theme else CLEAR_CONTOURS_DARK_ICON))
        for action in self.export_plot_actions.values():
            action.setIcon(icon)
        for action in self.capture_contour_actions.values():
            action.setIcon(capture_icon)
        for action in self.clear_contour_actions.values():
            action.setIcon(clear_icon)

    @Slot()
    def _save_frequency_settings(self) -> None:
        self.settings.setValue("solve/freq_min_hz", int(self.freq_min_spin.value()))
        self.settings.setValue("solve/freq_max_hz", int(self.freq_max_spin.value()))
        self.settings.setValue("solve/freq_count", int(self.freq_count_spin.value()))

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

    def _ath_mesh_payload(self, *, absolute_paths: bool) -> dict:
        return {}

    def _mesh_config_dialog_entries(self) -> tuple[MeshDialogEntry, ...]:
        entries = []
        for script in self.ath_scripts:
            result = self.ath_results_by_script_id.get(script.id)
            if result is None:
                continue
            entries.append(
                MeshDialogEntry(
                    name=script.mesh_name,
                    source_file=str(result.solver_msh_path),
                    scale_factor=float(script.mesh_scale_factor),
                    translation_mm=script.mesh_translation_mm,
                    enabled=script.mesh_enabled,
                    locked=True,
                )
            )
        entries.extend(self.imported_meshes)
        return tuple(entries)

    def _apply_mesh_config_dialog_entries(self, meshes: tuple[MeshDialogEntry, ...]) -> None:
        imported_meshes = []
        scripts = self.ath_scripts
        for mesh in meshes:
            script = self._script_for_mesh_name(mesh.name)
            if script is not None:
                scripts = replace_script(
                    scripts,
                    script.id,
                    mesh_enabled=bool(mesh.enabled),
                    mesh_translation_mm=mesh.translation_mm,
                    mesh_scale_factor=float(mesh.scale_factor),
                )
            else:
                imported_meshes.append(replace(mesh, locked=False))
        self.ath_scripts = scripts
        self.imported_meshes = tuple(imported_meshes)

    def _project_imported_meshes_payload(self) -> list[dict]:
        return [self._mesh_entry_to_payload(mesh, absolute_paths=True) for mesh in self.imported_meshes]

    def _mesh_entry_to_payload(self, mesh: MeshDialogEntry, *, absolute_paths: bool) -> dict:
        source_file = str(Path(mesh.source_file).resolve()) if absolute_paths and mesh.source_file else mesh.source_file
        cleaned_file = (
            None
            if mesh.cleaned_file is None
            else str(Path(mesh.cleaned_file).resolve())
            if absolute_paths
            else mesh.cleaned_file
        )
        return {
            "name": mesh.name,
            "source_file": source_file,
            "cleaned_file": cleaned_file,
            "scale_factor": float(mesh.scale_factor),
            "translation_mm": [int(round(value)) for value in mesh.translation_mm],
            "enabled": bool(mesh.enabled),
        }

    @staticmethod
    def _mesh_scale_from_payload(payload: object) -> float:
        if not isinstance(payload, dict):
            return DEFAULT_MESH_SCALE_FACTOR
        try:
            scale_factor = float(payload.get("scale_factor", DEFAULT_MESH_SCALE_FACTOR))
        except (TypeError, ValueError):
            return DEFAULT_MESH_SCALE_FACTOR
        return scale_factor if scale_factor > 0.0 else DEFAULT_MESH_SCALE_FACTOR

    def _script_for_mesh_name(self, mesh_name: str) -> AthScriptState | None:
        return next((script for script in self.ath_scripts if script.mesh_name == mesh_name), None)

    def _has_solver_meshes(self) -> bool:
        return bool(self._enabled_ath_results()) or bool(self._active_imported_meshes())

    def _enabled_ath_results(self) -> tuple[tuple[AthScriptState, AthRunResult], ...]:
        pairs = []
        for script in self.ath_scripts:
            if not script.mesh_enabled:
                continue
            result = self.ath_results_by_script_id.get(script.id)
            if result is not None:
                pairs.append((script, result))
        return tuple(pairs)

    def _all_radiators(self) -> tuple[RadiatorConfig, ...]:
        radiators = []
        for script, result in self._enabled_ath_results():
            radiators.extend(replace(radiator, mesh=script.mesh_name) for radiator in result.radiators)
        radiators.extend(self.imported_radiators)
        return tuple(radiators)

    def _apply_radiators_to_results(self, radiators: tuple[RadiatorConfig, ...]) -> None:
        generated_mesh_names = {script.mesh_name for script in self.ath_scripts}
        for script, result in tuple(self._enabled_ath_results()):
            updated = [
                replace(radiator, mesh=script.mesh_name) for radiator in radiators if radiator.mesh == script.mesh_name
            ]
            self.ath_results_by_script_id[script.id] = replace(result, radiators=tuple(updated))
        self.imported_radiators = tuple(radiator for radiator in radiators if radiator.mesh not in generated_mesh_names)

    def _mesh_entries_from_payload(self, payload: object) -> tuple[MeshDialogEntry, ...]:
        if not isinstance(payload, list):
            return ()

        meshes = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            source_file = str(item.get("source_file", "")).strip()
            name = str(item.get("name", "")).strip()
            if not source_file or not name or name == ATH_MESH_NAME:
                continue
            translation = item.get("translation_mm", [0.0, 0.0, 0.0])
            if not isinstance(translation, list) or len(translation) != 3:
                translation = [0.0, 0.0, 0.0]
            meshes.append(
                MeshDialogEntry(
                    name=name,
                    source_file=source_file,
                    cleaned_file=None if item.get("cleaned_file") is None else str(item.get("cleaned_file")),
                    scale_factor=self._mesh_scale_from_payload(item),
                    translation_mm=tuple(float(int(round(float(value)))) for value in translation),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return tuple(meshes)

    def _clean_imported_meshes(self, meshes: tuple[MeshDialogEntry, ...]) -> tuple[MeshDialogEntry, ...]:
        cleaned_meshes = []
        for mesh in meshes:
            if not mesh.enabled:
                cleaned_meshes.append(mesh)
                continue
            source_path = Path(mesh.source_file)
            if source_path.suffix.lower() != ".msh":
                raise ValueError(f"Only .msh mesh files can be imported: {source_path}")
            if not source_path.exists():
                raise FileNotFoundError(f"Imported mesh not found: {source_path}")

            cleaned_path = Path(mesh.cleaned_file) if mesh.cleaned_file else self._cleaned_imported_mesh_path(mesh)
            if not cleaned_path.exists() or source_path.stat().st_mtime > cleaned_path.stat().st_mtime:
                cleaned_path.parent.mkdir(parents=True, exist_ok=True)
                clean_mesh_file(
                    str(source_path),
                    str(cleaned_path),
                    merge_tol=MERGE_TOL,
                    area_tol=AREA_TOL,
                    mirror_x=False,
                    binary=False,
                )

            cleaned_meshes.append(replace(mesh, cleaned_file=str(cleaned_path)))
        return tuple(cleaned_meshes)

    def _server_health_matches_preferences(self, preferences: GuiPreferences | None = None) -> bool:
        prefs = preferences or self.preferences
        if prefs.solve_backend != "server" or self.server_health_payload is None or self.server_health_url is None:
            return False
        return self.server_health_url.rstrip("/") == prefs.solve_server_url.rstrip("/")

    def _backend_supports_symmetry(
        self,
        backend_id: str,
        *,
        preferences: GuiPreferences | None = None,
        server_health_payload: dict | None = None,
    ) -> bool:
        if backend_id != "server":
            return backend_info(backend_id).capabilities.supports_symmetry
        if server_health_payload is not None:
            return server_health_supports_symmetry(server_health_payload)
        if self._server_health_matches_preferences(preferences):
            return server_health_supports_symmetry(self.server_health_payload)
        return False

    def _effective_symmetry_for_preferences(
        self,
        symmetry: str,
        preferences: GuiPreferences,
        *,
        server_health_payload: dict | None = None,
    ) -> str:
        if symmetry == "off" or self._backend_supports_symmetry(
            preferences.solve_backend,
            preferences=preferences,
            server_health_payload=server_health_payload,
        ):
            return symmetry
        return "off"

    def _selected_backend_supports_symmetry(self) -> bool:
        return self._backend_supports_symmetry(self.preferences.solve_backend)

    def _disable_symmetry_if_backend_unsupported(self) -> bool:
        effective_symmetry = self._effective_symmetry_for_preferences(self.symmetry, self.preferences)
        if effective_symmetry == self.symmetry:
            return False
        self.symmetry = effective_symmetry
        return True

    def _imported_mesh_needs_reload(self, mesh: MeshDialogEntry) -> bool:
        if not mesh.enabled:
            return False
        source_path = Path(mesh.source_file)
        if source_path.suffix.lower() != ".msh" or not source_path.exists():
            return False
        cleaned_path = Path(mesh.cleaned_file) if mesh.cleaned_file else self._cleaned_imported_mesh_path(mesh)
        if not cleaned_path.exists():
            return True
        return source_path.stat().st_mtime_ns > cleaned_path.stat().st_mtime_ns

    def _updated_imported_mesh_names(self) -> tuple[str, ...]:
        return tuple(mesh.name for mesh in self.imported_meshes if self._imported_mesh_needs_reload(mesh))

    def _reload_updated_imported_meshes_on_focus(self) -> None:
        if not self.imported_meshes or self.solve_thread is not None:
            return

        now = time.monotonic()
        if now - self._last_imported_mesh_focus_check_at < 0.5:
            return
        self._last_imported_mesh_focus_check_at = now

        updated_names = self._updated_imported_mesh_names()
        if not updated_names:
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.status_label.setText(f"Reloading updated mesh file{'s' if len(updated_names) != 1 else ''}...")
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            self.mesh_state_changed.emit("imported_mesh_files_reloaded")
            self.solve_results_invalidated.emit("imported_mesh_files_reloaded")
            names = ", ".join(updated_names)
            self.status_label.setText(f"Reloaded updated mesh file{'s' if len(updated_names) != 1 else ''}: {names}")
        except Exception as exc:
            self.status_label.setText(f"Imported mesh reload failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def _cleaned_imported_mesh_path(self, mesh: MeshDialogEntry) -> Path:
        source_path = Path(mesh.source_file)
        source_hash = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
        safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in mesh.name).strip("_")
        safe_name = safe_name or "mesh"
        return Path.cwd() / "runs" / "imported_meshes" / f"{safe_name}_{source_hash}_clean.msh"

    def _stitch_candidate_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return (*self._ath_solver_mesh_configs_for_symmetry(self.symmetry), *self._imported_solver_mesh_configs())

    def _should_use_stitched_mesh(self) -> bool:
        return self.stitch_imported_meshes and len(self._stitch_candidate_mesh_configs()) > 1

    def _stitched_mesh_path(self, mesh_configs: tuple[MeshConfig, ...]) -> Path:
        payload = {
            "symmetry": self.symmetry,
            "ignored_boundary_axes": self._stitch_ignored_boundary_axes(),
            "tol_mm": round(float(self.preferences.stitch_tolerance_mm), 6),
            "meshes": [
                {
                    "name": mesh.name,
                    "file": str(Path(mesh.file).resolve()),
                    "mtime_ns": Path(mesh.file).stat().st_mtime_ns,
                    "translation_m": mesh.translation_m,
                    "scale_factor": mesh.scale_factor,
                }
                for mesh in mesh_configs
            ],
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        return Path.cwd() / "runs" / "imported_meshes" / f"stitched_{digest}.msh"

    def _mesh_for_stitching(self, mesh_cfg: MeshConfig) -> meshio.Mesh:
        mesh = meshio.read(mesh_cfg.file)
        scale_factor = 0.001 if mesh_cfg.scale_factor is None else float(mesh_cfg.scale_factor)
        points_m = np.asarray(mesh.points, dtype=float) * scale_factor + np.asarray(mesh_cfg.translation_m, dtype=float)
        return meshio.Mesh(
            points=points_m / 0.001,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=mesh.cell_data,
            field_data=mesh.field_data,
        )

    def _stitch_ignored_boundary_axes(self) -> tuple[str, ...]:
        if self.symmetry == "x":
            return ("x",)
        if self.symmetry == "xy":
            return ("x", "y")
        return ()

    def _stitched_solver_mesh_config(self) -> MeshConfig | None:
        if not self._should_use_stitched_mesh():
            return None

        mesh_configs = self._stitch_candidate_mesh_configs()
        stitched_path = self._stitched_mesh_path(mesh_configs)
        if not stitched_path.exists():
            stitched_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                stitched_mesh, _result = stitch_meshes(
                    tuple(self._mesh_for_stitching(mesh_cfg) for mesh_cfg in mesh_configs),
                    stitch_tol=float(self.preferences.stitch_tolerance_mm),
                    area_tol=AREA_TOL,
                    ignored_boundary_axes=self._stitch_ignored_boundary_axes(),
                )
                meshio.write(stitched_path, stitched_mesh, file_format="gmsh22", binary=False)
            except Exception as exc:
                raise RuntimeError(STITCH_FAILURE_MESSAGE) from exc

        return MeshConfig(name=STITCHED_MESH_NAME, file=str(stitched_path), scale_factor=DEFAULT_MESH_SCALE_FACTOR)

    def _active_imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        return tuple(mesh for mesh in self.imported_meshes if mesh.enabled)

    def _ath_result_for_solver_symmetry(
        self,
        script: AthScriptState,
        result: AthRunResult,
        symmetry: str,
    ) -> AthRunResult:
        if symmetry == "off":
            return result
        if result.reduced_cleaned_msh_path is not None and result.reduced_cleaned_msh_path.exists():
            return result
        updated = clean_ath_reduced_mesh_output(result)
        self.ath_results_by_script_id[script.id] = updated
        return updated

    def _ath_solver_mesh_configs_for_symmetry(self, symmetry: str) -> tuple[MeshConfig, ...]:
        configs = []
        for script, result in self._enabled_ath_results():
            solver_result = self._ath_result_for_solver_symmetry(script, result, symmetry)
            configs.append(
                MeshConfig(
                    name=script.mesh_name,
                    file=str(solver_result.solver_msh_path_for_symmetry(symmetry)),
                    scale_factor=float(script.mesh_scale_factor),
                    translation_m=tuple(value / 1000.0 for value in script.mesh_translation_mm),
                )
            )
        return tuple(configs)

    def _ath_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return self._ath_solver_mesh_configs_for_symmetry(self.symmetry)

    def _imported_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        configs = []
        for mesh in self._active_imported_meshes():
            mesh_file = self._mesh_file_for_imported(mesh)
            configs.append(
                MeshConfig(
                    name=mesh.name,
                    file=mesh_file,
                    scale_factor=float(mesh.scale_factor),
                    translation_m=tuple(value / 1000.0 for value in mesh.translation_mm),
                )
            )
        return tuple(configs)

    def _mesh_file_for_imported(self, mesh: MeshDialogEntry) -> str:
        if mesh.cleaned_file and Path(mesh.cleaned_file).exists():
            return mesh.cleaned_file
        return mesh.source_file

    def _solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        stitched_mesh = self._stitched_solver_mesh_config()
        if stitched_mesh is not None:
            return (stitched_mesh,)
        return (*self._ath_solver_mesh_configs(), *self._imported_solver_mesh_configs())

    def _unique_stitched_surface_name(
        self,
        surface_name: str,
        used_surface_names: set[str],
        mesh_index: int,
    ) -> str:
        if surface_name not in used_surface_names:
            return surface_name
        suffix = 2
        candidate = f"{surface_name}_mesh{mesh_index + 1}"
        while candidate in used_surface_names:
            candidate = f"{surface_name}_mesh{mesh_index + 1}_{suffix}"
            suffix += 1
        return candidate

    def _used_surface_tags_for_mesh(self, mesh_cfg: MeshConfig) -> tuple[int, ...]:
        mesh = meshio.read(mesh_cfg.file)
        physical_by_cell_type = mesh.cell_data_dict.get("gmsh:physical", {})
        triangle_tags = physical_by_cell_type.get("triangle")
        if triangle_tags is None:
            return ()
        return tuple(sorted(int(tag) for tag in np.unique(triangle_tags)))

    def _stitched_radiator_map(self) -> dict[tuple[str | None, int], tuple[str, int]]:
        mapping: dict[tuple[str | None, int], tuple[str, int]] = {}
        used_surface_names: set[str] = set()
        used_surface_tags: set[int] = set()
        next_surface_tag = 1

        for mesh_index, mesh_cfg in enumerate(self._stitch_candidate_mesh_configs()):
            names_by_tag = {
                tag: surface_name for surface_name, tag in read_surface_physical_names(Path(mesh_cfg.file)).items()
            }
            for old_tag in self._used_surface_tags_for_mesh(mesh_cfg):
                surface_name = names_by_tag.get(old_tag, f"mesh{mesh_index + 1}_surface_{old_tag}")
                stitched_surface_name = self._unique_stitched_surface_name(
                    surface_name,
                    used_surface_names,
                    mesh_index,
                )
                used_surface_names.add(stitched_surface_name)
                if old_tag not in used_surface_tags:
                    new_tag = old_tag
                else:
                    while next_surface_tag in used_surface_tags:
                        next_surface_tag += 1
                    new_tag = next_surface_tag
                used_surface_tags.add(new_tag)
                mapping[(mesh_cfg.name, old_tag)] = (f"{STITCHED_MESH_NAME}:{stitched_surface_name}", new_tag)

        return mapping

    def _radiators_for_solver_meshes(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[RadiatorConfig, ...]:
        if len(mesh_configs) != 1 or mesh_configs[0].name != STITCHED_MESH_NAME:
            return radiators

        stitched_map = self._stitched_radiator_map()
        resolved = []
        for radiator in radiators:
            stitched_surface = stitched_map.get((radiator.mesh, radiator.tag))
            if stitched_surface is None:
                resolved.append(replace(radiator, mesh=STITCHED_MESH_NAME))
                continue
            stitched_name, stitched_tag = stitched_surface
            resolved.append(
                replace(
                    radiator,
                    name=stitched_name,
                    mesh=STITCHED_MESH_NAME,
                    tag=stitched_tag,
                )
            )
        return tuple(resolved)

    def _show_stitch_or_generic_error(self, title: str, exc: Exception) -> None:
        if str(exc) != STITCH_FAILURE_MESSAGE:
            QMessageBox.critical(self, title, str(exc))
            return

        message = QMessageBox(QMessageBox.Critical, title, STITCH_FAILURE_MESSAGE, QMessageBox.Ok, self)
        if exc.__cause__ is not None:
            message.setDetailedText(str(exc.__cause__))
        message.exec()

    def _show_mesh_quality_warning(self, result: AthRunResult) -> None:
        warning = result.quality_warning
        if warning is None or not warning.has_warnings:
            return

        QMessageBox.warning(
            self,
            "Mesh quality warning",
            (
                "The cleaned mesh contains extremely thin triangles that may make the "
                "BEAT Engine produced non-finite results.\n\n"
                f"Thin triangles: {warning.sliver_triangles}\n"
                f"Float32-singular triangles: {warning.float32_singular_triangles}\n"
                f"Worst triangle: {warning.worst_triangle_index}\n"
                f"Worst altitude/edge ratio: {warning.worst_altitude_edge_ratio:.3g}\n\n"
                "Try increasing mesh resolution around sharp transitions or adjusting the Ath geometry "
                "to avoid long, needle-like triangles."
            ),
        )

    def _surface_tags_for_meshes(self) -> dict[str, tuple[str, int]]:
        surface_tags: dict[str, tuple[str, int]] = {}
        for mesh_cfg in self._solver_mesh_configs():
            for surface_name, tag in read_surface_physical_names(Path(mesh_cfg.file)).items():
                surface_tags[f"{mesh_cfg.name}:{surface_name}"] = (mesh_cfg.name, tag)
        return surface_tags

    def _refresh_mesh_preview(self) -> None:
        if not self._has_solver_meshes():
            self.preview.clear()
            return
        try:
            mesh_configs = self._solver_mesh_configs()
            if not mesh_configs:
                self.preview.clear()
                return
            surface_tags_by_mesh = {
                mesh_cfg.name: read_surface_physical_names(Path(mesh_cfg.file)) for mesh_cfg in mesh_configs
            }
            self.preview.load_mesh_configs(
                mesh_configs,
                driven_surfaces={
                    (radiator.mesh, radiator.tag)
                    for radiator in self._radiators_for_solver_meshes(mesh_configs, self._all_radiators())
                },
                surface_tags_by_mesh=surface_tags_by_mesh,
                symmetry=self.symmetry,
            )
        except Exception as exc:
            if str(exc) == STITCH_FAILURE_MESSAGE and self.stitch_imported_meshes:
                self._refresh_unstitched_mesh_preview_after_stitch_failure()
                return
            self.preview.clear()

    def _refresh_unstitched_mesh_preview_after_stitch_failure(self) -> None:
        try:
            mesh_configs = self._stitch_candidate_mesh_configs()
            if not mesh_configs:
                self.preview.clear()
                return
            surface_tags_by_mesh = {
                mesh_cfg.name: read_surface_physical_names(Path(mesh_cfg.file)) for mesh_cfg in mesh_configs
            }
            self.preview.load_mesh_configs(
                mesh_configs,
                driven_surfaces={(radiator.mesh, radiator.tag) for radiator in self._all_radiators()},
                surface_tags_by_mesh=surface_tags_by_mesh,
                symmetry=self.symmetry,
            )
            self.status_label.setText("Mesh preview showing unstitched meshes; stitching failed")
        except Exception:
            self.preview.clear()

    def _load_source_config_by_name(self) -> dict[str, dict]:
        return load_source_config_by_name(self.settings)

    def _save_source_config(
        self, surface_tags: dict[str, tuple[str, int]], radiators: tuple[RadiatorConfig, ...]
    ) -> None:
        save_source_config(self.settings, surface_tags, radiators)

    def _load_channel_config_by_name(self) -> dict[str, dict]:
        return load_channel_config_by_name(self.settings)

    def _save_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        save_channel_config(self.settings, channels)

    def _channel_configs(self) -> tuple[ChannelConfig, ...]:
        return channel_configs(self.settings)

    def _channels_for_solver_radiators(
        self,
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[ChannelConfig, ...]:
        return channels_for_solver_radiators(self._channel_configs(), radiators)

    def _channel_configs_for_current_radiators(self) -> tuple[ChannelConfig, ...]:
        return self._channels_for_solver_radiators(self._all_radiators())

    def _discard_channel_config_dialog(self) -> None:
        dialog = self.channel_config_dialog
        self.channel_config_dialog = None
        if dialog is not None:
            dialog.deleteLater()

    def _apply_saved_source_config_to_result(self, result: AthRunResult | None, mesh_name: str) -> AthRunResult | None:
        return apply_saved_source_config_to_result(result, mesh_name, self._load_source_config_by_name())

    def _apply_saved_source_config(self, result: AthRunResult | None) -> AthRunResult | None:
        return self._apply_saved_source_config_to_result(result, ATH_MESH_NAME)

    def _apply_saved_imported_source_config(self, surface_tags: dict[str, tuple[str, int]]) -> None:
        generated_mesh_names = {script.mesh_name for script in self.ath_scripts}
        self.imported_radiators = apply_saved_imported_source_config(
            surface_tags=surface_tags,
            generated_mesh_names=generated_mesh_names,
            existing_radiators=self.imported_radiators,
            config_by_name=self._load_source_config_by_name(),
        )

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._confirm_unsaved_project_changes("close"):
            event.ignore()
            return
        self._save_frequency_settings()
        self._save_preferences()
        self._save_window_state()
        super().closeEvent(event)

    def _result_from_script_state(self, script: AthScriptState) -> AthRunResult | None:
        if not script.output_dir or not script.msh_path:
            return None
        msh_path = Path(script.msh_path)
        if not msh_path.exists():
            return None
        cleaned_path = Path(script.cleaned_msh_path) if script.cleaned_msh_path else None
        solver_path = cleaned_path if cleaned_path is not None and cleaned_path.exists() else msh_path
        try:
            driven_tag = find_physical_tag_by_name(solver_path, "SD1D1001")
            return AthRunResult(
                output_dir=Path(script.output_dir),
                msh_path=msh_path,
                config_path=Path(script.config_path) if script.config_path else Path(script.output_dir) / "config.txt",
                driven_tag=driven_tag,
                radiators=detect_ath_radiators(solver_path),
                cleaned_msh_path=cleaned_path if cleaned_path is not None and cleaned_path.exists() else None,
            )
        except Exception:
            return None

    def _find_ath_exe(self) -> Path:
        bundled = ATH_BUNDLE_DIR / "ath.exe"
        if bundled.exists():
            return bundled
        for root in (Path.cwd(), Path.cwd().parent):
            candidate = root / "ath.exe"
            if candidate.exists():
                return candidate
        return bundled

    def _ensure_ath_runtime_config(self) -> None:
        ath_exe = self._find_ath_exe()
        ath_cfg = ath_exe.parent / "ath.cfg"
        if not ath_cfg.exists():
            return
        write_ath_output_root(ath_cfg, ATH_OUTPUT_ROOT)
        write_ath_gmsh_path(ath_cfg, GMSH_BUNDLE_EXE)

    @Slot()
    def import_config(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Import Ath config",
            str(Path.cwd()),
            "Ath config files (*.cfg);;All files (*)",
        )
        if not path_text:
            return

        self._import_config_path(Path(path_text))

    def _import_config_path(self, path: Path, *, script_id: str | None = None) -> None:
        try:
            config_text = path.read_text(encoding="utf-8")
            script = (
                next((item for item in self.ath_scripts if item.id == script_id), None)
                if script_id
                else self._active_script()
            )
            if script is None:
                script = new_script(unique_script_name(path.stem, self.ath_scripts), config_text)
                self.ath_scripts = (*self.ath_scripts, script)
                self.active_ath_script_id = script.id
            else:
                self.ath_scripts = replace_script(self.ath_scripts, script.id, config_text=config_text)
            self._rebuild_ath_script_tabs()
            self.status_label.setText(f"Imported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    @Slot()
    def new_project(self) -> None:
        if not self._confirm_unsaved_project_changes("new_project"):
            return
        self._discard_channel_config_dialog()
        self.project_path = None
        self.ath_scripts = default_scripts("")
        self.active_ath_script_id = self.ath_scripts[0].id if self.ath_scripts else None
        self.ath_results_by_script_id = {}
        self._rebuild_ath_script_tabs()
        self.imported_meshes = ()
        self.imported_radiators = ()
        self.stitch_imported_meshes = False
        self.symmetry = "off"
        clear_source_channel_configs(self.settings)
        self.project_state_changed.emit("new_project")
        self.solve_results_invalidated.emit("new_project")
        self._mark_project_clean()
        self.status_label.setText("New project")

    @Slot()
    def save_project(self) -> bool:
        if self.project_path is None:
            return self.save_project_as()
        return self._save_project_to_path(self.project_path)

    @Slot()
    def save_project_as(self) -> bool:
        default_path = self.project_path or (Path.cwd() / PROJECT_DEFAULT_NAME)
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            str(default_path),
            PROJECT_FILE_FILTER,
        )
        if not path_text:
            return False
        return self._save_project_to_path(normalize_project_path(path_text))

    def _save_project_to_path(self, path: Path) -> bool:
        try:
            project_path = write_project_file(path, self._project_payload())
            self.project_path = project_path
            self._remember_recent_project(project_path)
            self._mark_project_clean()
            self.status_label.setText(f"Saved project {project_path}")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))
            return False

    @Slot()
    def load_project(self) -> None:
        if not self._confirm_unsaved_project_changes("open_project"):
            return
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Open Project",
            str(Path.cwd()),
            PROJECT_FILE_FILTER,
        )
        if not path_text:
            return

        self._load_project_from_path(Path(path_text))

    @Slot()
    def open_recent_project(self, path: Path) -> None:
        if not path.exists():
            self._remove_recent_project(path)
            QMessageBox.warning(self, "Open project failed", f"Recent project not found:\n{path}")
            return
        if not self._confirm_unsaved_project_changes("open_project"):
            return
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: Path) -> None:
        try:
            payload = read_project_file(path)
            self._apply_project_payload(payload)
            self.project_path = path
            self._remember_recent_project(path)
            self._mark_project_clean()
            self.status_label.setText(f"Opened project {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))

    def _project_payload(self) -> dict:
        active_script = self._active_script()
        return build_project_payload(
            ath_config_text="" if active_script is None else active_script.config_text,
            ath_mesh=self._ath_mesh_payload(absolute_paths=True),
            imported_meshes=self._project_imported_meshes_payload(),
            stitch_imported_meshes=self.stitch_imported_meshes,
            symmetry=self.symmetry,
            source_config_by_name=self._load_source_config_by_name(),
            ath_scripts=[script_to_payload(script, absolute_paths=True) for script in self.ath_scripts],
            active_ath_script_id=self.active_ath_script_id,
            channel_config_by_name=self._load_channel_config_by_name(),
        )

    def _canonical_project_payload(self) -> dict:
        payload = json.loads(json.dumps(self._project_payload(), sort_keys=True))
        for script in payload.get("ath_scripts", []):
            if not isinstance(script, dict):
                continue
            for field in ("output_dir", "msh_path", "cleaned_msh_path", "config_path"):
                script.pop(field, None)
        return payload

    def _mark_project_clean(self) -> None:
        self._project_clean_payload = self._canonical_project_payload()

    def _has_unsaved_project_changes(self) -> bool:
        if self._project_clean_payload is None:
            return False
        return self._canonical_project_payload() != self._project_clean_payload

    def _confirm_unsaved_project_changes(self, action: str) -> bool:
        if not self._has_unsaved_project_changes():
            return True
        message_text = (
            "You have unsaved changes. Are you sure you want to close?"
            if action == "close"
            else "You have unsaved changes. Save before continuing?"
        )
        message = QMessageBox(
            QMessageBox.Warning,
            "Unsaved Changes",
            message_text,
            QMessageBox.NoButton,
            self,
        )
        save_button = message.addButton("Save", QMessageBox.AcceptRole)
        discard_button = message.addButton("Discard", QMessageBox.DestructiveRole)
        cancel_button = message.addButton("Cancel", QMessageBox.RejectRole)
        message.setDefaultButton(cancel_button)
        message.exec()
        clicked = message.clickedButton()
        if clicked is save_button:
            return self.save_project()
        if clicked is discard_button:
            return True
        return False

    def _apply_project_payload(self, payload: dict) -> None:
        self._discard_channel_config_dialog()
        source_config = payload.get("source_config_by_name", {})
        if not isinstance(source_config, dict):
            source_config = {}
        save_source_config_by_name(self.settings, source_config)
        channel_config = payload.get("channel_config_by_name", {})
        if not isinstance(channel_config, dict):
            channel_config = {}
        save_channel_config_by_name(self.settings, channel_config)

        self.ath_scripts = scripts_from_payload(
            payload.get("ath_scripts"),
            fallback_config_text=str(payload.get("ath_config_text", "")),
        )
        active_id = payload.get("active_ath_script_id")
        self.active_ath_script_id = (
            active_id
            if any(script.id == active_id for script in self.ath_scripts)
            else (self.ath_scripts[0].id if self.ath_scripts else None)
        )
        self.ath_results_by_script_id = {}
        for script in self.ath_scripts:
            result = self._result_from_script_state(script)
            if result is not None:
                self.ath_results_by_script_id[script.id] = self._apply_saved_source_config_to_result(
                    result, script.mesh_name
                )
        self._rebuild_ath_script_tabs()
        self.imported_meshes = self._mesh_entries_from_payload(payload.get("imported_meshes", []))
        self.stitch_imported_meshes = bool(payload.get("stitch_imported_meshes", False))
        self.symmetry = str(payload.get("symmetry", "off")).strip().lower()
        if self.symmetry not in {"off", "x", "xy"}:
            self.symmetry = "off"
        self._disable_symmetry_if_backend_unsupported()
        self.imported_radiators = ()
        try:
            self._apply_saved_imported_source_config(self._surface_tags_for_meshes())
        except Exception:
            self.imported_radiators = ()

        self.project_state_changed.emit("project_loaded")
        self.solve_results_invalidated.emit("project_loaded")

    @Slot()
    def export_config(self) -> None:
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Export Ath config",
            str(Path.cwd() / "waveguide.cfg"),
            "Ath config files (*.cfg);;All files (*)",
        )
        if not path_text:
            return

        path = Path(path_text)
        if path.suffix == "":
            path = path.with_suffix(".cfg")

        try:
            script = self._active_script()
            path.write_text("" if script is None else script.config_text, encoding="utf-8")
            self.status_label.setText(f"Exported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    @Slot(str)
    def export_plot(self, plot_id: str) -> None:
        dataset = self._prepared_live_plot_dataset(
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None,
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES if plot_id in {"horizontal_isobar", "vertical_isobar"} else None,
        )
        if dataset is None:
            QMessageBox.warning(self, "No plot data", "Run a solve before exporting a plot.")
            return

        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        if entry is None:
            return

        path_text, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {entry.title}",
            str(Path.cwd() / entry.default_filename),
            "PNG images (*.png);;All files (*)",
        )
        if not path_text:
            return

        output_path = Path(path_text)
        if output_path.suffix == "":
            output_path = output_path.with_suffix(".png")
        try:
            entry.update(dataset)
            figure = getattr(entry.widget, "figure")
            output_path = export_plot_png(figure, output_path, dpi=VisualizerConfig.figure_dpi)
            self.status_label.setText(f"Exported {entry.title} to {output_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export plot failed", str(exc))

    @Slot()
    def export_polar_data(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No polar data", "Run a solve before exporting polar data.")
            return

        dir_text = QFileDialog.getExistingDirectory(
            self,
            "Export polar data",
            str(Path.cwd()),
        )
        if not dir_text:
            return

        output_dir = Path(dir_text)
        try:
            self.live_dataset.set_channel_synthesis(
                self._channel_configs(),
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            written = export_polar_text_files(self.live_dataset, output_dir)
            self.status_label.setText(f"Exported {len(written)} polar files to {output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export polar data failed", str(exc))

    @Slot()
    def open_balloon_plot(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No balloon data", "Run a solve before opening the balloon plot.")
            return

        self.live_dataset.set_channel_synthesis(
            self._channel_configs(),
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
        )
        raw_balloon = self.live_dataset.as_balloon_raw_bundle()
        if raw_balloon is None:
            QMessageBox.warning(
                self,
                "No balloon data",
                "Enable spherical sampling in Preferences before running a solve.",
            )
            return

        try:
            from blab.ui.balloon import BalloonPlotWindow

            self.balloon_window = BalloonPlotWindow(
                raw_balloon,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
                polar_smoothing=self.preferences.polar_smoothing,
                parent=self,
            )
            self.balloon_window.show()
            self.balloon_window.raise_()
        except Exception as exc:
            QMessageBox.critical(self, "Balloon plot failed", str(exc))

    @Slot()
    def open_preferences(self) -> None:
        previous_preferences = self.preferences
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() != QDialog.Accepted:
            return
        preferences = dialog.preferences()
        dialog.deleteLater()
        checked_server_health = None
        if dialog.server_health_payload is not None and dialog.server_health_url == preferences.solve_server_url.rstrip("/"):
            checked_server_health = dialog.server_health_payload
        symmetry_will_be_disabled = (
            self._effective_symmetry_for_preferences(
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
            return

        self.preferences = preferences
        if checked_server_health is not None and preferences.solve_backend == "server":
            self.server_health_payload = checked_server_health
            self.server_health_url = preferences.solve_server_url.rstrip("/")
        elif (
            preferences.solve_backend != previous_preferences.solve_backend
            or preferences.solve_server_url != previous_preferences.solve_server_url
        ):
            self.server_health_payload = None
            self.server_health_url = None
        self._save_preferences()
        symmetry_disabled = self._disable_symmetry_if_backend_unsupported()
        QTimer.singleShot(0, self._apply_theme)
        self.mesh_state_changed.emit("preferences_changed")
        if symmetry_disabled or preferences_require_solve_invalidation(previous_preferences, self.preferences):
            self.solve_results_invalidated.emit("preferences_changed")
        elif preferences_require_visualization_refresh(previous_preferences, self.preferences):
            self.visualization_settings_changed.emit("preferences_changed")
        self.status_label.setText("Preferences updated")

    @Slot()
    def open_diagnostics(self) -> None:
        dialog = DiagnosticsDialog(self.preferences, self)
        dialog.exec()

    @Slot()
    def open_donate(self) -> None:
        dialog = DonateDialog(self)
        dialog.exec()

    @Slot()
    def open_help(self) -> None:
        if not HELP_GUIDE_PDF.exists():
            QMessageBox.warning(
                self,
                "Help guide missing",
                f"The Boundary Lab guide PDF could not be found:\n{HELP_GUIDE_PDF}",
            )
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(HELP_GUIDE_PDF))):
            QMessageBox.warning(
                self,
                "Help guide failed",
                "Unable to open the Boundary Lab guide PDF in the default viewer.",
            )

    @Slot()
    def open_mesh_config(self) -> None:
        self._disable_symmetry_if_backend_unsupported()
        symmetry_enabled = self._selected_backend_supports_symmetry()
        dialog = MeshConfigDialog(
            self._mesh_config_dialog_entries(),
            stitch_imported_meshes=self.stitch_imported_meshes,
            symmetry=self.symmetry,
            symmetry_enabled=symmetry_enabled,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        meshes = dialog.meshes()
        stitch_imported_meshes = dialog.stitch_imported_meshes()
        symmetry = dialog.symmetry() if symmetry_enabled else self.symmetry
        config_changed = (
            meshes != self._mesh_config_dialog_entries()
            or stitch_imported_meshes != self.stitch_imported_meshes
            or symmetry != self.symmetry
        )
        if not config_changed:
            self.status_label.setText("Mesh config unchanged")
            return
        if not self._confirm_clear_solved_data():
            return

        try:
            self.status_label.setText("Cleaning imported meshes...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._apply_mesh_config_dialog_entries(meshes)
            self.stitch_imported_meshes = stitch_imported_meshes
            if symmetry_enabled:
                self.symmetry = symmetry
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            self.mesh_state_changed.emit("mesh_config_changed")
            self.solve_results_invalidated.emit("mesh_config_changed")
            self.status_label.setText(
                f"Mesh config updated: {len(self._active_imported_meshes())}/{len(self.imported_meshes)} meshes enabled"
            )
        except Exception as exc:
            self.status_label.setText("Mesh config failed")
            QMessageBox.critical(self, "Mesh config failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def open_channel_config(self) -> None:
        if self.channel_config_dialog is not None:
            self.channel_config_dialog.show()
            self.channel_config_dialog.raise_()
            self.channel_config_dialog.activateWindow()
            return

        dialog = ChannelConfigDialog(self._channel_configs_for_current_radiators(), self)
        self.channel_config_dialog = dialog
        dialog.channelsApplied.connect(self._apply_channel_config)
        dialog.destroyed.connect(lambda *_args: setattr(self, "channel_config_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

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

    @Slot(object)
    def _apply_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        channels = tuple(channels)
        channel_config_changed = channels != self._channel_configs()
        previous_radiator_assignments = tuple(
            (radiator.mesh, radiator.tag, radiator.channel) for radiator in self._all_radiators()
        )
        valid_names = {channel.name for channel in channels}
        fallback = channels[0].name
        radiator_assignments_changed = any(radiator.channel not in valid_names for radiator in self._all_radiators())
        can_resynthesize = (
            not radiator_assignments_changed
            and self.live_dataset is not None
            and self.live_dataset.supports_channel_resynthesis
        )
        if not channel_config_changed and not radiator_assignments_changed:
            self.status_label.setText("Channel config unchanged")
            return
        if not can_resynthesize and not self._confirm_clear_solved_data():
            return

        self._save_channel_config(channels)
        for script_id, result in tuple(self.ath_results_by_script_id.items()):
            self.ath_results_by_script_id[script_id] = replace(
                result,
                radiators=tuple(
                    radiator if radiator.channel in valid_names else replace(radiator, channel=fallback)
                    for radiator in result.radiators
                ),
            )
        try:
            self._save_source_config(self._surface_tags_for_meshes(), self._all_radiators())
        except Exception:
            pass
        current_radiator_assignments = tuple(
            (radiator.mesh, radiator.tag, radiator.channel) for radiator in self._all_radiators()
        )
        radiator_assignments_changed = current_radiator_assignments != previous_radiator_assignments
        self.source_config_changed.emit("channel_config_changed")
        if can_resynthesize:
            self.live_dataset.set_channel_synthesis(
                channels,
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            self._refresh_plots()
            self.balloon_plot_action.setEnabled(self.live_dataset.as_balloon_raw_bundle() is not None)
            self.status_label.setText(f"Channel config updated: {len(channels)} channels; plots resynthesized")
        else:
            self.solve_results_invalidated.emit("channel_config_changed")
            self.status_label.setText(f"Channel config updated: {len(channels)} channels")

    @Slot()
    def open_source_config(self) -> None:
        if not self._has_solver_meshes():
            QMessageBox.warning(self, "No mesh", "Generate or load a mesh before configuring sources.")
            return

        try:
            surface_tags = self._surface_tags_for_meshes()
            self._apply_saved_imported_source_config(surface_tags)
        except Exception as exc:
            self._show_stitch_or_generic_error("Source config failed", exc)
            return

        dialog = SourceConfigDialog(
            surface_tags, self._all_radiators(), self._channel_configs_for_current_radiators(), self
        )
        if dialog.exec() != QDialog.Accepted:
            return

        radiators = dialog.radiators()
        if radiators == self._all_radiators():
            self.status_label.setText("Source config unchanged")
            return
        if not self._confirm_clear_solved_data():
            return
        self._apply_radiators_to_results(radiators)
        self._save_source_config(surface_tags, radiators)
        self.source_config_changed.emit("source_config_changed")
        self.solve_results_invalidated.emit("source_config_changed")
        self.status_label.setText(f"Source config updated: {len(radiators)} driven surfaces")

    @Slot()
    def generate_geometry(self) -> None:
        if self.ath_worker is not None or self.solve_worker is not None:
            return
        script = self._active_script()
        if script is None:
            QMessageBox.warning(self, "No Ath script", "Add an Ath script before generating.")
            return
        case_name = f"{script.mesh_name}_{script.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_root = ATH_OUTPUT_ROOT
        try:
            self._ensure_ath_runtime_config()
            ath_exe = self._find_ath_exe()
        except Exception as exc:
            self.status_label.setText("Generate failed")
            QMessageBox.critical(self, "Ath generation failed", str(exc))
            return

        self.solve_results_invalidated.emit("geometry_generation_started")
        self.status_label.setText(f"Generating {script.name}...")
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.source_config_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self._set_contour_button_states()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.ath_generation_script_id = script.id
        self.ath_generation_mesh_name = script.mesh_name
        self.ath_generation_cancel_requested = False
        self.ath_thread = QThread(self)
        self.ath_worker = AthGenerationWorker(
            ath_exe=ath_exe,
            config_text=script.config_text,
            run_root=run_root,
            case_name=case_name,
        )
        self.ath_worker.moveToThread(self.ath_thread)
        self.ath_thread.started.connect(self.ath_worker.run)
        self.ath_worker.generated.connect(self._on_ath_generated)
        self.ath_worker.status.connect(self.status_label.setText)
        self.ath_worker.failed.connect(self._on_ath_generation_failed)
        self.ath_worker.cancelled.connect(self._on_ath_generation_cancelled)
        self.ath_worker.finished.connect(self._on_ath_generation_finished)
        self.ath_worker.finished.connect(self.ath_thread.quit)
        self.ath_worker.finished.connect(self.ath_worker.deleteLater)
        self.ath_thread.finished.connect(self.ath_thread.deleteLater)
        self.ath_thread.start()
        QTimer.singleShot(3000, self._enable_ath_cancel_if_active)

    @Slot()
    def _enable_ath_cancel_if_active(self) -> None:
        if self.ath_worker is not None and not self.ath_generation_cancel_requested:
            self.cancel_button.setEnabled(True)

    @Slot(object)
    def _on_ath_generated(self, generated_result: AthRunResult) -> None:
        script_id = self.ath_generation_script_id
        mesh_name = self.ath_generation_mesh_name
        if script_id is None or mesh_name is None:
            return
        result = self._apply_saved_source_config_to_result(generated_result, mesh_name)
        self.ath_results_by_script_id[script_id] = result
        self.ath_scripts = replace_script(
            self.ath_scripts,
            script_id,
            output_dir=str(result.output_dir),
            msh_path=str(result.msh_path),
            cleaned_msh_path=None if result.cleaned_msh_path is None else str(result.cleaned_msh_path),
            config_path=str(result.config_path),
        )
        self.mesh_state_changed.emit("ath_mesh_generated")
        self.status_label.setText(f"Generated and cleaned {result.output_dir}")
        self._show_mesh_quality_warning(result)

    @Slot(str)
    def _on_ath_generation_failed(self, message: str) -> None:
        self.status_label.setText("Generate failed")
        QMessageBox.critical(self, "Ath generation failed", message)

    @Slot()
    def _on_ath_generation_cancelled(self) -> None:
        self.status_label.setText("Ath generation stopped")

    @Slot()
    def _on_ath_generation_finished(self) -> None:
        QApplication.restoreOverrideCursor()
        self.solve_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.mesh_config_button.setEnabled(True)
        self.channel_config_button.setEnabled(True)
        self.source_config_button.setEnabled(self._has_solver_meshes())
        self.cancel_button.setEnabled(False)
        self.ath_worker = None
        self.ath_thread = None
        self.ath_generation_script_id = None
        self.ath_generation_mesh_name = None
        self.ath_generation_cancel_requested = False

    @Slot()
    def start_solve(self) -> None:
        if self.ath_worker is not None:
            return
        if not self._has_solver_meshes():
            QMessageBox.warning(self, "No mesh", "Enable at least one generated or imported mesh before solving.")
            return
        radiators = self._all_radiators()
        if not radiators:
            QMessageBox.warning(
                self, "No driven surfaces", "Open Source Config and mark at least one surface as Driven."
            )
            return
        if self._disable_symmetry_if_backend_unsupported():
            self.mesh_state_changed.emit("symmetry_disabled_for_backend")

        try:
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            mesh_configs = self._solver_mesh_configs()
            radiators = self._radiators_for_solver_meshes(mesh_configs, radiators)
        except Exception as exc:
            self._show_stitch_or_generic_error("Imported mesh preparation failed", exc)
            return
        try:
            validate_reduced_mesh_configs(mesh_configs, self.symmetry)
        except SymmetryValidationError as exc:
            QMessageBox.warning(self, "Symmetry validation failed", str(exc))
            return

        freq_min = float(min(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_max = float(max(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_count = int(self.freq_count_spin.value())
        freqs = build_log_frequencies(freq_min, freq_max, freq_count)
        ordered_freqs = order_frequencies_for_live_plotting(freqs)

        channels = self._channels_for_solver_radiators(radiators)
        config = SimulationConfig(
            mesh_file=mesh_configs[0].file if mesh_configs else "",
            freq_min=freq_min,
            freq_max=freq_max,
            freq_count=freq_count,
            tag_throat=radiators[0].tag,
            meshes=mesh_configs,
            radiators=radiators,
            channels=channels,
            distance=self.preferences.polar_observation_distance_m,
            step_size=self.preferences.polar_angle_step_deg,
            use_burton_miller=self.preferences.use_burton_miller,
            gmres_tolerance=self.preferences.gmres_tolerance,
            workers=1,
            flat_target_normalization_enabled=self.preferences.normalized_channel_correction,
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
            spherical_sampling_points=balloon_sampling_points(self.preferences.balloon_angle_precision_deg),
            symmetry=self.symmetry,
        )

        self.live_dataset = None
        self._clear_plots()
        self.balloon_plot_action.setEnabled(False)
        self.solve_expected_count = int(ordered_freqs.size)
        self.solve_failed = False
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.source_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self._set_contour_button_states()
        self.solve_started_at = time.perf_counter()
        self.solve_cancel_requested = False
        self.status_label.setText("Initializing Solver...")

        self.solve_thread = QThread(self)
        self.solve_worker = SolveWorker(
            config,
            ordered_freqs,
            worker_count=1,
            backend_id=self.preferences.solve_backend,
            server_url=self.preferences.solve_server_url,
        )
        self.solve_worker.moveToThread(self.solve_thread)
        self.solve_thread.started.connect(self.solve_worker.run)
        self.solve_worker.initialized.connect(self._on_solver_initialized)
        self.solve_worker.result_ready.connect(self._on_frequency_result)
        self.solve_worker.status.connect(self.status_label.setText)
        self.solve_worker.failed.connect(self._on_solve_failed)
        self.solve_worker.finished.connect(self._on_solve_finished)
        self.solve_worker.finished.connect(self.solve_thread.quit)
        self.solve_worker.finished.connect(self.solve_worker.deleteLater)
        self.solve_thread.finished.connect(self.solve_thread.deleteLater)
        self.solve_thread.start()

    @Slot()
    def cancel_current_operation(self) -> None:
        if self.ath_worker is not None:
            self.cancel_ath_generation()
            return
        self.cancel_solve()

    @Slot()
    def cancel_ath_generation(self) -> None:
        self.ath_generation_cancel_requested = True
        self.cancel_button.setEnabled(False)
        if self.ath_worker is not None:
            self.ath_worker.stop()
            self.status_label.setText("Stop requested; ending Ath generation...")

    @Slot()
    def cancel_solve(self) -> None:
        self.solve_cancel_requested = True
        if self.solve_worker is not None:
            self.solve_worker.stop()
            self.status_label.setText("Stop requested; waiting for current frequency...")

    @Slot(object, object)
    def _on_solver_initialized(
        self,
        angles: np.ndarray,
        radiator_names: np.ndarray,
        sphere_metadata: dict[str, np.ndarray] | None,
    ) -> None:
        sphere_metadata = sphere_metadata or {}
        self.live_dataset = LiveSolveDataset(
            polar_angle_deg=np.asarray(angles, dtype=np.float32),
            radiator_names=np.asarray(radiator_names),
            channel_configs=self._channel_configs(),
            flat_target_normalization_enabled=self.preferences.normalized_channel_correction,
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            sphere_r_distance_m=sphere_metadata.get("r_distance_m"),
            sphere_theta_polar_rad=sphere_metadata.get("theta_polar_rad"),
            sphere_phi_azimuth_rad=sphere_metadata.get("phi_azimuth_rad"),
        )
        self.status_label.setText("Solving...")

    @Slot(object)
    def _on_frequency_result(self, result: FrequencyResult) -> None:
        if self.live_dataset is None:
            return
        self.live_dataset.add(result)
        self.status_label.setText(
            f"Solved {self.live_dataset.solved_count}/{self.freq_count_spin.value()} "
            f"({result.freq_hz:.1f} Hz) | {format_frequency_solve_timings(result)}"
        )
        if not self.preferences.live_plot_streaming:
            return
        self._refresh_plots()

    @Slot(str)
    def _on_solve_failed(self, message: str) -> None:
        self.solve_failed = True
        QMessageBox.critical(self, "Solve failed", message)
        self.status_label.setText("Solve failed")

    @Slot()
    def _on_solve_finished(self) -> None:
        self.solve_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.mesh_config_button.setEnabled(True)
        self.channel_config_button.setEnabled(True)
        self.source_config_button.setEnabled(self._has_solver_meshes())
        self.cancel_button.setEnabled(False)
        elapsed_s = None if self.solve_started_at is None else time.perf_counter() - self.solve_started_at
        self.solve_started_at = None
        if self.live_dataset is not None and self.live_dataset.solved_count > 0:
            solved_count = self.live_dataset.solved_count
            solve_completed = (
                not self.solve_cancel_requested
                and not self.solve_failed
                and self.solve_expected_count > 0
                and solved_count >= self.solve_expected_count
            )
            self._use_final_isobar_resolution = solve_completed
            if solve_completed:
                self.status_label.setText("Rendering final high-resolution plots...")
                QApplication.processEvents()
            if self.preferences.live_plot_streaming or solve_completed:
                self._refresh_plots()
            self._final_isobar_plots_rendered = solve_completed and bool(self._visible_isobar_plots())
            self._set_export_plot_actions_enabled(True)
            self.export_polar_data_action.setEnabled(True)
            self.balloon_plot_action.setEnabled(self.live_dataset.as_balloon_raw_bundle() is not None)
            self._set_contour_button_states()
            elapsed_text = "" if elapsed_s is None else f" in {elapsed_s:.1f} s"
            if self.solve_cancel_requested:
                self.status_label.setText(f"Solve stopped: {self.live_dataset.solved_count} frequencies{elapsed_text}")
                self.solve_cancel_requested = False
                self.solve_worker = None
                self.solve_thread = None
                self.solve_failed = False
                self.solve_expected_count = 0
                return
            if self.solve_failed:
                self.status_label.setText(f"Solve failed after {solved_count} frequencies{elapsed_text}")
                self.solve_failed = False
                self.solve_expected_count = 0
                self.solve_worker = None
                self.solve_thread = None
                return
            self.status_label.setText(f"Solve complete: {solved_count} frequencies{elapsed_text}")
        elif self.solve_cancel_requested:
            self.status_label.setText("Solve stopped")
        self.solve_cancel_requested = False
        self.solve_failed = False
        self.solve_expected_count = 0
        self.solve_worker = None
        self.solve_thread = None
        self._set_contour_button_states()

    def _clear_plots(self) -> None:
        self.live_dataset = None
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        for entry in self.plot_entries:
            entry.widget._draw_empty()
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.balloon_plot_action.setEnabled(False)
        self._set_contour_button_states()

    def _set_plot_visible(self, plot_id: str, visible: bool) -> None:
        for entry in self.plot_entries:
            if entry.plot_id != plot_id:
                continue
            dock = self.plot_docks.get(plot_id)
            if dock is not None and dock.isVisible() != visible:
                dock.setVisible(visible)
            if visible:
                self._refresh_plots()
                if self._use_final_isobar_resolution and plot_id in {"horizontal_isobar", "vertical_isobar"}:
                    self._final_isobar_plots_rendered = True
            self._set_contour_button_states()
            break

    def _sync_plot_view_action(self, plot_id: str) -> None:
        action = self.plot_view_actions.get(plot_id)
        dock = self.plot_docks.get(plot_id)
        if action is None or dock is None:
            return
        with QSignalBlocker(action):
            action.setChecked(not dock.isHidden())
        if not dock.isHidden():
            self._refresh_plots()
            if self._use_final_isobar_resolution and plot_id in {"horizontal_isobar", "vertical_isobar"}:
                self._final_isobar_plots_rendered = True
        self._set_contour_button_states()

    def _set_export_plot_actions_enabled(self, enabled: bool) -> None:
        for action in self.export_plot_actions.values():
            action.setEnabled(enabled)

    def _visible_isobar_plots(self) -> tuple[IsobarCanvas, ...]:
        plots: list[IsobarCanvas] = []
        horizontal_dock = self.plot_docks.get("horizontal_isobar")
        vertical_dock = self.plot_docks.get("vertical_isobar")
        if horizontal_dock is not None and not horizontal_dock.isHidden():
            plots.append(self.horizontal_plot)
        if vertical_dock is not None and not vertical_dock.isHidden():
            plots.append(self.vertical_plot)
        return tuple(plots)

    def _set_contour_button_states(self) -> None:
        capture_base_enabled = bool(
            self.live_dataset is not None and self._use_final_isobar_resolution and self._final_isobar_plots_rendered
        )
        for plot_id, plot in (
            ("horizontal_isobar", self.horizontal_plot),
            ("vertical_isobar", self.vertical_plot),
        ):
            dock = self.plot_docks.get(plot_id)
            visible = dock is not None and not dock.isHidden()
            capture_action = self.capture_contour_actions.get(plot_id)
            clear_action = self.clear_contour_actions.get(plot_id)
            if capture_action is not None:
                capture_action.setEnabled(capture_base_enabled and visible)
            if clear_action is not None:
                clear_action.setEnabled(plot.has_captured_contours)

    @Slot(str)
    def capture_isobar_contours(self, plot_id: str) -> None:
        plot = self._isobar_plot_for_id(plot_id)
        if plot is None:
            return
        if plot.capture_contours():
            entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
            self.status_label.setText(f"Captured contours for {entry.title if entry is not None else 'isobar plot'}")
        self._set_contour_button_states()

    @Slot(str)
    def clear_isobar_contours(self, plot_id: str) -> None:
        plot = self._isobar_plot_for_id(plot_id)
        if plot is None:
            return
        plot.clear_contours()
        entry = next((item for item in self.plot_entries if item.plot_id == plot_id), None)
        self.status_label.setText(f"Cleared contours for {entry.title if entry is not None else 'isobar plot'}")
        self._set_contour_button_states()

    def _isobar_plot_for_id(self, plot_id: str) -> IsobarCanvas | None:
        if plot_id == "horizontal_isobar":
            return self.horizontal_plot
        if plot_id == "vertical_isobar":
            return self.vertical_plot
        return None

    def _prepared_live_plot_dataset(
        self,
        *,
        angle_samples: int | None = None,
        freq_samples: int | None = None,
    ) -> dict[str, np.ndarray] | None:
        if self.live_dataset is None:
            return None
        if angle_samples is None:
            angle_samples = live_plot_angle_samples(self.preferences.live_plot_quality)
        if freq_samples is None:
            freq_samples = live_plot_freq_samples(self.preferences.live_plot_quality)
        self.live_dataset.set_channel_synthesis(
            self._channel_configs(),
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
        )
        return self.live_dataset.as_visualization_dataset(
            PrepConfig(
                angle_samples=angle_samples,
                freq_samples=freq_samples,
                octave_smoothing=self.preferences.polar_smoothing,
                hor_ref_angle=self.preferences.horizontal_normalization_angle,
                vert_ref_angle=self.preferences.vertical_normalization_angle,
                spin_hor_ref_angle=self.preferences.spin_horizontal_reference_angle,
                spin_vert_ref_angle=self.preferences.spin_vertical_reference_angle,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
                normalize_polar=True,
                auto_db_span=False,
            )
        )

    def _refresh_plots(self) -> None:
        visible_entries = [
            entry
            for entry in self.plot_entries
            if (dock := self.plot_docks.get(entry.plot_id)) is not None and not dock.isHidden()
        ]
        if not visible_entries:
            return

        dataset = self._prepared_live_plot_dataset(
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_angle_samples(self.preferences.live_plot_quality),
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_freq_samples(self.preferences.live_plot_quality),
        )
        if dataset is None:
            return

        for entry in visible_entries:
            entry.update(dataset)

    def _update_horizontal_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.horizontal_plot.update_plot(
            dataset["isobar_freq_hz"],
            dataset["isobar_angle_deg"],
            dataset["horizontal_isobar_db"],
            float(dataset["clip_min_db"]),
            float(dataset["clip_max_db"]),
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
        )

    def _update_vertical_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.vertical_plot.update_plot(
            dataset["isobar_freq_hz"],
            dataset["isobar_angle_deg"],
            dataset["vertical_isobar_db"],
            float(dataset["clip_min_db"]),
            float(dataset["clip_max_db"]),
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
        )

    def _update_impedance_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.impedance_plot.update_plot(
            dataset["impedance_freq_hz"],
            dataset["impedance_radiator_names"],
            dataset["impedance_real"],
            dataset["impedance_imag"],
        )

    def _update_on_axis_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.on_axis_plot.update_plot(
            dataset["freq_hz"],
            dataset["polar_angle_deg"],
            dataset["horizontal_spl_db"],
            dataset.get("channel_on_axis_names"),
            dataset.get("channel_on_axis_spl_db"),
        )

    def _update_spinorama_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.spinorama_plot.update_plot(
            dataset["freq_hz"],
            dataset["polar_angle_deg"],
            dataset["horizontal_spl_db"],
            dataset["vertical_spl_db"],
            horizontal_reference_angle_deg=float(dataset["spin_horizontal_reference_angle_deg"]),
            vertical_reference_angle_deg=float(dataset["spin_vertical_reference_angle_deg"]),
        )
