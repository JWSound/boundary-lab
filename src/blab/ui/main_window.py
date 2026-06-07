"""Main Qt window and user workflow orchestration."""

from __future__ import annotations

import json
import hashlib
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import meshio
import numpy as np
from PySide6.QtCore import QEvent, QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from blab import __version__
from blab.ath import (
    AthRunResult,
    clean_ath_mesh_output,
    clean_ath_reduced_mesh_output,
    detect_ath_radiators,
    find_physical_tag_by_name,
    read_surface_physical_names,
    run_ath,
    write_ath_gmsh_path,
    write_ath_output_root,
)
from blab.config import ChannelConfig, CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
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
from blab.solvers.registry import backend_info, normalize_backend_id
from blab.symmetry import SymmetryValidationError, effective_symmetry_for_backend, validate_reduced_mesh_configs
from blab.ui.diagnostics import DiagnosticsDialog
from blab.ui.dialogs import (
    ChannelConfigDialog,
    DonateDialog,
    MeshConfigDialog,
    MeshDialogEntry,
    PreferencesDialog,
    SourceConfigDialog,
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
    script_from_payload,
    script_to_payload,
    scripts_from_payload,
    unique_script_name,
)
from blab.ui.settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    GuiPreferences,
    balloon_angle_precision_from_points,
    balloon_sampling_points,
    live_plot_angle_samples,
    live_plot_freq_samples,
    normalize_balloon_angle_precision_deg,
    normalize_live_plot_quality,
    settings_bool,
    settings_float,
    settings_int,
    settings_optional_int,
    settings_str,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
)
from blab.ui.solve_worker import SolveWorker


ATH_MESH_NAME = "ath"
STITCHED_MESH_NAME = "stitched"
DEFAULT_MESH_SCALE_FACTOR = 0.001
STITCH_FAILURE_MESSAGE = (
    "Error - unable to stitch separate mesh entities. "
    "Refer to help documentation for more info on multi-mesh workflows."
)


def _format_frequency_solve_timings(result: FrequencyResult) -> str:
    timings = result.timings
    return (
        f"Assembly {timings.assembly_s:.2f}s | "
        f"Solve {timings.solve_s:.2f}s | "
        f"Field {timings.field_s:.2f}s"
    )


RECENT_PROJECTS_SETTINGS_KEY = "projects/recent"
MAX_RECENT_PROJECTS = 10
APP_ROOT = Path(__file__).resolve().parents[3]
ATH_BUNDLE_DIR = APP_ROOT / "ath"
ATH_OUTPUT_ROOT = APP_ROOT / "runs" / "ath_output"
GMSH_BUNDLE_EXE = APP_ROOT / "gmsh" / "gmsh-4.15.2-Windows64" / "gmsh.exe"
HELP_GUIDE_PDF = APP_ROOT / "docs" / "Boundary Lab Guide.pdf"
EDITOR_RAIL_WIDTH = 24
ADD_SCRIPT_TAB_LABEL = "+"


@dataclass(frozen=True)
class PlotEntry:
    plot_id: str
    title: str
    default_filename: str
    widget: QWidget
    update: Callable[[dict[str, np.ndarray]], None]


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
        self._apply_theme()
        self.ath_scripts: tuple[AthScriptState, ...] = default_scripts("")
        self.active_ath_script_id: str | None = self.ath_scripts[0].id if self.ath_scripts else None
        self.ath_results_by_script_id: dict[str, AthRunResult] = {}
        self.imported_radiators: tuple[RadiatorConfig, ...] = ()
        self.live_dataset: LiveSolveDataset | None = None
        self.balloon_window: QDialog | None = None
        self.channel_config_dialog: ChannelConfigDialog | None = None
        self.project_path: Path | None = None
        self.solve_thread: QThread | None = None
        self.solve_worker: SolveWorker | None = None
        self.solve_expected_count = 0
        self.solve_failed = False
        self.solve_started_at: float | None = None
        self.solve_cancel_requested = False
        self._use_final_isobar_resolution = False
        self._editor_collapsed = settings_bool(self.settings, "window/ath_editor_collapsed", False)
        self._last_editor_width = settings_int(self.settings, "window/ath_editor_width", 420)
        self._last_imported_mesh_focus_check_at = 0.0
        startup("Preparing Ath runtime config...")
        self._ensure_ath_runtime_config()

        startup("Building script editor...")
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.currentChanged.connect(self._on_active_ath_tab_changed)
        self.editor_tabs.tabCloseRequested.connect(self._remove_ath_script_at)
        self.editor_tabs.tabBar().installEventFilter(self)
        self.collapse_editor_button = QToolButton()
        self.collapse_editor_button.setToolTip("Collapse Ath script editor")
        self.collapse_editor_button.setAutoRaise(True)
        self.collapse_editor_button.setFixedWidth(EDITOR_RAIL_WIDTH)
        self.collapse_editor_button.clicked.connect(self.toggle_editor_panel)
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
        freq_max = min(max(settings_int(self.settings, "solve/freq_max_hz", 20000), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ)
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

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt override
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

        export_plot_menu = file_menu.addMenu("Export Plot")
        for entry in self.plot_entries:
            action = QAction(entry.title, self)
            action.setEnabled(False)
            action.triggered.connect(lambda _checked=False, plot_id=entry.plot_id: self.export_plot(plot_id))
            export_plot_menu.addAction(action)
            self.export_plot_actions[entry.plot_id] = action

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
        for entry in self.plot_entries:
            action = QAction(entry.title, self)
            action.setCheckable(True)
            action.setChecked(settings_bool(self.settings, f"plots/{entry.plot_id}/visible", True))
            action.toggled.connect(lambda visible, plot_id=entry.plot_id: self._set_plot_visible(plot_id, visible))
            view_menu.addAction(action)
            self.plot_view_actions[entry.plot_id] = action
            entry.widget.setVisible(action.isChecked())

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
        editor_container_layout.addWidget(self.collapse_editor_button)

        plot_content = QWidget()
        plot_layout = QVBoxLayout(plot_content)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)
        for entry in self.plot_entries:
            plot_layout.addWidget(entry.widget)

        plot_panel = QScrollArea()
        plot_panel.setWidgetResizable(True)
        plot_panel.setFrameShape(QFrame.NoFrame)
        plot_panel.setWidget(plot_content)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setOpaqueResize(False)
        self.main_splitter.addWidget(self.editor_container)
        self.main_splitter.addWidget(self.preview)
        self.main_splitter.addWidget(plot_panel)
        self.main_splitter.setStretchFactor(0, 1)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 1)

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
        layout.addWidget(self.main_splitter, stretch=1)
        layout.addWidget(controls)
        layout.addWidget(self.status_label)
        self.setCentralWidget(central)

    def _wire_controls(self) -> None:
        self.generate_button.clicked.connect(self.generate_geometry)
        self.solve_button.clicked.connect(self.start_solve)
        self.cancel_button.clicked.connect(self.cancel_solve)
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

    @Slot(str)
    def _on_visualization_settings_changed(self, _reason: str) -> None:
        self._refresh_plots()

    def _rebuild_ath_script_tabs(self) -> None:
        self.editor_tabs.blockSignals(True)
        self.editor_tabs.clear()
        for script in self.ath_scripts:
            editor = QPlainTextEdit()
            editor.setFont(QFont("Consolas", 10))
            editor.setPlainText(script.config_text)
            editor.textChanged.connect(lambda script_id=script.id, editor=editor: self._update_script_text(script_id, editor))
            self.editor_tabs.addTab(editor, script.name)
        add_tab = QWidget()
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
        self.active_ath_script_id = self.ath_scripts[min(index, len(self.ath_scripts) - 1)].id if self.ath_scripts else None
        self._rebuild_ath_script_tabs()
        self.mesh_state_changed.emit("ath_script_removed")
        self.solve_results_invalidated.emit("ath_script_removed")

    @Slot()
    def toggle_editor_panel(self) -> None:
        self._set_editor_collapsed(not self._editor_collapsed)

    def _set_editor_collapsed(self, collapsed: bool) -> None:
        if hasattr(self, "main_splitter") and not collapsed and self._editor_collapsed:
            sizes = self.main_splitter.sizes()
            plot_width = sizes[2] if len(sizes) >= 3 else 400
            current_editor_width = sizes[0] if sizes else EDITOR_RAIL_WIDTH
            preview_width = sizes[1] if len(sizes) >= 2 else max(self.preview.width(), 400)
            editor_delta = max(self._last_editor_width - current_editor_width, 0)
            self.editor_container.setMinimumWidth(0)
            self.editor_container.setMaximumWidth(16777215)
            self.main_splitter.setSizes(
                [
                    self._last_editor_width,
                    max(preview_width - editor_delta, 1),
                    plot_width,
                ]
            )
        elif hasattr(self, "main_splitter") and collapsed:
            sizes = self.main_splitter.sizes()
            plot_width = sizes[2] if len(sizes) >= 3 else 400
            preview_width = sizes[1] if len(sizes) >= 2 else max(self.preview.width(), 400)
            if sizes and sizes[0] > EDITOR_RAIL_WIDTH:
                self._last_editor_width = sizes[0]
            editor_delta = max(self._last_editor_width - EDITOR_RAIL_WIDTH, 0)
            self.editor_container.setMinimumWidth(EDITOR_RAIL_WIDTH)
            self.editor_container.setMaximumWidth(EDITOR_RAIL_WIDTH)
            self.main_splitter.setSizes(
                [
                    EDITOR_RAIL_WIDTH,
                    preview_width + editor_delta,
                    plot_width,
                ]
            )

        self._editor_collapsed = collapsed
        self.editor_panel.setVisible(not collapsed)
        self.collapse_editor_button.setText(">" if collapsed else "<")
        self.collapse_editor_button.setToolTip(
            "Expand Ath script editor" if collapsed else "Collapse Ath script editor"
        )
        self.settings.setValue("window/ath_editor_collapsed", collapsed)
        self.settings.setValue("window/ath_editor_width", self._last_editor_width)

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
        defaults = GuiPreferences()
        return GuiPreferences(
            theme=self._normalized_theme(settings_str(self.settings, "preferences/theme", defaults.theme)),
            solve_backend=normalize_backend_id(
                settings_str(self.settings, "preferences/solve_backend", defaults.solve_backend)
            ),
            solve_server_url=settings_str(self.settings, "preferences/solve_server_url", defaults.solve_server_url),
            live_plot_quality=normalize_live_plot_quality(
                settings_str(self.settings, "preferences/live_plot_quality", defaults.live_plot_quality)
            ),
            gmres_tolerance=settings_float(self.settings, "preferences/gmres_tolerance", defaults.gmres_tolerance),
            polar_angle_step_deg=settings_float(
                self.settings,
                "preferences/polar_angle_step_deg",
                defaults.polar_angle_step_deg,
            ),
            polar_observation_distance_m=settings_float(
                self.settings,
                "preferences/polar_observation_distance_m",
                defaults.polar_observation_distance_m,
            ),
            use_burton_miller=settings_bool(
                self.settings,
                "preferences/use_burton_miller",
                defaults.use_burton_miller,
            ),
            worker_count=settings_int(self.settings, "preferences/worker_count", defaults.worker_count),
            polar_smoothing=settings_optional_int(
                self.settings,
                "preferences/polar_smoothing",
                defaults.polar_smoothing,
            ),
            horizontal_normalization_angle=settings_float(
                self.settings,
                "preferences/horizontal_normalization_angle",
                defaults.horizontal_normalization_angle,
            ),
            vertical_normalization_angle=settings_float(
                self.settings,
                "preferences/vertical_normalization_angle",
                defaults.vertical_normalization_angle,
            ),
            spin_horizontal_reference_angle=settings_float(
                self.settings,
                "preferences/spin_horizontal_reference_angle",
                defaults.spin_horizontal_reference_angle,
            ),
            spin_vertical_reference_angle=settings_float(
                self.settings,
                "preferences/spin_vertical_reference_angle",
                defaults.spin_vertical_reference_angle,
            ),
            spl_max_db=settings_float(self.settings, "preferences/spl_max_db", defaults.spl_max_db),
            spl_min_db=settings_float(self.settings, "preferences/spl_min_db", defaults.spl_min_db),
            stitch_tolerance_mm=settings_float(
                self.settings,
                "preferences/stitch_tolerance_mm",
                defaults.stitch_tolerance_mm,
            ),
            spherical_sampling_enabled=settings_bool(
                self.settings,
                "preferences/spherical_sampling_enabled",
                defaults.spherical_sampling_enabled,
            ),
            balloon_angle_precision_deg=self._load_balloon_angle_precision_deg(defaults),
        )

    def _save_preferences(self) -> None:
        self.settings.setValue("preferences/theme", self.preferences.theme)
        self.settings.setValue("preferences/solve_backend", self.preferences.solve_backend)
        self.settings.setValue("preferences/solve_server_url", self.preferences.solve_server_url)
        self.settings.setValue("preferences/live_plot_quality", self.preferences.live_plot_quality)
        self.settings.setValue("preferences/gmres_tolerance", self.preferences.gmres_tolerance)
        self.settings.setValue("preferences/polar_angle_step_deg", self.preferences.polar_angle_step_deg)
        self.settings.setValue(
            "preferences/polar_observation_distance_m",
            self.preferences.polar_observation_distance_m,
        )
        self.settings.setValue("preferences/use_burton_miller", self.preferences.use_burton_miller)
        self.settings.setValue("preferences/worker_count", self.preferences.worker_count)
        self.settings.setValue("preferences/polar_smoothing", self.preferences.polar_smoothing)
        self.settings.setValue(
            "preferences/horizontal_normalization_angle",
            self.preferences.horizontal_normalization_angle,
        )
        self.settings.setValue(
            "preferences/vertical_normalization_angle",
            self.preferences.vertical_normalization_angle,
        )
        self.settings.setValue(
            "preferences/spin_horizontal_reference_angle",
            self.preferences.spin_horizontal_reference_angle,
        )
        self.settings.setValue(
            "preferences/spin_vertical_reference_angle",
            self.preferences.spin_vertical_reference_angle,
        )
        self.settings.setValue("preferences/spl_max_db", self.preferences.spl_max_db)
        self.settings.setValue("preferences/spl_min_db", self.preferences.spl_min_db)
        self.settings.setValue("preferences/stitch_tolerance_mm", self.preferences.stitch_tolerance_mm)
        self.settings.setValue("preferences/spherical_sampling_enabled", self.preferences.spherical_sampling_enabled)
        self.settings.setValue(
            "preferences/balloon_angle_precision_deg",
            self.preferences.balloon_angle_precision_deg,
        )

    def _load_balloon_angle_precision_deg(self, defaults: GuiPreferences) -> float:
        if self.settings.contains("preferences/balloon_angle_precision_deg"):
            return normalize_balloon_angle_precision_deg(
                settings_float(
                    self.settings,
                    "preferences/balloon_angle_precision_deg",
                    defaults.balloon_angle_precision_deg,
                )
            )
        if self.settings.contains("preferences/spherical_sampling_points"):
            return balloon_angle_precision_from_points(
                settings_int(
                    self.settings,
                    "preferences/spherical_sampling_points",
                    balloon_sampling_points(defaults.balloon_angle_precision_deg),
                )
            )
        return defaults.balloon_angle_precision_deg

    @staticmethod
    def _normalized_theme(theme: str) -> str:
        normalized = str(theme).strip().lower()
        return normalized if normalized in {"system", "light", "dark"} else "system"

    def _apply_theme(self) -> None:
        app = QApplication.instance()
        if app is None:
            return

        theme = self._normalized_theme(self.preferences.theme)
        app.setStyleSheet("")
        dark_text = QColor(30, 30, 30)
        light_text = QColor(245, 245, 245)
        if theme == "system":
            palette = app.style().standardPalette()
            window_color = palette.color(QPalette.Window)
            base_color = palette.color(QPalette.Base)
            text_color = dark_text if window_color.lightness() >= 128 else light_text
            self._set_palette_text_colors(palette, text_color)
            app.setPalette(palette)
            app.setStyleSheet(self._theme_stylesheet(text_color, window_color, base_color))
        elif theme == "dark":
            palette = app.style().standardPalette()
            window_color = QColor(45, 45, 48)
            base_color = QColor(30, 30, 30)
            palette.setColor(QPalette.Window, QColor(45, 45, 48))
            palette.setColor(QPalette.Base, base_color)
            palette.setColor(QPalette.AlternateBase, QColor(45, 45, 48))
            palette.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
            palette.setColor(QPalette.Button, QColor(45, 45, 48))
            palette.setColor(QPalette.BrightText, QColor(255, 80, 80))
            palette.setColor(QPalette.Highlight, QColor(61, 126, 154))
            palette.setColor(QPalette.HighlightedText, light_text)
            self._set_palette_text_colors(palette, light_text)
            app.setPalette(palette)
            app.setStyleSheet(self._theme_stylesheet(light_text, window_color, base_color))
        else:
            palette = app.style().standardPalette()
            window_color = QColor(245, 245, 245)
            base_color = QColor(255, 255, 255)
            palette.setColor(QPalette.Window, window_color)
            palette.setColor(QPalette.Base, Qt.white)
            palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
            palette.setColor(QPalette.ToolTipBase, Qt.white)
            palette.setColor(QPalette.Button, QColor(245, 245, 245))
            palette.setColor(QPalette.BrightText, Qt.red)
            palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
            palette.setColor(QPalette.HighlightedText, Qt.white)
            self._set_palette_text_colors(palette, dark_text)
            app.setPalette(palette)
            app.setStyleSheet(self._theme_stylesheet(dark_text, window_color, base_color))

        self._refresh_theme_widgets(app)

    @staticmethod
    def _set_palette_text_colors(palette: QPalette, color: QColor) -> None:
        roles = (
            QPalette.WindowText,
            QPalette.Text,
            QPalette.ButtonText,
            QPalette.ToolTipText,
        )
        if hasattr(QPalette, "PlaceholderText"):
            roles = (*roles, QPalette.PlaceholderText)

        disabled_color = QColor(color)
        disabled_color.setAlpha(140)
        for group, group_color in (
            (QPalette.Active, color),
            (QPalette.Inactive, color),
            (QPalette.Disabled, disabled_color),
        ):
            for role in roles:
                palette.setColor(group, role, group_color)

    def _refresh_theme_widgets(self, app: QApplication) -> None:
        style = app.style()
        for widget in app.allWidgets():
            style.unpolish(widget)
            style.polish(widget)
            widget.update()
        app.processEvents()

    @staticmethod
    def _theme_stylesheet(text_color: QColor, window_color: QColor, base_color: QColor) -> str:
        text = text_color.name()
        window = window_color.name()
        base = base_color.name()
        border = QColor(85, 85, 85).name() if text_color.lightness() > 128 else QColor(190, 190, 190).name()
        selected = QColor(61, 126, 154).name() if text_color.lightness() > 128 else QColor(0, 120, 215).name()
        selected_text = QColor(255, 255, 255).name()
        hover = QColor(65, 65, 68).name() if text_color.lightness() > 128 else QColor(225, 225, 225).name()
        disabled = QColor(text_color)
        disabled.setAlpha(150)
        disabled_css = f"rgba({disabled.red()}, {disabled.green()}, {disabled.blue()}, {disabled.alpha()})"

        return f"""
            QWidget {{
                color: {text};
            }}
            QMenuBar, QMenuBar::item, QMenu {{
                background-color: {window};
                color: {text};
            }}
            QMenuBar::item:selected, QMenu::item:selected {{
                background-color: {hover};
                color: {text};
            }}
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
            QTableWidget, QTableView, QListView, QTreeView {{
                background-color: {base};
                color: {text};
                border: 1px solid {border};
                selection-background-color: {selected};
                selection-color: {selected_text};
            }}
            QHeaderView::section {{
                background-color: {window};
                color: {text};
                border: 1px solid {border};
            }}
            QWidget:disabled {{
                color: {disabled_css};
            }}
            QToolTip {{
                background-color: {base};
                color: {text};
                border: 1px solid {border};
            }}
        """

    @Slot()
    def _save_frequency_settings(self) -> None:
        self.settings.setValue("solve/freq_min_hz", int(self.freq_min_spin.value()))
        self.settings.setValue("solve/freq_max_hz", int(self.freq_max_spin.value()))
        self.settings.setValue("solve/freq_count", int(self.freq_count_spin.value()))

    def _restore_window_state(self) -> None:
        geometry = self.settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        splitter_state = self.settings.value("window/main_splitter")
        if splitter_state is not None:
            self.main_splitter.restoreState(splitter_state)
        self._set_editor_collapsed(self._editor_collapsed)

    def _save_window_state(self) -> None:
        sizes = self.main_splitter.sizes()
        if sizes and not self._editor_collapsed:
            self._last_editor_width = sizes[0]
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/main_splitter", self.main_splitter.saveState())
        self.settings.setValue("window/ath_editor_collapsed", self._editor_collapsed)
        self.settings.setValue("window/ath_editor_width", self._last_editor_width)
        self.settings.sync()

    def _load_recent_project_paths(self) -> list[Path]:
        raw_paths = self.settings.value(RECENT_PROJECTS_SETTINGS_KEY, "[]")
        try:
            values = json.loads(str(raw_paths))
        except json.JSONDecodeError:
            values = []
        if not isinstance(values, list):
            return []

        paths: list[Path] = []
        seen = set()
        for value in values:
            path_text = str(value).strip()
            if not path_text:
                continue
            path = Path(path_text)
            key = str(path).casefold()
            if key in seen:
                continue
            seen.add(key)
            paths.append(path)
            if len(paths) >= MAX_RECENT_PROJECTS:
                break
        return paths

    def _save_recent_project_paths(self, paths: list[Path]) -> None:
        self.settings.setValue(
            RECENT_PROJECTS_SETTINGS_KEY,
            json.dumps([str(path) for path in paths[:MAX_RECENT_PROJECTS]]),
        )
        self.settings.sync()
        if hasattr(self, "open_recent_menu"):
            self._rebuild_open_recent_menu()

    def _remember_recent_project(self, path: Path) -> None:
        try:
            normalized = path.resolve()
        except OSError:
            normalized = path

        recent = [
            existing
            for existing in self._load_recent_project_paths()
            if str(existing).casefold() != str(normalized).casefold()
        ]
        self._save_recent_project_paths([normalized, *recent])

    def _remove_recent_project(self, path: Path) -> None:
        self._save_recent_project_paths(
            [
                existing
                for existing in self._load_recent_project_paths()
                if str(existing).casefold() != str(path).casefold()
            ]
        )

    def _clear_recent_projects(self) -> None:
        self._save_recent_project_paths([])

    def _rebuild_open_recent_menu(self) -> None:
        self.open_recent_menu.clear()
        recent_paths = self._load_recent_project_paths()
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
            else str(Path(mesh.cleaned_file).resolve()) if absolute_paths else mesh.cleaned_file
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
                replace(radiator, mesh=script.mesh_name)
                for radiator in radiators
                if radiator.mesh == script.mesh_name
            ]
            self.ath_results_by_script_id[script.id] = replace(result, radiators=tuple(updated))
        self.imported_radiators = tuple(
            radiator for radiator in radiators if radiator.mesh not in generated_mesh_names
        )

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
            if (
                not cleaned_path.exists()
                or source_path.stat().st_mtime > cleaned_path.stat().st_mtime
            ):
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

    def _selected_backend_supports_symmetry(self) -> bool:
        return backend_info(self.preferences.solve_backend).capabilities.supports_symmetry

    def _disable_symmetry_if_backend_unsupported(self) -> bool:
        effective_symmetry = effective_symmetry_for_backend(self.symmetry, self.preferences.solve_backend)
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
                tag: surface_name
                for surface_name, tag in read_surface_physical_names(Path(mesh_cfg.file)).items()
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
                "Afterburner produced non-finite results.\n\n"
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
                mesh_cfg.name: read_surface_physical_names(Path(mesh_cfg.file))
                for mesh_cfg in mesh_configs
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
                mesh_cfg.name: read_surface_physical_names(Path(mesh_cfg.file))
                for mesh_cfg in mesh_configs
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
        raw_config = self.settings.value("source/config_by_name", "{}")
        try:
            loaded = json.loads(str(raw_config))
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _save_source_config(self, surface_tags: dict[str, tuple[str, int]], radiators: tuple[RadiatorConfig, ...]) -> None:
        radiators_by_name = {radiator.name: radiator for radiator in radiators}
        config_by_name = self._load_source_config_by_name()
        for surface_name in surface_tags:
            radiator = radiators_by_name.get(surface_name)
            config_by_name[surface_name] = {
                "driven": radiator is not None,
                "channel": "main" if radiator is None else radiator.channel,
                "velocity_offset_db": 0.0 if radiator is None else float(radiator.velocity_offset_db),
            }
        self.settings.setValue("source/config_by_name", json.dumps(config_by_name, sort_keys=True))
        self.settings.sync()

    def _load_channel_config_by_name(self) -> dict[str, dict]:
        raw_config = self.settings.value("channel/config_by_name", "{}")
        try:
            loaded = json.loads(str(raw_config))
        except json.JSONDecodeError:
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _save_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        payload = {
            channel.name: {
                "level_db": float(channel.level_db),
                "polarity": int(channel.polarity),
                "delay_ms": float(channel.delay_ms),
                "hpf": self._crossover_settings(channel.hpf),
                "lpf": self._crossover_settings(channel.lpf),
            }
            for channel in channels
        }
        self.settings.setValue("channel/config_by_name", json.dumps(payload, sort_keys=True))
        self.settings.sync()

    def _channel_configs(self) -> tuple[ChannelConfig, ...]:
        raw = self._load_channel_config_by_name()
        if not raw:
            return (ChannelConfig(name="main"),)
        channels = []
        for name, payload in sorted(raw.items()):
            payload = payload if isinstance(payload, dict) else {}
            channels.append(
                ChannelConfig(
                    name=str(name),
                    level_db=float(payload.get("level_db", 0.0)),
                    polarity=int(payload.get("polarity", 1)),
                    delay_ms=float(payload.get("delay_ms", 0.0)),
                    hpf=self._saved_crossover(payload.get("hpf"), crossover_type="highpass"),
                    lpf=self._saved_crossover(payload.get("lpf"), crossover_type="lowpass"),
                )
            )
        return tuple(channels) or (ChannelConfig(name="main"),)

    def _channels_for_solver_radiators(
        self,
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[ChannelConfig, ...]:
        channels = list(self._channel_configs())
        configured_names = {channel.name for channel in channels}
        for radiator in radiators:
            if radiator.channel in configured_names:
                continue
            channels.append(ChannelConfig(name=radiator.channel))
            configured_names.add(radiator.channel)
        return tuple(channels)

    def _channel_configs_for_current_radiators(self) -> tuple[ChannelConfig, ...]:
        return self._channels_for_solver_radiators(self._all_radiators())

    def _discard_channel_config_dialog(self) -> None:
        dialog = self.channel_config_dialog
        self.channel_config_dialog = None
        if dialog is not None:
            dialog.close()

    def _crossover_settings(self, crossover: CrossoverConfig | None) -> dict:
        if crossover is None or crossover.type.lower() == "none":
            return {}
        return {
            "type": crossover.type,
            "filter": crossover.filter,
            "order": int(crossover.order),
            "frequency_hz": None if crossover.frequency_hz is None else float(crossover.frequency_hz),
        }

    def _saved_crossover(self, raw: object, *, crossover_type: str) -> CrossoverConfig:
        if not isinstance(raw, dict) or raw.get("frequency_hz") is None:
            return CrossoverConfig()
        return CrossoverConfig(
            type=crossover_type,
            filter=str(raw.get("filter", "butterworth")).lower(),
            order=int(raw.get("order", 1)),
            frequency_hz=float(raw["frequency_hz"]),
        )

    def _apply_saved_source_config_to_result(self, result: AthRunResult | None, mesh_name: str) -> AthRunResult | None:
        if result is None:
            return None
        try:
            surface_tags = {
                f"{mesh_name}:{surface_name}": (mesh_name, tag)
                for surface_name, tag in read_surface_physical_names(result.solver_msh_path).items()
            }
        except Exception:
            return result

        config_by_name = self._load_source_config_by_name()
        existing_by_tag = {radiator.tag: radiator for radiator in result.radiators}
        radiators = []
        for surface_name, (mesh_name, tag) in sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
            saved = config_by_name.get(surface_name)
            if isinstance(saved, dict):
                if not bool(saved.get("driven", False)):
                    continue
                radiators.append(
                    RadiatorConfig(
                        name=surface_name,
                        mesh=mesh_name,
                        tag=tag,
                        channel=str(saved.get("channel", "main")),
                        velocity_offset_db=float(saved.get("velocity_offset_db", 0.0)),
                    )
                )
                continue

            existing = existing_by_tag.get(tag)
            if existing is not None:
                radiators.append(
                    replace(
                        existing,
                        name=surface_name,
                        mesh=mesh_name,
                        tag=tag,
                        channel=existing.channel or "main",
                        velocity_offset_db=float(existing.velocity_offset_db),
                    )
                )

        return replace(result, radiators=tuple(radiators))

    def _apply_saved_source_config(self, result: AthRunResult | None) -> AthRunResult | None:
        return self._apply_saved_source_config_to_result(result, ATH_MESH_NAME)

    def _apply_saved_imported_source_config(self, surface_tags: dict[str, tuple[str, int]]) -> None:
        generated_mesh_names = {script.mesh_name for script in self.ath_scripts}
        config_by_name = self._load_source_config_by_name()
        existing_by_key = {(radiator.mesh, radiator.tag): radiator for radiator in self.imported_radiators}
        radiators = []
        for surface_name, (mesh_name, tag) in sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
            if mesh_name in generated_mesh_names:
                continue
            saved = config_by_name.get(surface_name)
            existing = existing_by_key.get((mesh_name, tag))
            if isinstance(saved, dict):
                if not bool(saved.get("driven", False)):
                    continue
                radiators.append(
                    RadiatorConfig(
                        name=surface_name,
                        mesh=mesh_name,
                        tag=tag,
                        channel=str(saved.get("channel", "main")),
                        velocity_offset_db=float(saved.get("velocity_offset_db", 0.0)),
                    )
                )
            elif existing is not None:
                radiators.append(existing)
        self.imported_radiators = tuple(radiators)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
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

        path = Path(path_text)
        try:
            script = self._active_script()
            if script is None:
                script = new_script(unique_script_name(path.stem, self.ath_scripts), path.read_text(encoding="utf-8"))
                self.ath_scripts = (*self.ath_scripts, script)
                self.active_ath_script_id = script.id
            else:
                self.ath_scripts = replace_script(self.ath_scripts, script.id, config_text=path.read_text(encoding="utf-8"))
            self._rebuild_ath_script_tabs()
            self.status_label.setText(f"Imported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    @Slot()
    def new_project(self) -> None:
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
        self.settings.setValue("source/config_by_name", "{}")
        self.settings.setValue("channel/config_by_name", "{}")
        self.settings.sync()
        self.project_state_changed.emit("new_project")
        self.solve_results_invalidated.emit("new_project")
        self.status_label.setText("New project")

    @Slot()
    def save_project(self) -> None:
        if self.project_path is None:
            self.save_project_as()
            return
        self._save_project_to_path(self.project_path)

    @Slot()
    def save_project_as(self) -> None:
        default_path = self.project_path or (Path.cwd() / PROJECT_DEFAULT_NAME)
        path_text, _ = QFileDialog.getSaveFileName(
            self,
            "Save Project",
            str(default_path),
            PROJECT_FILE_FILTER,
        )
        if not path_text:
            return
        self._save_project_to_path(normalize_project_path(path_text))

    def _save_project_to_path(self, path: Path) -> None:
        try:
            project_path = write_project_file(path, self._project_payload())
            self.project_path = project_path
            self._remember_recent_project(project_path)
            self.status_label.setText(f"Saved project {project_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))

    @Slot()
    def load_project(self) -> None:
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
        self._load_project_from_path(path)

    def _load_project_from_path(self, path: Path) -> None:
        try:
            payload = read_project_file(path)
            self._apply_project_payload(payload)
            self.project_path = path
            self._remember_recent_project(path)
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

    def _apply_project_payload(self, payload: dict) -> None:
        self._discard_channel_config_dialog()
        source_config = payload.get("source_config_by_name", {})
        if not isinstance(source_config, dict):
            source_config = {}
        self.settings.setValue("source/config_by_name", json.dumps(source_config, sort_keys=True))
        channel_config = payload.get("channel_config_by_name", {})
        if not isinstance(channel_config, dict):
            channel_config = {}
        self.settings.setValue("channel/config_by_name", json.dumps(channel_config, sort_keys=True))
        self.settings.sync()

        self.ath_scripts = scripts_from_payload(
            payload.get("ath_scripts"),
            fallback_config_text=str(payload.get("ath_config_text", "")),
        )
        active_id = payload.get("active_ath_script_id")
        self.active_ath_script_id = active_id if any(script.id == active_id for script in self.ath_scripts) else (
            self.ath_scripts[0].id if self.ath_scripts else None
        )
        self.ath_results_by_script_id = {}
        for script in self.ath_scripts:
            result = self._result_from_script_state(script)
            if result is not None:
                self.ath_results_by_script_id[script.id] = self._apply_saved_source_config_to_result(result, script.mesh_name)
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
        dataset = self._prepared_live_plot_dataset()
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
        self.preferences = dialog.preferences()
        dialog.deleteLater()
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

        try:
            self.status_label.setText("Cleaning imported meshes...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._apply_mesh_config_dialog_entries(dialog.meshes())
            self.stitch_imported_meshes = dialog.stitch_imported_meshes()
            if symmetry_enabled:
                self.symmetry = dialog.symmetry()
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

    @Slot(object)
    def _apply_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        channels = tuple(channels)
        previous_radiator_assignments = tuple(
            (radiator.mesh, radiator.tag, radiator.channel)
            for radiator in self._all_radiators()
        )
        self._save_channel_config(channels)
        valid_names = {channel.name for channel in channels}
        fallback = channels[0].name
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
            (radiator.mesh, radiator.tag, radiator.channel)
            for radiator in self._all_radiators()
        )
        radiator_assignments_changed = current_radiator_assignments != previous_radiator_assignments
        self.source_config_changed.emit("channel_config_changed")
        if (
            not radiator_assignments_changed
            and self.live_dataset is not None
            and self.live_dataset.supports_channel_resynthesis
        ):
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

        dialog = SourceConfigDialog(surface_tags, self._all_radiators(), self._channel_configs_for_current_radiators(), self)
        if dialog.exec() != QDialog.Accepted:
            return

        radiators = dialog.radiators()
        self._apply_radiators_to_results(radiators)
        self._save_source_config(surface_tags, radiators)
        self.source_config_changed.emit("source_config_changed")
        self.solve_results_invalidated.emit("source_config_changed")
        self.status_label.setText(f"Source config updated: {len(radiators)} driven surfaces")

    @Slot()
    def generate_geometry(self) -> None:
        script = self._active_script()
        if script is None:
            QMessageBox.warning(self, "No Ath script", "Add an Ath script before generating.")
            return
        case_name = f"{script.mesh_name}_{script.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_root = ATH_OUTPUT_ROOT
        self.solve_results_invalidated.emit("geometry_generation_started")
        self.status_label.setText(f"Generating {script.name}...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._ensure_ath_runtime_config()
            raw_result = run_ath(
                ath_exe=self._find_ath_exe(),
                config_text=script.config_text,
                run_root=run_root,
                case_name=case_name,
            )
            self.status_label.setText("Cleaning generated mesh...")
            result = self._apply_saved_source_config_to_result(clean_ath_mesh_output(raw_result), script.mesh_name)
            self.ath_results_by_script_id[script.id] = result
            self.ath_scripts = replace_script(
                self.ath_scripts,
                script.id,
                output_dir=str(result.output_dir),
                msh_path=str(result.msh_path),
                cleaned_msh_path=None if result.cleaned_msh_path is None else str(result.cleaned_msh_path),
                config_path=str(result.config_path),
            )
            self.mesh_state_changed.emit("ath_mesh_generated")
            self.status_label.setText(f"Generated and cleaned {result.output_dir}")
            self._show_mesh_quality_warning(result)
        except Exception as exc:
            self.status_label.setText("Generate failed")
            QMessageBox.critical(self, "Ath generation failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def start_solve(self) -> None:
        if not self._has_solver_meshes():
            QMessageBox.warning(self, "No mesh", "Enable at least one generated or imported mesh before solving.")
            return
        radiators = self._all_radiators()
        if not radiators:
            QMessageBox.warning(self, "No driven surfaces", "Open Source Config and mark at least one surface as Driven.")
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
            workers=self.preferences.worker_count,
            flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
            spherical_sampling_points=balloon_sampling_points(self.preferences.balloon_angle_precision_deg),
            symmetry=self.symmetry,
        )

        self.live_dataset = None
        self.balloon_plot_action.setEnabled(False)
        self.solve_expected_count = int(ordered_freqs.size)
        self.solve_failed = False
        self._use_final_isobar_resolution = False
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.source_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.solve_started_at = time.perf_counter()
        self.solve_cancel_requested = False
        self.status_label.setText("Initializing Solver...")

        self.solve_thread = QThread(self)
        self.solve_worker = SolveWorker(
            config,
            ordered_freqs,
            worker_count=self.preferences.worker_count,
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
            f"({result.freq_hz:.1f} Hz) | {_format_frequency_solve_timings(result)}"
        )
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
            self._refresh_plots()
            self._set_export_plot_actions_enabled(True)
            self.export_polar_data_action.setEnabled(True)
            self.balloon_plot_action.setEnabled(self.live_dataset.as_balloon_raw_bundle() is not None)
            elapsed_text = "" if elapsed_s is None else f" in {elapsed_s:.1f} s"
            if self.solve_cancel_requested:
                self.status_label.setText(
                    f"Solve stopped: {self.live_dataset.solved_count} frequencies{elapsed_text}"
                )
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
            self.status_label.setText(
                f"Solve complete: {solved_count} frequencies{elapsed_text}"
            )
        elif self.solve_cancel_requested:
            self.status_label.setText("Solve stopped")
        self.solve_cancel_requested = False
        self.solve_failed = False
        self.solve_expected_count = 0
        self.solve_worker = None
        self.solve_thread = None

    def _clear_plots(self) -> None:
        self.live_dataset = None
        self._use_final_isobar_resolution = False
        for entry in self.plot_entries:
            entry.widget._draw_empty()
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.balloon_plot_action.setEnabled(False)

    def _set_plot_visible(self, plot_id: str, visible: bool) -> None:
        for entry in self.plot_entries:
            if entry.plot_id != plot_id:
                continue
            entry.widget.setVisible(visible)
            self.settings.setValue(f"plots/{plot_id}/visible", visible)
            self.settings.sync()
            if visible:
                self._refresh_plots()
            break

    def _set_export_plot_actions_enabled(self, enabled: bool) -> None:
        for action in self.export_plot_actions.values():
            action.setEnabled(enabled)

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
        visible_entries = [entry for entry in self.plot_entries if entry.widget.isVisible()]
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
