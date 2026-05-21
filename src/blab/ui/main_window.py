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
from PySide6.QtCore import QEvent, QSettings, QSignalBlocker, Qt, QThread, Slot
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from blab.ath import (
    AthRunResult,
    clean_ath_mesh_output,
    discover_ath_output,
    read_surface_physical_names,
    run_ath,
    write_ath_gmsh_path,
    write_ath_output_root,
)
from blab.config import CrossoverConfig, MeshConfig, RadiatorConfig, SimulationConfig
from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
    build_log_frequencies,
    export_polar_text_files,
    order_frequencies_for_live_plotting,
)
from blab.mesh_clean import AREA_TOL, MERGE_TOL, clean_mesh_file, stitch_meshes
from blab.plotting import VisualizerConfig
from blab.postprocess import PrepConfig
from blab.ui.balloon import BalloonPlotWindow
from blab.ui.dialogs import MeshConfigDialog, MeshDialogEntry, PreferencesDialog, SourceConfigDialog
from blab.ui.mesh_preview import MeshPreview
from blab.ui.plots import (
    AUDIO_FREQ_MAX_HZ,
    AUDIO_FREQ_MIN_HZ,
    FREQ_SLIDER_STEPS,
    LIVE_ISOBAR_ANGLE_SAMPLES,
    LIVE_ISOBAR_FREQ_SAMPLES,
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
from blab.ui.settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    GuiPreferences,
    settings_bool,
    settings_float,
    settings_int,
    settings_optional_int,
    settings_str,
)
from blab.ui.remote_solve_worker import RemoteSolveWorker
from blab.ui.solve_worker import SolveWorker


ATH_MESH_NAME = "ath"
STITCHED_MESH_NAME = "stitched"
IMPORTED_MESH_SETTINGS_KEY = "mesh/imported_meshes"
ATH_MESH_SETTINGS_KEY = "mesh/ath_mesh"
APP_ROOT = Path(__file__).resolve().parents[3]
ATH_BUNDLE_DIR = APP_ROOT / "ath"
ATH_OUTPUT_ROOT = APP_ROOT / "runs" / "ath_output"
GMSH_BUNDLE_EXE = APP_ROOT / "gmsh" / "gmsh-4.15.2-Windows64" / "gmsh.exe"


@dataclass(frozen=True)
class PlotEntry:
    plot_id: str
    title: str
    default_filename: str
    widget: QWidget
    update: Callable[[dict[str, np.ndarray]], None]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.setWindowTitle("Ath4 Live BEM Solver")
        self.resize(1500, 900)
        self.imported_meshes: tuple[MeshDialogEntry, ...] = self._load_imported_meshes()
        self.ath_mesh_enabled, self.ath_mesh_translation_mm = self._load_ath_mesh_settings()
        self.preferences = self._load_preferences()
        self.ath_result: AthRunResult | None = self._apply_saved_source_config(self._load_existing_sample_output())
        self.live_dataset: LiveSolveDataset | None = None
        self.balloon_window: BalloonPlotWindow | None = None
        self.project_path: Path | None = None
        self.solve_thread: QThread | None = None
        self.solve_worker: SolveWorker | None = None
        self.solve_started_at: float | None = None
        self._last_imported_mesh_focus_check_at = 0.0
        self._ensure_ath_runtime_config()

        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.setPlainText(self._load_initial_config_text())

        self.preview = MeshPreview()
        if self.ath_result is not None:
            self._refresh_mesh_preview()

        self.generate_button = QPushButton("Generate")
        self.solve_button = QPushButton("Solve")
        self.cancel_button = QPushButton("Stop")
        self.cancel_button.setEnabled(False)
        self.mesh_config_button = QPushButton("Mesh Config")
        self.source_config_button = QPushButton("Source Config")
        self.source_config_button.setEnabled(self.ath_result is not None)

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

        self._wire_controls()
        self._build_menu_bar()
        self._build_layout()
        self._restore_window_state()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._reload_updated_imported_meshes_on_focus()

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

        load_project_action = QAction("Load Project", self)
        load_project_action.triggered.connect(self.load_project)
        file_menu.addAction(load_project_action)

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

    def _build_layout(self) -> None:
        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.addWidget(self.editor)

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
        self.main_splitter.addWidget(editor_panel)
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
            solve_backend=settings_str(self.settings, "preferences/solve_backend", defaults.solve_backend),
            solve_server_url=settings_str(self.settings, "preferences/solve_server_url", defaults.solve_server_url),
            gmres_tolerance=settings_float(self.settings, "preferences/gmres_tolerance", defaults.gmres_tolerance),
            polar_angle_step_deg=settings_float(
                self.settings,
                "preferences/polar_angle_step_deg",
                defaults.polar_angle_step_deg,
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
            spl_max_db=settings_float(self.settings, "preferences/spl_max_db", defaults.spl_max_db),
            spl_min_db=settings_float(self.settings, "preferences/spl_min_db", defaults.spl_min_db),
            stitch_imported_meshes=settings_bool(
                self.settings,
                "preferences/stitch_imported_meshes",
                defaults.stitch_imported_meshes,
            ),
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
            spherical_sampling_points=settings_int(
                self.settings,
                "preferences/spherical_sampling_points",
                defaults.spherical_sampling_points,
            ),
        )

    def _save_preferences(self) -> None:
        self.settings.setValue("preferences/solve_backend", self.preferences.solve_backend)
        self.settings.setValue("preferences/solve_server_url", self.preferences.solve_server_url)
        self.settings.setValue("preferences/gmres_tolerance", self.preferences.gmres_tolerance)
        self.settings.setValue("preferences/polar_angle_step_deg", self.preferences.polar_angle_step_deg)
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
        self.settings.setValue("preferences/spl_max_db", self.preferences.spl_max_db)
        self.settings.setValue("preferences/spl_min_db", self.preferences.spl_min_db)
        self.settings.setValue("preferences/stitch_imported_meshes", self.preferences.stitch_imported_meshes)
        self.settings.setValue("preferences/stitch_tolerance_mm", self.preferences.stitch_tolerance_mm)
        self.settings.setValue("preferences/spherical_sampling_enabled", self.preferences.spherical_sampling_enabled)
        self.settings.setValue("preferences/spherical_sampling_points", self.preferences.spherical_sampling_points)

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

    def _save_window_state(self) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/main_splitter", self.main_splitter.saveState())
        self.settings.sync()

    def _load_imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        raw_config = self.settings.value(IMPORTED_MESH_SETTINGS_KEY, "[]")
        try:
            loaded = json.loads(str(raw_config))
        except json.JSONDecodeError:
            return ()
        if not isinstance(loaded, list):
            return ()

        meshes = []
        for item in loaded:
            if not isinstance(item, dict):
                continue
            source_file = str(item.get("source_file", ""))
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
                    translation_mm=tuple(float(int(round(float(value)))) for value in translation),
                    enabled=bool(item.get("enabled", True)),
                )
            )
        return tuple(meshes)

    def _save_imported_meshes(self) -> None:
        payload = [
            self._mesh_entry_to_payload(mesh, absolute_paths=False)
            for mesh in self.imported_meshes
        ]
        self.settings.setValue(IMPORTED_MESH_SETTINGS_KEY, json.dumps(payload, sort_keys=True))
        self.settings.sync()

    def _load_ath_mesh_settings(self) -> tuple[bool, tuple[float, float, float]]:
        raw_config = self.settings.value(ATH_MESH_SETTINGS_KEY, "{}")
        try:
            loaded = json.loads(str(raw_config))
        except json.JSONDecodeError:
            loaded = {}
        if not isinstance(loaded, dict):
            loaded = {}

        translation = loaded.get("translation_mm", [0.0, 0.0, 0.0])
        if not isinstance(translation, list) or len(translation) != 3:
            translation = [0.0, 0.0, 0.0]
        return (
            bool(loaded.get("enabled", True)),
            tuple(float(int(round(float(value)))) for value in translation),
        )

    def _save_ath_mesh_settings(self) -> None:
        self.settings.setValue(
            ATH_MESH_SETTINGS_KEY,
            json.dumps(self._ath_mesh_payload(absolute_paths=False), sort_keys=True),
        )
        self.settings.sync()

    def _ath_mesh_payload(self, *, absolute_paths: bool) -> dict:
        source_file = ""
        if self.ath_result is not None:
            source_file = str(self.ath_result.solver_msh_path)
            if absolute_paths:
                source_file = str(Path(source_file).resolve())
        return {
            "name": ATH_MESH_NAME,
            "source_file": source_file,
            "cleaned_file": None,
            "translation_mm": [int(round(value)) for value in self.ath_mesh_translation_mm],
            "enabled": bool(self.ath_mesh_enabled),
        }

    def _mesh_config_dialog_entries(self) -> tuple[MeshDialogEntry, ...]:
        entries = []
        if self.ath_result is not None:
            entries.append(
                MeshDialogEntry(
                    name=ATH_MESH_NAME,
                    source_file=str(self.ath_result.solver_msh_path),
                    translation_mm=self.ath_mesh_translation_mm,
                    enabled=self.ath_mesh_enabled,
                    locked=True,
                )
            )
        entries.extend(self.imported_meshes)
        return tuple(entries)

    def _apply_mesh_config_dialog_entries(self, meshes: tuple[MeshDialogEntry, ...]) -> None:
        imported_meshes = []
        for mesh in meshes:
            if mesh.name == ATH_MESH_NAME:
                self.ath_mesh_enabled = bool(mesh.enabled)
                self.ath_mesh_translation_mm = mesh.translation_mm
            else:
                imported_meshes.append(replace(mesh, locked=False))
        self.imported_meshes = tuple(imported_meshes)
        self._save_ath_mesh_settings()

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
            "translation_mm": [int(round(value)) for value in mesh.translation_mm],
            "enabled": bool(mesh.enabled),
        }

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
            self._save_imported_meshes()
            if self.ath_result is not None:
                self.ath_result = self._apply_saved_source_config(self.ath_result)
                self._refresh_mesh_preview()
            self._clear_plots()
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
        if self.ath_result is None:
            return self._imported_solver_mesh_configs()
        return (*self._ath_solver_mesh_configs(), *self._imported_solver_mesh_configs())

    def _should_use_stitched_mesh(self) -> bool:
        return self.preferences.stitch_imported_meshes and len(self._stitch_candidate_mesh_configs()) > 1

    def _stitched_mesh_path(self, mesh_configs: tuple[MeshConfig, ...]) -> Path:
        payload = {
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

    def _stitched_solver_mesh_config(self) -> MeshConfig | None:
        if not self._should_use_stitched_mesh():
            return None

        mesh_configs = self._stitch_candidate_mesh_configs()
        stitched_path = self._stitched_mesh_path(mesh_configs)
        if not stitched_path.exists():
            stitched_path.parent.mkdir(parents=True, exist_ok=True)
            stitched_mesh, _result = stitch_meshes(
                tuple(self._mesh_for_stitching(mesh_cfg) for mesh_cfg in mesh_configs),
                stitch_tol=float(self.preferences.stitch_tolerance_mm),
                area_tol=AREA_TOL,
            )
            meshio.write(stitched_path, stitched_mesh, file_format="gmsh22", binary=False)

        return MeshConfig(name=STITCHED_MESH_NAME, file=str(stitched_path), scale_factor=0.001)

    def _active_imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        return tuple(mesh for mesh in self.imported_meshes if mesh.enabled)

    def _ath_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        if self.ath_result is None or not self.ath_mesh_enabled:
            return ()
        return (
            MeshConfig(
                name=ATH_MESH_NAME,
                file=str(self.ath_result.solver_msh_path),
                scale_factor=0.001,
                translation_m=tuple(value / 1000.0 for value in self.ath_mesh_translation_mm),
            ),
        )

    def _imported_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        configs = []
        for mesh in self._active_imported_meshes():
            mesh_file = self._mesh_file_for_imported(mesh)
            configs.append(
                MeshConfig(
                    name=mesh.name,
                    file=mesh_file,
                    scale_factor=0.001,
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
        if self.ath_result is None:
            return self._imported_solver_mesh_configs()
        return (*self._ath_solver_mesh_configs(), *self._imported_solver_mesh_configs())

    def _surface_tags_for_meshes(self) -> dict[str, tuple[str, int]]:
        surface_tags: dict[str, tuple[str, int]] = {}
        for mesh_cfg in self._solver_mesh_configs():
            for surface_name, tag in read_surface_physical_names(Path(mesh_cfg.file)).items():
                surface_tags[f"{mesh_cfg.name}:{surface_name}"] = (mesh_cfg.name, tag)
        return surface_tags

    def _refresh_mesh_preview(self) -> None:
        if self.ath_result is None:
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
                driven_surfaces={(radiator.mesh or ATH_MESH_NAME, radiator.tag) for radiator in self.ath_result.radiators},
                surface_tags_by_mesh=surface_tags_by_mesh,
            )
        except Exception:
            if self.ath_mesh_enabled:
                self.preview.load_ath_result(self.ath_result)
            else:
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
                "level_db": 0.0 if radiator is None else float(radiator.level_db),
                "polarity": 1 if radiator is None else int(radiator.polarity),
                "delay_ms": 0.0 if radiator is None else float(radiator.delay_ms),
                "hpf": self._crossover_settings(None if radiator is None else radiator.hpf),
                "lpf": self._crossover_settings(None if radiator is None else radiator.lpf),
            }
        self.settings.setValue("source/config_by_name", json.dumps(config_by_name, sort_keys=True))
        self.settings.sync()

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

    def _apply_saved_source_config(self, result: AthRunResult | None) -> AthRunResult | None:
        if result is None:
            return None
        original_result = self.ath_result if hasattr(self, "ath_result") else None
        try:
            self.ath_result = result
            surface_tags = self._surface_tags_for_meshes()
        except Exception:
            return result
        finally:
            self.ath_result = original_result

        config_by_name = self._load_source_config_by_name()
        existing_by_tag = {radiator.tag: radiator for radiator in result.radiators}
        radiators = []
        for surface_name, (mesh_name, tag) in sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0])):
            legacy_surface_name = surface_name.split(":", maxsplit=1)[1]
            saved = config_by_name.get(surface_name)
            if saved is None:
                saved = config_by_name.get(legacy_surface_name)
            if saved is None:
                matching_saved = [
                    value
                    for key, value in config_by_name.items()
                    if key.endswith(f":{legacy_surface_name}")
                ]
                saved = matching_saved[0] if matching_saved else None
            if isinstance(saved, dict):
                if not bool(saved.get("driven", False)):
                    continue
                radiators.append(
                    RadiatorConfig(
                        name=surface_name,
                        mesh=mesh_name,
                        tag=tag,
                        level_db=float(saved.get("level_db", 0.0)),
                        polarity=int(saved.get("polarity", 1)),
                        delay_ms=float(saved.get("delay_ms", 0.0)),
                        hpf=self._saved_crossover(saved.get("hpf"), crossover_type="highpass"),
                        lpf=self._saved_crossover(saved.get("lpf"), crossover_type="lowpass"),
                    )
                )
                continue

            existing = existing_by_tag.get(tag)
            if existing is not None and mesh_name == ATH_MESH_NAME:
                radiators.append(replace(existing, name=surface_name, mesh=mesh_name, tag=tag))

        return replace(result, radiators=tuple(radiators))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._save_frequency_settings()
        self._save_preferences()
        self._save_window_state()
        super().closeEvent(event)

    def _load_initial_config_text(self) -> str:
        for path in (Path.cwd() / "sampleathscript.cfg", Path.cwd().parent / "sampleathscript.cfg"):
            if path.exists():
                return path.read_text(encoding="utf-8")
        return ""

    def _load_existing_sample_output(self) -> AthRunResult | None:
        for root in (Path.cwd(), Path.cwd().parent):
            sample_dir = root / "sampleathscript"
            if sample_dir.exists():
                try:
                    return clean_ath_mesh_output(
                        discover_ath_output(
                            run_root=root,
                            case_name="sampleathscript",
                            config_path=root / "sampleathscript.cfg",
                        )
                    )
                except Exception:
                    return None
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
            self.editor.setPlainText(path.read_text(encoding="utf-8"))
            self.status_label.setText(f"Imported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    @Slot()
    def new_project(self) -> None:
        self.project_path = None
        self.editor.clear()
        self.imported_meshes = ()
        self.ath_mesh_enabled = True
        self.ath_mesh_translation_mm = (0.0, 0.0, 0.0)
        self.ath_result = None
        self.settings.setValue("source/config_by_name", "{}")
        self.settings.sync()
        self._save_imported_meshes()
        self._save_ath_mesh_settings()
        self.source_config_button.setEnabled(False)
        self.preview.clear()
        self._clear_plots()
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
            self.status_label.setText(f"Saved project {project_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save project failed", str(exc))

    @Slot()
    def load_project(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Load Project",
            str(Path.cwd()),
            PROJECT_FILE_FILTER,
        )
        if not path_text:
            return

        path = Path(path_text)
        try:
            payload = read_project_file(path)
            self._apply_project_payload(payload)
            self.project_path = path
            self.status_label.setText(f"Loaded project {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Load project failed", str(exc))

    def _project_payload(self) -> dict:
        return build_project_payload(
            ath_config_text=self.editor.toPlainText(),
            ath_mesh=self._ath_mesh_payload(absolute_paths=True),
            imported_meshes=self._project_imported_meshes_payload(),
            source_config_by_name=self._load_source_config_by_name(),
        )

    def _apply_project_payload(self, payload: dict) -> None:
        self.editor.setPlainText(str(payload.get("ath_config_text", "")))
        ath_mesh = payload.get("ath_mesh", {})
        if isinstance(ath_mesh, dict):
            self.ath_mesh_enabled = bool(ath_mesh.get("enabled", True))
            translation = ath_mesh.get("translation_mm", [0.0, 0.0, 0.0])
            if not isinstance(translation, list) or len(translation) != 3:
                translation = [0.0, 0.0, 0.0]
            self.ath_mesh_translation_mm = tuple(float(int(round(float(value)))) for value in translation)
            self._save_ath_mesh_settings()
        self.imported_meshes = self._mesh_entries_from_payload(payload.get("imported_meshes", []))
        self._save_imported_meshes()

        source_config = payload.get("source_config_by_name", {})
        if not isinstance(source_config, dict):
            source_config = {}
        self.settings.setValue("source/config_by_name", json.dumps(source_config, sort_keys=True))
        self.settings.sync()

        if self.ath_result is not None:
            self.ath_result = self._apply_saved_source_config(self.ath_result)
            self._refresh_mesh_preview()
        self.source_config_button.setEnabled(self.ath_result is not None)
        self._clear_plots()

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
            path.write_text(self.editor.toPlainText(), encoding="utf-8")
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
            figure.savefig(output_path, dpi=VisualizerConfig.figure_dpi)
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
            written = export_polar_text_files(self.live_dataset, output_dir)
            self.status_label.setText(f"Exported {len(written)} polar files to {output_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Export polar data failed", str(exc))

    @Slot()
    def open_balloon_plot(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No balloon data", "Run a solve before opening the balloon plot.")
            return

        raw_balloon = self.live_dataset.as_balloon_raw_bundle()
        if raw_balloon is None:
            QMessageBox.warning(
                self,
                "No balloon data",
                "Enable spherical sampling in Preferences before running a solve.",
            )
            return

        try:
            self.balloon_window = BalloonPlotWindow(
                raw_balloon,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
                parent=self,
            )
            self.balloon_window.show()
            self.balloon_window.raise_()
        except Exception as exc:
            QMessageBox.critical(self, "Balloon plot failed", str(exc))

    @Slot()
    def open_preferences(self) -> None:
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.preferences = dialog.preferences()
        self._save_preferences()
        if self.ath_result is not None:
            self.ath_result = self._apply_saved_source_config(self.ath_result)
            self._refresh_mesh_preview()
        self._refresh_plots()
        self.status_label.setText("Preferences updated")

    @Slot()
    def open_mesh_config(self) -> None:
        dialog = MeshConfigDialog(self._mesh_config_dialog_entries(), self)
        if dialog.exec() != QDialog.Accepted:
            return

        try:
            self.status_label.setText("Cleaning imported meshes...")
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._apply_mesh_config_dialog_entries(dialog.meshes())
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            self._save_imported_meshes()
            if self.ath_result is not None:
                self.ath_result = self._apply_saved_source_config(self.ath_result)
                self._refresh_mesh_preview()
            self.status_label.setText(
                f"Mesh config updated: {len(self._active_imported_meshes())}/{len(self.imported_meshes)} meshes enabled"
            )
        except Exception as exc:
            self.status_label.setText("Mesh config failed")
            QMessageBox.critical(self, "Mesh config failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def open_source_config(self) -> None:
        if self.ath_result is None:
            QMessageBox.warning(self, "No mesh", "Generate or load a mesh before configuring sources.")
            return

        try:
            surface_tags = self._surface_tags_for_meshes()
        except Exception as exc:
            QMessageBox.critical(self, "Source config failed", str(exc))
            return

        dialog = SourceConfigDialog(surface_tags, self.ath_result.radiators, self)
        if dialog.exec() != QDialog.Accepted:
            return

        radiators = dialog.radiators()
        self.ath_result = replace(self.ath_result, radiators=radiators)
        self._save_source_config(surface_tags, radiators)
        self._refresh_mesh_preview()
        self.status_label.setText(f"Source config updated: {len(radiators)} driven surfaces")

    @Slot()
    def generate_geometry(self) -> None:
        case_name = f"waveguide_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_root = ATH_OUTPUT_ROOT
        self._clear_plots()
        self.status_label.setText("Generating waveguide...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._ensure_ath_runtime_config()
            raw_result = run_ath(
                ath_exe=self._find_ath_exe(),
                config_text=self.editor.toPlainText(),
                run_root=run_root,
                case_name=case_name,
            )
            self.status_label.setText("Cleaning generated mesh...")
            self.ath_result = self._apply_saved_source_config(clean_ath_mesh_output(raw_result))
            self._refresh_mesh_preview()
            self.source_config_button.setEnabled(True)
            self.status_label.setText(f"Generated and cleaned {self.ath_result.output_dir}")
        except Exception as exc:
            self.status_label.setText("Generate failed")
            QMessageBox.critical(self, "Ath generation failed", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    @Slot()
    def start_solve(self) -> None:
        if self.ath_result is None:
            QMessageBox.warning(self, "No mesh", "Generate or load an Ath output before solving.")
            return
        if not self.ath_result.radiators:
            QMessageBox.warning(self, "No driven surfaces", "Open Source Config and mark at least one surface as Driven.")
            return

        try:
            self.imported_meshes = self._clean_imported_meshes(self.imported_meshes)
            self._save_imported_meshes()
            mesh_configs = self._solver_mesh_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Imported mesh preparation failed", str(exc))
            return

        freq_min = float(min(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_max = float(max(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_count = int(self.freq_count_spin.value())
        freqs = build_log_frequencies(freq_min, freq_max, freq_count)
        ordered_freqs = order_frequencies_for_live_plotting(freqs)

        config = SimulationConfig(
            mesh_file=str(self.ath_result.solver_msh_path),
            freq_min=freq_min,
            freq_max=freq_max,
            freq_count=freq_count,
            tag_throat=self.ath_result.driven_tag,
            meshes=mesh_configs,
            radiators=self.ath_result.radiators,
            step_size=self.preferences.polar_angle_step_deg,
            use_burton_miller=self.preferences.use_burton_miller,
            gmres_tolerance=self.preferences.gmres_tolerance,
            workers=self.preferences.worker_count,
            spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
            spherical_sampling_points=self.preferences.spherical_sampling_points,
        )

        self.live_dataset = None
        self.balloon_plot_action.setEnabled(False)
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.source_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.solve_started_at = time.perf_counter()
        self.status_label.setText("Initializing solver...")

        self.solve_thread = QThread(self)
        if self.preferences.solve_backend == "server":
            self.solve_worker = RemoteSolveWorker(config, ordered_freqs, self.preferences.solve_server_url)
        else:
            self.solve_worker = SolveWorker(config, ordered_freqs, worker_count=self.preferences.worker_count)
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
            f"({result.freq_hz:.1f} Hz)"
        )
        self._refresh_plots()

    @Slot(str)
    def _on_solve_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Solve failed", message)
        self.status_label.setText("Solve failed")

    @Slot()
    def _on_solve_finished(self) -> None:
        self.solve_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.mesh_config_button.setEnabled(True)
        self.source_config_button.setEnabled(self.ath_result is not None)
        self.cancel_button.setEnabled(False)
        elapsed_s = None if self.solve_started_at is None else time.perf_counter() - self.solve_started_at
        self.solve_started_at = None
        if self.live_dataset is not None and self.live_dataset.solved_count > 0:
            self._set_export_plot_actions_enabled(True)
            self.export_polar_data_action.setEnabled(True)
            self.balloon_plot_action.setEnabled(self.live_dataset.as_balloon_raw_bundle() is not None)
            elapsed_text = "" if elapsed_s is None else f" in {elapsed_s:.1f} s"
            self.status_label.setText(
                f"Solve complete: {self.live_dataset.solved_count} frequencies{elapsed_text}"
            )
        self.solve_worker = None
        self.solve_thread = None

    def _clear_plots(self) -> None:
        self.live_dataset = None
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
            break

    def _set_export_plot_actions_enabled(self, enabled: bool) -> None:
        for action in self.export_plot_actions.values():
            action.setEnabled(enabled)

    def _prepared_live_plot_dataset(self) -> dict[str, np.ndarray] | None:
        if self.live_dataset is None:
            return None
        return self.live_dataset.as_visualization_dataset(
            PrepConfig(
                angle_samples=LIVE_ISOBAR_ANGLE_SAMPLES,
                freq_samples=LIVE_ISOBAR_FREQ_SAMPLES,
                octave_smoothing=self.preferences.polar_smoothing,
                hor_ref_angle=self.preferences.horizontal_normalization_angle,
                vert_ref_angle=self.preferences.vertical_normalization_angle,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
                normalize_polar=True,
                auto_db_span=False,
            )
        )

    def _refresh_plots(self) -> None:
        dataset = self._prepared_live_plot_dataset()
        if dataset is None:
            return

        for entry in self.plot_entries:
            entry.update(dataset)

    def _update_horizontal_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.horizontal_plot.update_plot(
            dataset["isobar_freq_hz"],
            dataset["isobar_angle_deg"],
            dataset["horizontal_isobar_db"],
            float(dataset["clip_min_db"]),
            float(dataset["clip_max_db"]),
        )

    def _update_vertical_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.vertical_plot.update_plot(
            dataset["isobar_freq_hz"],
            dataset["isobar_angle_deg"],
            dataset["vertical_isobar_db"],
            float(dataset["clip_min_db"]),
            float(dataset["clip_max_db"]),
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
        )

    def _update_spinorama_plot(self, dataset: dict[str, np.ndarray]) -> None:
        self.spinorama_plot.update_plot(
            dataset["freq_hz"],
            dataset["polar_angle_deg"],
            dataset["horizontal_spl_norm_db"],
            dataset["vertical_spl_norm_db"],
        )
