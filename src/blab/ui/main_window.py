"""Main Qt window and user workflow orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QByteArray, QEvent, QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFont, QIcon, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
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
    read_surface_physical_names,
    write_ath_gmsh_path,
    write_ath_output_root,
)
from blab.config import ChannelConfig, MeshConfig, RadiatorConfig
from blab.exporting import (
    default_on_axis_filename,
    export_on_axis_text_files,
    export_plot_png,
    export_polar_text_files,
)
from blab.generators.ath import ATH_PROVIDER_ID, ath_source_text, with_ath_source_text
from blab.generators.base import GeneratedGeometry, GenerationCompleted, GenerationRequest, GeneratorDocument
from blab.generators.postprocess import ensure_reduced_geometry
from blab.generators.registry import create_generator, generator_info
from blab.live import (
    FrequencyResult,
    LiveSolveDataset,
)
from blab.physical_model import (
    AcousticRegionKind,
    BoundaryKind,
    MeshPurpose,
    PhysicalSolveKind,
    PhysicalSystem,
    infer_physical_solve_kind,
    physical_system_from_dict,
    physical_system_to_dict,
)
from blab.plotting import VisualizerConfig
from blab.solvers.http_server import server_health_supports_symmetry
from blab.solvers.registry import backend_info
from blab.symmetry import SymmetryValidationError
from blab.ui.application_state import OperationPhase, SolveCompletion, solve_invalidation_policy
from blab.ui.diagnostics import DiagnosticsDialog
from blab.ui.dialogs import (
    ChannelConfigDialog,
    DonateDialog,
    MeshConfigDialog,
    MeshDialogEntry,
    PreferencesDialog,
)
from blab.ui.exterior_system import exterior_bem_inputs
from blab.ui.file_dialogs import FileDialogService
from blab.ui.main_window_widgets import (
    AthScriptEditor,
    DockTitleBar,
    PlotEntry,
    format_frequency_solve_timings,
)
from blab.ui.mesh_assembly import (
    STITCH_FAILURE_MESSAGE,
    STITCHED_MESH_NAME,
    MeshAssemblyService,
    PreparedMeshAssembly,
)
from blab.ui.operation_controllers import (
    GeometryController,
    SolveController,
    SolveRequest,
)
from blab.ui.physical_system_migration import AUTO_SEEDED_EXTERIOR_KEY, seed_exterior_system
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
    ImportedMeshState,
    ProjectDocument,
    ProjectPreferencesState,
    generator_document_to_payload,
    generator_documents_from_payload,
    generator_mesh_name,
    new_generator_document,
    new_project_document,
    replace_generator_document,
    unique_generator_name,
)
from blab.ui.result_projection import (
    ProjectionOptions,
    ResultProjectionService,
    VisualizationProjection,
)
from blab.ui.server_credentials import load_server_access_token
from blab.ui.server_health_worker import ServerHealthCheckWorker
from blab.ui.settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    GuiPreferences,
    balloon_sampling_points,
    gui_preferences_with_project_preferences,
    live_plot_angle_samples,
    live_plot_freq_samples,
    load_gui_preferences,
    preferences_require_solve_invalidation,
    preferences_require_visualization_refresh,
    project_preferences_from_gui,
    save_gui_preferences,
    settings_int,
)
from blab.ui.simulation_assembler import SimulationAssembler, SimulationParameters
from blab.ui.source_channel_config import (
    apply_saved_imported_source_config,
    apply_saved_source_config_to_result,
    channel_config_payload,
    channel_configs_from_payload,
    channels_for_solver_radiators,
)
from blab.ui.system_config import (
    SystemConfigDialog,
    inspect_system_meshes,
    sync_physical_system_meshes,
)
from blab.ui.system_solve import prepare_coupled_ui_solve
from blab.ui.theme import apply_application_theme

DEFAULT_MESH_SCALE_FACTOR = 0.001
LIVE_PLOT_REFRESH_INTERVAL_MS = 250


APP_ROOT = Path(__file__).resolve().parents[3]
ATH_BUNDLE_DIR = APP_ROOT / "ath"
GENERATED_GEOMETRY_ROOT = APP_ROOT / "runs" / "generated_geometry"
GMSH_BUNDLE_EXE = APP_ROOT / "gmsh" / "gmsh-4.15.2-Windows64" / "gmsh.exe"
HELP_GUIDE_PDF = APP_ROOT / "docs" / "Boundary Lab Guide.pdf"
SAVE_DARK_ICON = APP_ROOT / "assets" / "save_dark.ico"
SAVE_LIGHT_ICON = APP_ROOT / "assets" / "save_light.ico"
CAPTURE_CONTOURS_DARK_ICON = APP_ROOT / "assets" / "capturecontours_dark.ico"
CAPTURE_CONTOURS_LIGHT_ICON = APP_ROOT / "assets" / "capturecontours_light.ico"
CLEAR_CONTOURS_DARK_ICON = APP_ROOT / "assets" / "clearcontours_dark.ico"
CLEAR_CONTOURS_LIGHT_ICON = APP_ROOT / "assets" / "clearcontours_light.ico"
FEM_PREVIEW_DARK_ICON = APP_ROOT / "assets" / "FEMTetra_dark.ico"
FEM_PREVIEW_LIGHT_ICON = APP_ROOT / "assets" / "FEMTetra_light.ico"
BEM_PREVIEW_DARK_ICON = APP_ROOT / "assets" / "BEMTri_dark.ico"
BEM_PREVIEW_LIGHT_ICON = APP_ROOT / "assets" / "BEMTri_light.ico"
ADD_DESIGN_TAB_LABEL = "+"
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


def _mesh_entries_with_file_overrides(
    meshes: tuple[MeshDialogEntry, ...],
    overrides_by_name: dict[str, str],
) -> tuple[MeshDialogEntry, ...]:
    return tuple(
        replace(mesh, cleaned_file=overrides_by_name.get(mesh.name, mesh.cleaned_file))
        for mesh in meshes
    )


def _physical_system_preview_metadata(
    system: PhysicalSystem | None,
    surface_tags_by_mesh: dict[str, dict[str, int]],
) -> tuple[set[tuple[str, int]], set[tuple[str, int]], dict[str, str], bool]:
    if system is None:
        return set(), set(), {}, False

    meshes_by_id = {mesh.id: mesh for mesh in system.meshes}
    boundaries_by_id = {boundary.id: boundary for boundary in system.boundaries}
    mesh_regions = {
        mesh.name: "interior" if mesh.purpose == MeshPurpose.FEM_VOLUME else "exterior"
        for mesh in system.meshes
    }
    has_interior_region = any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in system.regions)

    def resolved_surface(boundary_id: str) -> tuple[str, int] | None:
        boundary = boundaries_by_id.get(boundary_id)
        if boundary is None:
            return None
        mesh = meshes_by_id.get(boundary.group.mesh_id)
        if mesh is None:
            return None
        tag = boundary.group.tag
        if tag is None and boundary.group.name is not None:
            tag = surface_tags_by_mesh.get(mesh.name, {}).get(boundary.group.name)
        if tag is None:
            return None
        return mesh.name, int(tag)

    interface_surfaces: set[tuple[str, int]] = set()
    for boundary in system.boundaries:
        if boundary.kind != BoundaryKind.INTERFACE:
            continue
        surface = resolved_surface(boundary.id)
        if surface is not None:
            interface_surfaces.add(surface)

    component_surfaces: set[tuple[str, int]] = set()
    for component in system.components:
        for boundary_id in component.boundary_ids:
            boundary = boundaries_by_id.get(boundary_id)
            if boundary is None or boundary.kind != BoundaryKind.MOVING:
                continue
            surface = resolved_surface(boundary_id)
            if surface is not None:
                component_surfaces.add(surface)
    return interface_surfaces, component_surfaces, mesh_regions, has_interior_region


class MainWindow(QMainWindow):
    mesh_state_changed = Signal(str)
    source_config_changed = Signal(str)
    project_state_changed = Signal(str)
    solve_results_invalidated = Signal(str)
    visualization_settings_changed = Signal(str)

    def _project_document(self) -> ProjectDocument:
        project = getattr(self, "project", None)
        if project is None:
            project = new_project_document()
            self.project = project
        return project

    @property
    def generator_documents(self) -> tuple[GeneratorDocument, ...]:
        return self._project_document().generator_documents

    @generator_documents.setter
    def generator_documents(self, value: tuple[GeneratorDocument, ...]) -> None:
        self._project_document().generator_documents = tuple(value)

    @property
    def active_generator_document_id(self) -> str | None:
        return self._project_document().active_generator_document_id

    @active_generator_document_id.setter
    def active_generator_document_id(self, value: str | None) -> None:
        self._project_document().active_generator_document_id = value

    @property
    def imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        return tuple(
            MeshDialogEntry(
                name=mesh.name,
                source_file=mesh.source_file,
                cleaned_file=mesh.cleaned_file,
                scale_factor=mesh.scale_factor,
                translation_mm=mesh.translation_mm,
                enabled=mesh.enabled,
            )
            for mesh in self._project_document().imported_meshes
        )

    @imported_meshes.setter
    def imported_meshes(self, value: tuple[MeshDialogEntry, ...]) -> None:
        self._project_document().imported_meshes = tuple(
            ImportedMeshState(
                name=mesh.name,
                source_file=mesh.source_file,
                cleaned_file=mesh.cleaned_file,
                scale_factor=mesh.scale_factor,
                translation_mm=mesh.translation_mm,
                enabled=mesh.enabled,
            )
            for mesh in value
        )

    @property
    def stitch_imported_meshes(self) -> bool:
        return self._project_document().stitch_imported_meshes

    @stitch_imported_meshes.setter
    def stitch_imported_meshes(self, value: bool) -> None:
        self._project_document().stitch_imported_meshes = bool(value)

    @property
    def symmetry(self) -> str:
        return self._project_document().symmetry

    @symmetry.setter
    def symmetry(self, value: str) -> None:
        self._project_document().symmetry = value

    def __init__(self, startup_status: Callable[[str], None] | None = None):
        super().__init__()

        def startup(stage: str) -> None:
            if startup_status is not None:
                startup_status(stage)

        startup("Loading saved settings...")
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.file_dialogs = FileDialogService(self.settings)
        self.setWindowTitle(f"Boundary Lab Beta {__version__}")
        self.resize(1500, 900)
        self.preferences = self._load_preferences()
        self.project: ProjectDocument = new_project_document()
        self.simulation_assembler = SimulationAssembler()
        self.mesh_assembly_service = MeshAssemblyService(Path.cwd() / "runs" / "imported_meshes")
        self.result_projection_service = ResultProjectionService()
        self.geometry_controller = GeometryController(self)
        self.solve_controller = SolveController(self)
        self.server_health_payload: dict | None = None
        self.server_health_url: str | None = None
        self.server_health_thread: QThread | None = None
        self.server_health_worker: ServerHealthCheckWorker | None = None
        self._apply_theme()
        self.generated_geometry_by_document_id: dict[str, GeneratedGeometry] = {}
        self.imported_radiators: tuple[RadiatorConfig, ...] = ()
        self.live_dataset: LiveSolveDataset | None = None
        self._last_completed_visualization_dataset: VisualizationProjection | None = None
        self.balloon_window: QWidget | None = None
        self.channel_config_dialog: ChannelConfigDialog | None = None
        self.project_path: Path | None = None
        self._project_clean_payload: dict | None = None
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self._last_imported_mesh_focus_check_at = 0.0
        self._plot_dpi_screen = None
        self._plot_dpi_window_handle = None
        self._plot_dpi_refresh_pending = False
        self._live_plot_refresh_dirty = False
        self._live_plot_refresh_timer = QTimer(self)
        self._live_plot_refresh_timer.setSingleShot(True)
        self._live_plot_refresh_timer.setInterval(LIVE_PLOT_REFRESH_INTERVAL_MS)
        self._live_plot_refresh_timer.timeout.connect(self._flush_live_plot_refresh)
        startup("Building design editor...")
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.currentChanged.connect(self._on_active_generator_tab_changed)
        self.editor_tabs.tabCloseRequested.connect(self._remove_generator_document_at)
        self.editor_tabs.tabBar().installEventFilter(self)
        self._rebuild_generator_document_tabs()

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
        self.mesh_config_button = QPushButton("Meshes")
        self.system_config_button = QPushButton("System")
        self.channel_config_button = QPushButton("Channels")
        self.system_config_button.setEnabled(self._has_solver_meshes())

        freq_min = min(max(settings_int(self.settings, "solve/freq_min_hz", 200), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ)
        freq_max = min(
            max(settings_int(self.settings, "solve/freq_max_hz", 20000), AUDIO_FREQ_MIN_HZ), AUDIO_FREQ_MAX_HZ
        )
        freq_count = min(max(settings_int(self.settings, "solve/freq_count", 41), 3), 200)

        self.freq_min_slider = self._make_slider(0, FREQ_SLIDER_STEPS, frequency_to_slider_value(freq_min))
        self.freq_max_slider = self._make_slider(0, FREQ_SLIDER_STEPS, frequency_to_slider_value(freq_max))
        self.freq_count_slider = self._make_slider(3, 200, freq_count)
        self.freq_count_slider.setSingleStep(2)

        self.freq_min_spin = self._make_spin(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ, freq_min)
        self.freq_max_spin = self._make_spin(AUDIO_FREQ_MIN_HZ, AUDIO_FREQ_MAX_HZ, freq_max)
        self.freq_count_spin = self._make_spin(3, 200, freq_count)

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
        self._connect_operation_controllers()
        startup("Restoring window layout...")
        self._restore_window_state()
        startup("Starting new project...")
        self.new_project()
        QTimer.singleShot(0, self._check_configured_server_health_on_startup)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._refresh_plot_export_icons()
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
            if index == len(self.generator_documents):
                self.add_generator_document()
                return True
        if watched is self.editor_tabs.tabBar() and event.type() == QEvent.Type.MouseButtonDblClick:
            index = self.editor_tabs.tabBar().tabAt(event.position().toPoint())
            if 0 <= index < len(self.generator_documents):
                self.editor_tabs.setCurrentIndex(index)
                self.rename_active_generator_document()
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

        import_action = QAction("Import Waveguide Design...", self)
        import_action.triggered.connect(self.import_config)
        file_menu.addAction(import_action)

        export_cfg_action = QAction("Export Waveguide Design...", self)
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

        self.export_on_axis_data_action = QAction("Export On-Axis Data", self)
        self.export_on_axis_data_action.setEnabled(False)
        self.export_on_axis_data_action.triggered.connect(self.export_on_axis_data)
        file_menu.addAction(self.export_on_axis_data_action)

        view_menu = self.menuBar().addMenu("View")
        self.balloon_plot_action = QAction("Balloon Plot", self)
        self.balloon_plot_action.setEnabled(False)
        self.balloon_plot_action.triggered.connect(self.open_balloon_plot)
        view_menu.addAction(self.balloon_plot_action)
        view_menu.addSeparator()
        for dock_id, title in (
            ("editor", "Waveguide Design Panel"),
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
        self.editor_dock = self._make_panel_dock(
            "ath_editor_dock",
            "Waveguide Design",
            self.editor_container,
        )
        self.show_interior_regions_action = QAction("Show interior regions", self)
        self.show_interior_regions_action.setToolTip("Show interior regions")
        self.show_interior_regions_action.setCheckable(True)
        self.show_interior_regions_action.setEnabled(False)
        self.show_interior_regions_action.triggered.connect(self._on_show_interior_regions)
        self.show_exterior_region_action = QAction("Show exterior region", self)
        self.show_exterior_region_action.setToolTip("Show exterior region")
        self.show_exterior_region_action.setCheckable(True)
        self.show_exterior_region_action.setEnabled(False)
        self.show_exterior_region_action.triggered.connect(self._on_show_exterior_region)
        self.preview_dock = self._make_panel_dock(
            "mesh_preview_dock",
            "Mesh Preview",
            self.preview,
            tool_actions=(self.show_interior_regions_action, self.show_exterior_region_action),
        )
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
        controls_layout.addWidget(self.system_config_button)
        controls_layout.addWidget(self.channel_config_button)
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
        self.system_config_button.clicked.connect(self.open_system_config)
        self.channel_config_button.clicked.connect(self.open_channel_config)

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

    def _connect_operation_controllers(self) -> None:
        self.geometry_controller.completed.connect(self._on_geometry_generated)
        self.geometry_controller.status.connect(self.status_label.setText)
        self.geometry_controller.failed.connect(self._on_geometry_generation_failed)
        self.geometry_controller.cancelled.connect(self._on_geometry_generation_cancelled)
        self.geometry_controller.finished.connect(self._on_geometry_generation_finished)
        self.solve_controller.initialized.connect(self._on_solver_initialized)
        self.solve_controller.result_ready.connect(self._on_frequency_result)
        self.solve_controller.status.connect(self.status_label.setText)
        self.solve_controller.failed.connect(self._on_solve_failed)
        self.solve_controller.finished.connect(self._on_solve_finished)

    @Slot(str)
    def _on_mesh_state_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()
        self.system_config_button.setEnabled(self._has_solver_meshes())

    @Slot(str)
    def _on_source_config_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()

    @Slot(str)
    def _on_project_state_changed(self, _reason: str) -> None:
        self._refresh_mesh_preview()
        self.system_config_button.setEnabled(self._has_solver_meshes())

    @Slot(bool)
    def _on_show_interior_regions(self, checked: bool) -> None:
        if checked:
            blocker = QSignalBlocker(self.show_exterior_region_action)
            self.show_exterior_region_action.setChecked(False)
            del blocker
        self._apply_preview_region_action_state()

    @Slot(bool)
    def _on_show_exterior_region(self, checked: bool) -> None:
        if checked:
            blocker = QSignalBlocker(self.show_interior_regions_action)
            self.show_interior_regions_action.setChecked(False)
            del blocker
        self._apply_preview_region_action_state()

    def _apply_preview_region_action_state(self) -> None:
        if self.show_interior_regions_action.isChecked():
            mode = "interior"
        elif self.show_exterior_region_action.isChecked():
            mode = "exterior"
        else:
            mode = "all"
        self.preview.set_region_visibility_mode(mode)

    def _sync_preview_region_actions(self) -> None:
        if not hasattr(self, "show_interior_regions_action"):
            return
        system = getattr(self._project_document(), "physical_system", None)
        has_interior_region = (
            system is not None
            and any(region.kind == AcousticRegionKind.BOUNDED_AIR for region in system.regions)
        )
        self.show_interior_regions_action.setEnabled(has_interior_region)
        self.show_exterior_region_action.setEnabled(has_interior_region)
        if has_interior_region:
            return
        interior_blocker = QSignalBlocker(self.show_interior_regions_action)
        exterior_blocker = QSignalBlocker(self.show_exterior_region_action)
        self.show_interior_regions_action.setChecked(False)
        self.show_exterior_region_action.setChecked(False)
        del interior_blocker, exterior_blocker
        self.preview.set_region_visibility_mode("all")

    @Slot(str)
    def _on_solve_results_invalidated(self, reason: str) -> None:
        policy = solve_invalidation_policy(reason)
        if policy.clear_solve_results:
            self._clear_plots()
        if policy.clear_comparison_history:
            self._clear_plot_comparison_history()

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

    def _rebuild_generator_document_tabs(self) -> None:
        self.editor_tabs.blockSignals(True)
        self.editor_tabs.clear()
        for document in self.generator_documents:
            editor = AthScriptEditor()
            editor.setFont(QFont("Consolas", 10))
            if document.provider_id == ATH_PROVIDER_ID:
                editor.setPlainText(ath_source_text(document))
                editor.textChanged.connect(
                    lambda document_id=document.id, editor=editor: self._update_generator_source_text(
                        document_id, editor
                    )
                )
                editor.configDropped.connect(
                    lambda path, document_id=document.id: self._import_config_path(Path(path), document_id=document_id)
                )
            else:
                editor.setPlainText(json.dumps(document.source, indent=2, sort_keys=True))
                editor.setReadOnly(True)
            self.editor_tabs.addTab(editor, document.name)
        add_tab = AthScriptEditor()
        add_tab.setReadOnly(True)
        add_tab.configDropped.connect(lambda path: self._import_config_path(Path(path)))
        add_index = self.editor_tabs.addTab(add_tab, ADD_DESIGN_TAB_LABEL)
        self.editor_tabs.tabBar().setTabButton(add_index, QTabBar.ButtonPosition.RightSide, None)
        self.editor_tabs.tabBar().setTabToolTip(add_index, "Add waveguide design")
        active_index = self._active_generator_document_index()
        if active_index >= 0:
            self.editor_tabs.setCurrentIndex(active_index)
        self.editor_tabs.blockSignals(False)

    def _active_generator_document_index(self) -> int:
        for index, document in enumerate(self.generator_documents):
            if document.id == self.active_generator_document_id:
                return index
        return 0 if self.generator_documents else -1

    def _active_generator_document(self) -> GeneratorDocument | None:
        if not self.generator_documents:
            return None
        index = self._active_generator_document_index()
        return self.generator_documents[index] if index >= 0 else None

    def _update_generator_source_text(self, document_id: str, editor: QPlainTextEdit) -> None:
        self.generator_documents = tuple(
            with_ath_source_text(document, editor.toPlainText()) if document.id == document_id else document
            for document in self.generator_documents
        )

    def _on_active_generator_tab_changed(self, index: int) -> None:
        if index == len(self.generator_documents):
            self.add_generator_document()
            return
        if 0 <= index < len(self.generator_documents):
            self.active_generator_document_id = self.generator_documents[index].id

    @Slot()
    def add_generator_document(self) -> None:
        name = unique_generator_name("waveguide", self.generator_documents)
        document = new_generator_document(name, "")
        self.generator_documents = (*self.generator_documents, document)
        self.active_generator_document_id = document.id
        self._rebuild_generator_document_tabs()

    @Slot()
    def rename_active_generator_document(self) -> None:
        document = self._active_generator_document()
        if document is None:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Rename Waveguide Design",
            "Design name:",
            text=document.name,
        )
        if not accepted:
            return
        name = name.strip()
        if not name:
            return
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document.id,
            name=unique_generator_name(
                name,
                tuple(item for item in self.generator_documents if item.id != document.id),
            ),
        )
        self._rebuild_generator_document_tabs()
        self.mesh_state_changed.emit("generator_document_renamed")
        self.solve_results_invalidated.emit("generator_document_renamed")

    def _remove_generator_document_at(self, index: int) -> None:
        if not (0 <= index < len(self.generator_documents)):
            return
        document = self.generator_documents[index]
        self.generator_documents = tuple(item for item in self.generator_documents if item.id != document.id)
        self.generated_geometry_by_document_id.pop(document.id, None)
        self.active_generator_document_id = (
            self.generator_documents[min(index, len(self.generator_documents) - 1)].id
            if self.generator_documents
            else None
        )
        self._rebuild_generator_document_tabs()
        self.mesh_state_changed.emit("generator_document_removed")
        self.solve_results_invalidated.emit("generator_document_removed")

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

    @Slot()
    def _save_frequency_settings(self) -> None:
        self.settings.setValue("solve/freq_min_hz", int(self.freq_min_spin.value()))
        self.settings.setValue("solve/freq_max_hz", int(self.freq_max_spin.value()))
        self.settings.setValue("solve/freq_count", int(self.freq_count_spin.value()))
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

    def _mesh_config_dialog_entries(self) -> tuple[MeshDialogEntry, ...]:
        return self._mesh_config_dialog_entries_for_symmetry("off")

    def _mesh_config_dialog_entries_for_symmetry(
        self,
        symmetry: str,
    ) -> tuple[MeshDialogEntry, ...]:
        """Return mesh entries backed by solve-ready generated geometry."""

        entries = []
        for document in self.generator_documents:
            result = self.generated_geometry_by_document_id.get(document.id)
            if result is None:
                continue
            solver_result = self._generated_geometry_for_solver_symmetry(document, result, symmetry)
            entries.append(
                MeshDialogEntry(
                    name=generator_mesh_name(document),
                    source_file=str(solver_result.solver_mesh_path_for_symmetry(symmetry)),
                    scale_factor=float(document.mesh_scale_factor),
                    translation_mm=document.mesh_translation_mm,
                    enabled=document.mesh_enabled,
                    locked=True,
                )
            )
        entries.extend(self.imported_meshes)
        return tuple(entries)

    def _apply_mesh_config_dialog_entries(self, meshes: tuple[MeshDialogEntry, ...]) -> None:
        imported_meshes = []
        documents = self.generator_documents
        for mesh in meshes:
            document = self._generator_document_for_mesh_name(mesh.name)
            if document is not None:
                documents = replace_generator_document(
                    documents,
                    document.id,
                    mesh_enabled=bool(mesh.enabled),
                    mesh_translation_mm=mesh.translation_mm,
                    mesh_scale_factor=float(mesh.scale_factor),
                )
            else:
                imported_meshes.append(replace(mesh, locked=False))
        self.generator_documents = documents
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

    def _generator_document_for_mesh_name(self, mesh_name: str) -> GeneratorDocument | None:
        return next(
            (document for document in self.generator_documents if generator_mesh_name(document) == mesh_name),
            None,
        )

    def _has_solver_meshes(self) -> bool:
        return bool(self._enabled_generated_geometry()) or bool(self._active_imported_meshes())

    def _enabled_generated_geometry(self) -> tuple[tuple[GeneratorDocument, GeneratedGeometry], ...]:
        pairs = []
        for document in self.generator_documents:
            if not document.mesh_enabled:
                continue
            result = self.generated_geometry_by_document_id.get(document.id)
            if result is not None:
                pairs.append((document, result))
        return tuple(pairs)

    def _all_radiators(self) -> tuple[RadiatorConfig, ...]:
        radiators = []
        for document, result in self._enabled_generated_geometry():
            mesh_name = generator_mesh_name(document)
            radiators.extend(replace(radiator, mesh=mesh_name) for radiator in result.radiators)
        radiators.extend(self.imported_radiators)
        return tuple(radiators)

    def _ensure_seeded_exterior_system(self) -> None:
        current = self.project.physical_system
        if current is not None and not bool(current.metadata.get(AUTO_SEEDED_EXTERIOR_KEY, False)):
            return
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
            if not any(not mesh.has_tetrahedra for mesh in meshes):
                return
            system, component_channels = seed_exterior_system(
                meshes,
                self._source_radiators_for_system_seed(),
            )
        except (OSError, ValueError):
            return
        self.project.physical_system = system
        self.project.component_channel_by_id = component_channels
        if all(
            not document.mesh_enabled or document.id in self.generated_geometry_by_document_id
            for document in self.generator_documents
        ):
            self.project.source_config_by_name = {}

    def _source_radiators_for_system_seed(self) -> tuple[RadiatorConfig, ...]:
        radiators = self._all_radiators()
        if not any(radiator.mesh == STITCHED_MESH_NAME for radiator in radiators):
            return radiators
        try:
            source_meshes = self._stitch_candidate_mesh_configs()
            stitched_map = self._mesh_service().stitched_radiator_map(source_meshes)
            source_name_by_key = {
                (mesh.name, int(tag)): f"{mesh.name}:{name}"
                for mesh in source_meshes
                for name, tag in read_surface_physical_names(Path(mesh.file)).items()
            }
        except (OSError, ValueError):
            return radiators
        reverse = {
            (str(stitched_name), int(stitched_tag)): (str(mesh_name), int(source_tag))
            for (mesh_name, source_tag), (stitched_name, stitched_tag) in stitched_map.items()
        }
        resolved = []
        for radiator in radiators:
            if radiator.mesh != STITCHED_MESH_NAME:
                resolved.append(radiator)
                continue
            source_key = reverse.get((radiator.name, int(radiator.tag)))
            if source_key is None:
                source_key = next(
                    (
                        candidate
                        for (stitched_name, stitched_tag), candidate in reverse.items()
                        if stitched_tag == int(radiator.tag)
                    ),
                    None,
                )
            if source_key is None:
                continue
            resolved.append(
                replace(
                    radiator,
                    name=source_name_by_key.get(source_key, radiator.name),
                    mesh=source_key[0],
                    tag=source_key[1],
                )
            )
        return tuple(resolved)

    def _apply_radiators_to_results(self, radiators: tuple[RadiatorConfig, ...]) -> None:
        generated_mesh_names = {generator_mesh_name(document) for document in self.generator_documents}
        for document, result in tuple(self._enabled_generated_geometry()):
            mesh_name = generator_mesh_name(document)
            updated = [replace(radiator, mesh=mesh_name) for radiator in radiators if radiator.mesh == mesh_name]
            self.generated_geometry_by_document_id[document.id] = replace(result, radiators=tuple(updated))
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
            generated_mesh_names = {generator_mesh_name(document) for document in self.generator_documents}
            if not source_file or not name or name in generated_mesh_names:
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
        states = tuple(
            ImportedMeshState(
                name=mesh.name,
                source_file=mesh.source_file,
                cleaned_file=mesh.cleaned_file,
                scale_factor=mesh.scale_factor,
                translation_mm=mesh.translation_mm,
                enabled=mesh.enabled,
            )
            for mesh in meshes
        )
        cleaned = self._mesh_service().clean_imported_meshes(states)
        return tuple(
            replace(mesh, cleaned_file=state.cleaned_file) for mesh, state in zip(meshes, cleaned, strict=True)
        )

    def _mesh_service(self) -> MeshAssemblyService:
        service = getattr(self, "mesh_assembly_service", None)
        if service is None:
            service = MeshAssemblyService(Path.cwd() / "runs" / "imported_meshes")
            self.mesh_assembly_service = service
        return service

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

    @Slot()
    def _check_configured_server_health_on_startup(self) -> None:
        if self.preferences.solve_backend != "server" or self.server_health_thread is not None:
            return
        worker = ServerHealthCheckWorker(
            self.preferences.solve_server_url,
            access_token=load_server_access_token(self.preferences.solve_server_url),
            timeout_s=5.0,
        )
        thread = QThread(self)
        self.server_health_thread = thread
        self.server_health_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.succeeded.connect(self._on_startup_server_health_succeeded)
        worker.failed.connect(lambda _message: None)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_startup_server_health_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(str, object)
    def _on_startup_server_health_succeeded(self, server_url: str, payload: dict) -> None:
        if self.preferences.solve_backend != "server":
            return
        normalized_url = server_url.rstrip("/")
        if normalized_url != self.preferences.solve_server_url.rstrip("/"):
            return
        previously_supported = self._selected_backend_supports_symmetry()
        self.server_health_payload = payload
        self.server_health_url = normalized_url
        if self._selected_backend_supports_symmetry() != previously_supported:
            self.mesh_state_changed.emit("server_health_checked")

    @Slot()
    def _on_startup_server_health_finished(self) -> None:
        self.server_health_thread = None
        self.server_health_worker = None

    def _imported_mesh_needs_reload(self, mesh: MeshDialogEntry) -> bool:
        if not mesh.enabled:
            return False
        source_path = Path(mesh.source_file)
        if source_path.suffix.lower() != ".msh" or not source_path.exists():
            return False
        if self._mesh_service().is_volume_mesh(source_path):
            return False
        cleaned_path = Path(mesh.cleaned_file) if mesh.cleaned_file else self._cleaned_imported_mesh_path(mesh)
        if not cleaned_path.exists():
            return True
        return source_path.stat().st_mtime_ns > cleaned_path.stat().st_mtime_ns

    def _updated_imported_mesh_names(self) -> tuple[str, ...]:
        return tuple(mesh.name for mesh in self.imported_meshes if self._imported_mesh_needs_reload(mesh))

    def _reload_updated_imported_meshes_on_focus(self) -> None:
        if not self.imported_meshes or self.solve_controller.active:
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
        return self._mesh_service().cleaned_imported_mesh_path(
            ImportedMeshState(name=mesh.name, source_file=mesh.source_file)
        )

    def _stitch_candidate_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return (*self._generated_solver_mesh_configs_for_symmetry(self.symmetry), *self._imported_solver_mesh_configs())

    def _should_use_stitched_mesh(self) -> bool:
        return self.stitch_imported_meshes and len(self._stitch_candidate_mesh_configs()) > 1

    def _stitched_mesh_path(self, mesh_configs: tuple[MeshConfig, ...]) -> Path:
        return self._mesh_service().stitched_mesh_path(
            mesh_configs,
            self.preferences.stitch_tolerance_mm,
            self.symmetry,
        )

    def _mesh_for_stitching(self, mesh_cfg: MeshConfig):
        return self._mesh_service().mesh_for_stitching(mesh_cfg)

    def _stitch_ignored_boundary_axes(self) -> tuple[str, ...]:
        return self._mesh_service().ignored_boundary_axes(self.symmetry)

    def _stitched_solver_mesh_config(self) -> MeshConfig | None:
        if not self._should_use_stitched_mesh():
            return None
        assembly = self._prepare_mesh_assembly(())
        return assembly.mesh_configs[0] if assembly.mesh_configs else None

    def _active_imported_meshes(self) -> tuple[MeshDialogEntry, ...]:
        return tuple(mesh for mesh in self.imported_meshes if mesh.enabled)

    def _generated_geometry_for_solver_symmetry(
        self,
        document: GeneratorDocument,
        result: GeneratedGeometry,
        symmetry: str,
    ) -> GeneratedGeometry:
        if symmetry == "off":
            return result
        updated = ensure_reduced_geometry(result)
        self.generated_geometry_by_document_id[document.id] = updated
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document.id,
            artifact=updated.to_reference(),
        )
        return updated

    def _generated_solver_mesh_configs_for_symmetry(self, symmetry: str) -> tuple[MeshConfig, ...]:
        configs = []
        for document, result in self._enabled_generated_geometry():
            solver_result = self._generated_geometry_for_solver_symmetry(document, result, symmetry)
            configs.append(
                MeshConfig(
                    name=generator_mesh_name(document),
                    file=str(solver_result.solver_mesh_path_for_symmetry(symmetry)),
                    scale_factor=float(document.mesh_scale_factor),
                    translation_m=tuple(value / 1000.0 for value in document.mesh_translation_mm),
                )
            )
        return tuple(configs)

    def _generated_solver_mesh_configs(self) -> tuple[MeshConfig, ...]:
        return self._generated_solver_mesh_configs_for_symmetry(self.symmetry)

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
        return self._prepare_mesh_assembly(self._all_radiators()).mesh_configs

    def _prepare_mesh_assembly(
        self,
        radiators: tuple[RadiatorConfig, ...],
    ) -> PreparedMeshAssembly:
        assembly = self._mesh_service().prepare(
            generated_mesh_configs=self._generated_solver_mesh_configs(),
            imported_meshes=self._project_document().imported_meshes,
            radiators=radiators,
            stitch_imported_meshes=self.stitch_imported_meshes,
            stitch_tolerance_mm=self.preferences.stitch_tolerance_mm,
            symmetry=self.symmetry,
        )
        self._project_document().imported_meshes = assembly.imported_meshes
        return assembly

    def _unique_stitched_surface_name(
        self,
        surface_name: str,
        used_surface_names: set[str],
        mesh_index: int,
    ) -> str:
        return self._mesh_service().unique_surface_name(surface_name, used_surface_names, mesh_index)

    def _used_surface_tags_for_mesh(self, mesh_cfg: MeshConfig) -> tuple[int, ...]:
        return self._mesh_service().used_surface_tags(mesh_cfg)

    def _stitched_radiator_map(self) -> dict[tuple[str | None, int], tuple[str, int]]:
        return self._mesh_service().stitched_radiator_map(self._stitch_candidate_mesh_configs())

    def _radiators_for_solver_meshes(
        self,
        mesh_configs: tuple[MeshConfig, ...],
        radiators: tuple[RadiatorConfig, ...],
    ) -> tuple[RadiatorConfig, ...]:
        if len(mesh_configs) != 1 or mesh_configs[0].name != STITCHED_MESH_NAME:
            return radiators
        return self._mesh_service().radiators_for_stitched_mesh(
            self._stitch_candidate_mesh_configs(),
            radiators,
        )

    def _show_stitch_or_generic_error(self, title: str, exc: Exception) -> None:
        if str(exc) != STITCH_FAILURE_MESSAGE:
            QMessageBox.critical(self, title, str(exc))
            return

        message = QMessageBox(QMessageBox.Critical, title, STITCH_FAILURE_MESSAGE, QMessageBox.Ok, self)
        if exc.__cause__ is not None:
            message.setDetailedText(str(exc.__cause__))
        message.exec()

    def _show_mesh_quality_warning(self, result: GeneratedGeometry) -> None:
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
                "Try increasing mesh resolution around sharp transitions or adjusting the geometry "
                "to avoid long, needle-like triangles."
            ),
        )

    def _surface_tags_for_meshes(self) -> dict[str, tuple[str, int]]:
        return self._prepare_mesh_assembly(self._all_radiators()).surface_tags

    def _refresh_mesh_preview(self) -> None:
        self._sync_preview_region_actions()
        if not self._has_solver_meshes():
            self.preview.clear()
            return
        try:
            assembly = self._prepare_mesh_assembly(self._all_radiators())
            mesh_configs = assembly.mesh_configs
            if not mesh_configs:
                self.preview.clear()
                return
            interface_surfaces, component_surfaces, mesh_regions, _has_interior = (
                _physical_system_preview_metadata(
                    self._project_document().physical_system,
                    assembly.surface_tags_by_mesh,
                )
            )
            driven_surfaces = {
                (radiator.mesh, radiator.tag) for radiator in assembly.radiators
            } | component_surfaces
            self.preview.load_mesh_configs(
                mesh_configs,
                driven_surfaces=driven_surfaces,
                surface_tags_by_mesh=assembly.surface_tags_by_mesh,
                interface_surfaces=interface_surfaces,
                mesh_regions=mesh_regions,
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
            interface_surfaces, component_surfaces, mesh_regions, _has_interior = (
                _physical_system_preview_metadata(
                    self._project_document().physical_system,
                    surface_tags_by_mesh,
                )
            )
            driven_surfaces = {
                (radiator.mesh, radiator.tag) for radiator in self._all_radiators()
            } | component_surfaces
            self.preview.load_mesh_configs(
                mesh_configs,
                driven_surfaces=driven_surfaces,
                surface_tags_by_mesh=surface_tags_by_mesh,
                interface_surfaces=interface_surfaces,
                mesh_regions=mesh_regions,
                symmetry=self.symmetry,
            )
            self.status_label.setText("Mesh preview showing unstitched meshes; stitching failed")
        except Exception:
            self.preview.clear()

    def _load_source_config_by_name(self) -> dict[str, dict]:
        return self.project.source_config_by_name

    def _load_channel_config_by_name(self) -> dict[str, dict]:
        return self.project.channel_config_by_name

    def _save_channel_config(self, channels: tuple[ChannelConfig, ...]) -> None:
        self.project.channel_config_by_name = channel_config_payload(channels)

    def _channel_configs(self) -> tuple[ChannelConfig, ...]:
        return channel_configs_from_payload(self.project.channel_config_by_name)

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

    def _apply_saved_source_config_to_result(
        self,
        result: GeneratedGeometry | None,
        mesh_name: str,
    ) -> GeneratedGeometry | None:
        return apply_saved_source_config_to_result(result, mesh_name, self._load_source_config_by_name())

    def _apply_saved_imported_source_config(self, surface_tags: dict[str, tuple[str, int]]) -> None:
        generated_mesh_names = {generator_mesh_name(document) for document in self.generator_documents}
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

    def _result_from_generator_document(self, document: GeneratorDocument) -> GeneratedGeometry | None:
        try:
            return create_generator(document.provider_id).restore(document)
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
        write_ath_output_root(ath_cfg, GENERATED_GEOMETRY_ROOT)
        write_ath_gmsh_path(ath_cfg, GMSH_BUNDLE_EXE)

    @Slot()
    def import_config(self) -> None:
        path = self.file_dialogs.open_file(
            self,
            "Import Waveguide Design",
            "Ath config files (*.cfg);;All files (*)",
        )
        if path is None:
            return

        self._import_config_path(path)

    def _import_config_path(self, path: Path, *, document_id: str | None = None) -> None:
        try:
            config_text = path.read_text(encoding="utf-8")
            document = (
                next((item for item in self.generator_documents if item.id == document_id), None)
                if document_id
                else self._active_generator_document()
            )
            if document is None:
                document = new_generator_document(
                    unique_generator_name(path.stem, self.generator_documents),
                    config_text,
                )
                self.generator_documents = (*self.generator_documents, document)
                self.active_generator_document_id = document.id
            else:
                self.generator_documents = tuple(
                    with_ath_source_text(item, config_text) if item.id == document.id else item
                    for item in self.generator_documents
                )
            self._rebuild_generator_document_tabs()
            self.status_label.setText(f"Imported {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    @Slot()
    def new_project(self) -> None:
        if not self._confirm_unsaved_project_changes("new_project"):
            return
        self._discard_channel_config_dialog()
        self.project_path = None
        self.project = new_project_document(project_preferences=self._current_project_preferences())
        self.generated_geometry_by_document_id = {}
        self._rebuild_generator_document_tabs()
        self.imported_radiators = ()
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
        suggested_filename = PROJECT_DEFAULT_NAME if self.project_path is None else self.project_path.name
        path = self.file_dialogs.save_file(
            self,
            "Save Project",
            PROJECT_FILE_FILTER,
            suggested_filename,
        )
        if path is None:
            return False
        return self._save_project_to_path(normalize_project_path(path))

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
        path = self.file_dialogs.open_file(
            self,
            "Open Project",
            PROJECT_FILE_FILTER,
        )
        if path is None:
            return

        self._load_project_from_path(path)

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
            project_preferences = ProjectPreferencesState.from_payload(payload.get("project_preferences"))
            if self._confirm_apply_project_preferences(project_preferences):
                self._apply_project_preferences(project_preferences)
            self._apply_project_payload(payload)
            self.project_path = path
            self._remember_recent_project(path)
            self._mark_project_clean()
            self.status_label.setText(f"Opened project {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Open project failed", str(exc))

    def _project_payload(self) -> dict:
        project_preferences = self._current_project_preferences()
        self.project.project_preferences = project_preferences
        return build_project_payload(
            generator_documents=[
                generator_document_to_payload(document, absolute_paths=True) for document in self.generator_documents
            ],
            active_generator_document_id=self.active_generator_document_id,
            imported_meshes=self._project_imported_meshes_payload(),
            stitch_imported_meshes=self.stitch_imported_meshes,
            symmetry=self.symmetry,
            source_config_by_name=self._load_source_config_by_name(),
            channel_config_by_name=self._load_channel_config_by_name(),
            project_preferences=project_preferences.to_payload(),
            physical_system=(
                None if self.project.physical_system is None else physical_system_to_dict(self.project.physical_system)
            ),
            component_channel_by_id=self.project.component_channel_by_id,
        )

    def _current_project_preferences(self) -> ProjectPreferencesState:
        return project_preferences_from_gui(
            self.preferences,
            freq_min_hz=int(self.freq_min_spin.value()),
            freq_max_hz=int(self.freq_max_spin.value()),
            freq_count=int(self.freq_count_spin.value()),
        )

    def _confirm_apply_project_preferences(self, project_preferences: ProjectPreferencesState | None) -> bool:
        if project_preferences is None or project_preferences == self._current_project_preferences():
            return False
        return (
            QMessageBox.question(
                self,
                "Project Preferences",
                "This project file contains unique application preferences. Would you like to apply them?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            == QMessageBox.Yes
        )

    def _apply_project_preferences(self, project_preferences: ProjectPreferencesState) -> None:
        self.preferences = gui_preferences_with_project_preferences(self.preferences, project_preferences)
        controls = (
            self.freq_min_spin,
            self.freq_max_spin,
            self.freq_count_spin,
            self.freq_min_slider,
            self.freq_max_slider,
            self.freq_count_slider,
        )
        blockers = [QSignalBlocker(control) for control in controls]
        self.freq_min_spin.setValue(project_preferences.freq_min_hz)
        self.freq_max_spin.setValue(project_preferences.freq_max_hz)
        self.freq_count_spin.setValue(project_preferences.freq_count)
        self.freq_min_slider.setValue(frequency_to_slider_value(project_preferences.freq_min_hz))
        self.freq_max_slider.setValue(frequency_to_slider_value(project_preferences.freq_max_hz))
        self.freq_count_slider.setValue(project_preferences.freq_count)
        del blockers
        self._save_preferences()
        self._save_frequency_settings()
        self.project.project_preferences = self._current_project_preferences()

    def _canonical_project_payload(self) -> dict:
        payload = json.loads(json.dumps(self._project_payload(), sort_keys=True))
        for document in payload.get("generator_documents", []):
            if not isinstance(document, dict):
                continue
            document.pop("artifact", None)
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
        channel_config = payload.get("channel_config_by_name", {})
        if not isinstance(channel_config, dict):
            channel_config = {}

        documents = generator_documents_from_payload(payload.get("generator_documents"))
        active_id = payload.get("active_generator_document_id")
        active_id = (
            active_id
            if any(document.id == active_id for document in documents)
            else (documents[0].id if documents else None)
        )
        symmetry = str(payload.get("symmetry", "off")).strip().lower()
        if symmetry not in {"off", "x", "xy"}:
            symmetry = "off"
        imported_meshes = self._mesh_entries_from_payload(payload.get("imported_meshes", []))
        raw_physical_system = payload.get("physical_system")
        physical_system = (
            physical_system_from_dict(raw_physical_system) if isinstance(raw_physical_system, dict) else None
        )
        if physical_system is not None and not bool(
            physical_system.metadata.get(AUTO_SEEDED_EXTERIOR_KEY, False)
        ):
            source_config = {}
        component_channels = payload.get("component_channel_by_id", {})
        if not isinstance(component_channels, dict):
            component_channels = {}
        self.project = ProjectDocument(
            generator_documents=documents,
            active_generator_document_id=active_id,
            imported_meshes=tuple(
                ImportedMeshState(
                    name=mesh.name,
                    source_file=mesh.source_file,
                    cleaned_file=mesh.cleaned_file,
                    scale_factor=mesh.scale_factor,
                    translation_mm=mesh.translation_mm,
                    enabled=mesh.enabled,
                )
                for mesh in imported_meshes
            ),
            stitch_imported_meshes=bool(
                payload.get("stitch_exterior_meshes", payload.get("stitch_imported_meshes", False))
            ),
            symmetry=symmetry,
            source_config_by_name=source_config,
            channel_config_by_name=channel_config,
            project_preferences=self._current_project_preferences(),
            physical_system=physical_system,
            component_channel_by_id={
                str(component_id): str(channel_name)
                for component_id, channel_name in component_channels.items()
            },
        )
        self.generated_geometry_by_document_id = {}
        for document in self.generator_documents:
            result = self._result_from_generator_document(document)
            if result is not None:
                self.generated_geometry_by_document_id[document.id] = self._apply_saved_source_config_to_result(
                    result,
                    generator_mesh_name(document),
                )
        self._rebuild_generator_document_tabs()
        self._disable_symmetry_if_backend_unsupported()
        self.imported_radiators = ()
        try:
            self._apply_saved_imported_source_config(self._surface_tags_for_meshes())
        except Exception:
            self.imported_radiators = ()
        self._ensure_seeded_exterior_system()

        self.project_state_changed.emit("project_loaded")
        self.solve_results_invalidated.emit("project_loaded")

    @Slot()
    def export_config(self) -> None:
        path = self.file_dialogs.save_file(
            self,
            "Export Waveguide Design",
            "Ath config files (*.cfg);;All files (*)",
            "waveguide.cfg",
        )
        if path is None:
            return

        if path.suffix == "":
            path = path.with_suffix(".cfg")

        try:
            document = self._active_generator_document()
            path.write_text("" if document is None else ath_source_text(document), encoding="utf-8")
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

        output_path = self.file_dialogs.save_file(
            self,
            f"Export {entry.title}",
            "PNG images (*.png);;All files (*)",
            entry.default_filename,
        )
        if output_path is None:
            return

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

        output_dir = self.file_dialogs.select_directory(
            self,
            "Export polar data",
        )
        if output_dir is None:
            return

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
    def export_on_axis_data(self) -> None:
        if self.live_dataset is None or self.live_dataset.solved_count == 0:
            QMessageBox.warning(self, "No on-axis data", "Run a solve before exporting on-axis data.")
            return

        try:
            self.live_dataset.set_channel_synthesis(
                self._channel_configs(),
                flat_target_reference_angle_deg=self.preferences.horizontal_normalization_angle,
            )
            _freqs, channel_names, _spl_db, _phase_deg = self.live_dataset.as_channel_on_axis_export_arrays()
        except Exception as exc:
            QMessageBox.critical(self, "Export on-axis data failed", str(exc))
            return

        if channel_names.size == 1:
            output_target = self.file_dialogs.save_file(
                self,
                "Export on-axis data",
                "Text files (*.txt);;All files (*)",
                default_on_axis_filename(str(channel_names[0])),
            )
        else:
            output_target = self.file_dialogs.select_directory(
                self,
                "Export on-axis channel data",
            )
        if output_target is None:
            return

        try:
            written = export_on_axis_text_files(self.live_dataset, output_target)
            if len(written) == 1:
                self.status_label.setText(f"Exported on-axis data to {written[0]}")
            else:
                self.status_label.setText(f"Exported {len(written)} on-axis channel files to {output_target}")
        except Exception as exc:
            QMessageBox.critical(self, "Export on-axis data failed", str(exc))

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
                raw_balloon_data_provider=lambda: (
                    None if self.live_dataset is None else self.live_dataset.as_balloon_raw_bundle()
                ),
                file_dialog_service=self.file_dialogs,
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
        checked_server_health = None
        if (
            dialog.server_health_payload is not None
            and dialog.server_health_url == preferences.solve_server_url.rstrip("/")
            and dialog.server_health_access_token == dialog.server_access_token_edit.text().strip()
        ):
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
            dialog.deleteLater()
            return

        access_token = dialog.server_access_token_edit.text().strip()
        credential_persisted = dialog.persist_server_access_token()
        dialog.deleteLater()
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
        self.project.project_preferences = self._current_project_preferences()
        symmetry_disabled = self._disable_symmetry_if_backend_unsupported()
        QTimer.singleShot(0, self._apply_theme)
        self.mesh_state_changed.emit("preferences_changed")
        if symmetry_disabled or preferences_require_solve_invalidation(previous_preferences, self.preferences):
            self.solve_results_invalidated.emit("preferences_changed")
        elif preferences_require_visualization_refresh(previous_preferences, self.preferences):
            self.visualization_settings_changed.emit("preferences_changed")
        self.status_label.setText("Preferences updated")
        if access_token and not credential_persisted:
            QMessageBox.warning(
                self,
                "Server access token",
                "The operating system credential vault is unavailable. "
                "The server access token will work for this session but could not be saved.",
            )

    @Slot()
    def open_diagnostics(self) -> None:
        dialog = DiagnosticsDialog(
            self.preferences,
            self,
            context_provider=self._diagnostic_context,
        )
        dialog.exec()

    def _diagnostic_context(self) -> dict[str, object]:
        backend = backend_info(self.preferences.solve_backend)
        backend_details: dict[str, object] = {
            "id": backend.backend_id,
            "label": backend.label,
            "remote": backend.capabilities.is_remote,
            "supports symmetry": backend.capabilities.supports_symmetry,
            "supports spherical sampling": backend.capabilities.supports_spherical_sampling,
            "supports channel resynthesis": backend.capabilities.supports_channel_resynthesis,
        }
        if backend.capabilities.is_remote:
            backend_details["server health"] = (
                "reachable (cached)" if self._server_health_matches_preferences() else "not confirmed"
            )
            if self._server_health_matches_preferences() and self.server_health_payload is not None:
                backend_details["server solver"] = (
                    self.server_health_payload.get("solver_label")
                    or self.server_health_payload.get("solver")
                    or "unknown"
                )

        geometry_state = self.geometry_controller.state
        solve_state = self.solve_controller.state
        operations: dict[str, object] = {
            "geometry": {
                "phase": geometry_state.phase.value,
                "message": geometry_state.message or "none",
                "last error": self.geometry_controller.last_error or "none",
            },
            "solve": {
                "phase": solve_state.phase.value,
                "message": solve_state.message or "none",
                "last error": self.solve_controller.last_error or "none",
            },
        }
        completion = self.solve_controller.last_completion
        if completion is not None:
            solve_details = operations["solve"]
            assert isinstance(solve_details, dict)
            solve_details.update(
                {
                    "solved frequencies": completion.solved_count,
                    "requested frequencies": completion.expected_count,
                    "elapsed seconds": round(completion.elapsed_s, 3),
                }
            )

        enabled_generated_meshes = sum(
            1
            for script in self.generator_documents
            if script.mesh_enabled and script.id in self.generated_geometry_by_document_id
        )
        enabled_imported_meshes = sum(1 for mesh in self.imported_meshes if mesh.enabled)
        result_details: dict[str, object] = {
            "solved frequencies": 0 if self.live_dataset is None else self.live_dataset.solved_count,
        }
        context: dict[str, object] = {
            "backend": backend_details,
            "project": {
                "file": self.project_path.name if self.project_path is not None else "unsaved",
                "modified": self._has_unsaved_project_changes(),
                "waveguide designs": len(self.generator_documents),
                "enabled meshes": enabled_generated_meshes + enabled_imported_meshes,
                "imported meshes": len(self.imported_meshes),
                "radiators": len(self._all_radiators()),
                "channels": len(self._channel_configs()),
                "symmetry": self.symmetry,
                "stitch exterior meshes": self.stitch_imported_meshes,
                "frequency minimum Hz": int(self.freq_min_spin.value()),
                "frequency maximum Hz": int(self.freq_max_spin.value()),
                "frequency count": int(self.freq_count_spin.value()),
            },
            "operations": operations,
            "results": result_details,
        }

        if self.live_dataset is not None and self.live_dataset.results:
            latest_frequency = next(reversed(self.live_dataset.results))
            latest_result = self.live_dataset.results[latest_frequency]
            result_details.update(
                {
                    "latest frequency Hz": round(float(latest_result.freq_hz), 3),
                    "latest assembly seconds": round(float(latest_result.timings.assembly_s), 3),
                    "latest solve seconds": round(float(latest_result.timings.solve_s), 3),
                    "latest field seconds": round(float(latest_result.timings.field_s), 3),
                }
            )
            if latest_result.diagnostics is not None:
                result_details["latest convergence info"] = latest_result.diagnostics.convergence_info
                result_details["latest solver message"] = latest_result.diagnostics.message or "none"

        return context

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
            file_dialog_service=self.file_dialogs,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return

        meshes = dialog.meshes()
        replaced_mesh_names = dialog.replaced_mesh_names()
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
            if replaced_mesh_names:
                if self.project.physical_system is not None:
                    inspected_meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
                    self.project.physical_system = sync_physical_system_meshes(
                        self.project.physical_system,
                        inspected_meshes,
                    )
                self._apply_saved_imported_source_config(self._surface_tags_for_meshes())
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
    def open_system_config(self) -> None:
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries())
            symmetry_analysis_meshes = inspect_system_meshes(
                self._mesh_config_dialog_entries_for_symmetry(self.symmetry)
            )
        except Exception as exc:
            QMessageBox.critical(self, "System", f"Could not inspect the enabled meshes:\n{exc}")
            return
        if not meshes:
            QMessageBox.warning(self, "System", "Enable at least one mesh before configuring the system.")
            return
        self._ensure_seeded_exterior_system()
        system = self.project.physical_system
        if system is not None:
            system = sync_physical_system_meshes(system, meshes)
        dialog = SystemConfigDialog(
            meshes,
            system,
            tuple(channel.name for channel in self._channel_configs()),
            self.project.component_channel_by_id,
            self,
            stitch_exterior_meshes=self.stitch_imported_meshes,
            interface_output_root=self._mesh_service().output_root,
            symmetry_mode=self.symmetry,
            symmetry_analysis_meshes=symmetry_analysis_meshes,
        )
        dialog.systemApplied.connect(self._apply_system_config)
        dialog.exec()

    @Slot(object)
    def _apply_system_config(self, configuration) -> None:
        mesh_file_overrides = dict(getattr(configuration, "mesh_file_overrides_by_name", {}))
        updated_imported_meshes = _mesh_entries_with_file_overrides(
            self.imported_meshes,
            mesh_file_overrides,
        )
        if (
            configuration.system == self.project.physical_system
            and configuration.component_channel_by_id == self.project.component_channel_by_id
            and configuration.stitch_exterior_meshes == self.stitch_imported_meshes
            and updated_imported_meshes == self.imported_meshes
        ):
            self.status_label.setText("System unchanged")
            return
        self.imported_meshes = updated_imported_meshes
        metadata = dict(configuration.system.metadata)
        metadata.pop(AUTO_SEEDED_EXTERIOR_KEY, None)
        self.project.physical_system = replace(configuration.system, metadata=metadata)
        self.project.component_channel_by_id = dict(configuration.component_channel_by_id)
        self.project.source_config_by_name = {}
        self.stitch_imported_meshes = bool(configuration.stitch_exterior_meshes)
        reason = "system_interface_mesh_built" if mesh_file_overrides else "system_config_changed"
        self.project_state_changed.emit(reason)
        self.solve_results_invalidated.emit("system_config_changed")
        self.status_label.setText("System updated")
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
        for document_id, result in tuple(self.generated_geometry_by_document_id.items()):
            self.generated_geometry_by_document_id[document_id] = replace(
                result,
                radiators=tuple(
                    radiator if radiator.channel in valid_names else replace(radiator, channel=fallback)
                    for radiator in result.radiators
                ),
            )
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
    def generate_geometry(self) -> None:
        if self.geometry_controller.active or self.solve_controller.active:
            return
        document = self._active_generator_document()
        if document is None:
            QMessageBox.warning(self, "No waveguide design", "Add a waveguide design before generating.")
            return
        mesh_name = generator_mesh_name(document)
        case_name = f"{mesh_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{document.id}"
        run_root = GENERATED_GEOMETRY_ROOT
        provider_options = {}
        try:
            provider = generator_info(document.provider_id)
            if document.provider_id == ATH_PROVIDER_ID:
                self._ensure_ath_runtime_config()
                provider_options["ath_exe"] = str(self._find_ath_exe())
        except Exception as exc:
            self.status_label.setText("Generate failed")
            QMessageBox.critical(self, "Geometry generation failed", str(exc))
            return

        self.solve_results_invalidated.emit("geometry_generation_started")
        self.status_label.setText(f"Generating {document.name} with {provider.label}...")
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.system_config_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.export_on_axis_data_action.setEnabled(False)
        self._set_contour_button_states()
        QApplication.setOverrideCursor(Qt.WaitCursor)

        self.geometry_controller.start(
            GenerationRequest(
                provider_id=document.provider_id,
                document_id=document.id,
                mesh_name=mesh_name,
                source=document.source,
                run_root=run_root,
                case_name=case_name,
                provider_options=provider_options,
            )
        )
        QTimer.singleShot(3000, self._enable_geometry_cancel_if_active)

    @Slot()
    def _enable_geometry_cancel_if_active(self) -> None:
        if self.geometry_controller.state.phase == OperationPhase.RUNNING:
            self.cancel_button.setEnabled(True)

    @Slot(object)
    def _on_geometry_generated(self, completed: GenerationCompleted) -> None:
        document_id = completed.request.document_id
        mesh_name = completed.request.mesh_name
        result = self._apply_saved_source_config_to_result(completed.result, mesh_name)
        assert result is not None
        self.generated_geometry_by_document_id[document_id] = result
        self.generator_documents = replace_generator_document(
            self.generator_documents,
            document_id,
            artifact=result.to_reference(),
        )
        self._ensure_seeded_exterior_system()
        self.mesh_state_changed.emit("geometry_generated")
        self.status_label.setText(f"Generated and cleaned {result.output_dir}")
        self._show_mesh_quality_warning(result)

    @Slot(str)
    def _on_geometry_generation_failed(self, message: str) -> None:
        self.status_label.setText("Generate failed")
        QMessageBox.critical(self, "Geometry generation failed", message)

    @Slot()
    def _on_geometry_generation_cancelled(self) -> None:
        self.status_label.setText("Geometry generation stopped")

    @Slot()
    def _on_geometry_generation_finished(self) -> None:
        QApplication.restoreOverrideCursor()
        self.solve_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.mesh_config_button.setEnabled(True)
        self.channel_config_button.setEnabled(True)
        self.system_config_button.setEnabled(self._has_solver_meshes())
        self.cancel_button.setEnabled(False)

    @Slot()
    def start_solve(self) -> None:
        if self.geometry_controller.active or self.solve_controller.active:
            return
        if not self._has_solver_meshes():
            QMessageBox.warning(self, "No mesh", "Enable at least one generated or imported mesh before solving.")
            return
        self._ensure_seeded_exterior_system()
        if self.project.physical_system is not None:
            try:
                solve_kind = infer_physical_solve_kind(self.project.physical_system)
            except ValueError as exc:
                QMessageBox.warning(self, "System solve", str(exc))
                return
            if solve_kind == PhysicalSolveKind.EXTERIOR_BEM:
                self._start_exterior_system_solve()
            else:
                self._start_coupled_system_solve()
            return
        radiators = self._all_radiators()
        if not radiators:
            QMessageBox.warning(
                self,
                "No driven surfaces",
                "Open System and add a prescribed-velocity component to a moving boundary.",
            )
            return
        if self._disable_symmetry_if_backend_unsupported():
            self.mesh_state_changed.emit("symmetry_disabled_for_backend")

        try:
            assembly = self._prepare_mesh_assembly(radiators)
            mesh_configs = assembly.mesh_configs
            radiators = assembly.radiators
        except Exception as exc:
            self._show_stitch_or_generic_error("Imported mesh preparation failed", exc)
            return
        freq_min = float(min(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_max = float(max(self.freq_min_spin.value(), self.freq_max_spin.value()))
        freq_count = int(self.freq_count_spin.value())
        channels = self._channels_for_solver_radiators(radiators)
        try:
            prepared_simulation = self.simulation_assembler.prepare(
                mesh_configs=mesh_configs,
                radiators=radiators,
                channels=channels,
                parameters=SimulationParameters(
                    freq_min_hz=freq_min,
                    freq_max_hz=freq_max,
                    freq_count=freq_count,
                    observation_distance_m=self.preferences.polar_observation_distance_m,
                    polar_angle_step_deg=self.preferences.polar_angle_step_deg,
                    use_burton_miller=self.preferences.use_burton_miller,
                    gmres_tolerance=self.preferences.gmres_tolerance,
                    normalized_channel_correction=self.preferences.normalized_channel_correction,
                    horizontal_normalization_angle_deg=self.preferences.horizontal_normalization_angle,
                    spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
                    spherical_sampling_points=balloon_sampling_points(self.preferences.balloon_angle_precision_deg),
                    symmetry=self.symmetry,
                ),
            )
        except SymmetryValidationError as exc:
            QMessageBox.warning(self, "Symmetry validation failed", str(exc))
            return
        config = prepared_simulation.config
        ordered_freqs = prepared_simulation.ordered_frequencies

        self.live_dataset = None
        self._clear_plots()
        self._apply_last_completed_plot_comparison()
        self.balloon_plot_action.setEnabled(False)
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.system_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.export_on_axis_data_action.setEnabled(False)
        self._set_contour_button_states()
        self.status_label.setText("Initializing Solver...")
        self.solve_controller.start(
            SolveRequest(
                config=config,
                ordered_frequencies=ordered_freqs,
                worker_count=1,
                backend_id=self.preferences.solve_backend,
                server_url=self.preferences.solve_server_url,
                server_access_token=load_server_access_token(self.preferences.solve_server_url),
            )
        )

    def _start_exterior_system_solve(self) -> None:
        if self._disable_symmetry_if_backend_unsupported():
            self.mesh_state_changed.emit("symmetry_disabled_for_backend")
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries_for_symmetry(self.symmetry))
            system = sync_physical_system_meshes(self.project.physical_system, meshes)
            self.project.physical_system = system
            inputs = exterior_bem_inputs(
                system,
                component_channel_by_id=self.project.component_channel_by_id,
                symmetry_mode=self.symmetry,
            )
            mesh_configs, radiators = self._mesh_service().prepare_mesh_configs(
                inputs.mesh_configs,
                inputs.radiators,
                stitch_meshes_enabled=self.stitch_imported_meshes,
                stitch_tolerance_mm=self.preferences.stitch_tolerance_mm,
                symmetry=self.symmetry,
            )
            prepared_simulation = self.simulation_assembler.prepare(
                mesh_configs=mesh_configs,
                radiators=radiators,
                channels=self._channels_for_solver_radiators(radiators),
                parameters=SimulationParameters(
                    freq_min_hz=float(self.freq_min_spin.value()),
                    freq_max_hz=float(self.freq_max_spin.value()),
                    freq_count=int(self.freq_count_spin.value()),
                    observation_distance_m=self.preferences.polar_observation_distance_m,
                    polar_angle_step_deg=self.preferences.polar_angle_step_deg,
                    use_burton_miller=self.preferences.use_burton_miller,
                    gmres_tolerance=self.preferences.gmres_tolerance,
                    normalized_channel_correction=self.preferences.normalized_channel_correction,
                    horizontal_normalization_angle_deg=self.preferences.horizontal_normalization_angle,
                    spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
                    spherical_sampling_points=balloon_sampling_points(
                        self.preferences.balloon_angle_precision_deg
                    ),
                    symmetry=self.symmetry,
                ),
            )
        except (ValueError, OSError, SymmetryValidationError) as exc:
            self._show_stitch_or_generic_error("Exterior system preparation failed", exc)
            return

        self.live_dataset = None
        self._clear_plots()
        self._apply_last_completed_plot_comparison()
        self.balloon_plot_action.setEnabled(False)
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.system_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.export_on_axis_data_action.setEnabled(False)
        self._set_contour_button_states()
        self.status_label.setText("Initializing exterior solver...")
        self.solve_controller.start(
            SolveRequest(
                config=prepared_simulation.config,
                ordered_frequencies=prepared_simulation.ordered_frequencies,
                worker_count=1,
                backend_id=self.preferences.solve_backend,
                server_url=self.preferences.solve_server_url,
                server_access_token=load_server_access_token(self.preferences.solve_server_url),
            )
        )

    def _start_coupled_system_solve(self) -> None:
        try:
            meshes = inspect_system_meshes(self._mesh_config_dialog_entries_for_symmetry(self.symmetry))
            system = sync_physical_system_meshes(self.project.physical_system, meshes)
            self.project.physical_system = system
            prepared = prepare_coupled_ui_solve(
                system,
                freq_min_hz=float(self.freq_min_spin.value()),
                freq_max_hz=float(self.freq_max_spin.value()),
                freq_count=int(self.freq_count_spin.value()),
                observation_distance_m=self.preferences.polar_observation_distance_m,
                polar_angle_step_deg=self.preferences.polar_angle_step_deg,
                spherical_sampling_enabled=self.preferences.spherical_sampling_enabled,
                spherical_sampling_points=balloon_sampling_points(self.preferences.balloon_angle_precision_deg),
                component_channel_by_id=self.project.component_channel_by_id,
                backend_id=self.preferences.solve_backend,
                symmetry_mode=self.symmetry,
            )
        except Exception as exc:
            QMessageBox.warning(self, "Coupled solve", str(exc))
            return

        self.live_dataset = None
        self._clear_plots()
        self._apply_last_completed_plot_comparison()
        self.balloon_plot_action.setEnabled(False)
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        self.solve_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.mesh_config_button.setEnabled(False)
        self.system_config_button.setEnabled(False)
        self.channel_config_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.export_on_axis_data_action.setEnabled(False)
        self._set_contour_button_states()
        self.status_label.setText("Initializing coupled solver...")
        self.solve_controller.start(prepared)

    @Slot()
    def cancel_current_operation(self) -> None:
        if self.geometry_controller.active:
            self.cancel_geometry_generation()
            return
        self.cancel_solve()

    @Slot()
    def cancel_geometry_generation(self) -> None:
        self.cancel_button.setEnabled(False)
        if self.geometry_controller.active:
            self.geometry_controller.cancel()
            self.status_label.setText("Stop requested; ending geometry generation...")

    @Slot()
    def cancel_solve(self) -> None:
        if self.solve_controller.active:
            self.solve_controller.cancel()
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
        self._request_live_plot_refresh()

    def _request_live_plot_refresh(self) -> None:
        self._live_plot_refresh_dirty = True
        if not self._live_plot_refresh_timer.isActive():
            self._live_plot_refresh_timer.start()

    @Slot()
    def _flush_live_plot_refresh(self) -> None:
        if not self._live_plot_refresh_dirty:
            return
        self._live_plot_refresh_dirty = False
        self._refresh_plots(active_only=True)

    def _cancel_live_plot_refresh(self) -> None:
        self._live_plot_refresh_dirty = False
        self._live_plot_refresh_timer.stop()

    @Slot(str)
    def _on_solve_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Solve failed", message)
        self.status_label.setText("Solve failed")

    @Slot(object)
    def _on_solve_finished(self, completion: SolveCompletion) -> None:
        self._cancel_live_plot_refresh()
        self.solve_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.mesh_config_button.setEnabled(True)
        self.channel_config_button.setEnabled(True)
        self.system_config_button.setEnabled(self._has_solver_meshes())
        self.cancel_button.setEnabled(False)
        elapsed_s = completion.elapsed_s
        if self.live_dataset is not None and self.live_dataset.solved_count > 0:
            solved_count = self.live_dataset.solved_count
            solve_completed = completion.completed
            self._use_final_isobar_resolution = solve_completed
            if solve_completed:
                self.status_label.setText("Rendering final high-resolution plots...")
            refreshed_dataset = None
            if self.preferences.live_plot_streaming or solve_completed:
                refreshed_dataset = self._refresh_plots()
            if solve_completed:
                if refreshed_dataset is None:
                    refreshed_dataset = self._prepared_live_plot_dataset(
                        angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES,
                        freq_samples=FINAL_ISOBAR_FREQ_SAMPLES,
                    )
                if refreshed_dataset is not None:
                    self._last_completed_visualization_dataset = refreshed_dataset.snapshot()
            self._final_isobar_plots_rendered = solve_completed and bool(self._visible_isobar_plots())
            self._set_export_plot_actions_enabled(True)
            self.export_polar_data_action.setEnabled(True)
            self.export_on_axis_data_action.setEnabled(self.live_dataset.supports_channel_resynthesis)
            self.balloon_plot_action.setEnabled(self.live_dataset.as_balloon_raw_bundle() is not None)
            self._set_contour_button_states()
            elapsed_text = f" in {elapsed_s:.1f} s"
            if completion.phase == OperationPhase.CANCELLED:
                self.status_label.setText(f"Solve stopped: {self.live_dataset.solved_count} frequencies{elapsed_text}")
                return
            if completion.phase == OperationPhase.FAILED:
                self.status_label.setText(f"Solve failed after {solved_count} frequencies{elapsed_text}")
                return
            self.status_label.setText(f"Solve complete: {solved_count} frequencies{elapsed_text}")
        elif completion.phase == OperationPhase.CANCELLED:
            self.status_label.setText("Solve stopped")
        self._set_contour_button_states()

    def _clear_plots(self) -> None:
        self._cancel_live_plot_refresh()
        self.live_dataset = None
        self._use_final_isobar_resolution = False
        self._final_isobar_plots_rendered = False
        for entry in self.plot_entries:
            entry.widget._draw_empty()
        self._set_export_plot_actions_enabled(False)
        self.export_polar_data_action.setEnabled(False)
        self.export_on_axis_data_action.setEnabled(False)
        self.balloon_plot_action.setEnabled(False)
        self._set_contour_button_states()

    def _apply_last_completed_plot_comparison(self) -> None:
        dataset = self._last_completed_visualization_dataset
        if dataset is None:
            for entry in self.plot_entries:
                entry.widget.clear_comparison_plot()
            return
        isobar = dataset.isobar
        options = {
            "shading": FINAL_ISOBAR_SHADING,
            "contour_step_db": self.preferences.isobar_contour_step_db,
        }
        self.horizontal_plot.set_comparison_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.horizontal_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            **options,
        )
        self.vertical_plot.set_comparison_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.vertical_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            **options,
        )
        impedance = dataset.impedance
        self.impedance_plot.set_comparison_plot(
            impedance.freq_hz,
            impedance.radiator_names,
            impedance.real,
            impedance.imaginary,
        )
        response = dataset.response
        self.on_axis_plot.set_comparison_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.channel_on_axis_names,
            response.channel_on_axis_spl_db,
        )
        self.spinorama_plot.set_comparison_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.vertical_spl_db,
            horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
            vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
        )

    def _clear_plot_comparison_history(self) -> None:
        self._last_completed_visualization_dataset = None
        for entry in self.plot_entries:
            entry.widget.clear_comparison_plot()

    def _set_plot_visible(self, plot_id: str, visible: bool) -> None:
        for entry in self.plot_entries:
            if entry.plot_id != plot_id:
                continue
            dock = self.plot_docks.get(plot_id)
            if dock is not None and dock.isVisible() != visible:
                dock.setVisible(visible)
            if visible:
                if self.solve_controller.active:
                    if self.preferences.live_plot_streaming:
                        self._request_live_plot_refresh()
                else:
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
            if self.solve_controller.active:
                if self.preferences.live_plot_streaming:
                    self._request_live_plot_refresh()
            else:
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
    ) -> VisualizationProjection | None:
        if self.live_dataset is None:
            return None
        if angle_samples is None:
            angle_samples = live_plot_angle_samples(self.preferences.live_plot_quality)
        if freq_samples is None:
            freq_samples = live_plot_freq_samples(self.preferences.live_plot_quality)
        return self.result_projection_service.prepare(
            self.live_dataset,
            self._channel_configs(),
            ProjectionOptions(
                angle_samples=angle_samples,
                freq_samples=freq_samples,
                octave_smoothing=self.preferences.polar_smoothing,
                horizontal_reference_angle_deg=self.preferences.horizontal_normalization_angle,
                vertical_reference_angle_deg=self.preferences.vertical_normalization_angle,
                spin_horizontal_reference_angle_deg=self.preferences.spin_horizontal_reference_angle,
                spin_vertical_reference_angle_deg=self.preferences.spin_vertical_reference_angle,
                min_db=self.preferences.spl_min_db,
                max_db=self.preferences.spl_max_db,
            ),
        )

    def _plot_entry_is_actively_visible(self, entry: PlotEntry) -> bool:
        dock = self.plot_docks.get(entry.plot_id)
        if dock is None or dock.isHidden():
            return False
        return not entry.widget.visibleRegion().isEmpty()

    def _refresh_plots(self, *, active_only: bool = False) -> VisualizationProjection | None:
        visible_entries = [
            entry
            for entry in self.plot_entries
            if (dock := self.plot_docks.get(entry.plot_id)) is not None
            and not dock.isHidden()
            and (not active_only or self._plot_entry_is_actively_visible(entry))
        ]
        if not visible_entries:
            return None

        dataset = self._prepared_live_plot_dataset(
            angle_samples=FINAL_ISOBAR_ANGLE_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_angle_samples(self.preferences.live_plot_quality),
            freq_samples=FINAL_ISOBAR_FREQ_SAMPLES
            if self._use_final_isobar_resolution
            else live_plot_freq_samples(self.preferences.live_plot_quality),
        )
        if dataset is None:
            return None

        for entry in visible_entries:
            entry.update(dataset)
        return dataset

    def _update_horizontal_plot(self, dataset: VisualizationProjection) -> None:
        isobar = dataset.isobar
        self.horizontal_plot.update_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.horizontal_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
            contour_step_db=self.preferences.isobar_contour_step_db,
        )

    def _update_vertical_plot(self, dataset: VisualizationProjection) -> None:
        isobar = dataset.isobar
        self.vertical_plot.update_plot(
            isobar.freq_hz,
            isobar.angle_deg,
            isobar.vertical_db,
            isobar.clip_min_db,
            isobar.clip_max_db,
            shading=FINAL_ISOBAR_SHADING if self._use_final_isobar_resolution else LIVE_ISOBAR_SHADING,
            contour_step_db=self.preferences.isobar_contour_step_db,
        )

    def _update_impedance_plot(self, dataset: VisualizationProjection) -> None:
        impedance = dataset.impedance
        self.impedance_plot.update_plot(
            impedance.freq_hz,
            impedance.radiator_names,
            impedance.real,
            impedance.imaginary,
        )

    def _update_on_axis_plot(self, dataset: VisualizationProjection) -> None:
        response = dataset.response
        self.on_axis_plot.update_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.channel_on_axis_names,
            response.channel_on_axis_spl_db,
        )

    def _update_spinorama_plot(self, dataset: VisualizationProjection) -> None:
        response = dataset.response
        self.spinorama_plot.update_plot(
            response.freq_hz,
            response.angle_deg,
            response.horizontal_spl_db,
            response.vertical_spl_db,
            horizontal_reference_angle_deg=response.spin_horizontal_reference_angle_deg,
            vertical_reference_angle_deg=response.spin_vertical_reference_angle_deg,
        )
