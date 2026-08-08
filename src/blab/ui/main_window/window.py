"""The MainWindow shell: construction, Qt overrides, and project state properties."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QEvent, QSettings, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget,
    QLabel,
    QMainWindow,
    QPushButton,
    QTabWidget,
    QWidget,
)

from blab import __version__
from blab.config import RadiatorConfig
from blab.generators.base import GeneratedGeometry, GeneratorDocument
from blab.live import (
    LiveSolveDataset,
)
from blab.ui.dialogs import (
    ChannelConfigDialog,
    MeshDialogEntry,
)
from blab.ui.file_dialogs import FileDialogService
from blab.ui.main_window.backend_health import BackendHealthController
from blab.ui.main_window.channels import ChannelsMixin
from blab.ui.main_window.constants import (
    LIVE_PLOT_REFRESH_INTERVAL_MS,
)
from blab.ui.main_window.dialog_actions import DialogActionsMixin
from blab.ui.main_window.exports import ExportsMixin
from blab.ui.main_window.generator_documents import GeneratorDocumentsMixin
from blab.ui.main_window.geometry_store import GeometryStore
from blab.ui.main_window.geometry_workflow import GeometryWorkflowController
from blab.ui.main_window.mesh_workflow import MeshWorkflowMixin
from blab.ui.main_window.panels import PanelVisibilityMixin
from blab.ui.main_window.plot_presenter import PlotPresenterMixin
from blab.ui.main_window.preferences import PreferencesMixin
from blab.ui.main_window.project_session import ProjectSession
from blab.ui.main_window.project_workflow import ProjectWorkflowController
from blab.ui.main_window.radiators import RadiatorsMixin
from blab.ui.main_window.solve_session import SolveSession
from blab.ui.main_window.solve_workflow import SolveWorkflowController
from blab.ui.main_window.state_sync import StateSyncMixin
from blab.ui.main_window.view_builder import ViewBuilderMixin
from blab.ui.main_window_widgets import (
    PlotEntry,
)
from blab.ui.mesh_assembly import (
    MeshAssemblyService,
)
from blab.ui.operation_controllers import (
    GeometryController,
    SolveController,
)
from blab.ui.plots import (
    AUDIO_FREQ_MAX_HZ,
    AUDIO_FREQ_MIN_HZ,
    FREQ_SLIDER_STEPS,
    ImpedanceCanvas,
    IsobarCanvas,
    OnAxisResponseCanvas,
    SpinoramaCanvas,
    frequency_to_slider_value,
)
from blab.ui.project_state import (
    ImportedMeshState,
    ProjectDocument,
)
from blab.ui.result_projection import (
    ResultProjectionService,
    VisualizationProjection,
)
from blab.ui.settings import (
    SETTINGS_APP,
    SETTINGS_ORG,
    load_syntax_highlighting_enabled,
    settings_int,
)
from blab.ui.simulation_assembler import SimulationAssembler


class MainWindow(
    ViewBuilderMixin,
    PanelVisibilityMixin,
    PreferencesMixin,
    GeneratorDocumentsMixin,
    StateSyncMixin,
    MeshWorkflowMixin,
    RadiatorsMixin,
    ChannelsMixin,
    ExportsMixin,
    DialogActionsMixin,
    PlotPresenterMixin,
    QMainWindow,
):
    mesh_state_changed = Signal(str)

    project_state_changed = Signal(str)

    solve_results_invalidated = Signal(str)

    visualization_settings_changed = Signal(str)

    def _project_document(self) -> ProjectDocument:
        return self._project_session().document

    def _project_session(self) -> ProjectSession:
        session = getattr(self, "project_session", None)
        if session is None:
            session = ProjectSession()
            self.project_session = session
        return session

    # The open project, its file and its save state live on ProjectSession;
    # these properties keep the historical attribute names working.

    @property
    def project(self) -> ProjectDocument:
        return self._project_session().document

    @project.setter
    def project(self, value: ProjectDocument) -> None:
        self._project_session().document = value

    @property
    def project_path(self) -> Path | None:
        return self._project_session().path

    @project_path.setter
    def project_path(self, value: Path | None) -> None:
        self._project_session().path = value

    @property
    def _project_clean_payload(self) -> dict | None:
        return self._project_session().clean_payload

    @_project_clean_payload.setter
    def _project_clean_payload(self, value: dict | None) -> None:
        self._project_session().clean_payload = value

    def _geometry_store(self) -> GeometryStore:
        store = getattr(self, "geometry_store", None)
        if store is None:
            store = GeometryStore()
            self.geometry_store = store
        return store

    def _solve_session(self) -> SolveSession:
        session = getattr(self, "solve_session", None)
        if session is None:
            session = SolveSession()
            self.solve_session = session
        return session

    # The live solve results are shared by seven modules; they live on
    # SolveSession, and these properties keep the historical attribute names
    # working for every reader.

    @property
    def live_dataset(self) -> LiveSolveDataset | None:
        return self._solve_session().live_dataset

    @live_dataset.setter
    def live_dataset(self, value: LiveSolveDataset | None) -> None:
        self._solve_session().live_dataset = value

    @property
    def _use_final_isobar_resolution(self) -> bool:
        return self._solve_session().use_final_isobar_resolution

    @_use_final_isobar_resolution.setter
    def _use_final_isobar_resolution(self, value: bool) -> None:
        self._solve_session().use_final_isobar_resolution = bool(value)

    @property
    def _final_isobar_plots_rendered(self) -> bool:
        return self._solve_session().final_isobar_plots_rendered

    @_final_isobar_plots_rendered.setter
    def _final_isobar_plots_rendered(self, value: bool) -> None:
        self._solve_session().final_isobar_plots_rendered = bool(value)

    @property
    def _last_completed_visualization_dataset(self) -> VisualizationProjection | None:
        return self._solve_session().last_completed_visualization

    @_last_completed_visualization_dataset.setter
    def _last_completed_visualization_dataset(self, value: VisualizationProjection | None) -> None:
        self._solve_session().last_completed_visualization = value

    @property
    def generated_geometry_by_document_id(self) -> dict[str, GeneratedGeometry]:
        return self._geometry_store().generated_by_document_id

    @generated_geometry_by_document_id.setter
    def generated_geometry_by_document_id(self, value: dict[str, GeneratedGeometry]) -> None:
        self._geometry_store().generated_by_document_id = dict(value)

    @property
    def imported_radiators(self) -> tuple[RadiatorConfig, ...]:
        return self._geometry_store().imported_radiators

    @imported_radiators.setter
    def imported_radiators(self, value: tuple[RadiatorConfig, ...]) -> None:
        self._geometry_store().imported_radiators = tuple(value)

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
        # Needed before the design tabs are built.
        self.syntax_highlighting_enabled = load_syntax_highlighting_enabled(self.settings)
        self.project_session = ProjectSession()
        self.simulation_assembler = SimulationAssembler()
        self.mesh_assembly_service = MeshAssemblyService(Path.cwd() / "runs" / "imported_meshes")
        self.result_projection_service = ResultProjectionService()
        self.geometry_controller = GeometryController(self)
        self.solve_controller = SolveController(self)
        self.backend_health = BackendHealthController(self, preferences=lambda: self.preferences)
        self.backend_health.capability_changed.connect(self.mesh_state_changed)
        self._apply_theme()
        self.geometry_store = GeometryStore()
        self.solve_session = SolveSession()
        self.solve_workflow = SolveWorkflowController(
            self,
            view=self,
            plots=self,
            inputs=self,
            session=self.solve_session,
            project=self._project_document,
            preferences=lambda: self.preferences,
            assembler=self.simulation_assembler,
            geometry_controller=self.geometry_controller,
            solve_controller=self.solve_controller,
        )
        self.solve_workflow.mesh_state_changed.connect(self.mesh_state_changed)
        self.geometry_workflow = GeometryWorkflowController(
            self,
            view=self,
            plots=self,
            inputs=self,
            geometry_controller=self.geometry_controller,
            solve_controller=self.solve_controller,
        )
        self.geometry_workflow.mesh_state_changed.connect(self.mesh_state_changed)
        self.geometry_workflow.solve_results_invalidated.connect(self.solve_results_invalidated)
        self.project_workflow = ProjectWorkflowController(
            self,
            view=self,
            inputs=self,
            session=self.project_session,
            geometry_store=self.geometry_store,
            preferences=lambda: self.preferences,
            set_preferences=self._set_preferences,
            save_preferences=lambda: self._save_preferences(),
            save_frequency_settings=lambda: self._save_frequency_settings(),
            remember_recent=lambda path: self._remember_recent_project(path),
            forget_recent=lambda path: self._remove_recent_project(path),
        )
        self.project_workflow.project_state_changed.connect(self.project_state_changed)
        self.project_workflow.solve_results_invalidated.connect(self.solve_results_invalidated)
        self.balloon_window: QWidget | None = None
        self.channel_config_dialog: ChannelConfigDialog | None = None
        self._last_imported_mesh_focus_check_at = 0.0
        self._imported_mesh_source_fingerprints: dict[str, tuple[int, int]] = {}
        self._plot_dpi_screen = None
        self._plot_dpi_window_handle = None
        self._plot_dpi_refresh_pending = False
        self._live_plot_refresh_dirty = False
        self._live_plot_refresh_timer = QTimer(self)
        self._live_plot_refresh_timer.setSingleShot(True)
        self._live_plot_refresh_timer.setInterval(LIVE_PLOT_REFRESH_INTERVAL_MS)
        self._live_plot_refresh_timer.timeout.connect(self.flush_live_refresh)
        startup("Building design editor...")
        self.editor_tabs = QTabWidget()
        self.editor_tabs.setDocumentMode(True)
        self.editor_tabs.setTabsClosable(True)
        self.editor_tabs.currentChanged.connect(self._on_active_generator_tab_changed)
        self.editor_tabs.tabCloseRequested.connect(self._remove_generator_document_at)
        self.editor_tabs.tabBar().installEventFilter(self)
        self.rebuild_generator_document_tabs()

        startup("Creating mesh preview...")
        from blab.ui.mesh_preview import MeshPreview

        self.preview = MeshPreview()
        if self.has_solver_meshes():
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
        self.system_config_button.setEnabled(self.has_solver_meshes())

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
        QTimer.singleShot(0, self.backend_health.check_on_startup)

    # -- solve workflow ----------------------------------------------------
    # Buttons, menu actions and the F5/Shift+F5 shortcuts are connected to the
    # window; these forward to the controller that now owns the behaviour.

    @Slot()
    def start_solve(self) -> None:
        self.solve_workflow.start_solve()

    @Slot()
    def cancel_current_operation(self) -> None:
        self.solve_workflow.cancel_current_operation()

    @Slot()
    def cancel_geometry_generation(self) -> None:
        self.solve_workflow.cancel_geometry_generation()

    @Slot()
    def cancel_solve(self) -> None:
        self.solve_workflow.cancel_solve()

    @Slot()
    def generate_geometry(self) -> None:
        self.geometry_workflow.generate_geometry()

    # -- project workflow --------------------------------------------------
    # File menu actions, the recent-project list and the drag-and-drop config
    # handlers are connected to the window; these forward to the controller
    # that now owns the behaviour.

    def _set_preferences(self, preferences) -> None:
        self.preferences = preferences

    @Slot()
    def new_project(self) -> None:
        self.project_workflow.new_project()

    @Slot()
    def save_project(self) -> bool:
        return self.project_workflow.save_project()

    @Slot()
    def save_project_as(self) -> bool:
        return self.project_workflow.save_project_as()

    @Slot()
    def load_project(self) -> None:
        self.project_workflow.load_project()

    @Slot()
    def open_recent_project(self, path: Path) -> None:
        self.project_workflow.open_recent_project(path)

    @Slot()
    def import_config(self) -> None:
        self.project_workflow.import_config()

    @Slot()
    def export_config(self) -> None:
        self.project_workflow.export_config()

    def import_config_path(self, path: Path, *, document_id: str | None = None) -> None:
        self.project_workflow.import_config_path(path, document_id=document_id)

    def _current_project_preferences(self):
        return self.project_workflow.current_project_preferences()

    def _has_unsaved_project_changes(self) -> bool:
        return self.project_workflow.has_unsaved_project_changes()

    def _confirm_unsaved_project_changes(self, action: str) -> bool:
        return self.project_workflow.confirm_unsaved_project_changes(action)

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange:
            self._refresh_plot_export_icons()
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            self._reload_updated_imported_meshes_on_focus()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().showEvent(event)
        self.connect_dpi_signals()

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

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._confirm_unsaved_project_changes("close"):
            event.ignore()
            return
        self._save_frequency_settings()
        self._save_preferences()
        self._save_window_state()
        super().closeEvent(event)
