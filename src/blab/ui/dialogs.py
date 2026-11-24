"""Configuration dialogs for the live solver GUI."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from blab.config import ChannelConfig, CrossoverConfig
from blab.mesh_data import MeshData
from blab.paths import APP_ROOT
from blab.project.model import ProjectPreferencesState
from blab.solvers.registry import backend_label_to_id, normalize_backend_id
from blab.ui.drag_drop import local_drop_paths
from blab.ui.file_dialogs import FileDialogService
from blab.ui.server_preferences import ServerPreferences
from blab.ui.settings import (
    FIELD_CACHE_SIZE_MAX_MB,
    FIELD_CACHE_SIZE_MIN_MB,
    FIELD_TRANSLATION_TARGET_FPS_MAX,
    FIELD_TRANSLATION_TARGET_FPS_MIN,
    GuiPreferences,
    normalize_field_cache_size_mb,
    normalize_field_translation_target_fps,
    normalize_live_plot_quality,
)

CROSSOVER_TYPE_OPTIONS = [
    ("Off", None),
    ("Butterworth 1st", ("butterworth", 1)),
    ("Butterworth 2nd", ("butterworth", 2)),
    ("Butterworth 4th", ("butterworth", 4)),
    ("Butterworth 6th", ("butterworth", 6)),
    ("Linkwitz-Riley 2nd", ("linkwitz_riley", 2)),
    ("Linkwitz-Riley 4th", ("linkwitz_riley", 4)),
    ("Linkwitz-Riley 6th", ("linkwitz_riley", 6)),
]
DONATE_QR_PATH = APP_ROOT / "assets" / "donateqr.png"
INFO_ICON_PATH = APP_ROOT / "assets" / "info-16.ico"
DONATE_URL = "https://www.paypal.com/donate/?hosted_button_id=ZVC2HAFBJNPDW"
DONATE_BLURB = (
    "Boundary Lab is free open source software. If you've found the tool helpful for your workflows and want "
    "to contribute, please consider a donation to support future development."
)


@dataclass(frozen=True)
class MeshDialogEntry:
    name: str
    source_file: str
    cleaned_file: str | None = None
    scale_factor: float = 0.001
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enabled: bool = True
    locked: bool = False
    mesh_data: MeshData | None = None
    assembly_id: str | None = None


class DonateDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Donate")

        qr_label = QLabel()
        qr_label.setAlignment(Qt.AlignCenter)
        qr_pixmap = QPixmap(str(DONATE_QR_PATH))
        if qr_pixmap.isNull():
            qr_label.setText(f"Donation QR code could not be loaded:\n{DONATE_QR_PATH}")
            qr_label.setWordWrap(True)
        else:
            qr_label.setPixmap(qr_pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        blurb_label = QLabel(DONATE_BLURB)
        blurb_label.setWordWrap(True)
        blurb_label.setAlignment(Qt.AlignCenter)

        donate_button = QPushButton("Donate with PayPal")
        donate_button.clicked.connect(self._open_donate_url)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(donate_button)
        button_row.addStretch(1)

        close_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        close_buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.addWidget(qr_label, alignment=Qt.AlignCenter)
        layout.addWidget(blurb_label)
        layout.addLayout(button_row)
        layout.addWidget(close_buttons)
        self.resize(420, 440)

    def _open_donate_url(self) -> None:
        if not QDesktopServices.openUrl(QUrl(DONATE_URL)):
            QMessageBox.warning(
                self,
                "Donation link failed",
                "Unable to open the donation page in the default browser.",
            )


def _crossover_label(crossover: CrossoverConfig) -> str:
    if crossover.type.lower() == "none" or crossover.frequency_hz is None:
        return "Off"
    for label, payload in CROSSOVER_TYPE_OPTIONS:
        if payload == (crossover.filter.lower(), crossover.order):
            return label
    return "Off"


def _build_crossover_type_combo(current_label: str) -> QComboBox:
    combo = QComboBox()
    for label, payload in CROSSOVER_TYPE_OPTIONS:
        combo.addItem(label, payload)
    combo.setCurrentText(current_label if current_label in {label for label, _ in CROSSOVER_TYPE_OPTIONS} else "Off")
    return combo


def _build_crossover_frequency_spin(frequency_hz: float | None) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(1.0, 200000.0)
    spin.setDecimals(1)
    spin.setSingleStep(10.0)
    spin.setSuffix(" Hz")
    spin.setValue(1000.0 if frequency_hz is None else float(frequency_hz))
    return spin


def _settings_section(title: str, rows: tuple[tuple[str, QWidget] | tuple[str, QWidget, str], ...]) -> QGroupBox:
    group = QGroupBox(title)
    form = QFormLayout(group)
    info_icon = QIcon(str(INFO_ICON_PATH))
    for row in rows:
        label_text, widget = row[:2]
        tooltip = row[2] if len(row) > 2 else ""
        label = QLabel(label_text)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
            label_row = QWidget()
            label_layout = QHBoxLayout(label_row)
            label_layout.setContentsMargins(0, 0, 0, 0)
            label_layout.setSpacing(4)
            icon_label = QLabel()
            icon_label.setPixmap(info_icon.pixmap(16, 16))
            icon_label.setToolTip(tooltip)
            label_layout.addWidget(icon_label)
            label_layout.addWidget(label)
            label_layout.addStretch(1)
            form.addRow(label_row, widget)
            continue
        form.addRow(label, widget)
    return group


class PreferencesDialog(QDialog):
    def __init__(self, preferences: GuiPreferences, parent: QWidget | None = None):
        super().__init__(parent)
        self._preferences = preferences
        self.setWindowTitle("Preferences")

        self.theme_combo = QComboBox()
        self.theme_options = {
            "System": "system",
            "Light": "light",
            "Dark": "dark",
        }
        self.theme_combo.addItems(self.theme_options.keys())
        theme_label = next(
            (label for label, value in self.theme_options.items() if value == preferences.theme),
            "System",
        )
        self.theme_combo.setCurrentText(theme_label)

        self.live_plot_quality_combo = QComboBox()
        self.live_plot_quality_options = {
            "Low": "low",
            "Medium": "medium",
            "High": "high",
        }
        self.live_plot_quality_combo.addItems(self.live_plot_quality_options.keys())
        live_plot_quality = normalize_live_plot_quality(preferences.live_plot_quality)
        live_plot_quality_label = next(
            (label for label, value in self.live_plot_quality_options.items() if value == live_plot_quality),
            "Medium",
        )
        self.live_plot_quality_combo.setCurrentText(live_plot_quality_label)

        self.cuda_worker_reuse_check = QCheckBox("Enabled")
        self.cuda_worker_reuse_check.setChecked(preferences.cuda_worker_reuse)
        self.live_plot_streaming_check = QCheckBox("Enabled")
        self.live_plot_streaming_check.setChecked(preferences.live_plot_streaming)
        self.live_plot_quality_combo.setEnabled(preferences.live_plot_streaming)
        self.live_plot_streaming_check.toggled.connect(self.live_plot_quality_combo.setEnabled)

        self.solve_backend_combo = QComboBox()
        self.solve_backend_options = backend_label_to_id()
        self.solve_backend_combo.addItems(self.solve_backend_options.keys())
        current_backend = normalize_backend_id(preferences.solve_backend)
        backend_label = next(
            (label for label, value in self.solve_backend_options.items() if value == current_backend),
            "BEAT Engine (CPU)",
        )
        self.solve_backend_combo.setCurrentText(backend_label)
        self.server_preferences = ServerPreferences(preferences)
        self.field_cache_size_spin = QSpinBox()
        self.field_cache_size_spin.setRange(FIELD_CACHE_SIZE_MIN_MB, FIELD_CACHE_SIZE_MAX_MB)
        self.field_cache_size_spin.setSingleStep(16)
        self.field_cache_size_spin.setSuffix(" MB")
        self.field_cache_size_spin.setValue(normalize_field_cache_size_mb(preferences.field_cache_size_mb))

        self.field_translation_target_fps_spin = QDoubleSpinBox()
        self.field_translation_target_fps_spin.setRange(
            FIELD_TRANSLATION_TARGET_FPS_MIN,
            FIELD_TRANSLATION_TARGET_FPS_MAX,
        )
        self.field_translation_target_fps_spin.setDecimals(1)
        self.field_translation_target_fps_spin.setSingleStep(0.5)
        self.field_translation_target_fps_spin.setSuffix(" FPS")
        self.field_translation_target_fps_spin.setValue(
            normalize_field_translation_target_fps(preferences.field_translation_target_fps)
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        compute = _settings_section(
            "Compute and connections",
            (
                ("Solver", self.solve_backend_combo),
                ("Server", self.server_preferences, "Use your server address and optional access key."),
                (
                    "Reuse CUDA Worker Memory",
                    self.cuda_worker_reuse_check,
                    "Reuse cached GPU memory between local CUDA solves.",
                ),
            ),
        )
        layout.addWidget(compute)

        def show_backend_options():
            backend = self.solve_backend_options[self.solve_backend_combo.currentText()]
            self.server_preferences.setEnabled(backend == "beat_remote")
            compute.layout().setRowVisible(self.server_preferences, backend == "beat_remote")
            compute.layout().setRowVisible(self.cuda_worker_reuse_check, backend == "beat_cuda")

        self.solve_backend_combo.currentTextChanged.connect(show_backend_options)
        show_backend_options()
        layout.addWidget(
            _settings_section(
                "Interactive performance",
                (
                    (
                        "Field Cache Size",
                        self.field_cache_size_spin,
                        "Memory available for evaluated observation-plane fields.",
                    ),
                    (
                        "Field Translation Target Framerate",
                        self.field_translation_target_fps_spin,
                        "Update rate for fields while moving an observation plane.",
                    ),
                    ("Live Plot Streaming", self.live_plot_streaming_check, ""),
                    ("Live Plot Quality", self.live_plot_quality_combo, ""),
                ),
            )
        )
        layout.addWidget(_settings_section("Appearance", (("Theme", self.theme_combo),)))
        layout.addWidget(buttons)

    def preferences(self) -> GuiPreferences:
        return replace(
            self._preferences,
            theme=self.theme_options[self.theme_combo.currentText()],
            solve_backend=self.solve_backend_options[self.solve_backend_combo.currentText()],
            solve_server_url=self.server_preferences.url.text().strip(),
            solve_server_access_key=self.server_preferences.access_key.text().strip(),
            cuda_worker_reuse=bool(self.cuda_worker_reuse_check.isChecked()),
            live_plot_streaming=bool(self.live_plot_streaming_check.isChecked()),
            live_plot_quality=self.live_plot_quality_options[self.live_plot_quality_combo.currentText()],
            field_cache_size_mb=int(self.field_cache_size_spin.value()),
            field_translation_target_fps=float(self.field_translation_target_fps_spin.value()),
        )


class ProjectConfigurationDialog(QDialog):
    def __init__(self, preferences: ProjectPreferencesState, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Configuration")
        self._preferences = preferences
        self.polar_step_spin = QDoubleSpinBox()
        self.polar_step_spin.setRange(0.5, 90.0)
        self.polar_step_spin.setDecimals(1)
        self.polar_step_spin.setSingleStep(0.5)
        self.polar_step_spin.setSuffix(" deg")
        self.polar_step_spin.setValue(preferences.polar_angle_step_deg)

        self.polar_distance_spin = QDoubleSpinBox()
        self.polar_distance_spin.setRange(0.01, 1000.0)
        self.polar_distance_spin.setDecimals(3)
        self.polar_distance_spin.setSingleStep(0.25)
        self.polar_distance_spin.setSuffix(" m")
        self.polar_distance_spin.setValue(preferences.polar_observation_distance_m)

        self.smoothing_combo = QComboBox()
        self.smoothing_options = {
            "off": None,
            "1/48": 48,
            "1/24": 24,
            "1/12": 12,
            "1/6": 6,
        }
        self.smoothing_combo.addItems(self.smoothing_options.keys())
        smoothing_label = "off" if preferences.polar_smoothing is None else f"1/{preferences.polar_smoothing:g}"
        self.smoothing_combo.setCurrentText(smoothing_label if smoothing_label in self.smoothing_options else "1/24")

        self.horizontal_norm_angle_spin = QDoubleSpinBox()
        self.horizontal_norm_angle_spin.setRange(-180.0, 180.0)
        self.horizontal_norm_angle_spin.setDecimals(1)
        self.horizontal_norm_angle_spin.setSingleStep(1.0)
        self.horizontal_norm_angle_spin.setSuffix(" deg")
        self.horizontal_norm_angle_spin.setValue(preferences.horizontal_normalization_angle)

        self.vertical_norm_angle_spin = QDoubleSpinBox()
        self.vertical_norm_angle_spin.setRange(-180.0, 180.0)
        self.vertical_norm_angle_spin.setDecimals(1)
        self.vertical_norm_angle_spin.setSingleStep(1.0)
        self.vertical_norm_angle_spin.setSuffix(" deg")
        self.vertical_norm_angle_spin.setValue(preferences.vertical_normalization_angle)

        self.spin_horizontal_ref_angle_spin = QDoubleSpinBox()
        self.spin_horizontal_ref_angle_spin.setRange(-180.0, 180.0)
        self.spin_horizontal_ref_angle_spin.setDecimals(1)
        self.spin_horizontal_ref_angle_spin.setSingleStep(1.0)
        self.spin_horizontal_ref_angle_spin.setSuffix(" deg")
        self.spin_horizontal_ref_angle_spin.setValue(preferences.spin_horizontal_reference_angle)

        self.spin_vertical_ref_angle_spin = QDoubleSpinBox()
        self.spin_vertical_ref_angle_spin.setRange(-180.0, 180.0)
        self.spin_vertical_ref_angle_spin.setDecimals(1)
        self.spin_vertical_ref_angle_spin.setSingleStep(1.0)
        self.spin_vertical_ref_angle_spin.setSuffix(" deg")
        self.spin_vertical_ref_angle_spin.setValue(preferences.spin_vertical_reference_angle)

        self.spl_max_spin = QDoubleSpinBox()
        self.spl_max_spin.setRange(-200.0, 200.0)
        self.spl_max_spin.setDecimals(1)
        self.spl_max_spin.setSingleStep(1.0)
        self.spl_max_spin.setSuffix(" dB")
        self.spl_max_spin.setValue(preferences.spl_max_db)

        self.spl_min_spin = QDoubleSpinBox()
        self.spl_min_spin.setRange(-200.0, 200.0)
        self.spl_min_spin.setDecimals(1)
        self.spl_min_spin.setSingleStep(1.0)
        self.spl_min_spin.setSuffix(" dB")
        self.spl_min_spin.setValue(preferences.spl_min_db)

        self.isobar_contour_step_spin = QDoubleSpinBox()
        self.isobar_contour_step_spin.setRange(0.0, 200.0)
        self.isobar_contour_step_spin.setDecimals(1)
        self.isobar_contour_step_spin.setSingleStep(0.5)
        self.isobar_contour_step_spin.setSuffix(" dB")
        self.isobar_contour_step_spin.setValue(preferences.isobar_contour_step_db)

        self.spherical_sampling_check = QCheckBox("Enabled")
        self.spherical_sampling_check.setChecked(preferences.spherical_sampling_enabled)

        self.balloon_angle_precision_spin = QDoubleSpinBox()
        self.balloon_angle_precision_spin.setRange(0.5, 15.0)
        self.balloon_angle_precision_spin.setDecimals(1)
        self.balloon_angle_precision_spin.setSingleStep(0.5)
        self.balloon_angle_precision_spin.setSuffix(" deg")
        self.balloon_angle_precision_spin.setValue(preferences.balloon_angle_precision_deg)

        tabs = QTabWidget()
        sampling = QWidget()
        form = QFormLayout(sampling)
        form.addRow("Observation distance", self.polar_distance_spin)
        form.addRow("Polar angle step", self.polar_step_spin)
        self.polar_step_spin.setToolTip(
            "Horizontal and vertical sample spacing. Spinorama requires 10 degrees or finer."
        )
        self.spherical_sampling_check.setText("Include 3D balloon samples")
        form.addRow(self.spherical_sampling_check)
        form.addRow("Balloon angular spacing", self.balloon_angle_precision_spin)
        form.setRowVisible(self.balloon_angle_precision_spin, preferences.spherical_sampling_enabled)
        self.spherical_sampling_check.toggled.connect(
            lambda enabled: form.setRowVisible(self.balloon_angle_precision_spin, enabled)
        )
        tabs.addTab(sampling, "Observation Sampling")

        presentation = QWidget()
        layout = QVBoxLayout(presentation)
        references = QGroupBox("Reference angles")
        grid = QGridLayout(references)
        grid.addWidget(QLabel("Horizontal"), 0, 1)
        grid.addWidget(QLabel("Vertical"), 0, 2)
        grid.addWidget(QLabel("Directivity normalization"), 1, 0)
        horizontal = QHBoxLayout()
        horizontal.addWidget(self.horizontal_norm_angle_spin)
        info = QLabel()
        info.setPixmap(QIcon(str(INFO_ICON_PATH)).pixmap(16, 16))
        note = "When Impose Channel Normalization is enabled, this horizontal angle also sets the reference axis for the per-channel magnitude correction."
        info.setToolTip(note)
        self.horizontal_norm_angle_spin.setToolTip(note)
        horizontal.addWidget(info)
        grid.addLayout(horizontal, 1, 1)
        grid.addWidget(self.vertical_norm_angle_spin, 1, 2)
        grid.addWidget(QLabel("Spinorama reference"), 2, 0)
        grid.addWidget(self.spin_horizontal_ref_angle_spin, 2, 1)
        grid.addWidget(self.spin_vertical_ref_angle_spin, 2, 2)
        layout.addWidget(references)
        layout.addWidget(
            _settings_section(
                "Directivity plots",
                (
                    ("Fractional-octave smoothing", self.smoothing_combo),
                    ("Display range minimum", self.spl_min_spin),
                    ("Display range maximum", self.spl_max_spin),
                    (
                        "Isobar contour interval",
                        self.isobar_contour_step_spin,
                        "Set to 0 dB for smoothly interpolated isobar colors.",
                    ),
                ),
            )
        )
        layout.addStretch()
        tabs.addTab(presentation, "Plot Presentation")
        self.use_as_defaults = QCheckBox("Use as defaults for new projects")
        self.use_as_defaults.setToolTip(
            "Use this project's sampling, presentation, frequency sweep, channel normalization and stitch tolerance when creating projects."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer = QVBoxLayout(self)
        outer.addWidget(tabs)
        outer.addWidget(self.use_as_defaults)
        outer.addWidget(buttons)
        self._initial_values = self._control_values()

    def configuration(self) -> ProjectPreferencesState:
        # Files can carry more precision than a spin box displays.
        changes = {key: value for key, value in self._control_values().items() if value != self._initial_values[key]}
        return replace(self._preferences, **changes)

    def _control_values(self):
        spl_min = float(self.spl_min_spin.value())
        spl_max = float(self.spl_max_spin.value())
        if spl_max <= spl_min:
            spl_max = spl_min + 1.0

        return dict(
            polar_angle_step_deg=float(self.polar_step_spin.value()),
            polar_observation_distance_m=float(self.polar_distance_spin.value()),
            polar_smoothing=self.smoothing_options[self.smoothing_combo.currentText()],
            horizontal_normalization_angle=float(self.horizontal_norm_angle_spin.value()),
            vertical_normalization_angle=float(self.vertical_norm_angle_spin.value()),
            spin_horizontal_reference_angle=float(self.spin_horizontal_ref_angle_spin.value()),
            spin_vertical_reference_angle=float(self.spin_vertical_ref_angle_spin.value()),
            spl_max_db=spl_max,
            spl_min_db=spl_min,
            isobar_contour_step_db=float(self.isobar_contour_step_spin.value()),
            spherical_sampling_enabled=bool(self.spherical_sampling_check.isChecked()),
            balloon_angle_precision_deg=float(self.balloon_angle_precision_spin.value()),
        )


class MeshDropTable(QTableWidget):
    meshFilesDropped = Signal(object)

    def __init__(self, rows: int, columns: int, parent: QWidget | None = None):
        super().__init__(rows, columns, parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        if self._msh_drop_paths(event):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self._msh_drop_paths(event):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = self._msh_drop_paths(event)
        if not paths:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        self.meshFilesDropped.emit(paths)

    @staticmethod
    def _msh_drop_paths(event) -> list[Path]:
        return [path for path in local_drop_paths(event) if path.suffix.lower() == ".msh"]


class MeshConfigDialog(QDialog):
    def __init__(
        self,
        meshes: tuple[MeshDialogEntry, ...],
        *,
        stitch_imported_meshes: bool = False,
        symmetry: str = "off",
        symmetry_enabled: bool = True,
        file_dialog_service: FileDialogService | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Meshes")
        self._meshes = list(meshes)
        self.file_dialogs = file_dialog_service or FileDialogService()

        self.table = MeshDropTable(0, 7)
        self.table.setHorizontalHeaderLabels(["Enabled", "Name", "Mesh File", "Scale", "X mm", "Y mm", "Z mm"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        for column in range(3, 7):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self.enabled_widgets: list[QCheckBox] = []
        self.name_items: list[QTableWidgetItem] = []
        self.file_items: list[QTableWidgetItem] = []
        self.scale_widgets: list[QDoubleSpinBox] = []
        self.x_widgets: list[QSpinBox] = []
        self.y_widgets: list[QSpinBox] = []
        self.z_widgets: list[QSpinBox] = []

        self.add_button = QPushButton("Import .msh")
        self.replace_button = QPushButton("Replace .msh")
        self.replace_button.setEnabled(False)
        self.remove_button = QPushButton("Remove")
        # Kept as a compatibility input while project files migrate; the control now
        # belongs to the exterior region in the System editor.
        self._stitch_imported_meshes = bool(stitch_imported_meshes)
        self.symmetry_combo = QComboBox()
        self.symmetry_options = {
            "Off": "off",
            "X": "x",
            "XY": "xy",
        }
        self.symmetry_combo.addItems(self.symmetry_options.keys())
        current_symmetry = str(symmetry or "off").strip().lower()
        current_label = next(
            (label for label, value in self.symmetry_options.items() if value == current_symmetry),
            "Off",
        )
        self.symmetry_combo.setCurrentText(current_label)
        self.symmetry_combo.setEnabled(symmetry_enabled)
        self.add_button.clicked.connect(self._add_mesh)
        self.replace_button.clicked.connect(self._replace_mesh)
        self.remove_button.clicked.connect(self._remove_selected_meshes)
        self.table.itemSelectionChanged.connect(self._update_replace_button)
        self.table.meshFilesDropped.connect(self._add_mesh_paths)

        button_row = QHBoxLayout()
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.replace_button)
        button_row.addWidget(self.remove_button)
        button_row.addSpacing(16)
        button_row.addWidget(QLabel("Symmetry"))
        button_row.addWidget(self.symmetry_combo)
        button_row.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(button_row)
        layout.addWidget(buttons)

        for mesh in self._meshes:
            self._append_row(mesh)
        self.resize(820, 360)

    def stitch_imported_meshes(self) -> bool:
        return self._stitch_imported_meshes

    def symmetry(self) -> str:
        return self.symmetry_options.get(self.symmetry_combo.currentText(), "off")

    def replaced_mesh_names(self) -> tuple[str, ...]:
        return tuple(
            self.name_items[row].text().strip()
            for row in range(self.table.rowCount())
            if bool(self.file_items[row].data(int(Qt.ItemDataRole.UserRole) + 1))
        )

    def accept(self) -> None:
        try:
            self.meshes()
        except ValueError as exc:
            QMessageBox.warning(self, "Meshes", str(exc))
            return
        super().accept()

    def _append_row(self, mesh: MeshDialogEntry) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        enabled_check = QCheckBox()
        enabled_check.setChecked(mesh.enabled)
        self.table.setCellWidget(row, 0, enabled_check)
        self.enabled_widgets.append(enabled_check)

        name_item = QTableWidgetItem(mesh.name)
        if mesh.locked:
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        self.table.setItem(row, 1, name_item)
        self.name_items.append(name_item)
        name_item.setData(Qt.ItemDataRole.UserRole, mesh.assembly_id)
        if mesh.assembly_id:
            name_item.setToolTip("Generated assembly: enabling, scale and position apply to all of its meshes.")

        file_item = QTableWidgetItem(mesh.source_file)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        file_item.setData(Qt.ItemDataRole.UserRole, mesh.cleaned_file)
        file_item.setData(int(Qt.ItemDataRole.UserRole) + 1, False)
        file_item.setData(int(Qt.ItemDataRole.UserRole) + 2, mesh.mesh_data)
        self.table.setItem(row, 2, file_item)
        self.file_items.append(file_item)

        scale_spin = QDoubleSpinBox()
        scale_spin.setRange(0.000001, 1000.0)
        scale_spin.setDecimals(6)
        scale_spin.setSingleStep(0.001)
        scale_spin.setValue(float(mesh.scale_factor))
        self.table.setCellWidget(row, 3, scale_spin)
        self.scale_widgets.append(scale_spin)

        x_mm, y_mm, z_mm = mesh.translation_mm
        for column, value, widgets in (
            (4, x_mm, self.x_widgets),
            (5, y_mm, self.y_widgets),
            (6, z_mm, self.z_widgets),
        ):
            spin = QSpinBox()
            spin.setRange(-1000000, 1000000)
            spin.setSingleStep(1)
            spin.setSuffix(" mm")
            spin.setValue(round(float(value)))
            self.table.setCellWidget(row, column, spin)
            widgets.append(spin)

        if mesh.assembly_id:
            for column in (0, 3, 4, 5, 6):
                widget = self.table.cellWidget(row, column)
                signal = widget.toggled if column == 0 else widget.valueChanged
                signal.connect(lambda value, group=mesh.assembly_id, col=column: self._sync_assembly(group, col, value))

    def _sync_assembly(self, assembly_id, column, value):
        for row, item in enumerate(self.name_items):
            if item.data(Qt.ItemDataRole.UserRole) != assembly_id:
                continue
            widget = self.table.cellWidget(row, column)
            blocked = widget.blockSignals(True)
            if column == 0:
                widget.setChecked(value)
            else:
                widget.setValue(value)
            widget.blockSignals(blocked)

    def _add_mesh(self) -> None:
        path = self.file_dialogs.open_file(
            self,
            "Import mesh",
            "Gmsh mesh files (*.msh)",
        )
        if path is None:
            return

        self._add_mesh_paths([path])

    def _add_mesh_paths(self, paths: list[Path]) -> None:
        unsupported = [path for path in paths if path.suffix.lower() != ".msh"]
        if unsupported:
            QMessageBox.warning(self, "Unsupported mesh", "Only .msh mesh files can be imported.")
            return
        for path in paths:
            self._append_mesh_path(path)

    def _replace_mesh(self) -> None:
        row = self._selected_replaceable_row()
        if row is None:
            return
        path = self.file_dialogs.open_file(
            self,
            "Replace mesh",
            "Gmsh mesh files (*.msh)",
        )
        if path is None:
            return
        if path.suffix.lower() != ".msh":
            QMessageBox.warning(self, "Unsupported mesh", "Only .msh mesh files can be imported.")
            return
        self.file_items[row].setText(str(path))
        self.file_items[row].setData(Qt.ItemDataRole.UserRole, None)
        self.file_items[row].setData(int(Qt.ItemDataRole.UserRole) + 1, True)

    def _selected_replaceable_row(self) -> int | None:
        rows = {index.row() for index in self.table.selectedIndexes()}
        if len(rows) != 1:
            return None
        row = next(iter(rows))
        if not bool(self.name_items[row].flags() & Qt.ItemIsEditable):
            return None
        return row

    def _update_replace_button(self) -> None:
        self.replace_button.setEnabled(self._selected_replaceable_row() is not None)

    def _append_mesh_path(self, path: Path) -> None:
        self._append_row(
            MeshDialogEntry(
                name=self._unique_mesh_name(path.stem),
                source_file=str(path),
                enabled=True,
            )
        )

    def _remove_selected_meshes(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            if not bool(self.name_items[row].flags() & Qt.ItemIsEditable):
                continue
            self.table.removeRow(row)
            del self.enabled_widgets[row]
            del self.name_items[row]
            del self.file_items[row]
            del self.scale_widgets[row]
            del self.x_widgets[row]
            del self.y_widgets[row]
            del self.z_widgets[row]

    def _unique_mesh_name(self, base_name: str) -> str:
        sanitized = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in base_name).strip("_")
        name = sanitized or "mesh"
        used = {item.text().strip() for item in self.name_items}
        if name not in used:
            return name

        suffix = 2
        while f"{name}_{suffix}" in used:
            suffix += 1
        return f"{name}_{suffix}"

    def meshes(self) -> tuple[MeshDialogEntry, ...]:
        meshes = []
        seen_names = set()
        for row in range(self.table.rowCount()):
            name = self.name_items[row].text().strip()
            if not name:
                raise ValueError("Each imported mesh must have a name.")
            is_generated_row = not bool(self.name_items[row].flags() & Qt.ItemIsEditable)
            if ":" in name:
                raise ValueError("Mesh names cannot contain ':'.")
            if name in seen_names:
                raise ValueError(f"Duplicate mesh name: {name}")
            seen_names.add(name)

            meshes.append(
                MeshDialogEntry(
                    name=name,
                    source_file=self.file_items[row].text(),
                    cleaned_file=self._cleaned_file_for_row(row),
                    scale_factor=float(self.scale_widgets[row].value()),
                    translation_mm=(
                        float(int(self.x_widgets[row].value())),
                        float(int(self.y_widgets[row].value())),
                        float(int(self.z_widgets[row].value())),
                    ),
                    enabled=bool(self.enabled_widgets[row].isChecked()),
                    locked=is_generated_row,
                    mesh_data=self.file_items[row].data(int(Qt.ItemDataRole.UserRole) + 2),
                    assembly_id=self.name_items[row].data(Qt.ItemDataRole.UserRole),
                )
            )
        return tuple(meshes)

    def _cleaned_file_for_row(self, row: int) -> str | None:
        value = self.file_items[row].data(Qt.ItemDataRole.UserRole)
        return None if value is None else str(value)


class ChannelConfigDialog(QDialog):
    channelsApplied = Signal(object, bool)
    closeRequested = Signal()

    def __init__(
        self,
        channels: tuple[ChannelConfig, ...],
        parent: QWidget | None = None,
        *,
        embedded: bool = False,
        normalized_channel_correction: bool = False,
        prescribed_velocity_channel_names: frozenset[str] = frozenset(),
    ):
        super().__init__(parent)
        self.setWindowTitle("Channel Config")
        self.setAttribute(Qt.WA_DeleteOnClose, not embedded)
        self._embedded = bool(embedded)
        self._channels = list(channels) or [ChannelConfig(name="main")]
        self._prescribed_velocity_channel_names = frozenset(prescribed_velocity_channel_names)

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(
            [
                "Name",
                "Voltage",
                "Trim dB",
                "Polarity",
                "Delay ms",
                "HPF Type",
                "HPF Frequency",
                "LPF Type",
                "LPF Frequency",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 9):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self.name_items: list[QTableWidgetItem] = []
        self.voltage_widgets: list[QDoubleSpinBox] = []
        self.level_widgets: list[QDoubleSpinBox] = []
        self.polarity_widgets: list[QComboBox] = []
        self.delay_widgets: list[QDoubleSpinBox] = []
        self.hpf_type_widgets: list[QComboBox] = []
        self.hpf_freq_widgets: list[QDoubleSpinBox] = []
        self.lpf_type_widgets: list[QComboBox] = []
        self.lpf_freq_widgets: list[QDoubleSpinBox] = []

        for channel in self._channels:
            self._append_row(channel)

        add_button = QPushButton("Add Channel")
        remove_button = QPushButton("Remove")
        add_button.clicked.connect(self._add_channel)
        remove_button.clicked.connect(self._remove_selected_channels)

        button_row = QHBoxLayout()
        button_row.addWidget(add_button)
        button_row.addWidget(remove_button)
        self.normalization_check = QCheckBox("Impose Channel Normalization")
        self.normalization_check.setToolTip(
            "Force a flat reference-axis magnitude for all channels before applying gain, delay, and crossover filters."
        )
        self.normalization_check.setChecked(normalized_channel_correction)
        button_row.addWidget(self.normalization_check)
        button_row.addStretch(1)

        button_flags = QDialogButtonBox.Apply if self._embedded else QDialogButtonBox.Apply | QDialogButtonBox.Close
        buttons = QDialogButtonBox(button_flags)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.closeRequested.emit if self._embedded else self.reject)
        button_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addLayout(button_row)
        if not self._embedded:
            self.resize(1080, min(520, 160 + 34 * max(1, len(self._channels))))

    def apply(self) -> bool:
        try:
            channels = self.channels()
        except ValueError as exc:
            QMessageBox.warning(self, "Channel config", str(exc))
            return False
        self.channelsApplied.emit(channels, self.normalization_check.isChecked())
        return True

    def _append_row(self, channel: ChannelConfig) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        name_item = QTableWidgetItem(channel.name)
        self.table.setItem(row, 0, name_item)
        self.name_items.append(name_item)

        voltage_spin = QDoubleSpinBox()
        voltage_spin.setRange(0.0, 1000.0)
        voltage_spin.setDecimals(2)
        voltage_spin.setSingleStep(0.1)
        voltage_spin.setSuffix(" V")
        voltage_spin.setValue(float(channel.voltage_v))
        voltage_editable = channel.name not in self._prescribed_velocity_channel_names
        voltage_spin.setEnabled(voltage_editable)
        if not voltage_editable:
            voltage_spin.setToolTip(
                "Voltage is unavailable because this channel contains a prescribed-velocity source."
            )
        self.table.setCellWidget(row, 1, voltage_spin)
        self.voltage_widgets.append(voltage_spin)

        level_spin = QDoubleSpinBox()
        level_spin.setRange(-120.0, 60.0)
        level_spin.setDecimals(2)
        level_spin.setSingleStep(0.5)
        level_spin.setValue(float(channel.level_db))
        self.table.setCellWidget(row, 2, level_spin)
        self.level_widgets.append(level_spin)

        polarity_combo = QComboBox()
        polarity_combo.addItem("+", 1)
        polarity_combo.addItem("-", -1)
        polarity_combo.setCurrentIndex(0 if channel.polarity >= 0 else 1)
        self.table.setCellWidget(row, 3, polarity_combo)
        self.polarity_widgets.append(polarity_combo)

        delay_spin = QDoubleSpinBox()
        delay_spin.setRange(-1000.0, 1000.0)
        delay_spin.setDecimals(3)
        delay_spin.setSingleStep(0.01)
        delay_spin.setValue(float(channel.delay_ms))
        self.table.setCellWidget(row, 4, delay_spin)
        self.delay_widgets.append(delay_spin)

        hpf_type_combo = _build_crossover_type_combo(_crossover_label(channel.hpf))
        self.table.setCellWidget(row, 5, hpf_type_combo)
        self.hpf_type_widgets.append(hpf_type_combo)

        hpf_freq_spin = _build_crossover_frequency_spin(channel.hpf.frequency_hz)
        self.table.setCellWidget(row, 6, hpf_freq_spin)
        self.hpf_freq_widgets.append(hpf_freq_spin)

        lpf_type_combo = _build_crossover_type_combo(_crossover_label(channel.lpf))
        self.table.setCellWidget(row, 7, lpf_type_combo)
        self.lpf_type_widgets.append(lpf_type_combo)

        lpf_freq_spin = _build_crossover_frequency_spin(channel.lpf.frequency_hz)
        self.table.setCellWidget(row, 8, lpf_freq_spin)
        self.lpf_freq_widgets.append(lpf_freq_spin)

        hpf_type_combo.currentTextChanged.connect(lambda _text, row=row: self._set_frequency_controls_enabled(row))
        lpf_type_combo.currentTextChanged.connect(lambda _text, row=row: self._set_frequency_controls_enabled(row))
        self._set_frequency_controls_enabled(row)

    def _add_channel(self) -> None:
        used = {item.text().strip() for item in self.name_items}
        index = 1
        while f"channel_{index}" in used:
            index += 1
        self._append_row(ChannelConfig(name=f"channel_{index}"))

    def _remove_selected_channels(self) -> None:
        rows = sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True)
        for row in rows:
            if self.table.rowCount() <= 1:
                return
            self.table.removeRow(row)
            del self.name_items[row]
            del self.voltage_widgets[row]
            del self.level_widgets[row]
            del self.polarity_widgets[row]
            del self.delay_widgets[row]
            del self.hpf_type_widgets[row]
            del self.hpf_freq_widgets[row]
            del self.lpf_type_widgets[row]
            del self.lpf_freq_widgets[row]

    def _set_frequency_controls_enabled(self, row: int) -> None:
        self.hpf_freq_widgets[row].setEnabled(self.hpf_type_widgets[row].currentData() is not None)
        self.lpf_freq_widgets[row].setEnabled(self.lpf_type_widgets[row].currentData() is not None)

    def _crossover_config(self, row: int, *, highpass: bool) -> CrossoverConfig:
        type_widget = self.hpf_type_widgets[row] if highpass else self.lpf_type_widgets[row]
        freq_widget = self.hpf_freq_widgets[row] if highpass else self.lpf_freq_widgets[row]
        payload = type_widget.currentData()
        if payload is None:
            return CrossoverConfig()
        filter_name, order = payload
        return CrossoverConfig(
            type="highpass" if highpass else "lowpass",
            filter=filter_name,
            order=int(order),
            frequency_hz=float(freq_widget.value()),
        )

    def channels(self) -> tuple[ChannelConfig, ...]:
        channels = []
        seen = set()
        for row in range(self.table.rowCount()):
            name = self.name_items[row].text().strip()
            if not name:
                raise ValueError("Each channel must have a name.")
            if ":" in name:
                raise ValueError("Channel names cannot contain ':'.")
            if name in seen:
                raise ValueError(f"Duplicate channel name: {name}")
            seen.add(name)
            channels.append(
                ChannelConfig(
                    name=name,
                    voltage_v=float(self.voltage_widgets[row].value()),
                    level_db=float(self.level_widgets[row].value()),
                    polarity=int(self.polarity_widgets[row].currentData()),
                    delay_ms=float(self.delay_widgets[row].value()),
                    hpf=self._crossover_config(row, highpass=True),
                    lpf=self._crossover_config(row, highpass=False),
                )
            )
        return tuple(channels)
