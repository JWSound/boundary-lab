"""Configuration dialogs for the live solver GUI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.config import ChannelConfig, CrossoverConfig, RadiatorConfig
from blab.solvers.http_server import query_server_health
from blab.solvers.registry import backend_info, backend_label_to_id, normalize_backend_id
from blab.ui.drag_drop import local_drop_paths
from blab.ui.settings import GuiPreferences, normalize_live_plot_quality

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
APP_ROOT = Path(__file__).resolve().parents[3]
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


class PreferencesDialog(QDialog):
    def __init__(self, preferences: GuiPreferences, parent: QWidget | None = None):
        super().__init__(parent)
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
            "Bempp (OpenCL CPU)",
        )
        self.solve_backend_combo.setCurrentText(backend_label)

        self.solve_server_url_edit = QLineEdit()
        self.solve_server_url_edit.setText(preferences.solve_server_url)
        self.server_health_payload: dict | None = None
        self.server_health_url: str | None = None
        self.check_server_button = QPushButton("Check Server")
        self.check_server_button.clicked.connect(self._check_server)

        self.gmres_spin = QDoubleSpinBox()
        self.gmres_spin.setRange(1e-8, 1e-2)
        self.gmres_spin.setDecimals(8)
        self.gmres_spin.setSingleStep(1e-4)
        self.gmres_spin.setValue(preferences.gmres_tolerance)

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

        self.normalized_channel_correction_check = QCheckBox("Enabled")
        self.normalized_channel_correction_check.setChecked(preferences.normalized_channel_correction)

        self.burton_miller_check = QCheckBox("Enabled")
        self.burton_miller_check.setChecked(preferences.use_burton_miller)

        def update_backend_fields(label: str) -> None:
            backend_id = self.solve_backend_options.get(label, "local")
            uses_bempp = backend_id in {"local", "server"}
            uses_remote = backend_info(backend_id).capabilities.is_remote
            self.solve_server_url_edit.setEnabled(uses_remote)
            self.check_server_button.setEnabled(uses_remote)
            self.gmres_spin.setEnabled(uses_bempp)
            self.burton_miller_check.setEnabled(uses_bempp)

        update_backend_fields(backend_label)
        self.solve_backend_combo.currentTextChanged.connect(update_backend_fields)

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

        self.stitch_tolerance_spin = QDoubleSpinBox()
        self.stitch_tolerance_spin.setRange(0.001, 1000.0)
        self.stitch_tolerance_spin.setDecimals(3)
        self.stitch_tolerance_spin.setSingleStep(0.5)
        self.stitch_tolerance_spin.setSuffix(" mm")
        self.stitch_tolerance_spin.setValue(preferences.stitch_tolerance_mm)

        self.spherical_sampling_check = QCheckBox("Enabled")
        self.spherical_sampling_check.setChecked(preferences.spherical_sampling_enabled)

        self.balloon_angle_precision_spin = QDoubleSpinBox()
        self.balloon_angle_precision_spin.setRange(0.5, 15.0)
        self.balloon_angle_precision_spin.setDecimals(1)
        self.balloon_angle_precision_spin.setSingleStep(0.5)
        self.balloon_angle_precision_spin.setSuffix(" deg")
        self.balloon_angle_precision_spin.setValue(preferences.balloon_angle_precision_deg)
        self.balloon_angle_precision_spin.setEnabled(preferences.spherical_sampling_enabled)
        self.spherical_sampling_check.toggled.connect(self.balloon_angle_precision_spin.setEnabled)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        columns = QHBoxLayout()
        columns.setSpacing(12)
        left_column = QVBoxLayout()
        right_column = QVBoxLayout()
        left_column.setSpacing(8)
        right_column.setSpacing(8)

        left_column.addWidget(
            self._section(
                "Solver Config",
                (
                    ("BEM Solver", self.solve_backend_combo, ""),
                    ("Solve Server URL", self.solve_server_url_edit, ""),
                    (
                        "",
                        self.check_server_button,
                        "Query the configured solve server and update advertised capabilities.",
                    ),
                    (
                        "GMRES Tolerance",
                        self.gmres_spin,
                        "Solution accuracy for the BEMPP iterative solver. Lower values offer higher quality solves but take longer.",
                    ),
                    (
                        "Burton Miller Formulation",
                        self.burton_miller_check,
                        "Enable to resolve fictitious interior resonances when using BEMPP solver. Always enabled for BEAT Engine.",
                    ),
                    (
                        "Balloon Sampling",
                        self.spherical_sampling_check,
                        "Gather spherical observation data for 3d ballon viewer",
                    ),
                    (
                        "Balloon Angle Precision",
                        self.balloon_angle_precision_spin,
                        "Resolution of spherical sampling. 2.5 degrees = ~6,000 points.",
                    ),
                ),
            )
        )
        left_column.addWidget(
            self._section(
                "Observation Config",
                (
                    (
                        "Polar Angle Step",
                        self.polar_step_spin,
                        "Angular resolution for horizontal/vertical polars - spin plot requires a minimum of 10 degrees",
                    ),
                    ("Polar Observation Distance", self.polar_distance_spin, ""),
                    (
                        "Normalized Channel Correction",
                        self.normalized_channel_correction_check,
                        "Applies a per-channel reference-axis magnitude correction before channel gain, delay, and crossover filters.",
                    ),
                    ("Horizontal Normalization Angle", self.horizontal_norm_angle_spin, ""),
                    ("Vertical Normalization Angle", self.vertical_norm_angle_spin, ""),
                    ("Spin Horizontal Ref Angle", self.spin_horizontal_ref_angle_spin, ""),
                    ("Spin Vertical Ref Angle", self.spin_vertical_ref_angle_spin, ""),
                    ("Polar Smoothing", self.smoothing_combo, ""),
                    ("SPL Min", self.spl_min_spin, ""),
                    ("SPL Max", self.spl_max_spin, ""),
                ),
            )
        )
        left_column.addStretch(1)

        right_column.addWidget(
            self._section(
                "Mesh Config",
                (("Stitch Tolerance", self.stitch_tolerance_spin, ""),),
            )
        )
        right_column.addWidget(
            self._section(
                "Application",
                (
                    ("Theme", self.theme_combo, ""),
                    ("Live Plot Streaming", self.live_plot_streaming_check, ""),
                    ("Live Plot Quality", self.live_plot_quality_combo, ""),
                ),
            )
        )
        right_column.addStretch(1)

        columns.addLayout(left_column, 1)
        columns.addLayout(right_column, 1)
        layout.addLayout(columns)
        layout.addWidget(buttons)
        self.resize(820, 420)

    def _check_server(self) -> None:
        url = self.solve_server_url_edit.text().strip() or "http://127.0.0.1:8765"
        try:
            payload = query_server_health(url)
        except Exception as exc:
            self.server_health_payload = None
            self.server_health_url = None
            QMessageBox.warning(self, "Check Server", f"Failed to connect to solve server:\n{exc}")
            return

        self.server_health_payload = payload
        self.server_health_url = url.rstrip("/")
        capabilities = payload.get("capabilities", {}) if isinstance(payload.get("capabilities", {}), dict) else {}
        capability_lines = [
            f"Solver: {payload.get('solver_label') or payload.get('solver') or 'Unknown'}",
            f"Backend: {payload.get('backend') or payload.get('solver') or 'Unknown'}",
            f"Symmetry: {'yes' if capabilities.get('supports_symmetry') else 'no'}",
            f"Spherical sampling: {'yes' if capabilities.get('supports_spherical_sampling') else 'no'}",
            f"Channel resynthesis: {'yes' if capabilities.get('supports_channel_resynthesis') else 'no'}",
        ]
        QMessageBox.information(self, "Check Server", "Solve server is reachable.\n\n" + "\n".join(capability_lines))

    @staticmethod
    def _section(title: str, rows: tuple[tuple[str, QWidget] | tuple[str, QWidget, str], ...]) -> QGroupBox:
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

    def preferences(self) -> GuiPreferences:
        spl_min = float(self.spl_min_spin.value())
        spl_max = float(self.spl_max_spin.value())
        if spl_max <= spl_min:
            spl_max = spl_min + 1.0

        return GuiPreferences(
            theme=self.theme_options[self.theme_combo.currentText()],
            solve_backend=self.solve_backend_options[self.solve_backend_combo.currentText()],
            solve_server_url=self.solve_server_url_edit.text().strip() or "http://127.0.0.1:8765",
            live_plot_streaming=bool(self.live_plot_streaming_check.isChecked()),
            live_plot_quality=self.live_plot_quality_options[self.live_plot_quality_combo.currentText()],
            gmres_tolerance=float(self.gmres_spin.value()),
            polar_angle_step_deg=float(self.polar_step_spin.value()),
            polar_observation_distance_m=float(self.polar_distance_spin.value()),
            normalized_channel_correction=bool(self.normalized_channel_correction_check.isChecked()),
            use_burton_miller=bool(self.burton_miller_check.isChecked()),
            polar_smoothing=self.smoothing_options[self.smoothing_combo.currentText()],
            horizontal_normalization_angle=float(self.horizontal_norm_angle_spin.value()),
            vertical_normalization_angle=float(self.vertical_norm_angle_spin.value()),
            spin_horizontal_reference_angle=float(self.spin_horizontal_ref_angle_spin.value()),
            spin_vertical_reference_angle=float(self.spin_vertical_ref_angle_spin.value()),
            spl_max_db=spl_max,
            spl_min_db=spl_min,
            stitch_tolerance_mm=float(self.stitch_tolerance_spin.value()),
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
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Mesh Config")
        self._meshes = list(meshes)

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
        self.remove_button = QPushButton("Remove")
        self.stitch_imported_meshes_check = QCheckBox("Stitch Imported Meshes")
        self.stitch_imported_meshes_check.setChecked(stitch_imported_meshes)
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
        self.remove_button.clicked.connect(self._remove_selected_meshes)
        self.table.meshFilesDropped.connect(self._add_mesh_paths)

        button_row = QHBoxLayout()
        button_row.addWidget(self.add_button)
        button_row.addWidget(self.remove_button)
        button_row.addSpacing(16)
        button_row.addWidget(self.stitch_imported_meshes_check)
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
        return bool(self.stitch_imported_meshes_check.isChecked())

    def symmetry(self) -> str:
        return self.symmetry_options.get(self.symmetry_combo.currentText(), "off")

    def accept(self) -> None:
        try:
            self.meshes()
        except ValueError as exc:
            QMessageBox.warning(self, "Mesh config", str(exc))
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

        file_item = QTableWidgetItem(mesh.source_file)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
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

    def _add_mesh(self) -> None:
        path_text, _ = QFileDialog.getOpenFileName(
            self,
            "Import mesh",
            str(Path.cwd()),
            "Gmsh mesh files (*.msh)",
        )
        if not path_text:
            return

        path = Path(path_text)
        self._add_mesh_paths([path])

    def _add_mesh_paths(self, paths: list[Path]) -> None:
        unsupported = [path for path in paths if path.suffix.lower() != ".msh"]
        if unsupported:
            QMessageBox.warning(self, "Unsupported mesh", "Only .msh mesh files can be imported.")
            return
        for path in paths:
            self._append_mesh_path(path)

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
            if self.name_items[row].text().strip() == "ath":
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
        if name not in used and name != "ath":
            return name

        suffix = 2
        while f"{name}_{suffix}" in used or f"{name}_{suffix}" == "ath":
            suffix += 1
        return f"{name}_{suffix}"

    def meshes(self) -> tuple[MeshDialogEntry, ...]:
        meshes = []
        seen_names = set()
        for row in range(self.table.rowCount()):
            name = self.name_items[row].text().strip()
            if not name:
                raise ValueError("Each imported mesh must have a name.")
            is_ath_row = name == "ath"
            if is_ath_row and row != 0:
                raise ValueError("'ath' is reserved for the default Ath mesh.")
            if name == "ath" and row == 0:
                pass
            elif name == "ath":
                raise ValueError("'ath' is reserved for the default Ath mesh.")
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
                    locked=is_ath_row,
                )
            )
        return tuple(meshes)

    def _cleaned_file_for_row(self, row: int) -> str | None:
        for mesh in self._meshes:
            if mesh.source_file == self.file_items[row].text():
                return mesh.cleaned_file
        return None


class ChannelConfigDialog(QDialog):
    channelsApplied = Signal(object)
    closeRequested = Signal()

    def __init__(self, channels: tuple[ChannelConfig, ...], parent: QWidget | None = None, *, embedded: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Channel Config")
        self.setAttribute(Qt.WA_DeleteOnClose, not embedded)
        self._embedded = bool(embedded)
        self._channels = list(channels) or [ChannelConfig(name="main")]

        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Level dB", "Polarity", "Delay ms", "HPF Type", "HPF Frequency", "LPF Type", "LPF Frequency"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 8):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self.name_items: list[QTableWidgetItem] = []
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
        self.channelsApplied.emit(channels)
        return True

    def _append_row(self, channel: ChannelConfig) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        name_item = QTableWidgetItem(channel.name)
        self.table.setItem(row, 0, name_item)
        self.name_items.append(name_item)

        level_spin = QDoubleSpinBox()
        level_spin.setRange(-120.0, 60.0)
        level_spin.setDecimals(2)
        level_spin.setSingleStep(0.5)
        level_spin.setValue(float(channel.level_db))
        self.table.setCellWidget(row, 1, level_spin)
        self.level_widgets.append(level_spin)

        polarity_combo = QComboBox()
        polarity_combo.addItem("+", 1)
        polarity_combo.addItem("-", -1)
        polarity_combo.setCurrentIndex(0 if channel.polarity >= 0 else 1)
        self.table.setCellWidget(row, 2, polarity_combo)
        self.polarity_widgets.append(polarity_combo)

        delay_spin = QDoubleSpinBox()
        delay_spin.setRange(-1000.0, 1000.0)
        delay_spin.setDecimals(3)
        delay_spin.setSingleStep(0.01)
        delay_spin.setValue(float(channel.delay_ms))
        self.table.setCellWidget(row, 3, delay_spin)
        self.delay_widgets.append(delay_spin)

        hpf_type_combo = _build_crossover_type_combo(_crossover_label(channel.hpf))
        self.table.setCellWidget(row, 4, hpf_type_combo)
        self.hpf_type_widgets.append(hpf_type_combo)

        hpf_freq_spin = _build_crossover_frequency_spin(channel.hpf.frequency_hz)
        self.table.setCellWidget(row, 5, hpf_freq_spin)
        self.hpf_freq_widgets.append(hpf_freq_spin)

        lpf_type_combo = _build_crossover_type_combo(_crossover_label(channel.lpf))
        self.table.setCellWidget(row, 6, lpf_type_combo)
        self.lpf_type_widgets.append(lpf_type_combo)

        lpf_freq_spin = _build_crossover_frequency_spin(channel.lpf.frequency_hz)
        self.table.setCellWidget(row, 7, lpf_freq_spin)
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
                    level_db=float(self.level_widgets[row].value()),
                    polarity=int(self.polarity_widgets[row].currentData()),
                    delay_ms=float(self.delay_widgets[row].value()),
                    hpf=self._crossover_config(row, highpass=True),
                    lpf=self._crossover_config(row, highpass=False),
                )
            )
        return tuple(channels)


class SourceConfigDialog(QDialog):
    def __init__(
        self,
        surface_tags: dict[str, tuple[str, int]],
        radiators: tuple[RadiatorConfig, ...],
        channels: tuple[ChannelConfig, ...],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Source Config")
        self.surface_items = sorted(surface_tags.items(), key=lambda item: (item[1][0], item[1][1], item[0]))
        self.radiators_by_key = {(radiator.mesh, radiator.tag): radiator for radiator in radiators}
        self.channel_names = tuple(channel.name for channel in channels) or ("main",)

        self.table = QTableWidget(len(self.surface_items), 5)
        self.table.setHorizontalHeaderLabels(
            [
                "Surface",
                "Tag",
                "Mode",
                "Channel",
                "Velocity Offset dB",
            ]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeToContents)

        self.mode_widgets: list[QComboBox] = []
        self.channel_widgets: list[QComboBox] = []
        self.velocity_widgets: list[QDoubleSpinBox] = []

        for row, (surface_name, (mesh_name, tag)) in enumerate(self.surface_items):
            radiator = self.radiators_by_key.get((mesh_name, tag))

            name_item = QTableWidgetItem(surface_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)

            tag_item = QTableWidgetItem(str(tag))
            tag_item.setFlags(tag_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, tag_item)

            mode_combo = QComboBox()
            mode_combo.addItems(["Rigid", "Driven"])
            mode_combo.setCurrentText("Driven" if radiator is not None else "Rigid")
            self.table.setCellWidget(row, 2, mode_combo)
            self.mode_widgets.append(mode_combo)

            channel_combo = QComboBox()
            channel_combo.addItems(self.channel_names)
            channel_combo.setCurrentText(
                radiator.channel
                if radiator is not None and radiator.channel in self.channel_names
                else self.channel_names[0]
            )
            self.table.setCellWidget(row, 3, channel_combo)
            self.channel_widgets.append(channel_combo)

            velocity_spin = QDoubleSpinBox()
            velocity_spin.setRange(-120.0, 60.0)
            velocity_spin.setDecimals(2)
            velocity_spin.setSingleStep(0.5)
            velocity_spin.setValue(0.0 if radiator is None else float(radiator.velocity_offset_db))
            self.table.setCellWidget(row, 4, velocity_spin)
            self.velocity_widgets.append(velocity_spin)

            mode_combo.currentTextChanged.connect(
                lambda mode, row=row: self._set_drive_controls_enabled(row, mode == "Driven")
            )
            self._set_drive_controls_enabled(row, radiator is not None)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.table)
        layout.addWidget(buttons)
        self.resize(860, min(520, 160 + 34 * max(1, len(self.surface_items))))

    def _set_drive_controls_enabled(self, row: int, enabled: bool) -> None:
        self.channel_widgets[row].setEnabled(enabled)
        self.velocity_widgets[row].setEnabled(enabled)

    def radiators(self) -> tuple[RadiatorConfig, ...]:
        radiators = []
        for row, (surface_name, (mesh_name, tag)) in enumerate(self.surface_items):
            if self.mode_widgets[row].currentText() != "Driven":
                continue
            radiators.append(
                RadiatorConfig(
                    name=surface_name,
                    mesh=mesh_name,
                    tag=tag,
                    channel=str(self.channel_widgets[row].currentText()),
                    velocity_offset_db=float(self.velocity_widgets[row].value()),
                )
            )
        return tuple(radiators)
