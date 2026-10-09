"""Local package management; discovery displays manifests without executing code."""

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from blab.generators.catalog import provider_catalog


class ProviderPackagesDialog(QDialog):
    def __init__(self, enabled, parent=None, *, default_provider="ath", is_busy=lambda: False):
        super().__init__(parent)
        self.setWindowTitle("Generator Plugins")
        self._is_busy = is_busy
        self.enabled = set(enabled)
        self.catalog = provider_catalog()
        layout = QVBoxLayout(self)
        label = QLabel(
            "Enable plugins you trust: plugins execute Python inside Boundary Lab.\n"
            "Ath is built in. Use Rescan after adding plugins to the install folder."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Enabled", "Generator Plugin", "Version", "Status", "Folder"])
        layout.addWidget(self.table)
        self.default_provider_combo = QComboBox()
        self.default_provider_combo.addItem(default_provider, default_provider)
        form = QFormLayout()
        form.addRow("Default Generator Plugin", self.default_provider_combo)
        layout.addLayout(form)
        layout.addWidget(QLabel("Used for new designs; existing designs retain their plugin."))
        self.default_warning = QLabel()
        self.default_warning.setWordWrap(True)
        layout.addWidget(self.default_warning)
        actions = QHBoxLayout()
        rescan = QPushButton("Rescan")
        rescan.clicked.connect(self.rescan)
        actions.addWidget(rescan)
        root = self.catalog.roots[0]
        button = QPushButton("Open install folder")
        button.setToolTip(str(root))
        button.clicked.connect(lambda _checked=False: self.open_folder(root))
        actions.addWidget(button)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons = buttons
        self.default_provider_combo.currentIndexChanged.connect(self._validate_default)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.rescan()
        self.resize(950, 360)

    def open_folder(self, root):
        try:
            root.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))
        except OSError as exc:
            QMessageBox.warning(self, "Cannot open plugin folder", str(exc))

    def rescan(self):
        packages = self.catalog.scan()
        self.table.setRowCount(len(packages) + 1)
        builtin = QCheckBox()
        builtin.setChecked(True)
        builtin.setEnabled(False)
        self.table.setCellWidget(0, 0, builtin)
        for column, value in enumerate(("Ath", "", "Built-in", "Bundled with Boundary Lab"), 1):
            item = QTableWidgetItem(value)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(0, column, item)
        for row, package in enumerate(packages, 1):
            manifest = package.manifest
            check = QCheckBox()
            check.setChecked(bool(manifest and manifest.id in self.enabled))
            # A failed/changed package can still be disabled.
            check.setEnabled(manifest is not None and (not package.error or check.isChecked()))
            if manifest:
                check.toggled.connect(lambda checked, key=manifest.id: self._toggle(key, checked))
            self.table.setCellWidget(row, 0, check)
            for column, text in enumerate(
                (
                    f"{manifest.name} ({manifest.id})" if manifest else package.path.name,
                    manifest.version if manifest else "",
                    self.catalog.status(package, enabled=self.enabled),
                    str(package.path),
                ),
                1,
            ):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(text)
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._refresh_default_choices()

    def _toggle(self, provider_id, checked):
        if checked:
            self.enabled.add(provider_id)
        else:
            self.enabled.discard(provider_id)
        for row, package in enumerate(self.catalog.packages, 1):
            if package.manifest and package.manifest.id == provider_id:
                self.table.item(row, 3).setText(self.catalog.status(package, enabled=self.enabled))
        self._refresh_default_choices()

    @property
    def default_provider(self):
        return self.default_provider_combo.currentData()

    def _refresh_default_choices(self):
        selected = self.default_provider
        self.default_provider_combo.blockSignals(True)
        populate_provider_choices(self.default_provider_combo, self.enabled, selected)
        self.default_provider_combo.blockSignals(False)
        self._validate_default()

    def _validate_default(self):
        valid = self.default_provider == "ath" or any(
            package.manifest
            and package.manifest.id == self.default_provider
            and not package.error
            and package.manifest.id in self.enabled
            for package in self.catalog.packages
        )
        self.default_warning.setVisible(not valid)
        self.default_warning.setText(
            "" if valid else "Choose an enabled, available default Generator Plugin before saving."
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(valid)
        return valid

    def accept(self):
        if not self._validate_default():
            return
        if self._is_busy():
            QMessageBox.information(
                self, "Generator Plugins", "Wait for preparation, generation or solving to finish before saving."
            )
            return
        super().accept()


def populate_provider_choices(combo, enabled, selected):
    combo.clear()
    combo.addItem("Ath", "ath")
    for package in provider_catalog().packages:
        if package.manifest and not package.error and package.manifest.id in enabled:
            combo.addItem(package.manifest.name, package.manifest.id)
    index = combo.findData(selected)
    if index < 0:
        combo.addItem(f"{selected} (unavailable)", selected)
        index = combo.count() - 1
    combo.setCurrentIndex(index)
