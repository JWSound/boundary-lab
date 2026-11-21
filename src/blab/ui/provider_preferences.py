"""Local package management; discovery displays manifests without executing code."""

import sys

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
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
    def __init__(self, enabled, parent=None, *, is_busy=lambda: False):
        super().__init__(parent)
        self.setWindowTitle("Generator Plugins")
        self._is_busy = is_busy
        self.enabled = set(enabled)
        self.catalog = provider_catalog()
        layout = QVBoxLayout(self)
        label = QLabel(
            "Enable plugins you trust: plugins execute Python inside Boundary Lab.\n"
            + (
                "The Ath adapter is built in; configure external tools in Edit → Ath Setup…. "
                if sys.platform == "darwin"
                else "Ath is built in. "
            )
            + "Use Rescan after adding plugins to the install folder."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Enabled", "Generator Plugin", "Version", "Status", "Folder"])
        layout.addWidget(self.table)
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
        ath_status = "Built-in"
        if sys.platform == "darwin":
            from blab.ath_setup import ATH_PATH_KEY, WINE_PATH_KEY, validate_paths
            from blab.ui.settings import application_settings

            settings = application_settings()
            try:
                validate_paths(str(settings.value(ATH_PATH_KEY, "")), str(settings.value(WINE_PATH_KEY, "")))
                ath_status = "Configured"
            except ValueError:
                ath_status = "Setup required"
        for column, value in enumerate(
            (
                "Ath",
                "",
                ath_status,
                "External Ath and Wine (Edit → Ath Setup…)"
                if sys.platform == "darwin"
                else "Bundled with Boundary Lab",
            ),
            1,
        ):
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

    def _toggle(self, provider_id, checked):
        if checked:
            self.enabled.add(provider_id)
        else:
            self.enabled.discard(provider_id)
        for row, package in enumerate(self.catalog.packages, 1):
            if package.manifest and package.manifest.id == provider_id:
                self.table.item(row, 3).setText(self.catalog.status(package, enabled=self.enabled))

    def accept(self):
        if self._is_busy():
            QMessageBox.information(
                self, "Generator Plugins", "Wait for preparation, generation or solving to finish before saving."
            )
            return
        super().accept()
