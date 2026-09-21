"""Local package management; discovery displays manifests without executing code."""

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
    def __init__(self, enabled, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Geometry Providers")
        self.enabled = set(enabled)
        self.catalog = provider_catalog()
        layout = QVBoxLayout(self)
        label = QLabel(
            "Enable packages you trust: providers execute Python inside Boundary Lab.\n"
            "Ath is built in. Changes to loaded package code require an application restart."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Enabled", "Provider", "Version", "Status", "Folder"])
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
            QMessageBox.warning(self, "Cannot open provider folder", str(exc))

    def rescan(self):
        packages = self.catalog.scan()
        self.table.setRowCount(len(packages))
        for row, package in enumerate(packages):
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
        for row, package in enumerate(self.catalog.packages):
            if package.manifest and package.manifest.id == provider_id:
                self.table.item(row, 3).setText(self.catalog.status(package, enabled=self.enabled))


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
