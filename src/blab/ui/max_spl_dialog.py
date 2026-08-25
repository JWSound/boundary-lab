"""Channel-rating dialog used by the maximum-SPL plot."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from blab.max_spl import MaxSplLimit


class MaxSplLimitsDialog(QDialog):
    """Collect shared Xmax and Pmax ratings for eligible channels."""

    def __init__(
        self,
        channel_names: tuple[str, ...],
        saved_limits: dict[str, MaxSplLimit],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Maximum SPL Configuration")
        self.setModal(True)
        self._channel_names = tuple(channel_names)
        self._xmax_widgets: list[QDoubleSpinBox] = []
        self._pmax_widgets: list[QDoubleSpinBox] = []

        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Enter one-way peak Xmax and rated Pmax per physical driver. "
            "The values are applied to every electrodynamic component on the channel. "
            "Set both values to zero to disable a channel."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.table = QTableWidget(len(self._channel_names), 3)
        self.table.setHorizontalHeaderLabels(("Channel", "Xmax (mm, peak)", "Pmax (W / driver)"))
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        for row, channel_name in enumerate(self._channel_names):
            name_item = QTableWidgetItem(channel_name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            saved = saved_limits.get(channel_name)

            xmax_spin = QDoubleSpinBox()
            xmax_spin.setRange(0.0, 1000.0)
            xmax_spin.setDecimals(3)
            xmax_spin.setSingleStep(0.1)
            xmax_spin.setSuffix(" mm")
            xmax_spin.setValue(0.0 if saved is None else saved.xmax_mm)
            self.table.setCellWidget(row, 1, xmax_spin)
            self._xmax_widgets.append(xmax_spin)

            pmax_spin = QDoubleSpinBox()
            pmax_spin.setRange(0.0, 100000.0)
            pmax_spin.setDecimals(1)
            pmax_spin.setSingleStep(10.0)
            pmax_spin.setSuffix(" W")
            pmax_spin.setValue(0.0 if saved is None else saved.pmax_w)
            self.table.setCellWidget(row, 2, pmax_spin)
            self._pmax_widgets.append(pmax_spin)
        layout.addWidget(self.table)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.AcceptRole)
        apply_button.clicked.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(560, min(500, 150 + 38 * len(self._channel_names)))

    def limits(self) -> dict[str, MaxSplLimit]:
        return {
            channel_name: MaxSplLimit(
                xmax_mm=float(self._xmax_widgets[row].value()),
                pmax_w=float(self._pmax_widgets[row].value()),
            ).validated()
            for row, channel_name in enumerate(self._channel_names)
        }

    def _accept_if_valid(self) -> None:
        try:
            self.limits()
        except ValueError as exc:
            QMessageBox.warning(self, "Maximum SPL ratings", str(exc))
            return
        self.accept()


__all__ = ["MaxSplLimitsDialog"]
