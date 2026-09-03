"""Dialog for selecting automatic or manual plot-axis limits."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from blab.ui.numeric_locale import FlexibleDoubleValidator, format_decimal_number, parse_decimal_number
from blab.ui.plots import PlotAxisLimits


class PlotLimitsDialog(QDialog):
    """Collect one automatic/manual limit choice for a plot's primary axes."""

    def __init__(
        self,
        plot_title: str,
        current_limits: PlotAxisLimits,
        *,
        automatic: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{plot_title} Limits")
        self.setModal(True)

        layout = QVBoxLayout(self)
        self.auto_checkbox = QCheckBox("Auto")
        self.auto_checkbox.setToolTip("Automatically choose axis limits as plot data changes")
        self.auto_checkbox.setChecked(bool(automatic))
        layout.addWidget(self.auto_checkbox)

        self.limit_group = QGroupBox("Manual limits")
        form = QFormLayout(self.limit_group)
        validator = FlexibleDoubleValidator(self)
        self.x_min_edit = self._limit_edit(current_limits.x_min, validator)
        self.x_max_edit = self._limit_edit(current_limits.x_max, validator)
        self.y_min_edit = self._limit_edit(current_limits.y_min, validator)
        self.y_max_edit = self._limit_edit(current_limits.y_max, validator)
        form.addRow("X lower", self.x_min_edit)
        form.addRow("X upper", self.x_max_edit)
        form.addRow("Y lower", self.y_min_edit)
        form.addRow("Y upper", self.y_max_edit)
        layout.addWidget(self.limit_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.auto_checkbox.toggled.connect(self.limit_group.setDisabled)
        self.limit_group.setDisabled(self.auto_checkbox.isChecked())
        self.setMinimumWidth(280)

    @staticmethod
    def _limit_edit(value: float, validator: FlexibleDoubleValidator) -> QLineEdit:
        edit = QLineEdit()
        edit.setText(format_decimal_number(value, edit.locale()))
        edit.setValidator(validator)
        return edit

    def limits(self) -> PlotAxisLimits | None:
        if self.auto_checkbox.isChecked():
            return None
        values = tuple(
            self._finite_value(edit) for edit in (self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit)
        )
        return PlotAxisLimits(*values).validated()

    @staticmethod
    def _finite_value(edit: QLineEdit) -> float:
        text = edit.text().strip()
        if not text:
            raise ValueError("Enter all four limits or select Auto.")
        try:
            value = parse_decimal_number(text, edit.locale())
        except ValueError as exc:
            raise ValueError(f"'{text}' is not a valid numeric limit.") from exc
        return value

    def _accept_if_valid(self) -> None:
        try:
            self.limits()
        except ValueError as exc:
            QMessageBox.warning(self, "Plot limits", str(exc))
            return
        self.accept()


__all__ = ["PlotLimitsDialog"]
