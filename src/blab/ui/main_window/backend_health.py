"""Capabilities of the selected physical-system solver backend."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import QObject, Signal

from blab.solvers.registry import backend_info
from blab.ui.settings import GuiPreferences


class BackendHealthController(QObject):
    capability_changed = Signal(str)

    def __init__(self, parent: QObject | None, *, preferences: Callable[[], GuiPreferences]) -> None:
        super().__init__(parent)
        self._read_preferences = preferences

    def supports_symmetry(self, backend_id: str) -> bool:
        return backend_info(backend_id).capabilities.supports_symmetry

    def selected_backend_supports_symmetry(self) -> bool:
        return self.supports_symmetry(self._read_preferences().solve_backend)

    def effective_symmetry(self, symmetry: str, preferences: GuiPreferences) -> str:
        return symmetry if self.supports_symmetry(preferences.solve_backend) else "off"
