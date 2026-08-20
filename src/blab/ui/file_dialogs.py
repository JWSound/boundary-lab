"""Persistent, path-oriented wrappers around Qt file dialogs."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QWidget

from blab.ui.settings import application_settings

LAST_USED_DIRECTORY_KEY = "file_dialogs/last_used_directory"


class FileDialogService:
    """Open Qt file dialogs from one shared, persistent directory."""

    def __init__(
        self,
        settings: QSettings | None = None,
        *,
        fallback_directory: str | Path | None = None,
    ) -> None:
        self.settings = settings if settings is not None else application_settings()
        self.fallback_directory = Path.home() if fallback_directory is None else Path(fallback_directory)

    def open_file(
        self,
        parent: QWidget | None,
        title: str,
        file_filter: str = "All files (*)",
        suggested_filename: str | Path | None = None,
    ) -> Path | None:
        path_text, _selected_filter = QFileDialog.getOpenFileName(
            parent,
            title,
            str(self._initial_path(suggested_filename)),
            file_filter,
        )
        return self._accepted_file(path_text)

    def save_file(
        self,
        parent: QWidget | None,
        title: str,
        file_filter: str = "All files (*)",
        suggested_filename: str | Path | None = None,
    ) -> Path | None:
        path_text, _selected_filter = QFileDialog.getSaveFileName(
            parent,
            title,
            str(self._initial_path(suggested_filename)),
            file_filter,
        )
        return self._accepted_file(path_text)

    def select_directory(
        self,
        parent: QWidget | None,
        title: str,
        file_filter: str = "",
        suggested_filename: str | Path | None = None,
    ) -> Path | None:
        del file_filter  # Kept for a consistent service call signature.
        directory_text = QFileDialog.getExistingDirectory(
            parent,
            title,
            str(self._initial_path(suggested_filename)),
        )
        if not directory_text:
            return None
        directory = Path(directory_text)
        self._remember_directory(directory)
        return directory

    def last_used_directory(self) -> Path:
        remembered = self.settings.value(LAST_USED_DIRECTORY_KEY)
        remembered_text = "" if remembered is None else str(remembered).strip()
        candidates = (
            None if not remembered_text else Path(remembered_text),
            self.fallback_directory,
            Path.home(),
            Path.cwd(),
        )
        for candidate in candidates:
            if candidate is not None and candidate.is_dir():
                return candidate
        return Path.cwd()

    def _initial_path(self, suggested_filename: str | Path | None) -> Path:
        directory = self.last_used_directory()
        if suggested_filename is None:
            return directory
        filename = Path(suggested_filename).name
        return directory / filename if filename else directory

    def _accepted_file(self, path_text: str) -> Path | None:
        if not path_text:
            return None
        path = Path(path_text)
        self._remember_directory(path.parent)
        return path

    def _remember_directory(self, directory: Path) -> None:
        self.settings.setValue(LAST_USED_DIRECTORY_KEY, str(directory))
        self.settings.sync()
