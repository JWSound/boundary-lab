from __future__ import annotations

from pathlib import Path

from blab.ui import file_dialogs as file_dialogs_module
from blab.ui.file_dialogs import LAST_USED_DIRECTORY_KEY, FileDialogService


class MemorySettings:
    def __init__(self, values: dict[str, object] | None = None):
        self.values = dict(values or {})
        self.sync_count = 0

    def value(self, key: str, default: object = None) -> object:
        return self.values.get(key, default)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value

    def sync(self) -> None:
        self.sync_count += 1


def test_qt_file_dialog_usage_is_centralized_in_service() -> None:
    ui_root = Path("src/blab/ui")
    direct_users = [
        path.relative_to(ui_root).as_posix()
        for path in ui_root.glob("*.py")
        if path.name != "file_dialogs.py" and "QFileDialog" in path.read_text(encoding="utf-8")
    ]

    assert direct_users == []


def test_open_file_uses_and_updates_shared_last_directory(monkeypatch, tmp_path: Path) -> None:
    previous_directory = tmp_path / "previous"
    selected_directory = tmp_path / "selected"
    previous_directory.mkdir()
    selected_directory.mkdir()
    selected_path = selected_directory / "design.cfg"
    settings = MemorySettings({LAST_USED_DIRECTORY_KEY: str(previous_directory)})
    calls = []

    def get_open_file_name(parent, title, initial_path, file_filter):
        calls.append((parent, title, initial_path, file_filter))
        return str(selected_path), "Ath config files (*.cfg)"

    monkeypatch.setattr(file_dialogs_module.QFileDialog, "getOpenFileName", get_open_file_name)
    service = FileDialogService(settings, fallback_directory=tmp_path)

    result = service.open_file(None, "Import Waveguide Design", "Ath config files (*.cfg)")

    assert result == selected_path
    assert calls == [(None, "Import Waveguide Design", str(previous_directory), "Ath config files (*.cfg)")]
    assert settings.values[LAST_USED_DIRECTORY_KEY] == str(selected_directory)
    assert settings.sync_count == 1


def test_cancelled_dialog_does_not_update_last_directory(monkeypatch, tmp_path: Path) -> None:
    remembered_directory = tmp_path / "remembered"
    remembered_directory.mkdir()
    settings = MemorySettings({LAST_USED_DIRECTORY_KEY: str(remembered_directory)})
    monkeypatch.setattr(
        file_dialogs_module.QFileDialog,
        "getOpenFileName",
        lambda *_args: ("", ""),
    )
    service = FileDialogService(settings, fallback_directory=tmp_path)

    assert service.open_file(None, "Open Project", "Projects (*.blab.json)") is None
    assert settings.values[LAST_USED_DIRECTORY_KEY] == str(remembered_directory)
    assert settings.sync_count == 0


def test_save_file_combines_last_directory_with_suggested_filename(monkeypatch, tmp_path: Path) -> None:
    remembered_directory = tmp_path / "remembered"
    remembered_directory.mkdir()
    selected_path = remembered_directory / "renamed.blab.json"
    settings = MemorySettings({LAST_USED_DIRECTORY_KEY: str(remembered_directory)})
    initial_paths = []

    def get_save_file_name(_parent, _title, initial_path, _file_filter):
        initial_paths.append(initial_path)
        return str(selected_path), "Boundary Lab projects (*.blab.json)"

    monkeypatch.setattr(file_dialogs_module.QFileDialog, "getSaveFileName", get_save_file_name)
    service = FileDialogService(settings, fallback_directory=tmp_path)

    result = service.save_file(
        None,
        "Save Project",
        "Boundary Lab projects (*.blab.json)",
        tmp_path / "elsewhere" / "speaker.blab.json",
    )

    assert result == selected_path
    assert initial_paths == [str(remembered_directory / "speaker.blab.json")]
    assert settings.values[LAST_USED_DIRECTORY_KEY] == str(remembered_directory)
    assert settings.sync_count == 1


def test_select_directory_returns_path_and_remembers_selection(monkeypatch, tmp_path: Path) -> None:
    fallback_directory = tmp_path / "fallback"
    selected_directory = tmp_path / "exports"
    fallback_directory.mkdir()
    selected_directory.mkdir()
    settings = MemorySettings()
    initial_paths = []

    def get_existing_directory(_parent, _title, initial_path):
        initial_paths.append(initial_path)
        return str(selected_directory)

    monkeypatch.setattr(file_dialogs_module.QFileDialog, "getExistingDirectory", get_existing_directory)
    service = FileDialogService(settings, fallback_directory=fallback_directory)

    result = service.select_directory(None, "Export polar data")

    assert result == selected_directory
    assert initial_paths == [str(fallback_directory)]
    assert settings.values[LAST_USED_DIRECTORY_KEY] == str(selected_directory)
    assert settings.sync_count == 1


def test_missing_remembered_directory_falls_back_safely(monkeypatch, tmp_path: Path) -> None:
    fallback_directory = tmp_path / "fallback"
    fallback_directory.mkdir()
    settings = MemorySettings({LAST_USED_DIRECTORY_KEY: str(tmp_path / "removed")})
    initial_paths = []

    def get_open_file_name(_parent, _title, initial_path, _file_filter):
        initial_paths.append(initial_path)
        return "", ""

    monkeypatch.setattr(file_dialogs_module.QFileDialog, "getOpenFileName", get_open_file_name)
    service = FileDialogService(settings, fallback_directory=fallback_directory)

    assert service.open_file(None, "Open", "All files (*)", "example.txt") is None
    assert initial_paths == [str(fallback_directory / "example.txt")]
    assert settings.values[LAST_USED_DIRECTORY_KEY] == str(tmp_path / "removed")
    assert settings.sync_count == 0
