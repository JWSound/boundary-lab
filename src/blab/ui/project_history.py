"""Recent project path storage."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings


RECENT_PROJECTS_SETTINGS_KEY = "projects/recent"
MAX_RECENT_PROJECTS = 10


def load_recent_project_paths(settings: QSettings) -> list[Path]:
    raw_paths = settings.value(RECENT_PROJECTS_SETTINGS_KEY, "[]")
    try:
        values = json.loads(str(raw_paths))
    except json.JSONDecodeError:
        values = []
    if not isinstance(values, list):
        return []

    paths: list[Path] = []
    seen = set()
    for value in values:
        path_text = str(value).strip()
        if not path_text:
            continue
        path = Path(path_text)
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        paths.append(path)
        if len(paths) >= MAX_RECENT_PROJECTS:
            break
    return paths


def save_recent_project_paths(settings: QSettings, paths: list[Path]) -> None:
    settings.setValue(
        RECENT_PROJECTS_SETTINGS_KEY,
        json.dumps([str(path) for path in paths[:MAX_RECENT_PROJECTS]]),
    )
    settings.sync()


def remember_recent_project(settings: QSettings, path: Path) -> None:
    try:
        normalized = path.resolve()
    except OSError:
        normalized = path

    recent = [
        existing
        for existing in load_recent_project_paths(settings)
        if str(existing).casefold() != str(normalized).casefold()
    ]
    save_recent_project_paths(settings, [normalized, *recent])


def remove_recent_project(settings: QSettings, path: Path) -> None:
    save_recent_project_paths(
        settings,
        [
            existing
            for existing in load_recent_project_paths(settings)
            if str(existing).casefold() != str(path).casefold()
        ],
    )


def clear_recent_projects(settings: QSettings) -> None:
    save_recent_project_paths(settings, [])
