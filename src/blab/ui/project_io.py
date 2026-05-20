"""Readable project-file helpers for the Boundary Lab GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = 1
PROJECT_FILE_FILTER = "Boundary Lab project files (*.blab.json *.json);;JSON files (*.json);;All files (*)"
PROJECT_DEFAULT_NAME = "boundary_lab_project.blab.json"


def normalize_project_path(path: str | Path) -> Path:
    project_path = Path(path)
    if project_path.suffix == "":
        project_path = project_path.with_suffix(".blab.json")
    return project_path


def write_project_file(path: str | Path, payload: dict[str, Any]) -> Path:
    project_path = normalize_project_path(path)
    project_path.parent.mkdir(parents=True, exist_ok=True)
    project_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    return project_path


def read_project_file(path: str | Path) -> dict[str, Any]:
    project_path = Path(path)
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Project file must contain a JSON object.")

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version {schema_version}. "
            f"Expected {PROJECT_SCHEMA_VERSION}."
        )
    return payload


def build_project_payload(
    *,
    ath_config_text: str,
    ath_mesh: dict[str, Any],
    imported_meshes: list[dict[str, Any]],
    source_config_by_name: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "ath_config_text": ath_config_text,
        "ath_mesh": ath_mesh,
        "imported_meshes": imported_meshes,
        "source_config_by_name": source_config_by_name,
    }
