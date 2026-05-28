"""Readable project-file helpers for the Boundary Lab GUI.

Project files capture application workflow state: editor text, mesh choices,
and GUI source assignments. Solver-domain settings stay in ``blab.config`` and
wire/API serialization stays in ``blab.protocol``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_SCHEMA_VERSION = 1
PROJECT_FILE_FILTER = "Boundary Lab project files (*.blab.json *.json);;JSON files (*.json);;All files (*)"
PROJECT_DEFAULT_NAME = "boundary_lab_project.blab.json"
PROJECT_PAYLOAD_KEYS = (
    "schema_version",
    "ath_config_text",
    "ath_scripts",
    "active_ath_script_id",
    "ath_mesh",
    "imported_meshes",
    "stitch_imported_meshes",
    "source_config_by_name",
    "channel_config_by_name",
)


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

    return resolve_project_paths(migrate_project_payload(payload), project_path.parent)


def resolve_project_paths(payload: dict[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """Resolve project-relative file paths against ``base_dir``."""
    resolved = dict(payload)
    base_path = Path(base_dir)

    ath_mesh = _dict_or_empty(resolved.get("ath_mesh")).copy()
    _resolve_path_fields(ath_mesh, base_path, ("source_file", "cleaned_file"))
    resolved["ath_mesh"] = ath_mesh

    imported_meshes = []
    for item in _list_or_empty(resolved.get("imported_meshes")):
        if not isinstance(item, dict):
            imported_meshes.append(item)
            continue
        mesh = item.copy()
        _resolve_path_fields(mesh, base_path, ("source_file", "cleaned_file"))
        imported_meshes.append(mesh)
    resolved["imported_meshes"] = imported_meshes

    ath_scripts = []
    for item in _list_or_empty(resolved.get("ath_scripts")):
        if not isinstance(item, dict):
            ath_scripts.append(item)
            continue
        script = item.copy()
        _resolve_path_fields(
            script,
            base_path,
            ("output_dir", "stl_path", "msh_path", "cleaned_msh_path", "config_path"),
        )
        ath_scripts.append(script)
    resolved["ath_scripts"] = ath_scripts

    return resolved


def migrate_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized current-schema project payload."""
    schema_version = _schema_version(payload)
    if schema_version != PROJECT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported project schema version {schema_version}. "
            f"Expected {PROJECT_SCHEMA_VERSION}."
        )

    return _normalize_project_payload(dict(payload))


def _schema_version(payload: dict[str, Any]) -> int:
    if "schema_version" not in payload:
        raise ValueError("Project file is missing schema_version.")
    raw_version = payload["schema_version"]
    try:
        return int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Project schema_version must be an integer, got {raw_version!r}.") from exc


def _normalize_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "ath_config_text": str(payload.get("ath_config_text", "")),
        "ath_scripts": _list_or_empty(payload.get("ath_scripts")),
        "active_ath_script_id": _optional_str(payload.get("active_ath_script_id")),
        "ath_mesh": _dict_or_empty(payload.get("ath_mesh")),
        "imported_meshes": _list_or_empty(payload.get("imported_meshes")),
        "stitch_imported_meshes": bool(payload.get("stitch_imported_meshes", False)),
        "source_config_by_name": _dict_or_empty(payload.get("source_config_by_name")),
        "channel_config_by_name": _dict_or_empty(payload.get("channel_config_by_name")),
    }


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_path_fields(payload: dict[str, Any], base_dir: Path, fields: tuple[str, ...]) -> None:
    for field in fields:
        value = payload.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        path = Path(text)
        if path.is_absolute():
            continue
        payload[field] = str((base_dir / path).resolve())


def build_project_payload(
    *,
    ath_config_text: str,
    ath_mesh: dict[str, Any],
    imported_meshes: list[dict[str, Any]],
    source_config_by_name: dict[str, Any],
    stitch_imported_meshes: bool = False,
    ath_scripts: list[dict[str, Any]] | None = None,
    active_ath_script_id: str | None = None,
    channel_config_by_name: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "ath_config_text": ath_config_text,
        "ath_scripts": ath_scripts or [],
        "active_ath_script_id": active_ath_script_id,
        "ath_mesh": ath_mesh,
        "imported_meshes": imported_meshes,
        "stitch_imported_meshes": bool(stitch_imported_meshes),
        "source_config_by_name": source_config_by_name,
        "channel_config_by_name": channel_config_by_name or {},
    }
