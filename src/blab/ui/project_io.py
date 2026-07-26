"""Readable project-file helpers for the Boundary Lab GUI.

Project files capture application workflow state: editor text, mesh choices,
and GUI source assignments. Solver-domain settings stay in ``blab.config`` and
wire/API serialization stays in ``blab.protocol``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blab.ui.project_state import ProjectPreferencesState

PROJECT_SCHEMA_VERSION = 4
PROJECT_FILE_FILTER = "Boundary Lab project files (*.blab.json *.json);;JSON files (*.json);;All files (*)"
PROJECT_DEFAULT_NAME = "boundary_lab_project.blab.json"
PROJECT_PAYLOAD_KEYS = (
    "schema_version",
    "generator_documents",
    "active_generator_document_id",
    "imported_meshes",
    "stitch_imported_meshes",
    "symmetry",
    "source_config_by_name",
    "channel_config_by_name",
    "project_preferences",
    "physical_system",
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

    imported_meshes = []
    for item in _list_or_empty(resolved.get("imported_meshes")):
        if not isinstance(item, dict):
            imported_meshes.append(item)
            continue
        mesh = item.copy()
        _resolve_path_fields(mesh, base_path, ("source_file", "cleaned_file"))
        imported_meshes.append(mesh)
    resolved["imported_meshes"] = imported_meshes

    generator_documents = []
    for item in _list_or_empty(resolved.get("generator_documents")):
        if not isinstance(item, dict):
            generator_documents.append(item)
            continue
        document = item.copy()
        artifact = _dict_or_empty(document.get("artifact")).copy()
        _resolve_path_fields(
            artifact,
            base_path,
            ("output_dir", "mesh_path", "cleaned_mesh_path", "reduced_cleaned_mesh_path", "source_path"),
        )
        document["artifact"] = artifact or None
        generator_documents.append(document)
    resolved["generator_documents"] = generator_documents

    physical_system = resolved.get("physical_system")
    if isinstance(physical_system, dict):
        physical_system = physical_system.copy()
        meshes = []
        for item in _list_or_empty(physical_system.get("meshes")):
            if not isinstance(item, dict):
                meshes.append(item)
                continue
            mesh = item.copy()
            _resolve_path_fields(mesh, base_path, ("file",))
            meshes.append(mesh)
        physical_system["meshes"] = meshes
        resolved["physical_system"] = physical_system

    return resolved


def migrate_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized current-schema project payload."""
    schema_version = _schema_version(payload)
    if schema_version not in {1, 2, 3, PROJECT_SCHEMA_VERSION}:
        raise ValueError(
            f"Unsupported project schema version {schema_version}. Expected 1, 2, 3, or {PROJECT_SCHEMA_VERSION}."
        )
    migrated = dict(payload)
    if schema_version in {1, 2}:
        migrated = _migrate_ath_documents(migrated)
    return _normalize_project_payload(migrated)


def _schema_version(payload: dict[str, Any]) -> int:
    if "schema_version" not in payload:
        raise ValueError("Project file is missing schema_version.")
    raw_version = payload["schema_version"]
    try:
        return int(raw_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Project schema_version must be an integer, got {raw_version!r}.") from exc


def _normalize_project_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "generator_documents": _list_or_empty(payload.get("generator_documents")),
        "active_generator_document_id": _optional_str(payload.get("active_generator_document_id")),
        "imported_meshes": _list_or_empty(payload.get("imported_meshes")),
        "stitch_imported_meshes": bool(payload.get("stitch_imported_meshes", False)),
        "symmetry": _normalize_symmetry(payload.get("symmetry", "off")),
        "source_config_by_name": _dict_or_empty(payload.get("source_config_by_name")),
        "channel_config_by_name": _dict_or_empty(payload.get("channel_config_by_name")),
    }
    if payload.get("project_preferences") is not None:
        preferences = ProjectPreferencesState.from_payload(payload["project_preferences"])
        if preferences is not None:
            normalized["project_preferences"] = preferences.to_payload()
    if payload.get("physical_system") is not None:
        if not isinstance(payload["physical_system"], dict):
            raise ValueError("physical_system must be an object when provided.")
        normalized["physical_system"] = dict(payload["physical_system"])
    return normalized


def _migrate_ath_documents(payload: dict[str, Any]) -> dict[str, Any]:
    legacy_scripts = _list_or_empty(payload.get("ath_scripts"))
    if not legacy_scripts:
        legacy_scripts = [
            {
                "id": "legacy-ath",
                "name": "waveguide",
                "config_text": str(payload.get("ath_config_text", "")),
            }
        ]

    legacy_mesh = _dict_or_empty(payload.get("ath_mesh"))
    documents = []
    for index, item in enumerate(legacy_scripts):
        if not isinstance(item, dict):
            continue
        mesh_path = _optional_str(item.get("msh_path"))
        cleaned_mesh_path = _optional_str(item.get("cleaned_msh_path"))
        output_dir = _optional_str(item.get("output_dir"))
        if index == 0 and mesh_path is None:
            mesh_path = _optional_str(legacy_mesh.get("source_file"))
            cleaned_mesh_path = cleaned_mesh_path or _optional_str(legacy_mesh.get("cleaned_file"))
            if output_dir is None and mesh_path is not None:
                output_dir = str(Path(mesh_path).parent)
        artifact = None
        if mesh_path is not None and output_dir is not None:
            artifact = {
                "output_dir": output_dir,
                "mesh_path": mesh_path,
                "cleaned_mesh_path": cleaned_mesh_path,
                "reduced_cleaned_mesh_path": None,
                "source_path": _optional_str(item.get("config_path")),
            }
        documents.append(
            {
                "id": _optional_str(item.get("id")) or f"legacy-ath-{index + 1}",
                "name": _optional_str(item.get("name")) or "waveguide",
                "provider_id": "ath",
                "provider_schema_version": 1,
                "source": {"format": "ath_cfg", "text": str(item.get("config_text", ""))},
                "mesh_enabled": bool(item.get("mesh_enabled", True)),
                "mesh_scale_factor": item.get("mesh_scale_factor", 0.001),
                "mesh_translation_mm": item.get("mesh_translation_mm", [0, 0, 0]),
                "artifact": artifact,
            }
        )

    migrated = dict(payload)
    migrated["schema_version"] = PROJECT_SCHEMA_VERSION
    migrated["generator_documents"] = documents
    active_id = _optional_str(payload.get("active_ath_script_id"))
    migrated["active_generator_document_id"] = active_id or (documents[0]["id"] if documents else None)
    return migrated


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_or_empty(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_symmetry(value: Any) -> str:
    symmetry = str(value or "off").strip().lower()
    return symmetry if symmetry in {"off", "x", "xy"} else "off"


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
    generator_documents: list[dict[str, Any]],
    active_generator_document_id: str | None,
    imported_meshes: list[dict[str, Any]],
    source_config_by_name: dict[str, Any],
    stitch_imported_meshes: bool = False,
    symmetry: str = "off",
    channel_config_by_name: dict[str, Any] | None = None,
    project_preferences: dict[str, Any] | None = None,
    physical_system: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "generator_documents": generator_documents,
        "active_generator_document_id": active_generator_document_id,
        "imported_meshes": imported_meshes,
        "stitch_imported_meshes": bool(stitch_imported_meshes),
        "symmetry": _normalize_symmetry(symmetry),
        "source_config_by_name": source_config_by_name,
        "channel_config_by_name": channel_config_by_name or {},
    }
    if project_preferences is not None:
        normalized_preferences = ProjectPreferencesState.from_payload(project_preferences)
        if normalized_preferences is not None:
            payload["project_preferences"] = normalized_preferences.to_payload()
    if physical_system is not None:
        payload["physical_system"] = dict(physical_system)
    return payload
