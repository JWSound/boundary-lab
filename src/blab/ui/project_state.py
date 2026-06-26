"""Lightweight GUI project-state helpers."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_SCRIPT_NAME = "ath"
DEFAULT_MESH_SCALE_FACTOR = 0.001


@dataclass(frozen=True)
class AthScriptState:
    id: str
    name: str
    config_text: str
    mesh_enabled: bool = True
    mesh_scale_factor: float = DEFAULT_MESH_SCALE_FACTOR
    mesh_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    output_dir: str | None = None
    msh_path: str | None = None
    cleaned_msh_path: str | None = None
    config_path: str | None = None

    @property
    def mesh_name(self) -> str:
        return sanitized_name(self.name)


def new_script(name: str, config_text: str = "") -> AthScriptState:
    return AthScriptState(id=uuid.uuid4().hex[:12], name=name, config_text=config_text)


def default_scripts(config_text: str = "") -> tuple[AthScriptState, ...]:
    return (new_script(DEFAULT_SCRIPT_NAME, config_text),)


def sanitized_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in name).strip("_")
    return sanitized or DEFAULT_SCRIPT_NAME


def unique_script_name(base_name: str, scripts: tuple[AthScriptState, ...]) -> str:
    base = sanitized_name(base_name)
    used = {script.name for script in scripts}
    if base not in used:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def script_to_payload(script: AthScriptState, *, absolute_paths: bool = False) -> dict[str, Any]:
    return {
        "id": script.id,
        "name": script.name,
        "config_text": script.config_text,
        "mesh_enabled": bool(script.mesh_enabled),
        "mesh_scale_factor": float(script.mesh_scale_factor),
        "mesh_translation_mm": [int(round(value)) for value in script.mesh_translation_mm],
        "output_dir": _path_payload(script.output_dir, absolute_paths),
        "msh_path": _path_payload(script.msh_path, absolute_paths),
        "cleaned_msh_path": _path_payload(script.cleaned_msh_path, absolute_paths),
        "config_path": _path_payload(script.config_path, absolute_paths),
    }


def script_from_payload(payload: object) -> AthScriptState | None:
    if not isinstance(payload, dict):
        return None
    script_id = str(payload.get("id", "")).strip() or uuid.uuid4().hex[:12]
    name = str(payload.get("name", DEFAULT_SCRIPT_NAME)).strip() or DEFAULT_SCRIPT_NAME
    translation = payload.get("mesh_translation_mm", [0.0, 0.0, 0.0])
    if not isinstance(translation, list) or len(translation) != 3:
        translation = [0.0, 0.0, 0.0]
    return AthScriptState(
        id=script_id,
        name=name,
        config_text=str(payload.get("config_text", "")),
        mesh_enabled=bool(payload.get("mesh_enabled", True)),
        mesh_scale_factor=_positive_float(payload.get("mesh_scale_factor"), DEFAULT_MESH_SCALE_FACTOR),
        mesh_translation_mm=tuple(float(int(round(float(value)))) for value in translation),
        output_dir=_optional_path_text(payload.get("output_dir")),
        msh_path=_optional_path_text(payload.get("msh_path")),
        cleaned_msh_path=_optional_path_text(payload.get("cleaned_msh_path")),
        config_path=_optional_path_text(payload.get("config_path")),
    )


def scripts_from_payload(payload: object, *, fallback_config_text: str = "") -> tuple[AthScriptState, ...]:
    if isinstance(payload, list):
        scripts = tuple(script for item in payload if (script := script_from_payload(item)) is not None)
        if scripts:
            return scripts
    return default_scripts(fallback_config_text)


def replace_script(
    scripts: tuple[AthScriptState, ...],
    script_id: str,
    **changes,
) -> tuple[AthScriptState, ...]:
    return tuple(replace(script, **changes) if script.id == script_id else script for script in scripts)


def _path_payload(path_text: str | None, absolute_paths: bool) -> str | None:
    if not path_text:
        return None
    return str(Path(path_text).resolve()) if absolute_paths else path_text


def _optional_path_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _positive_float(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0.0 else default
