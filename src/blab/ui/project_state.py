"""Lightweight GUI project-state helpers."""

from __future__ import annotations

import math
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from blab.generators.base import GeneratedGeometryReference, GeneratorDocument
from blab.physical_model import PhysicalSystem

DEFAULT_DESIGN_NAME = "waveguide"
DEFAULT_MESH_SCALE_FACTOR = 0.001
DEFAULT_GENERATOR_PROVIDER_ID = "ath"
DEFAULT_GENERATOR_PROVIDER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProjectPreferencesState:
    """Project preferences that affect reproducibility or interpretation.

    Solver selection, server URL, GMRES tolerance, and Burton-Miller selection
    intentionally remain application-owned and are not part of this model.
    """

    freq_min_hz: int = 200
    freq_max_hz: int = 20000
    freq_count: int = 41
    polar_angle_step_deg: float = 10.0
    polar_observation_distance_m: float = 2.0
    normalized_channel_correction: bool = True
    fem_bulk_loss_factor: float = 0.0
    polar_smoothing: int | None = 48
    horizontal_normalization_angle: float = 10.0
    vertical_normalization_angle: float = 10.0
    spin_horizontal_reference_angle: float = 0.0
    spin_vertical_reference_angle: float = 0.0
    spl_max_db: float = 0.0
    spl_min_db: float = -30.0
    isobar_contour_step_db: float = 3.0
    stitch_tolerance_mm: float = 2.0
    spherical_sampling_enabled: bool = False
    balloon_angle_precision_deg: float = 2.5

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: object) -> ProjectPreferencesState | None:
        if payload is None:
            return None
        if not isinstance(payload, dict):
            raise ValueError("project_preferences must be an object when provided.")

        defaults = cls()
        freq_min_hz = _bounded_int(payload.get("freq_min_hz"), defaults.freq_min_hz, minimum=1)
        freq_max_hz = _bounded_int(payload.get("freq_max_hz"), defaults.freq_max_hz, minimum=1)
        if freq_min_hz > freq_max_hz:
            freq_min_hz, freq_max_hz = freq_max_hz, freq_min_hz
        return cls(
            freq_min_hz=freq_min_hz,
            freq_max_hz=freq_max_hz,
            freq_count=_bounded_int(payload.get("freq_count"), defaults.freq_count, minimum=1),
            polar_angle_step_deg=_finite_float(
                payload.get("polar_angle_step_deg"), defaults.polar_angle_step_deg, minimum=0.000001
            ),
            polar_observation_distance_m=_finite_float(
                payload.get("polar_observation_distance_m"),
                defaults.polar_observation_distance_m,
                minimum=0.000001,
            ),
            normalized_channel_correction=_boolean(
                payload.get("normalized_channel_correction"), defaults.normalized_channel_correction
            ),
            fem_bulk_loss_factor=_finite_float(
                payload.get("fem_bulk_loss_factor"),
                defaults.fem_bulk_loss_factor,
                minimum=0.0,
                maximum=1.0,
            ),
            polar_smoothing=_optional_int(payload.get("polar_smoothing"), defaults.polar_smoothing),
            horizontal_normalization_angle=_finite_float(
                payload.get("horizontal_normalization_angle"), defaults.horizontal_normalization_angle
            ),
            vertical_normalization_angle=_finite_float(
                payload.get("vertical_normalization_angle"), defaults.vertical_normalization_angle
            ),
            spin_horizontal_reference_angle=_finite_float(
                payload.get("spin_horizontal_reference_angle"), defaults.spin_horizontal_reference_angle
            ),
            spin_vertical_reference_angle=_finite_float(
                payload.get("spin_vertical_reference_angle"), defaults.spin_vertical_reference_angle
            ),
            spl_max_db=_finite_float(payload.get("spl_max_db"), defaults.spl_max_db),
            spl_min_db=_finite_float(payload.get("spl_min_db"), defaults.spl_min_db),
            isobar_contour_step_db=_finite_float(
                payload.get("isobar_contour_step_db"), defaults.isobar_contour_step_db, minimum=0.0
            ),
            stitch_tolerance_mm=_finite_float(
                payload.get("stitch_tolerance_mm"), defaults.stitch_tolerance_mm, minimum=0.0
            ),
            spherical_sampling_enabled=_boolean(
                payload.get("spherical_sampling_enabled"), defaults.spherical_sampling_enabled
            ),
            balloon_angle_precision_deg=_finite_float(
                payload.get("balloon_angle_precision_deg"), defaults.balloon_angle_precision_deg, minimum=0.000001
            ),
        )


@dataclass(frozen=True)
class ImportedMeshState:
    """Project-owned imported mesh state, independent of Qt dialog models."""

    name: str
    source_file: str
    cleaned_file: str | None = None
    scale_factor: float = DEFAULT_MESH_SCALE_FACTOR
    translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    enabled: bool = True


@dataclass
class ProjectDocument:
    """Canonical in-memory owner of project-local editable state."""

    generator_documents: tuple[GeneratorDocument, ...]
    active_generator_document_id: str | None
    imported_meshes: tuple[ImportedMeshState, ...] = ()
    stitch_imported_meshes: bool = False
    symmetry: str = "off"
    source_config_by_name: dict[str, dict] = field(default_factory=dict)
    channel_config_by_name: dict[str, dict] = field(default_factory=dict)
    project_preferences: ProjectPreferencesState | None = None
    physical_system: PhysicalSystem | None = None
    component_channel_by_id: dict[str, str] = field(default_factory=dict)


def new_generator_document(
    name: str,
    source_text: str = "",
    *,
    provider_id: str = DEFAULT_GENERATOR_PROVIDER_ID,
    provider_schema_version: int = DEFAULT_GENERATOR_PROVIDER_SCHEMA_VERSION,
    source: dict[str, Any] | None = None,
) -> GeneratorDocument:
    if source is None:
        source = (
            {"format": "ath_cfg", "text": str(source_text)}
            if provider_id == DEFAULT_GENERATOR_PROVIDER_ID
            else {"format": "structured", "values": {}}
        )
    return GeneratorDocument(
        id=uuid.uuid4().hex[:12],
        name=name,
        provider_id=provider_id,
        provider_schema_version=max(int(provider_schema_version), 1),
        source=dict(source),
    )


def default_generator_documents(config_text: str = "") -> tuple[GeneratorDocument, ...]:
    return (new_generator_document(DEFAULT_DESIGN_NAME, config_text),)


def new_project_document(
    config_text: str = "",
    *,
    project_preferences: ProjectPreferencesState | None = None,
) -> ProjectDocument:
    documents = default_generator_documents(config_text)
    return ProjectDocument(
        generator_documents=documents,
        active_generator_document_id=documents[0].id if documents else None,
        project_preferences=project_preferences,
    )


def sanitized_name(name: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in name).strip("_")
    return sanitized or DEFAULT_DESIGN_NAME


def generator_mesh_name(document: GeneratorDocument) -> str:
    return sanitized_name(document.name)


def unique_generator_name(base_name: str, documents: tuple[GeneratorDocument, ...]) -> str:
    base = sanitized_name(base_name)
    used = {document.name for document in documents}
    if base not in used:
        return base
    suffix = 2
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def generator_document_to_payload(
    document: GeneratorDocument,
    *,
    absolute_paths: bool = False,
) -> dict[str, Any]:
    return {
        "id": document.id,
        "name": document.name,
        "provider_id": document.provider_id,
        "provider_schema_version": int(document.provider_schema_version),
        "source": document.source,
        "mesh_enabled": bool(document.mesh_enabled),
        "mesh_scale_factor": float(document.mesh_scale_factor),
        "mesh_translation_mm": [int(round(value)) for value in document.mesh_translation_mm],
        "artifact": _artifact_to_payload(document.artifact, absolute_paths=absolute_paths),
    }


def generator_document_from_payload(payload: object) -> GeneratorDocument | None:
    if not isinstance(payload, dict):
        return None
    document_id = str(payload.get("id", "")).strip() or uuid.uuid4().hex[:12]
    name = str(payload.get("name", DEFAULT_DESIGN_NAME)).strip() or DEFAULT_DESIGN_NAME
    provider_id = (
        str(payload.get("provider_id", DEFAULT_GENERATOR_PROVIDER_ID)).strip() or DEFAULT_GENERATOR_PROVIDER_ID
    )
    try:
        provider_schema_version = int(payload.get("provider_schema_version", 1))
    except (TypeError, ValueError):
        provider_schema_version = 1
    source = payload.get("source")
    if not isinstance(source, dict):
        source = {"format": "ath_cfg", "text": str(payload.get("config_text", ""))}
    translation = payload.get("mesh_translation_mm", [0.0, 0.0, 0.0])
    if not isinstance(translation, list) or len(translation) != 3:
        translation = [0.0, 0.0, 0.0]
    return GeneratorDocument(
        id=document_id,
        name=name,
        provider_id=provider_id,
        provider_schema_version=max(provider_schema_version, 1),
        source=dict(source),
        mesh_enabled=bool(payload.get("mesh_enabled", True)),
        mesh_scale_factor=_positive_float(payload.get("mesh_scale_factor"), DEFAULT_MESH_SCALE_FACTOR),
        mesh_translation_mm=tuple(float(int(round(float(value)))) for value in translation),
        artifact=_artifact_from_payload(payload.get("artifact")),
    )


def generator_documents_from_payload(
    payload: object,
    *,
    fallback_config_text: str = "",
) -> tuple[GeneratorDocument, ...]:
    if isinstance(payload, list):
        documents = tuple(
            document for item in payload if (document := generator_document_from_payload(item)) is not None
        )
        if documents:
            return documents
    return default_generator_documents(fallback_config_text)


def replace_generator_document(
    documents: tuple[GeneratorDocument, ...],
    document_id: str,
    **changes,
) -> tuple[GeneratorDocument, ...]:
    return tuple(replace(document, **changes) if document.id == document_id else document for document in documents)


def _artifact_to_payload(
    artifact: GeneratedGeometryReference | None,
    *,
    absolute_paths: bool,
) -> dict[str, Any] | None:
    if artifact is None:
        return None
    return {
        "output_dir": _path_payload(artifact.output_dir, absolute_paths),
        "mesh_path": _path_payload(artifact.mesh_path, absolute_paths),
        "cleaned_mesh_path": _path_payload(artifact.cleaned_mesh_path, absolute_paths),
        "reduced_cleaned_mesh_path": _path_payload(artifact.reduced_cleaned_mesh_path, absolute_paths),
        "source_path": _path_payload(artifact.source_path, absolute_paths),
    }


def _artifact_from_payload(payload: object) -> GeneratedGeometryReference | None:
    if not isinstance(payload, dict):
        return None
    output_dir = _optional_path_text(payload.get("output_dir"))
    mesh_path = _optional_path_text(payload.get("mesh_path"))
    if output_dir is None or mesh_path is None:
        return None
    return GeneratedGeometryReference(
        output_dir=output_dir,
        mesh_path=mesh_path,
        cleaned_mesh_path=_optional_path_text(payload.get("cleaned_mesh_path")),
        reduced_cleaned_mesh_path=_optional_path_text(payload.get("reduced_cleaned_mesh_path")),
        source_path=_optional_path_text(payload.get("source_path")),
    )


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


def _finite_float(
    value: object,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if (
        not math.isfinite(parsed)
        or (minimum is not None and parsed < minimum)
        or (maximum is not None and parsed > maximum)
    ):
        return default
    return parsed


def _bounded_int(value: object, default: int, *, minimum: int | None = None) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and parsed < minimum:
        return default
    return parsed


def _optional_int(value: object, default: int | None) -> int | None:
    if value is None or str(value).strip().lower() in {"", "none", "off"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default
