"""Shared geometry-generator contracts and domain models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from blab.config import RadiatorConfig
from blab.mesh_clean import MeshQualityWarning


class GenerationCancelledError(RuntimeError):
    """Raised when an active geometry generation is cancelled."""


@dataclass(frozen=True)
class GeneratorCapabilities:
    source_formats: tuple[str, ...]
    editor_kind: str = "text"
    supports_import: bool = True
    supports_export: bool = True
    supports_cancellation: bool = True
    supports_symmetry: bool = True


@dataclass(frozen=True)
class GeneratedGeometryReference:
    """Serializable paths for the last artifact produced by a design."""

    output_dir: str
    mesh_path: str
    cleaned_mesh_path: str | None = None
    reduced_cleaned_mesh_path: str | None = None
    source_path: str | None = None


@dataclass(frozen=True)
class GeneratorDocument:
    """Project-owned editable input for one geometry generator provider."""

    id: str
    name: str
    provider_id: str
    provider_schema_version: int
    source: dict[str, Any]
    mesh_enabled: bool = True
    mesh_scale_factor: float = 0.001
    mesh_translation_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    artifact: GeneratedGeometryReference | None = None


@dataclass(frozen=True)
class GenerationRequest:
    provider_id: str
    document_id: str
    mesh_name: str
    source: Mapping[str, Any]
    run_root: Path
    case_name: str
    provider_options: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)


@dataclass(frozen=True)
class GeneratedGeometry:
    """Provider-neutral geometry artifact consumed by mesh assembly."""

    provider_id: str
    output_dir: Path
    mesh_path: Path
    radiators: tuple[RadiatorConfig, ...]
    source_path: Path | None = None
    cleaned_mesh_path: Path | None = None
    reduced_cleaned_mesh_path: Path | None = None
    mirror_axes: tuple[str, ...] = ()
    quality_warning: MeshQualityWarning | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def solver_mesh_path(self) -> Path:
        return self.cleaned_mesh_path or self.mesh_path

    def solver_mesh_path_for_symmetry(self, symmetry: str) -> Path:
        if str(symmetry or "off").strip().lower() == "off":
            return self.solver_mesh_path
        return self.reduced_cleaned_mesh_path or self.mesh_path

    def to_reference(self) -> GeneratedGeometryReference:
        return GeneratedGeometryReference(
            output_dir=str(self.output_dir),
            mesh_path=str(self.mesh_path),
            cleaned_mesh_path=None if self.cleaned_mesh_path is None else str(self.cleaned_mesh_path),
            reduced_cleaned_mesh_path=(
                None if self.reduced_cleaned_mesh_path is None else str(self.reduced_cleaned_mesh_path)
            ),
            source_path=None if self.source_path is None else str(self.source_path),
        )


@dataclass(frozen=True)
class GenerationCompleted:
    request: GenerationRequest
    result: GeneratedGeometry


class GeneratorSession(Protocol):
    def generate(
        self,
        *,
        status_callback: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> GeneratedGeometry: ...

    def stop(self) -> None: ...


class GeneratorBackend(Protocol):
    provider_id: str
    label: str
    capabilities: GeneratorCapabilities

    def create_session(self, request: GenerationRequest) -> GeneratorSession: ...

    def restore(self, document: GeneratorDocument) -> GeneratedGeometry | None: ...
