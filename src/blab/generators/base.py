"""Shared geometry-generator contracts and domain models."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from blab.config import RadiatorConfig
from blab.mesh_clean import MeshQualityWarning
from blab.mesh_data import MeshData

PROVIDER_API_VERSION = 1


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
    """Serializable mesh source for the last artifact produced by a design."""

    output_dir: str
    mesh_path: str
    cleaned_mesh_path: str | None = None
    reduced_cleaned_mesh_path: str | None = None
    source_path: str | None = None
    mesh_data: MeshData | None = None
    reduced_mesh_data: MeshData | None = None
    mirror_axes: tuple[str, ...] = ()


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
    schema_version: int = PROVIDER_API_VERSION
    request_id: str = field(default_factory=lambda: uuid4().hex)
    project_revision: str | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PROVIDER_API_VERSION:
            raise ValueError(f"Unsupported provider API version: {self.schema_version}")
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("A generation request ID is required.")
        # Never lend editable project dictionaries to a background provider.
        object.__setattr__(self, "source", deepcopy(self.source))
        object.__setattr__(self, "configuration", deepcopy(self.configuration))
        object.__setattr__(self, "provider_options", deepcopy(self.provider_options))


@dataclass(frozen=True)
class GeneratedGeometry:
    """Provider-neutral geometry artifact consumed by mesh assembly."""

    provider_id: str
    output_dir: Path
    mesh_path: Path | None
    radiators: tuple[RadiatorConfig, ...]
    source_path: Path | None = None
    cleaned_mesh_path: Path | None = None
    reduced_cleaned_mesh_path: Path | None = None
    mirror_axes: tuple[str, ...] = ()
    quality_warning: MeshQualityWarning | None = None
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)
    mesh_data: MeshData | None = None
    reduced_mesh_data: MeshData | None = None

    def __post_init__(self):
        if (self.mesh_path is None) == (self.mesh_data is None):
            raise ValueError("GeneratedGeometry requires exactly one of mesh_path and mesh_data.")
        if self.reduced_mesh_data is not None and self.mesh_data is None:
            raise ValueError("reduced_mesh_data requires an in-memory primary mesh.")
        if self.mesh_data is not None and (self.cleaned_mesh_path is not None or self.reduced_cleaned_mesh_path is not None):
            raise ValueError("In-memory geometry cannot have file-backed variants.")

    def solver_mesh_data_for_symmetry(self, symmetry: str) -> MeshData | None:
        if symmetry == "off" or self.mesh_data is None:
            return self.mesh_data
        if self.reduced_mesh_data is not None:
            return self.reduced_mesh_data
        if self.mirror_axes:
            raise ValueError("In-memory mirrored geometry requires an explicit reduced_mesh_data variant.")
        return self.mesh_data

    @property
    def solver_mesh_path(self) -> Path | None:
        return self.cleaned_mesh_path or self.mesh_path

    def solver_mesh_path_for_symmetry(self, symmetry: str) -> Path | None:
        if str(symmetry or "off").strip().lower() == "off":
            return self.solver_mesh_path
        return self.reduced_cleaned_mesh_path or self.mesh_path

    def to_reference(self) -> GeneratedGeometryReference:
        return GeneratedGeometryReference(
            output_dir=str(self.output_dir),
            mesh_path="" if self.mesh_data is not None else str(self.mesh_path),
            cleaned_mesh_path=None if self.cleaned_mesh_path is None else str(self.cleaned_mesh_path),
            reduced_cleaned_mesh_path=(
                None if self.reduced_cleaned_mesh_path is None else str(self.reduced_cleaned_mesh_path)
            ),
            source_path=None if self.source_path is None else str(self.source_path),
            mirror_axes=self.mirror_axes,
            mesh_data=self.mesh_data,
            reduced_mesh_data=self.reduced_mesh_data,
        )


@dataclass(frozen=True)
class GenerationCompleted:
    request: GenerationRequest
    result: GeneratedGeometry
    configuration_patch: Mapping[str, Any] = field(default_factory=dict)
    legacy_response: bool = False


@dataclass(frozen=True)
class GenerationResponse:
    """Versioned provider response; omitted configuration preserves host values.

    Geometry may use an immutable in-memory snapshot or a mesh file.
    Request IDs are echoed, never generated by the provider.
    """

    request_id: str
    geometry: GeneratedGeometry
    configuration_patch: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = PROVIDER_API_VERSION


def complete_generation(request: GenerationRequest, response: GenerationResponse | GeneratedGeometry) -> GenerationCompleted:
    """Validate correlation and adapt legacy built-in providers at one boundary."""
    legacy = isinstance(response, GeneratedGeometry)
    if legacy:
        response = GenerationResponse(request_id=request.request_id, geometry=response)
    if not isinstance(response, GenerationResponse):
        raise ValueError("Provider must return GenerationResponse or GeneratedGeometry.")
    if type(response.schema_version) is not int or response.schema_version != PROVIDER_API_VERSION:
        raise ValueError(f"Unsupported provider response version: {response.schema_version}")
    if response.request_id != request.request_id:
        raise ValueError("Provider response does not match the generation request ID.")
    if response.geometry.provider_id != request.provider_id:
        raise ValueError("Provider response has a different provider ID.")
    from blab.generators.configuration import validate_patch

    patch = validate_patch(response.configuration_patch)
    return GenerationCompleted(request, deepcopy(response.geometry), patch, legacy_response=legacy)


class GeneratorSession(Protocol):
    def generate(
        self,
        *,
        status_callback: Callable[[str], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> GeneratedGeometry | GenerationResponse: ...

    def stop(self) -> None: ...


class GeneratorBackend(Protocol):
    provider_id: str
    label: str
    capabilities: GeneratorCapabilities

    def create_session(self, request: GenerationRequest) -> GeneratorSession: ...

    def restore(self, document: GeneratorDocument) -> GeneratedGeometry | None: ...
